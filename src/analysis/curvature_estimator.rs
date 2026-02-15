// src/analysis/curvature_estimator.rs
//
// v4.13: Polynomial curvature estimation from YOLO-seg mask geometry.
//
// Instead of collapsing each lane marking mask to a single center_x scalar,
// this module extracts the mask's SPINE (ordered centerline points), fits a
// 2nd-degree polynomial x = ay² + by + c, and computes per-boundary curvature.
//
// Why this matters:
//   - On a road curve, BOTH lane boundaries curve in the same direction.
//     Their polynomial `a` coefficients have the same sign and similar magnitude.
//   - On a lane change, one boundary approaches while the other recedes.
//     Their `a` coefficients diverge.
//
// This provides a direct geometric signal that replaces the indirect boundary
// coherence heuristic (v4.11). Boundary coherence measures frame-to-frame
// co-movement of the center_x scalars — it's a velocity proxy. Polynomial
// curvature measures the SHAPE of the marking in a single frame — it's a
// position-domain measurement, immune to temporal aliasing and smoothing lag.
//
// Coordinate system:
//   - YOLO-seg mask is 160×160, representing the full 640×640 input image
//   - Input image is letterboxed: scale + pad_x/pad_y
//   - Mask pixel (mx, my) → original image: ((mx*4 - pad_x)/scale, (my*4 - pad_y)/scale)
//   - Polynomial uses IMAGE y as independent variable (increases downward)
//     and IMAGE x as dependent variable: x(y) = a·y² + b·y + c
//   - y is normalized to [0,1] before fitting for numerical stability
//
// References:
//   - Udacity Advanced Lane Finding: polynomial fit on bird's-eye warped lanes
//   - IPM-based curvature: R = f²/κ where f is focal length
//   - Bayesian curved lane estimation (Hal 2020): regional hyperbola fitting
//
// We skip the full IPM transform (requires camera calibration) and work
// directly in perspective image space. The curvature comparison between
// left and right boundaries is still valid because perspective distortion
// affects both boundaries equally for lane-width-scale separations.

use tracing::debug;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Minimum spine points required for a reliable polynomial fit.
/// A quadratic needs ≥3, but we require more for noise tolerance.
const MIN_SPINE_POINTS: usize = 8;

/// Minimum vertical span of spine points (in normalized [0,1] y-space).
/// Prevents fitting on a tiny cluster that doesn't represent the boundary shape.
const MIN_SPINE_VERTICAL_SPAN: f32 = 0.15;

/// Maximum residual (RMSE in pixels) for a polynomial fit to be trusted.
/// If the mask pixels are too scattered, the quadratic doesn't describe the shape.
const MAX_FIT_RMSE_PX: f32 = 25.0;

/// Curvature agreement threshold for road curve detection.
/// When agreement > this, both boundaries are bending the same way → road curve.
const CURVE_AGREEMENT_THRESHOLD: f32 = 0.50;

/// Minimum |a| coefficient (in image-pixel units) to count as curved at all.
/// Below this, the boundary is effectively straight.
const MIN_CURVATURE_MAGNITUDE: f32 = 1e-5;

// ============================================================================
// TYPES
// ============================================================================

/// Direction the road is curving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurveDirection {
    Left,
    Right,
    Straight,
}

impl CurveDirection {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Left => "LEFT",
            Self::Right => "RIGHT",
            Self::Straight => "STRAIGHT",
        }
    }
}

/// Polynomial fit for a single lane boundary: x = a·y² + b·y + c
/// where y is normalized to [0,1] over the spine's vertical extent.
#[derive(Debug, Clone, Copy)]
pub struct LanePolynomial {
    /// Quadratic coefficient — sign determines curvature direction.
    /// Positive a: boundary curves rightward as y increases (downward).
    /// Negative a: boundary curves leftward.
    pub a: f32,
    /// Linear coefficient — slope at top of spine region.
    pub b: f32,
    /// Constant — x-intercept at top of spine region.
    pub c: f32,
    /// RMS residual of the fit in original-image pixels.
    pub rmse_px: f32,
    /// Number of spine points used for fitting.
    pub num_points: usize,
    /// Vertical span in original-image pixels.
    pub vertical_span_px: f32,
}

/// Per-frame curvature estimate from both lane boundaries.
#[derive(Debug, Clone, Copy)]
pub struct CurvatureEstimate {
    /// Polynomial fit for the left boundary (if available).
    pub left_poly: Option<LanePolynomial>,
    /// Polynomial fit for the right boundary (if available).
    pub right_poly: Option<LanePolynomial>,
    /// Mean curvature (`a` coefficient) across both boundaries.
    /// Positive = curving right, negative = curving left.
    pub mean_curvature: f32,
    /// Agreement between left and right curvatures [0, 1].
    /// 1.0 = identical curvature (road curve).
    /// 0.0 = completely divergent (lane change or noise).
    pub curvature_agreement: f32,
    /// Whether this looks like a road curve (vs lane change or straight).
    pub is_curve: bool,
    /// Direction of the road curve (if any).
    pub curve_direction: CurveDirection,
    /// Overall confidence in the estimate [0, 1].
    /// Based on fit quality, point count, and vertical span.
    pub confidence: f32,
}

impl Default for CurvatureEstimate {
    fn default() -> Self {
        Self {
            left_poly: None,
            right_poly: None,
            mean_curvature: 0.0,
            curvature_agreement: 0.0,
            is_curve: false,
            curve_direction: CurveDirection::Straight,
            confidence: 0.0,
        }
    }
}

// ============================================================================
// MASK SPINE EXTRACTION
// ============================================================================

/// Extract the spine (centerline) of a lane marking from its segmentation mask.
///
/// For each row of the mask within the detection's bounding box region,
/// computes the centroid of active pixels and maps it back to original
/// image coordinates.
///
/// Returns ordered (x, y) points in original image space, sorted by y (top→bottom).
///
/// # Arguments
/// * `mask` — Binary mask (160×160), 255 = active, 0 = background
/// * `mask_w` / `mask_h` — Mask dimensions (both 160)
/// * `bbox` — Bounding box in original image coordinates [x1, y1, x2, y2]
/// * `scale` — Letterbox scale factor (original → 640)
/// * `pad_x` / `pad_y` — Letterbox padding in 640-space
pub fn extract_mask_spine(
    mask: &[u8],
    mask_w: usize,
    mask_h: usize,
    bbox: &[f32; 4],
    scale: f32,
    pad_x: f32,
    pad_y: f32,
) -> Vec<(f32, f32)> {
    // Map bbox from original image → 640-space → 160-space (mask coordinates)
    let ratio = mask_w as f32 / (mask_w as f32 * 4.0); // = 0.25 (160/640)

    let mask_x1 = ((bbox[0] * scale + pad_x) * ratio).floor().max(0.0) as usize;
    let mask_y1 = ((bbox[1] * scale + pad_y) * ratio).floor().max(0.0) as usize;
    let mask_x2 = ((bbox[2] * scale + pad_x) * ratio)
        .ceil()
        .min(mask_w as f32 - 1.0) as usize;
    let mask_y2 = ((bbox[3] * scale + pad_y) * ratio)
        .ceil()
        .min(mask_h as f32 - 1.0) as usize;

    let mut spine = Vec::with_capacity(mask_y2.saturating_sub(mask_y1) + 1);

    for my in mask_y1..=mask_y2 {
        let mut sum_x = 0.0f64;
        let mut count = 0u32;

        for mx in mask_x1..=mask_x2 {
            let idx = my * mask_w + mx;
            if idx < mask.len() && mask[idx] == 255 {
                sum_x += mx as f64;
                count += 1;
            }
        }

        if count > 0 {
            let centroid_mx = sum_x / count as f64;
            // Convert back: mask → 640-space → original image
            let orig_x = ((centroid_mx * 4.0 - pad_x as f64) / scale as f64) as f32;
            let orig_y = ((my as f64 * 4.0 - pad_y as f64) / scale as f64) as f32;
            spine.push((orig_x, orig_y));
        }
    }

    spine
}

// ============================================================================
// POLYNOMIAL FITTING (Least Squares)
// ============================================================================

/// Fit a 2nd-degree polynomial x = a·y² + b·y + c to a set of (x, y) points.
///
/// The y values are normalized to [0, 1] internally for numerical stability.
/// The returned coefficients are in NORMALIZED y-space, with `rmse_px` and
/// `vertical_span_px` in original-image pixel units.
///
/// Returns None if:
/// - Fewer than MIN_SPINE_POINTS points
/// - Vertical span too small
/// - System is singular (degenerate geometry)
/// - RMSE exceeds MAX_FIT_RMSE_PX
pub fn fit_lane_polynomial(spine: &[(f32, f32)]) -> Option<LanePolynomial> {
    if spine.len() < MIN_SPINE_POINTS {
        return None;
    }

    // Compute y-range for normalization
    let y_min = spine.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
    let y_max = spine.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);
    let y_range = y_max - y_min;

    if y_range < 1.0 {
        return None; // Degenerate — all points at same y
    }

    let vertical_span_norm = y_range; // in pixels
    let norm_span = vertical_span_norm / y_max.max(1.0);
    if norm_span < MIN_SPINE_VERTICAL_SPAN {
        return None;
    }

    // Normalize y to [0, 1]
    let n = spine.len() as f64;
    let s0: f64 = n;
    let mut s1: f64 = 0.0;
    let mut s2: f64 = 0.0;
    let mut s3: f64 = 0.0;
    let mut s4: f64 = 0.0;
    let mut sx0: f64 = 0.0;
    let mut sx1: f64 = 0.0;
    let mut sx2: f64 = 0.0;

    for &(x, y) in spine {
        let yn = ((y - y_min) / y_range) as f64; // normalized y ∈ [0, 1]
        let xd = x as f64;
        let yn2 = yn * yn;
        let yn3 = yn2 * yn;
        let yn4 = yn3 * yn;

        s1 += yn;
        s2 += yn2;
        s3 += yn3;
        s4 += yn4;
        sx0 += xd;
        sx1 += xd * yn;
        sx2 += xd * yn2;
    }

    // Solve 3×3 system via Gaussian elimination:
    //   | s4 s3 s2 | | a |   | sx2 |
    //   | s3 s2 s1 | | b | = | sx1 |
    //   | s2 s1 s0 | | c |   | sx0 |
    let (a, b, c) = solve_3x3([s4, s3, s2, s3, s2, s1, s2, s1, s0], [sx2, sx1, sx0])?;

    // Compute RMSE in original pixel space
    let mut sse = 0.0f64;
    for &(x, y) in spine {
        let yn = ((y - y_min) / y_range) as f64;
        let predicted = a * yn * yn + b * yn + c;
        let residual = x as f64 - predicted;
        sse += residual * residual;
    }
    let rmse = (sse / n).sqrt() as f32;

    if rmse > MAX_FIT_RMSE_PX {
        debug!(
            "  ⚠️ Poly fit rejected: RMSE={:.1}px > {:.1}px max ({} points, span={:.0}px)",
            rmse,
            MAX_FIT_RMSE_PX,
            spine.len(),
            y_range
        );
        return None;
    }

    Some(LanePolynomial {
        a: a as f32,
        b: b as f32,
        c: c as f32,
        rmse_px: rmse,
        num_points: spine.len(),
        vertical_span_px: y_range,
    })
}

/// Solve a 3×3 linear system Ax = b using Gaussian elimination with partial pivoting.
/// Matrix is row-major: [a00, a01, a02, a10, a11, a12, a20, a21, a22].
/// Returns None if the system is singular.
fn solve_3x3(mat: [f64; 9], rhs: [f64; 3]) -> Option<(f64, f64, f64)> {
    // Augmented matrix [A|b]
    let mut m = [
        [mat[0], mat[1], mat[2], rhs[0]],
        [mat[3], mat[4], mat[5], rhs[1]],
        [mat[6], mat[7], mat[8], rhs[2]],
    ];

    // Forward elimination with partial pivoting
    for col in 0..3 {
        // Find pivot
        let mut max_val = m[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..3 {
            if m[row][col].abs() > max_val {
                max_val = m[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-12 {
            return None; // Singular
        }

        // Swap rows
        if max_row != col {
            m.swap(col, max_row);
        }

        // Eliminate below
        for row in (col + 1)..3 {
            let factor = m[row][col] / m[col][col];
            for j in col..4 {
                m[row][j] -= factor * m[col][j];
            }
        }
    }

    // Back substitution
    if m[2][2].abs() < 1e-12 {
        return None;
    }
    let c = m[2][3] / m[2][2];
    let b = (m[1][3] - m[1][2] * c) / m[1][1];
    let a = (m[0][3] - m[0][2] * c - m[0][1] * b) / m[0][0];

    if a.is_finite() && b.is_finite() && c.is_finite() {
        Some((a, b, c))
    } else {
        None
    }
}

// ============================================================================
// CURVATURE ESTIMATION (Top-Level)
// ============================================================================

/// Parameters needed for mask→image coordinate conversion.
#[derive(Debug, Clone, Copy)]
pub struct MaskTransformParams {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
}

impl MaskTransformParams {
    /// Recompute from original image dimensions and model input size (640).
    pub fn from_image_dims(orig_w: usize, orig_h: usize) -> Self {
        let target = 640.0f32;
        let scale = (target / orig_w as f32).min(target / orig_h as f32);
        let pad_x = (target - orig_w as f32 * scale) / 2.0;
        let pad_y = (target - orig_h as f32 * scale) / 2.0;
        Self {
            scale,
            pad_x,
            pad_y,
        }
    }
}

/// Mask data for a single detected lane marking.
/// Contains the fields needed for spine extraction without requiring
/// the full DetectedRoadMarking struct (keeps this module decoupled).
pub struct MaskInput<'a> {
    pub mask: &'a [u8],
    pub mask_w: usize,
    pub mask_h: usize,
    pub bbox: [f32; 4],
    pub center_x: f32, // bbox center-x in original image coords
}

/// Estimate road curvature from a pair of lane boundary masks.
///
/// # Arguments
/// * `left_mask` — Mask data for the left boundary marking
/// * `right_mask` — Mask data for the right boundary marking
/// * `params` — Coordinate transform parameters
///
/// Returns a CurvatureEstimate even if only one boundary has a valid fit,
/// though confidence will be lower.
pub fn estimate_curvature_from_masks(
    left_mask: Option<&MaskInput>,
    right_mask: Option<&MaskInput>,
    params: &MaskTransformParams,
) -> CurvatureEstimate {
    let left_poly = left_mask.and_then(|m| {
        let spine = extract_mask_spine(
            m.mask,
            m.mask_w,
            m.mask_h,
            &m.bbox,
            params.scale,
            params.pad_x,
            params.pad_y,
        );
        let poly = fit_lane_polynomial(&spine);
        if let Some(ref p) = poly {
            debug!(
                "  📐 Left poly: a={:.6} b={:.2} c={:.1} | RMSE={:.1}px | pts={} span={:.0}px",
                p.a, p.b, p.c, p.rmse_px, p.num_points, p.vertical_span_px
            );
        }
        poly
    });

    let right_poly = right_mask.and_then(|m| {
        let spine = extract_mask_spine(
            m.mask,
            m.mask_w,
            m.mask_h,
            &m.bbox,
            params.scale,
            params.pad_x,
            params.pad_y,
        );
        let poly = fit_lane_polynomial(&spine);
        if let Some(ref p) = poly {
            debug!(
                "  📐 Right poly: a={:.6} b={:.2} c={:.1} | RMSE={:.1}px | pts={} span={:.0}px",
                p.a, p.b, p.c, p.rmse_px, p.num_points, p.vertical_span_px
            );
        }
        poly
    });

    compute_curvature(left_poly, right_poly)
}

/// Compute curvature estimate from polynomial fits of both boundaries.
fn compute_curvature(
    left_poly: Option<LanePolynomial>,
    right_poly: Option<LanePolynomial>,
) -> CurvatureEstimate {
    match (left_poly, right_poly) {
        (Some(lp), Some(rp)) => {
            let mean_a = (lp.a + rp.a) / 2.0;

            // Agreement: how similar are the curvatures?
            // Uses signed comparison so opposing curvatures give low agreement.
            let agreement = curvature_agreement(lp.a, rp.a);

            // Confidence from fit quality
            let left_conf = fit_confidence(&lp);
            let right_conf = fit_confidence(&rp);
            let confidence = (left_conf + right_conf) / 2.0;

            let is_curve = agreement > CURVE_AGREEMENT_THRESHOLD
                && mean_a.abs() > MIN_CURVATURE_MAGNITUDE
                && confidence > 0.3;

            let direction = if !is_curve || mean_a.abs() <= MIN_CURVATURE_MAGNITUDE {
                CurveDirection::Straight
            } else if mean_a > 0.0 {
                CurveDirection::Right
            } else {
                CurveDirection::Left
            };

            debug!(
                "  📐 Curvature: mean_a={:.6} agreement={:.2} conf={:.2} → {} ({})",
                mean_a,
                agreement,
                confidence,
                if is_curve { "CURVE" } else { "NOT_CURVE" },
                direction.as_str()
            );

            CurvatureEstimate {
                left_poly: Some(lp),
                right_poly: Some(rp),
                mean_curvature: mean_a,
                curvature_agreement: agreement,
                is_curve,
                curve_direction: direction,
                confidence,
            }
        }

        // Only one boundary — lower confidence, can still detect curvature
        (Some(poly), None) | (None, Some(poly)) => {
            let side = if left_poly.is_some() { "left" } else { "right" };
            let conf = fit_confidence(&poly) * 0.5; // halved for single-boundary

            let is_curve = poly.a.abs() > MIN_CURVATURE_MAGNITUDE * 5.0 && conf > 0.2;
            let direction = if !is_curve {
                CurveDirection::Straight
            } else if poly.a > 0.0 {
                CurveDirection::Right
            } else {
                CurveDirection::Left
            };

            debug!(
                "  📐 Curvature ({}‑only): a={:.6} conf={:.2} → {} ({})",
                side,
                poly.a,
                conf,
                if is_curve { "CURVE" } else { "NOT_CURVE" },
                direction.as_str()
            );

            CurvatureEstimate {
                left_poly,
                right_poly,
                mean_curvature: poly.a,
                curvature_agreement: 0.0, // can't compute agreement with one boundary
                is_curve,
                curve_direction: direction,
                confidence: conf,
            }
        }

        (None, None) => CurvatureEstimate::default(),
    }
}

/// Compute agreement between two curvature values.
///
/// Returns 1.0 when curvatures are identical in sign and magnitude.
/// Returns 0.0 when curvatures are opposite or very different.
///
/// Uses a signed metric: if a_left = -0.001 and a_right = +0.001,
/// agreement should be low (boundaries curving in opposite directions).
fn curvature_agreement(a_left: f32, a_right: f32) -> f32 {
    let abs_left = a_left.abs();
    let abs_right = a_right.abs();
    let max_abs = abs_left.max(abs_right);

    if max_abs < MIN_CURVATURE_MAGNITUDE {
        // Both essentially straight — perfect agreement (no curvature to disagree on)
        return 1.0;
    }

    // Sign agreement: same sign → positive contribution, opposite → negative
    let sign_factor = if a_left * a_right >= 0.0 { 1.0 } else { -1.0 };

    // Magnitude ratio: how similar in absolute value
    let min_abs = abs_left.min(abs_right);
    let mag_ratio = min_abs / max_abs; // ∈ [0, 1]

    // Combined: sign_factor * mag_ratio ∈ [-1, 1], mapped to [0, 1]
    let raw = sign_factor * mag_ratio;
    ((raw + 1.0) / 2.0).clamp(0.0, 1.0)
}

/// Confidence in a single polynomial fit based on quality metrics.
fn fit_confidence(poly: &LanePolynomial) -> f32 {
    // More points → more confident (saturates around 40)
    let point_score = (poly.num_points as f32 / 40.0).min(1.0);

    // Larger vertical span → more confident (saturates around 300px)
    let span_score = (poly.vertical_span_px / 300.0).min(1.0);

    // Lower RMSE → more confident (0 RMSE = 1.0, MAX_FIT_RMSE = 0.0)
    let rmse_score = 1.0 - (poly.rmse_px / MAX_FIT_RMSE_PX).min(1.0);

    // Geometric mean emphasizes that ALL factors need to be decent
    (point_score * span_score * rmse_score).powf(1.0 / 3.0)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_3x3_identity() {
        // Solve Ix = [1, 2, 3]
        let result = solve_3x3(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
        );
        let (a, b, c) = result.unwrap();
        assert!((a - 1.0).abs() < 1e-10);
        assert!((b - 2.0).abs() < 1e-10);
        assert!((c - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_3x3_singular() {
        // Two identical rows → singular
        let result = solve_3x3(
            [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.0, 1.0, 2.0],
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_fit_straight_line() {
        // x = 100 (constant) for y = 0..100 → a ≈ 0, c ≈ 100
        let spine: Vec<(f32, f32)> = (0..50).map(|i| (100.0, i as f32 * 2.0 + 300.0)).collect();
        let poly = fit_lane_polynomial(&spine).unwrap();
        assert!(
            poly.a.abs() < 0.01,
            "Straight line should have a ≈ 0, got {}",
            poly.a
        );
        assert!(poly.rmse_px < 1.0);
    }

    #[test]
    fn test_fit_parabola() {
        // x = 0.001 * y² + 100, y ∈ [300, 700] → a should be positive
        let spine: Vec<(f32, f32)> = (0..50)
            .map(|i| {
                let y = 300.0 + i as f32 * 8.0;
                let yn = (y - 300.0) / 400.0; // normalized
                let x = 50.0 * yn * yn + 100.0; // a=50 in normalized space
                (x, y)
            })
            .collect();
        let poly = fit_lane_polynomial(&spine).unwrap();
        assert!(
            poly.a > 0.0,
            "Rightward curve should have a > 0, got {}",
            poly.a
        );
        assert!(poly.rmse_px < 1.0);
    }

    #[test]
    fn test_curvature_agreement_same() {
        let ag = curvature_agreement(0.001, 0.001);
        assert!(ag > 0.95, "Identical curvatures should agree: {}", ag);
    }

    #[test]
    fn test_curvature_agreement_opposite() {
        let ag = curvature_agreement(0.001, -0.001);
        assert!(ag < 0.05, "Opposite curvatures should disagree: {}", ag);
    }

    #[test]
    fn test_curvature_agreement_similar() {
        let ag = curvature_agreement(0.001, 0.0008);
        assert!(ag > 0.7, "Similar curvatures should mostly agree: {}", ag);
    }

    #[test]
    fn test_insufficient_points() {
        let spine: Vec<(f32, f32)> = (0..3).map(|i| (100.0, i as f32 * 10.0)).collect();
        assert!(fit_lane_polynomial(&spine).is_none());
    }

    #[test]
    fn test_compute_curvature_both_curving_right() {
        let lp = LanePolynomial {
            a: 50.0,
            b: 2.0,
            c: 100.0,
            rmse_px: 2.0,
            num_points: 40,
            vertical_span_px: 350.0,
        };
        let rp = LanePolynomial {
            a: 48.0,
            b: 3.0,
            c: 500.0,
            rmse_px: 3.0,
            num_points: 35,
            vertical_span_px: 320.0,
        };
        let est = compute_curvature(Some(lp), Some(rp));
        assert!(
            est.is_curve,
            "Both curving right should be detected as curve"
        );
        assert_eq!(est.curve_direction, CurveDirection::Right);
        assert!(est.curvature_agreement > 0.8);
    }

    #[test]
    fn test_compute_curvature_diverging() {
        let lp = LanePolynomial {
            a: 50.0,
            b: 2.0,
            c: 100.0,
            rmse_px: 2.0,
            num_points: 40,
            vertical_span_px: 350.0,
        };
        let rp = LanePolynomial {
            a: -45.0,
            b: 3.0,
            c: 500.0,
            rmse_px: 3.0,
            num_points: 35,
            vertical_span_px: 320.0,
        };
        let est = compute_curvature(Some(lp), Some(rp));
        assert!(
            !est.is_curve,
            "Opposing curvatures should NOT be detected as curve"
        );
        assert!(est.curvature_agreement < 0.3);
    }
}
