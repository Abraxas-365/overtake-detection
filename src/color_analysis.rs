// src/color_analysis.rs
//
// v5.0: HSV-based color verification for road markings.
//
// Replaces the fragile RGB-ratio color check in lane_legality.rs with
// a proper HSV-space classifier that handles:
//   - Variable lighting (garúa fog, highland sun, dusk/dawn)
//   - Wet pavement reflections
//   - Camera white-balance shifts
//   - Faded/degraded paint (common in Peru)
//
// Peru MTC rules (Manual de Dispositivos de Control de Tránsito):
//   - YELLOW center lines → opposite-direction traffic (doble sentido)
//   - WHITE lane lines → same-direction lane separation
//   - RED edge lines → no-stopping zones (rare, urban only)

use tracing::debug;

// ============================================================================
// PUBLIC TYPES
// ============================================================================

/// Detected color of a road marking, based on pixel analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkingColor {
    White,
    Yellow,
    Red,
    Unknown,
}

impl MarkingColor {
    pub fn as_str(&self) -> &'static str {
        match self {
            MarkingColor::White => "WHITE",
            MarkingColor::Yellow => "YELLOW",
            MarkingColor::Red => "RED",
            MarkingColor::Unknown => "UNKNOWN",
        }
    }
}

/// Result of color analysis on a marking region.
#[derive(Debug, Clone)]
pub struct ColorAnalysisResult {
    pub detected_color: MarkingColor,
    /// Fraction of sampled pixels that voted for the winning color [0, 1]
    pub confidence: f32,
    /// Total pixels sampled (after filtering dark/shadow pixels)
    pub samples: u32,
    /// Average HSV values for diagnostics
    pub avg_hue: f32,
    pub avg_saturation: f32,
    pub avg_value: f32,
}

// ============================================================================
// HSV CONVERSION
// ============================================================================

/// Convert RGB to HSV.
/// Returns (H: 0-360, S: 0-100, V: 0-255).
#[inline]
pub fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let r_n = r / 255.0;
    let g_n = g / 255.0;
    let b_n = b / 255.0;

    let max = r_n.max(g_n).max(b_n);
    let min = r_n.min(g_n).min(b_n);
    let delta = max - min;

    // Hue
    let h = if delta < 1e-6 {
        0.0
    } else if (max - r_n).abs() < 1e-6 {
        60.0 * (((g_n - b_n) / delta) % 6.0)
    } else if (max - g_n).abs() < 1e-6 {
        60.0 * (((b_n - r_n) / delta) + 2.0)
    } else {
        60.0 * (((r_n - g_n) / delta) + 4.0)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };

    // Saturation (0-100)
    let s = if max < 1e-6 {
        0.0
    } else {
        (delta / max) * 100.0
    };

    // Value (0-255)
    let v = max * 255.0;

    (h, s, v)
}

// ============================================================================
// CORE COLOR CLASSIFIER
// ============================================================================

/// Classify the color of a road marking from a bounding box region.
///
/// Uses HSV-space analysis which is far more robust to lighting changes
/// than the previous RGB-ratio approach.
///
/// # Arguments
/// * `frame_rgb` - Raw RGB frame data
/// * `width`, `height` - Frame dimensions
/// * `bbox` - [x1, y1, x2, y2] bounding box of the marking
/// * `mask` - Optional segmentation mask (if available, samples only mask pixels)
/// * `mask_width`, `mask_height` - Mask dimensions (for coordinate mapping)
pub fn classify_marking_color(
    frame_rgb: &[u8],
    width: usize,
    height: usize,
    bbox: &[f32; 4],
    mask: Option<&[u8]>,
    mask_width: usize,
    mask_height: usize,
) -> ColorAnalysisResult {
    let x1 = (bbox[0] as usize).min(width.saturating_sub(1));
    let y1 = (bbox[1] as usize).min(height.saturating_sub(1));
    let x2 = (bbox[2] as usize).min(width.saturating_sub(1));
    let y2 = (bbox[3] as usize).min(height.saturating_sub(1));

    if x2 <= x1 || y2 <= y1 {
        return ColorAnalysisResult {
            detected_color: MarkingColor::Unknown,
            confidence: 0.0,
            samples: 0,
            avg_hue: 0.0,
            avg_saturation: 0.0,
            avg_value: 0.0,
        };
    }

    let bbox_w = x2 - x1;
    let bbox_h = y2 - y1;

    let mut votes_white: u32 = 0;
    let mut votes_yellow: u32 = 0;
    let mut votes_red: u32 = 0;
    let mut total_samples: u32 = 0;
    let mut sum_h: f64 = 0.0;
    let mut sum_s: f64 = 0.0;
    let mut sum_v: f64 = 0.0;

    // Sample every 2nd pixel for speed (still plenty of samples)
    let step = if bbox_w * bbox_h > 2000 { 3 } else { 2 };

    for y in (y1..=y2).step_by(step) {
        for x in (x1..=x2).step_by(step) {
            // If we have a mask, check that this pixel is part of the marking
            if let Some(m) = mask {
                if mask_width > 0 && mask_height > 0 {
                    let mx = ((x - x1) as f32 / bbox_w as f32 * mask_width as f32) as usize;
                    let my = ((y - y1) as f32 / bbox_h as f32 * mask_height as f32) as usize;
                    let mx = mx.min(mask_width - 1);
                    let my = my.min(mask_height - 1);
                    if m[my * mask_width + mx] == 0 {
                        continue; // Not part of marking mask
                    }
                }
            }

            let idx = (y * width + x) * 3;
            if idx + 2 >= frame_rgb.len() {
                continue;
            }

            let r = frame_rgb[idx] as f32;
            let g = frame_rgb[idx + 1] as f32;
            let b = frame_rgb[idx + 2] as f32;

            let (h, s, v) = rgb_to_hsv(r, g, b);

            // Skip dark pixels (shadow, pavement, underexposed)
            if v < 70.0 {
                continue;
            }

            total_samples += 1;
            sum_h += h as f64;
            sum_s += s as f64;
            sum_v += v as f64;

            // ----- YELLOW -----
            // Yellow road paint: hue 15-55°, moderate-high saturation, bright
            // Wide hue range accounts for faded yellow (shifts toward white/orange)
            if h >= 15.0 && h <= 55.0 && s > 25.0 && v > 90.0 {
                votes_yellow += 1;
            }
            // ----- WHITE -----
            // White paint: very low saturation, high brightness
            else if s < 30.0 && v > 150.0 {
                votes_white += 1;
            }
            // ----- RED -----
            // Red paint: hue <15° or >340°, moderate saturation
            else if (h < 15.0 || h > 340.0) && s > 30.0 && v > 80.0 {
                votes_red += 1;
            }
        }
    }

    if total_samples < 5 {
        return ColorAnalysisResult {
            detected_color: MarkingColor::Unknown,
            confidence: 0.0,
            samples: total_samples,
            avg_hue: 0.0,
            avg_saturation: 0.0,
            avg_value: 0.0,
        };
    }

    let avg_h = (sum_h / total_samples as f64) as f32;
    let avg_s = (sum_s / total_samples as f64) as f32;
    let avg_v = (sum_v / total_samples as f64) as f32;

    let yellow_ratio = votes_yellow as f32 / total_samples as f32;
    let white_ratio = votes_white as f32 / total_samples as f32;
    let red_ratio = votes_red as f32 / total_samples as f32;

    // Determine winning color — require at least 25% vote share
    let min_threshold = 0.25;
    let (detected_color, confidence) = if yellow_ratio >= min_threshold
        && yellow_ratio > white_ratio
        && yellow_ratio > red_ratio
    {
        (MarkingColor::Yellow, yellow_ratio)
    } else if white_ratio >= min_threshold && white_ratio > yellow_ratio && white_ratio > red_ratio
    {
        (MarkingColor::White, white_ratio)
    } else if red_ratio >= min_threshold && red_ratio > yellow_ratio && red_ratio > white_ratio {
        (MarkingColor::Red, red_ratio)
    } else {
        // No clear winner — use average HSV as tiebreaker
        let color = if avg_s < 25.0 && avg_v > 150.0 {
            MarkingColor::White
        } else if avg_h >= 15.0 && avg_h <= 55.0 && avg_s > 20.0 {
            MarkingColor::Yellow
        } else {
            MarkingColor::Unknown
        };
        let conf = yellow_ratio.max(white_ratio).max(red_ratio);
        (color, conf)
    };

    ColorAnalysisResult {
        detected_color,
        confidence,
        samples: total_samples,
        avg_hue: avg_h,
        avg_saturation: avg_s,
        avg_value: avg_v,
    }
}

// ============================================================================
// RECLASSIFICATION HELPERS
// ============================================================================

/// Given a YOLO class_id and the detected marking color, return the corrected
/// class_id if the color disagrees with what the model predicted.
///
/// Mapping (from lane_legality.rs):
///   4 → solid_single_white
///   5 → solid_single_yellow
///   6 → solid_single_red
///   7 → solid_double_white
///   8 → solid_double_yellow
///   9 → dashed_single_white
///  10 → dashed_single_yellow
pub fn correct_class_by_color(class_id: usize, detected_color: MarkingColor) -> usize {
    match detected_color {
        MarkingColor::Yellow => match class_id {
            4 => 5,  // solid_single_white → solid_single_yellow
            7 => 8,  // solid_double_white → solid_double_yellow
            9 => 10, // dashed_single_white → dashed_single_yellow
            _ => class_id,
        },
        MarkingColor::White => match class_id {
            5 => 4,  // solid_single_yellow → solid_single_white
            8 => 7,  // solid_double_yellow → solid_double_white
            10 => 9, // dashed_single_yellow → dashed_single_white
            _ => class_id,
        },
        MarkingColor::Red => match class_id {
            4 | 5 => 6, // solid_single_* → solid_single_red
            _ => class_id,
        },
        MarkingColor::Unknown => class_id,
    }
}

/// Full class_id → name mapping (mirrors lane_legality.rs).
pub fn class_id_to_name(class_id: usize) -> &'static str {
    match class_id {
        4 => "solid_single_white",
        5 => "solid_single_yellow",
        6 => "solid_single_red",
        7 => "solid_double_white",
        8 => "solid_double_yellow",
        9 => "dashed_single_white",
        10 => "dashed_single_yellow",
        99 => "mixed_double_yellow",
        _ => "other_marking",
    }
}

// ============================================================================
// DROP-IN REPLACEMENT FOR verify_line_color
// ============================================================================

/// **Drop-in replacement** for the `verify_line_color` function in `lane_legality.rs`.
///
/// Call this after YOLO postprocessing on each DetectedRoadMarking.
/// It uses HSV analysis (bidirectional) instead of the old RGB-ratio approach.
///
/// Unlike the original, this corrects BOTH directions:
///   - white → yellow (original behavior)
///   - yellow → white (NEW: handles warm lighting misclassification)
///   - any solid → red (NEW: for Peruvian red edge markings)
pub fn verify_and_correct_line_color(
    frame_rgb: &[u8],
    frame_width: usize,
    frame_height: usize,
    class_id: &mut usize,
    class_name: &mut String,
    bbox: &[f32; 4],
    mask: Option<&[u8]>,
    mask_width: usize,
    mask_height: usize,
) -> bool {
    // Only re-check lane line classes (not arrows, crosswalks, text, etc.)
    let is_lane_line = matches!(*class_id, 4 | 5 | 6 | 7 | 8 | 9 | 10);
    if !is_lane_line {
        return false;
    }

    let result = classify_marking_color(
        frame_rgb,
        frame_width,
        frame_height,
        bbox,
        mask,
        mask_width,
        mask_height,
    );

    // Only reclassify if we have enough confidence
    if result.confidence < 0.30 || result.samples < 10 {
        return false;
    }

    let corrected = correct_class_by_color(*class_id, result.detected_color);

    if corrected != *class_id {
        debug!(
            "🎨 HSV color correction: {} → {} (H={:.0}° S={:.0} V={:.0}, {}% conf, {} samples)",
            class_id_to_name(*class_id),
            class_id_to_name(corrected),
            result.avg_hue,
            result.avg_saturation,
            result.avg_value,
            (result.confidence * 100.0) as u32,
            result.samples,
        );
        *class_id = corrected;
        *class_name = class_id_to_name(corrected).to_string();
        return true;
    }

    false
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_hsv_red() {
        let (h, s, v) = rgb_to_hsv(255.0, 0.0, 0.0);
        assert!((h - 0.0).abs() < 1.0);
        assert!((s - 100.0).abs() < 1.0);
        assert!((v - 255.0).abs() < 1.0);
    }

    #[test]
    fn test_rgb_to_hsv_yellow() {
        let (h, s, v) = rgb_to_hsv(255.0, 255.0, 0.0);
        assert!((h - 60.0).abs() < 1.0);
        assert!((s - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_rgb_to_hsv_white() {
        let (h, s, v) = rgb_to_hsv(255.0, 255.0, 255.0);
        assert!(s < 1.0);
        assert!((v - 255.0).abs() < 1.0);
    }

    #[test]
    fn test_classify_yellow_marking() {
        // Create a small image of yellow-ish pixels (R=220, G=200, B=50)
        let w = 20;
        let h = 20;
        let mut img = vec![0u8; w * h * 3];
        for i in 0..w * h {
            img[i * 3] = 220;
            img[i * 3 + 1] = 200;
            img[i * 3 + 2] = 50;
        }
        let bbox = [0.0f32, 0.0, w as f32, h as f32];
        let result = classify_marking_color(&img, w, h, &bbox, None, 0, 0);
        assert_eq!(result.detected_color, MarkingColor::Yellow);
    }

    #[test]
    fn test_classify_white_marking() {
        let w = 20;
        let h = 20;
        let mut img = vec![0u8; w * h * 3];
        for i in 0..w * h {
            img[i * 3] = 230;
            img[i * 3 + 1] = 230;
            img[i * 3 + 2] = 230;
        }
        let bbox = [0.0f32, 0.0, w as f32, h as f32];
        let result = classify_marking_color(&img, w, h, &bbox, None, 0, 0);
        assert_eq!(result.detected_color, MarkingColor::White);
    }

    #[test]
    fn test_correct_white_to_yellow() {
        let corrected = correct_class_by_color(4, MarkingColor::Yellow); // solid_white → solid_yellow
        assert_eq!(corrected, 5);
    }

    #[test]
    fn test_correct_yellow_to_white() {
        let corrected = correct_class_by_color(5, MarkingColor::White); // solid_yellow → solid_white
        assert_eq!(corrected, 4);
    }

    #[test]
    fn test_no_correction_when_matching() {
        let corrected = correct_class_by_color(5, MarkingColor::Yellow); // yellow stays yellow
        assert_eq!(corrected, 5);
    }
}
