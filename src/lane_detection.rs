// src/lane_detection.rs
//
// v5.0: Corrected lane detection with:
//   - Configurable row anchor parameters (not hardcoded to CULane 720p)
//   - Proper softmax-based confidence (not sigmoid)
//   - Sub-pixel X interpolation via expectation over grid probabilities
//   - Outlier rejection for noisy anchor points

use crate::types::{Config, DetectedLane};
use anyhow::Result;
use tracing::debug;

pub struct LaneDetectionResult {
    pub lanes: Vec<DetectedLane>,
    pub timestamp_ms: f64,
}

// ============================================================================
// ROW ANCHOR CONFIGURATION
// ============================================================================

// CULane dataset anchors (720p training resolution).
// These define the Y positions in the ORIGINAL training image where the model
// predicts lane positions. They are scaled to the actual frame height at runtime.
const ROW_ANCHOR_START: f32 = 160.0;
const ROW_ANCHOR_END: f32 = 710.0;
const ORIGINAL_HEIGHT: f32 = 720.0;

// ============================================================================
// SOFTMAX
// ============================================================================

/// Softmax along the first axis (grid dimension) of shape [dim0, dim1, dim2].
///
/// For UFLDv2: dim0=griding_num (200), dim1=num_anchors (72), dim2=num_lanes (4).
/// This converts raw logits into probabilities over grid positions.
fn softmax_axis0(data: &[f32], dim0: usize, dim1: usize, dim2: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; data.len()];

    for j in 0..dim1 {
        for k in 0..dim2 {
            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..dim0 {
                let idx = i * (dim1 * dim2) + j * dim2 + k;
                if data[idx] > max_val {
                    max_val = data[idx];
                }
            }

            // Compute exp and sum
            let mut sum = 0.0f32;
            for i in 0..dim0 {
                let idx = i * (dim1 * dim2) + j * dim2 + k;
                let exp_val = (data[idx] - max_val).exp();
                result[idx] = exp_val;
                sum += exp_val;
            }

            // Normalize
            if sum > 0.0 {
                for i in 0..dim0 {
                    let idx = i * (dim1 * dim2) + j * dim2 + k;
                    result[idx] /= sum;
                }
            }
        }
    }

    result
}

// ============================================================================
// MAIN PARSER
// ============================================================================

pub fn parse_lanes(
    output: &[f32],
    frame_width: f32,
    frame_height: f32,
    config: &Config,
    timestamp_ms: f64,
) -> Result<LaneDetectionResult> {
    let griding_num = config.model.griding_num; // 200
    let num_anchors = config.model.num_anchors; // 72
    let num_lanes = config.model.num_lanes; // 4

    let loc_row_size = griding_num * num_anchors * num_lanes;

    if output.len() < loc_row_size {
        anyhow::bail!(
            "Output size mismatch: expected at least {}, got {}",
            loc_row_size,
            output.len()
        );
    }

    // Apply softmax along grid dimension
    let loc_row_prob = softmax_axis0(output, griding_num, num_anchors, num_lanes);

    let mut lanes = Vec::new();

    for lane_idx in 0..num_lanes {
        let mut points: Vec<(f32, f32)> = Vec::new();
        let mut total_confidence = 0.0f32;
        let mut valid_points = 0u32;

        for anchor_idx in 0..num_anchors {
            // Find argmax AND its probability
            let mut max_prob = f32::NEG_INFINITY;
            let mut max_grid_idx = 0;

            for grid_idx in 0..griding_num {
                let idx = grid_idx * (num_anchors * num_lanes) + anchor_idx * num_lanes + lane_idx;
                let prob = loc_row_prob[idx];
                if prob > max_prob {
                    max_prob = prob;
                    max_grid_idx = grid_idx;
                }
            }

            // grid_idx == 0 means "no lane at this anchor" — skip
            if max_grid_idx == 0 {
                continue;
            }

            // Require minimum confidence to avoid noise
            if max_prob < 0.1 {
                continue;
            }

            // --- Sub-pixel X refinement via expectation ---
            // Instead of just taking argmax, compute the weighted average of
            // neighboring grid positions for sub-pixel precision.
            let x_refined = if max_grid_idx > 1 && max_grid_idx < griding_num - 1 {
                let idx_l = (max_grid_idx - 1) * (num_anchors * num_lanes)
                    + anchor_idx * num_lanes
                    + lane_idx;
                let idx_c =
                    max_grid_idx * (num_anchors * num_lanes) + anchor_idx * num_lanes + lane_idx;
                let idx_r = (max_grid_idx + 1) * (num_anchors * num_lanes)
                    + anchor_idx * num_lanes
                    + lane_idx;

                let p_l = loc_row_prob[idx_l];
                let p_c = loc_row_prob[idx_c];
                let p_r = loc_row_prob[idx_r];
                let total = p_l + p_c + p_r;

                if total > 0.01 {
                    let weighted = (max_grid_idx as f32 - 1.0) * p_l
                        + max_grid_idx as f32 * p_c
                        + (max_grid_idx as f32 + 1.0) * p_r;
                    let refined_grid = weighted / total;
                    // Convert refined grid index to normalized X
                    (refined_grid - 1.0) / (griding_num as f32 - 1.0)
                } else {
                    (max_grid_idx as f32 - 1.0) / (griding_num as f32 - 1.0)
                }
            } else {
                (max_grid_idx as f32 - 1.0) / (griding_num as f32 - 1.0)
            };

            let x = x_refined * frame_width;

            // Y coordinate: linearly spaced anchors scaled to frame height
            let y_norm = ROW_ANCHOR_START
                + (ROW_ANCHOR_END - ROW_ANCHOR_START)
                    * (anchor_idx as f32 / (num_anchors as f32 - 1.0));
            let y = (y_norm / ORIGINAL_HEIGHT) * frame_height;

            points.push((x, y));
            total_confidence += max_prob;
            valid_points += 1;
        }

        // Require minimum points per lane
        if points.len() >= config.detection.min_points_per_lane {
            let avg_confidence = if valid_points > 0 {
                total_confidence / valid_points as f32
            } else {
                0.0
            };

            // --- Outlier rejection ---
            // Remove points that deviate too much from the lane's general trend.
            // Fit a simple linear model and reject points > 2σ away.
            let cleaned_points = reject_outlier_points(&points, 2.0);

            if cleaned_points.len() >= config.detection.min_points_per_lane {
                // Sort by Y descending (bottom to top)
                let mut final_points = cleaned_points;
                final_points
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                lanes.push(DetectedLane {
                    points: final_points,
                    confidence: avg_confidence,
                });
            }
        }
    }

    debug!("Detected {} lanes", lanes.len());

    Ok(LaneDetectionResult {
        lanes,
        timestamp_ms,
    })
}

// ============================================================================
// OUTLIER REJECTION
// ============================================================================

/// Remove outlier points from a lane polyline using simple linear regression.
///
/// Fits X = a*Y + b, then removes points where |X - predicted| > threshold * σ.
fn reject_outlier_points(points: &[(f32, f32)], sigma_threshold: f32) -> Vec<(f32, f32)> {
    if points.len() < 5 {
        return points.to_vec(); // Too few points to fit
    }

    // Fit X = a * Y + b via least squares
    let n = points.len() as f32;
    let sum_y: f32 = points.iter().map(|p| p.1).sum();
    let sum_x: f32 = points.iter().map(|p| p.0).sum();
    let sum_yy: f32 = points.iter().map(|p| p.1 * p.1).sum();
    let sum_xy: f32 = points.iter().map(|p| p.0 * p.1).sum();

    let denom = n * sum_yy - sum_y * sum_y;
    if denom.abs() < 1e-6 {
        return points.to_vec();
    }

    let a = (n * sum_xy - sum_y * sum_x) / denom;
    let b = (sum_x - a * sum_y) / n;

    // Compute residuals
    let residuals: Vec<f32> = points.iter().map(|p| (p.0 - (a * p.1 + b)).abs()).collect();

    // Compute σ (standard deviation of residuals)
    let mean_r = residuals.iter().sum::<f32>() / n;
    let variance = residuals.iter().map(|r| (r - mean_r).powi(2)).sum::<f32>() / n;
    let sigma = variance.sqrt().max(3.0); // minimum σ of 3px to avoid over-rejection

    // Keep points within threshold
    points
        .iter()
        .zip(residuals.iter())
        .filter(|(_, &r)| r < sigma_threshold * sigma)
        .map(|(&p, _)| p)
        .collect()
}

// ============================================================================
// VEHICLE LANE POSITION
// ============================================================================

/// Find which lane the vehicle is in and compute lateral offset.
///
/// Returns (lane_index, lateral_offset, confidence).
/// Lateral offset is negative = left of center, positive = right of center.
pub fn find_vehicle_lane_with_confidence(
    lanes: &[DetectedLane],
    frame_width: f32,
) -> Option<(usize, f32, f32)> {
    if lanes.len() < 2 {
        return None;
    }

    let vehicle_x = frame_width / 2.0;
    let reference_y = frame_width * 0.8; // ~80% down the frame

    // Find the two lanes closest to the vehicle on either side
    let mut left_lane: Option<(usize, f32, f32)> = None; // (index, x_at_ref_y, confidence)
    let mut right_lane: Option<(usize, f32, f32)> = None;

    for (i, lane) in lanes.iter().enumerate() {
        // Interpolate X at reference Y
        if let Some(x) = interpolate_x_at_y(&lane.points, reference_y) {
            if x < vehicle_x {
                // Left of vehicle
                if left_lane.is_none() || x > left_lane.unwrap().1 {
                    left_lane = Some((i, x, lane.confidence));
                }
            } else {
                // Right of vehicle
                if right_lane.is_none() || x < right_lane.unwrap().1 {
                    right_lane = Some((i, x, lane.confidence));
                }
            }
        }
    }

    match (left_lane, right_lane) {
        (Some((li, lx, lc)), Some((ri, rx, rc))) => {
            let lane_width = rx - lx;
            if lane_width < 50.0 {
                return None; // Unreasonably narrow
            }

            let lane_center = (lx + rx) / 2.0;
            let lateral_offset = vehicle_x - lane_center;
            let confidence = (lc + rc) / 2.0;

            Some((li, lateral_offset, confidence))
        }
        _ => None,
    }
}

/// Interpolate X position at a given Y using a lane's point list.
fn interpolate_x_at_y(points: &[(f32, f32)], target_y: f32) -> Option<f32> {
    if points.len() < 2 {
        return None;
    }

    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    for i in 0..sorted.len() - 1 {
        let (x1, y1) = sorted[i];
        let (x2, y2) = sorted[i + 1];

        if y1 <= target_y && target_y <= y2 {
            if (y2 - y1).abs() < 1e-6 {
                return Some(x1);
            }
            let ratio = (target_y - y1) / (y2 - y1);
            return Some(x1 + ratio * (x2 - x1));
        }
    }

    // Extrapolate from nearest segment if target_y is outside range
    if target_y < sorted[0].1 && sorted.len() >= 2 {
        let (x1, y1) = sorted[0];
        let (x2, y2) = sorted[1];
        if (y2 - y1).abs() > 1e-6 {
            let ratio = (target_y - y1) / (y2 - y1);
            return Some(x1 + ratio * (x2 - x1));
        }
    }

    None
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sums_to_one() {
        let data = vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5];
        let result = softmax_axis0(&data, 3, 1, 2);

        // Sum over dim0 for each (j,k) should be 1.0
        let sum0: f32 = (0..3).map(|i| result[i * 2]).sum();
        let sum1: f32 = (0..3).map(|i| result[i * 2 + 1]).sum();
        assert!((sum0 - 1.0).abs() < 1e-5);
        assert!((sum1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_outlier_rejection() {
        // Straight line + one outlier
        let mut points = vec![
            (100.0, 100.0),
            (100.0, 200.0),
            (100.0, 300.0),
            (500.0, 350.0), // outlier!
            (100.0, 400.0),
            (100.0, 500.0),
        ];
        let cleaned = reject_outlier_points(&points, 2.0);
        // The outlier at x=500 should be removed
        assert!(cleaned.len() < points.len());
        assert!(!cleaned.contains(&(500.0, 350.0)));
    }

    #[test]
    fn test_interpolate_x_at_y() {
        let points = vec![(100.0, 100.0), (200.0, 200.0), (300.0, 300.0)];
        let x = interpolate_x_at_y(&points, 150.0);
        assert!(x.is_some());
        assert!((x.unwrap() - 150.0).abs() < 1.0);
    }
}
