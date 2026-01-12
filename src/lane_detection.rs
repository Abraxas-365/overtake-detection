// src/lane_detection.rs

use crate::types::{Config, DetectedLane};
use anyhow::Result;
use tracing::debug;

pub struct LaneDetectionResult {
    pub lanes: Vec<DetectedLane>,
    pub timestamp_ms: f64,
}

/// Softmax along first axis
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
            for i in 0..dim0 {
                let idx = i * (dim1 * dim2) + j * dim2 + k;
                result[idx] /= sum;
            }
        }
    }

    result
}

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

    // Expected size for loc_row tensor [200, 72, 4]
    let loc_row_size = griding_num * num_anchors * num_lanes;

    if output.len() < loc_row_size {
        anyhow::bail!(
            "Output size mismatch: expected {}, got {}",
            loc_row_size,
            output.len()
        );
    }

    // Constants matching Python
    const ROW_ANCHOR_START: f32 = 160.0;
    const ROW_ANCHOR_END: f32 = 710.0;
    const ORIGINAL_HEIGHT: f32 = 720.0;

    // Apply softmax along grid dimension (like Python)
    let loc_row_prob = softmax_axis0(output, griding_num, num_anchors, num_lanes);

    let mut lanes = Vec::new();

    for lane_idx in 0..num_lanes {
        let mut points: Vec<(f32, f32)> = Vec::new();
        let mut total_confidence = 0.0;
        let mut valid_points = 0;

        for anchor_idx in 0..num_anchors {
            // Find argmax along grid dimension (after softmax)
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

            // Skip grid_idx == 0 (no lane class) - matches Python
            if max_grid_idx == 0 {
                continue;
            }

            // Only include points with reasonable confidence
            if max_prob < 0.1 {
                continue;
            }

            // Calculate X coordinate - matches Python exactly
            // Python: x_norm = (grid_idx - 1) / (num_grid_cells - 1)
            let x_norm = (max_grid_idx as f32 - 1.0) / (griding_num as f32 - 1.0);
            let x = x_norm * frame_width;

            // Calculate Y coordinate - matches Python exactly
            // Python: np.linspace(160, 710, 72)
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

            // Sort points by Y (bottom to top for consistency)
            points.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            lanes.push(DetectedLane {
                points,
                confidence: avg_confidence,
            });
        }
    }

    debug!("Detected {} lanes", lanes.len());

    Ok(LaneDetectionResult {
        lanes,
        timestamp_ms,
    })
}
