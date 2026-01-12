// src/lane_detection.rs

use crate::types::{Config, DetectedLane};
use anyhow::Result;
use tracing::info;

pub struct LaneDetectionResult {
    pub lanes: Vec<DetectedLane>,
    pub timestamp_ms: f64,
}

pub fn parse_lanes(
    output: &[f32],
    frame_width: f32,
    frame_height: f32,
    config: &Config,
    timestamp_ms: f64,
) -> Result<LaneDetectionResult> {
    let num_cls = config.model.griding_num; // Should be 201
    let num_anchors = config.model.num_anchors; // 72
    let num_lanes = config.model.num_lanes; // 4

    const ROW_ANCHOR_START: f32 = 160.0;
    const ROW_ANCHOR_END: f32 = 710.0;
    const ORIGINAL_HEIGHT: f32 = 720.0;

    let mut lanes = Vec::new();

    for lane_idx in 0..num_lanes {
        let mut points: Vec<(f32, f32)> = Vec::new();
        let mut total_confidence = 0.0;
        let mut point_count = 0;

        for anchor_idx in 0..num_anchors {
            let mut max_prob = f32::NEG_INFINITY;
            let mut max_grid_idx = 0;

            // Find the grid cell with highest probability
            for grid_idx in 0..num_cls {
                // Changed: iterate 0..201
                let idx = grid_idx * (num_anchors * num_lanes) + anchor_idx * num_lanes + lane_idx;
                let prob = output[idx];
                if prob > max_prob {
                    max_prob = prob;
                    max_grid_idx = grid_idx;
                }
            }

            let confidence = 1.0 / (1.0 + (-max_prob).exp());

            // Skip grid_idx == 0 (no lane class)
            if confidence >= 0.1 && max_grid_idx > 0 {
                // Fixed: use (num_cls - 1) = 200 as denominator
                let x = ((max_grid_idx as f32 - 1.0) / (num_cls as f32 - 2.0)) * frame_width;

                let y_norm = ROW_ANCHOR_START
                    + (ROW_ANCHOR_END - ROW_ANCHOR_START)
                        * (anchor_idx as f32 / (num_anchors as f32 - 1.0));
                let y = (y_norm / ORIGINAL_HEIGHT) * frame_height;

                points.push((x, y));
                total_confidence += confidence;
                point_count += 1;
            }
        }

        if points.len() >= config.detection.min_points_per_lane {
            let avg_confidence = if point_count > 0 {
                total_confidence / point_count as f32
            } else {
                0.0
            };

            // Apply confidence threshold at lane level (like Python)
            if avg_confidence >= config.detection.confidence_threshold {
                lanes.push(DetectedLane {
                    points,
                    confidence: avg_confidence,
                });
            }
        }
    }

    Ok(LaneDetectionResult {
        lanes,
        timestamp_ms,
    })
}

pub fn find_vehicle_lane_with_confidence(
    lanes: &[DetectedLane],
    frame_width: f32,
) -> Option<(usize, f32, f32)> {
    if lanes.len() < 2 {
        return None;
    }

    let vehicle_x = frame_width / 2.0;

    let mut lane_positions: Vec<(usize, f32, f32)> = lanes
        .iter()
        .enumerate()
        .filter_map(|(idx, lane)| lane.points.last().map(|p| (idx, p.0, lane.confidence)))
        .collect();

    lane_positions.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    for i in 0..lane_positions.len() - 1 {
        let (_, left_x, left_conf) = lane_positions[i];
        let (_, right_x, right_conf) = lane_positions[i + 1];

        if left_x <= vehicle_x && vehicle_x <= right_x {
            let lane_width = right_x - left_x;
            let offset_from_left = vehicle_x - left_x;
            let normalized_offset = (offset_from_left / lane_width - 0.5) * 2.0;
            let confidence = left_conf.min(right_conf);

            return Some((i, normalized_offset, confidence));
        }
    }

    None
}
