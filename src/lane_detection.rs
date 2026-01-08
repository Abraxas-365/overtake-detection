use crate::types::{Config, Lane, LanePoint};
use anyhow::Result;

pub struct LaneDetectionResult {
    pub lanes: Vec<Lane>,
    pub timestamp: f64,
}

pub fn parse_lanes(
    output: &[f32],
    frame_width: f32,
    frame_height: f32,
    config: &Config,
    timestamp: f64,
) -> Result<LaneDetectionResult> {
    // Model output shape: [1, griding_num, num_anchors, num_lanes]
    // = [1, 200, 72, 4]

    let griding_num = config.model.griding_num;
    let num_anchors = config.model.num_anchors;
    let num_lanes = config.model.num_lanes;

    let mut lanes = Vec::new();

    // Process each lane
    for lane_idx in 0..num_lanes {
        let mut points = Vec::new();

        // Process each anchor (row)
        for anchor_idx in 0..num_anchors {
            // Find the grid position with max probability
            let mut max_prob = f32::NEG_INFINITY;
            let mut max_grid_idx = 0;

            // Check each grid position
            for grid_idx in 0..griding_num {
                // Index calculation for shape [1, 200, 72, 4]
                // Skip batch dimension (0), so: [grid, anchor, lane]
                let idx = grid_idx * (num_anchors * num_lanes) + anchor_idx * num_lanes + lane_idx;

                let prob = output[idx];
                if prob > max_prob {
                    max_prob = prob;
                    max_grid_idx = grid_idx;
                }
            }

            // Apply softmax to get confidence
            let mut sum_exp = 0.0_f32;
            for grid_idx in 0..griding_num {
                let idx = grid_idx * (num_anchors * num_lanes) + anchor_idx * num_lanes + lane_idx;
                sum_exp += (output[idx] - max_prob).exp();
            }
            let confidence = 1.0 / sum_exp;

            // Only add point if confidence is high enough
            if confidence >= 0.5 && max_grid_idx < griding_num {
                // Convert grid position to pixel coordinates
                let x = (max_grid_idx as f32 / griding_num as f32) * frame_width;

                // Y coordinate from row anchor (scaled to frame height)
                let y = (config.model.row_anchors[anchor_idx] / config.model.input_height as f32)
                    * frame_height;

                points.push(LanePoint { x, y, confidence });
            }
        }

        // Only add lane if it has enough points
        if points.len() >= 5 {
            lanes.push(Lane {
                id: lane_idx,
                points,
            });
        }
    }

    Ok(LaneDetectionResult { lanes, timestamp })
}

pub fn find_vehicle_lane(lanes: &[Lane], frame_width: f32) -> Option<(usize, f32)> {
    if lanes.len() < 2 {
        return None;
    }

    let vehicle_x = frame_width / 2.0;

    // Sort lanes by x position at bottom of frame
    let mut lane_positions: Vec<(usize, f32)> = lanes
        .iter()
        .filter_map(|lane| lane.points.last().map(|p| (lane.id, p.x)))
        .collect();

    lane_positions.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Find which lane pair the vehicle is between
    for i in 0..lane_positions.len() - 1 {
        let (left_id, left_x) = lane_positions[i];
        let (right_id, right_x) = lane_positions[i + 1];

        if left_x <= vehicle_x && vehicle_x <= right_x {
            let lane_width = right_x - left_x;
            let offset_from_left = vehicle_x - left_x;
            let normalized_offset = (offset_from_left / lane_width - 0.5) * 2.0; // [-1, 1]

            return Some((i, normalized_offset));
        }
    }

    None
}
