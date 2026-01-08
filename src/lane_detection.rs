use crate::types::{Config, Lane};
use anyhow::Result;
use tracing::info;

pub struct LaneDetectionResult {
    pub lanes: Vec<Lane>,
    pub timestamp: f64,
}

// CULane row anchors (normalized to 320 height)
const ROW_ANCHORS: [f32; 72] = [
    121.0, 131.0, 141.0, 150.0, 160.0, 170.0, 180.0, 189.0, 199.0, 209.0, 219.0, 228.0, 238.0,
    248.0, 258.0, 267.0, 277.0, 287.0, 297.0, 306.0, 316.0, 326.0, 336.0, 345.0, 355.0, 365.0,
    375.0, 384.0, 394.0, 404.0, 414.0, 423.0, 433.0, 443.0, 453.0, 462.0, 472.0, 482.0, 492.0,
    501.0, 511.0, 521.0, 531.0, 540.0, 550.0, 560.0, 570.0, 579.0, 589.0, 599.0, 609.0, 618.0,
    628.0, 638.0, 648.0, 657.0, 667.0, 677.0, 687.0, 696.0, 706.0, 716.0, 726.0, 735.0, 745.0,
    755.0, 765.0, 774.0, 784.0, 794.0, 804.0, 813.0,
];

pub fn parse_lanes(
    output: &[f32],
    frame_width: f32,
    frame_height: f32,
    config: &Config,
    timestamp: f64,
) -> Result<LaneDetectionResult> {
    // DEBUG: Check output values
    let max_val = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_val = output.iter().copied().fold(f32::INFINITY, f32::min);
    let avg_val = output.iter().sum::<f32>() / output.len() as f32;

    info!(
        "Output stats - min: {:.4}, max: {:.4}, avg: {:.4}",
        min_val, max_val, avg_val
    );

    // Model output shape: [1, griding_num, num_anchors, num_lanes]
    // = [1, 200, 72, 4]

    let griding_num = config.model.griding_num;
    let num_anchors = config.model.num_anchors;
    let num_lanes = config.model.num_lanes;

    info!(
        "Config - griding: {}, anchors: {}, lanes: {}",
        griding_num, num_anchors, num_lanes
    );

    let mut lanes = Vec::new();

    // Process each lane
    for lane_idx in 0..num_lanes {
        let mut points = Vec::new();
        let mut total_confidence = 0.0;
        let mut point_count = 0;

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

            // Use sigmoid for confidence (simpler than softmax)
            let confidence = 1.0 / (1.0 + (-max_prob).exp());

            // LOWERED THRESHOLD for debugging
            if confidence >= 0.1 && max_grid_idx < griding_num {
                // Convert grid position to pixel coordinates
                let x = (max_grid_idx as f32 / griding_num as f32) * frame_width;

                // Y coordinate from row anchor (scaled to frame height)
                let y = (ROW_ANCHORS[anchor_idx] / config.model.input_height as f32) * frame_height;

                points.push((x, y));
                total_confidence += confidence;
                point_count += 1;
            }
        }

        // LOWERED THRESHOLD: Only need 3+ points
        if points.len() >= 3 {
            let avg_confidence = if point_count > 0 {
                total_confidence / point_count as f32
            } else {
                0.0
            };

            info!(
                "Lane {} detected with {} points, confidence: {:.4}",
                lane_idx,
                points.len(),
                avg_confidence
            );

            lanes.push(Lane {
                points,
                confidence: avg_confidence,
            });
        }
    }

    info!("Total lanes detected: {}", lanes.len());

    Ok(LaneDetectionResult { lanes, timestamp })
}

pub fn find_vehicle_lane(lanes: &[Lane], frame_width: f32) -> Option<(usize, f32)> {
    if lanes.len() < 2 {
        return None;
    }

    let vehicle_x = frame_width / 2.0;

    // Get x positions of lanes at bottom of frame
    let mut lane_positions: Vec<(usize, f32)> = lanes
        .iter()
        .enumerate()
        .filter_map(|(idx, lane)| {
            lane.points.last().map(|p| (idx, p.0)) // p.0 is x coordinate
        })
        .collect();

    lane_positions.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Find which lane pair the vehicle is between
    for i in 0..lane_positions.len() - 1 {
        let (_, left_x) = lane_positions[i];
        let (_, right_x) = lane_positions[i + 1];

        if left_x <= vehicle_x && vehicle_x <= right_x {
            let lane_width = right_x - left_x;
            let offset_from_left = vehicle_x - left_x;
            let normalized_offset = (offset_from_left / lane_width - 0.5) * 2.0; // [-1, 1]

            return Some((i, normalized_offset));
        }
    }

    None
}
