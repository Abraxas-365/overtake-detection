use crate::types::{Config, Lane, LaneDetection};
use anyhow::Result;

// CULane row anchors
const ROW_ANCHORS: [f32; 72] = [
    121.0, 131.0, 141.0, 150.0, 160.0, 170.0, 180.0, 189.0, 199.0, 209.0, 219.0, 228.0, 238.0,
    248.0, 258.0, 267.0, 277.0, 287.0, 297.0, 306.0, 316.0, 326.0, 336.0, 345.0, 355.0, 365.0,
    375.0, 384.0, 394.0, 404.0, 414.0, 423.0, 433.0, 443.0, 453.0, 462.0, 472.0, 482.0, 492.0,
    501.0, 511.0, 521.0, 531.0, 540.0, 550.0, 560.0, 570.0, 579.0, 589.0, 599.0, 609.0, 618.0,
    628.0, 638.0, 648.0, 657.0, 667.0, 677.0, 687.0, 696.0, 706.0, 716.0, 726.0, 735.0, 745.0,
    755.0, 765.0, 774.0, 784.0, 794.0, 804.0, 813.0,
];

pub fn parse_lanes(
    raw_output: &[f32],
    img_width: f32,
    img_height: f32,
    config: &Config,
    timestamp: f64,
) -> Result<LaneDetection> {
    let num_anchors = config.model.num_anchors;
    let num_lanes = config.model.num_lanes;
    let griding_num = config.model.griding_num;

    let mut lanes = Vec::new();

    for lane_idx in 0..num_lanes {
        let mut points = Vec::new();
        let mut confidences = Vec::new();

        for anchor_idx in 0..num_anchors {
            // Calculate offset in flat array
            // Layout: [anchor, lane, class]
            let offset = anchor_idx * num_lanes * (griding_num + 1) + lane_idx * (griding_num + 1);

            let logits = &raw_output[offset..offset + (griding_num + 1)];

            // Find argmax
            let (max_idx, max_val) = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();

            // If not "no lane" class
            if max_idx < griding_num {
                // Convert grid index to x-coordinate
                let x = (max_idx as f32 / griding_num as f32) * img_width;

                // Get y-coordinate from row anchor
                let y = ROW_ANCHORS[anchor_idx] * (img_height / 320.0);

                // Calculate confidence (simplified softmax)
                let sum_exp: f32 = logits.iter().map(|&v| (v - max_val).exp()).sum();
                let confidence = 1.0 / sum_exp;

                points.push((x, y));
                confidences.push(confidence);
            }
        }

        // Filter lanes with too few points
        if points.len() >= config.detection.min_points_per_lane {
            let avg_confidence = confidences.iter().sum::<f32>() / confidences.len() as f32;

            if avg_confidence >= config.detection.confidence_threshold {
                lanes.push(Lane {
                    points,
                    confidence: avg_confidence,
                });
            }
        }
    }

    Ok(LaneDetection { lanes, timestamp })
}

/// Find which lane the vehicle is in
pub fn find_vehicle_lane(lanes: &[Lane], img_width: f32) -> Option<(usize, f32)> {
    if lanes.len() < 2 {
        return None;
    }

    let vehicle_x = img_width / 2.0;
    let sample_y_ratio = 0.85; // Sample at 85% down the image

    // Get lane x-coordinates at sample height
    let mut lane_xs: Vec<(usize, f32)> = lanes
        .iter()
        .enumerate()
        .filter_map(|(i, lane)| interpolate_lane_at_y(lane, sample_y_ratio).map(|x| (i, x)))
        .collect();

    // Sort by x-coordinate
    lane_xs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Find which lane pair contains vehicle
    for window in lane_xs.windows(2) {
        let (left_idx, left_x) = window[0];
        let (_right_idx, right_x) = window[1];

        if left_x <= vehicle_x && vehicle_x <= right_x {
            // Calculate lateral offset [-1, 1]
            let lane_width = right_x - left_x;
            let offset_from_left = vehicle_x - left_x;
            let normalized = (offset_from_left / lane_width - 0.5) * 2.0;

            return Some((left_idx, normalized.clamp(-1.0, 1.0)));
        }
    }

    None
}

fn interpolate_lane_at_y(lane: &Lane, target_y_ratio: f32) -> Option<f32> {
    if lane.points.is_empty() {
        return None;
    }

    // Find the point closest to target y
    let target_y =
        lane.points.iter().map(|(_, y)| y).sum::<f32>() / lane.points.len() as f32 * target_y_ratio;

    let mut lower: Option<(f32, f32)> = None;
    let mut upper: Option<(f32, f32)> = None;

    for &(x, y) in &lane.points {
        if y <= target_y {
            if lower.is_none() || y > lower.unwrap().1 {
                lower = Some((x, y));
            }
        }
        if y >= target_y {
            if upper.is_none() || y < upper.unwrap().1 {
                upper = Some((x, y));
            }
        }
    }

    match (lower, upper) {
        (Some((x1, y1)), Some((x2, y2))) => {
            if (y2 - y1).abs() < 1.0 {
                Some((x1 + x2) / 2.0)
            } else {
                let t = (target_y - y1) / (y2 - y1);
                Some(x1 + t * (x2 - x1))
            }
        }
        _ => None,
    }
}
