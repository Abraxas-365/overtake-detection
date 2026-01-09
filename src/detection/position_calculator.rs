// src/detection/position_calculator.rs
use super::types::{Lane, VehiclePosition};

const MIN_LANE_CONFIDENCE: f32 = 0.6;
const MIN_POINTS_PER_LANE: usize = 5;

pub fn calculate_vehicle_position(
    lanes: &[Lane],
    image_width: u32,
    image_height: u32,
) -> VehiclePosition {
    let vehicle_center_x = image_width as f32 / 2.0;
    let sample_y = image_height as f32 * 0.85;

    // Filter lanes by confidence and point count
    let valid_lanes: Vec<&Lane> = lanes
        .iter()
        .filter(|lane| {
            lane.confidence > MIN_LANE_CONFIDENCE && lane.points.len() >= MIN_POINTS_PER_LANE
        })
        .collect();

    if valid_lanes.len() < 2 {
        return VehiclePosition::invalid();
    }

    // Get x-coordinates at sample height
    let mut lane_xs: Vec<(usize, f32)> = valid_lanes
        .iter()
        .enumerate()
        .filter_map(|(i, lane)| interpolate_lane_x(lane, sample_y).map(|x| (i, x)))
        .collect();

    // Sort by x-coordinate
    lane_xs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Filter out lanes too far from vehicle center (likely road edges)
    let max_distance = image_width as f32 * 0.6;
    lane_xs.retain(|(_, x)| (x - vehicle_center_x).abs() < max_distance);

    if lane_xs.len() < 2 {
        return VehiclePosition::invalid();
    }

    // Find which lane boundaries the vehicle is between
    for i in 0..lane_xs.len() - 1 {
        let (left_idx, left_x) = lane_xs[i];
        let (right_idx, right_x) = lane_xs[i + 1];

        if left_x <= vehicle_center_x && vehicle_center_x <= right_x {
            let lane_width = right_x - left_x;
            let offset_normalized = ((vehicle_center_x - left_x) / lane_width - 0.5) * 2.0;

            return VehiclePosition {
                lane_index: i as i32,
                lateral_offset: offset_normalized,
                left_boundary: left_x,
                right_boundary: right_x,
                confidence: valid_lanes[left_idx]
                    .confidence
                    .min(valid_lanes[right_idx].confidence),
                timestamp: std::time::Instant::now(),
            };
        }
    }

    VehiclePosition::invalid()
}

fn interpolate_lane_x(lane: &Lane, target_y: f32) -> Option<f32> {
    // Find two points bracketing target_y
    let mut lower: Option<&crate::detection::types::LanePoint> = None;
    let mut upper: Option<&crate::detection::types::LanePoint> = None;

    for point in &lane.points {
        if point.y <= target_y {
            if lower.is_none() || point.y > lower.unwrap().y {
                lower = Some(point);
            }
        }
        if point.y >= target_y {
            if upper.is_none() || point.y < upper.unwrap().y {
                upper = Some(point);
            }
        }
    }

    match (lower, upper) {
        (Some(p1), Some(p2)) if (p2.y - p1.y).abs() > 0.1 => {
            // Linear interpolation
            let t = (target_y - p1.y) / (p2.y - p1.y);
            Some(p1.x + t * (p2.x - p1.x))
        }
        (Some(p), None) | (None, Some(p)) => Some(p.x),
        _ => None,
    }
}
