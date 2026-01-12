// src/analysis/position_estimator.rs

use crate::types::{Lane, LanePosition, VehicleState};
use tracing::debug;

/// Estimates vehicle lateral position from lane detections
pub struct PositionEstimator {
    /// Vertical position for measurement (0=top, 1=bottom)
    pub reference_y_ratio: f32,
    /// Minimum acceptable lane width in pixels
    pub min_lane_width: f32,
    /// Maximum acceptable lane width in pixels
    pub max_lane_width: f32,
}

impl PositionEstimator {
    pub fn new(reference_y_ratio: f32) -> Self {
        Self {
            reference_y_ratio,
            min_lane_width: 100.0,
            max_lane_width: 1000.0,
        }
    }

    /// Estimate vehicle position from detected lanes
    pub fn estimate(&self, lanes: &[Lane], frame_width: u32, frame_height: u32) -> VehicleState {
        let vehicle_x = frame_width as f32 / 2.0;
        let reference_y = frame_height as f32 * self.reference_y_ratio;

        // Find ego lane boundaries (lanes closest to center on each side)
        let left_lane = self.find_ego_lane(lanes, vehicle_x, true);
        let right_lane = self.find_ego_lane(lanes, vehicle_x, false);

        // Calculate positions at reference Y
        let left_x = left_lane.and_then(|l| l.get_x_at_y(reference_y));
        let right_x = right_lane.and_then(|l| l.get_x_at_y(reference_y));

        // Calculate lane width and offset
        let mut lane_width: Option<f32> = None;
        let mut lateral_offset = 0.0f32;

        match (left_x, right_x) {
            (Some(lx), Some(rx)) => {
                let width = rx - lx;

                // Validate lane width
                if width >= self.min_lane_width && width <= self.max_lane_width {
                    lane_width = Some(width);
                    let lane_center = (lx + rx) / 2.0;
                    // Offset: positive = right of center, negative = left of center
                    lateral_offset = vehicle_x - lane_center;
                } else {
                    debug!(
                        "Invalid lane width: {:.1} (min: {:.1}, max: {:.1})",
                        width, self.min_lane_width, self.max_lane_width
                    );
                }
            }
            (Some(lx), None) => {
                // Only left lane visible
                lateral_offset = vehicle_x - lx;
            }
            (None, Some(rx)) => {
                // Only right lane visible
                lateral_offset = vehicle_x - rx;
            }
            (None, None) => {}
        }

        VehicleState {
            lateral_offset,
            lane_width,
            heading_offset: 0.0,
            frame_id: 0,
            timestamp_ms: 0.0,
        }
    }

    /// Find the ego lane boundary on one side
    fn find_ego_lane<'a>(
        &self,
        lanes: &'a [Lane],
        vehicle_x: f32,
        is_left: bool,
    ) -> Option<&'a Lane> {
        let mut candidates: Vec<(&Lane, f32)> = Vec::new();

        for lane in lanes {
            if lane.points.is_empty() {
                continue;
            }

            let avg_x = lane.avg_x();

            if is_left {
                // Left boundary should be to the left of vehicle
                if avg_x < vehicle_x {
                    let distance = vehicle_x - avg_x;
                    candidates.push((lane, distance));
                }
            } else {
                // Right boundary should be to the right of vehicle
                if avg_x > vehicle_x {
                    let distance = avg_x - vehicle_x;
                    candidates.push((lane, distance));
                }
            }
        }

        // Return closest lane
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.first().map(|(lane, _)| *lane)
    }
}

/// Temporal smoothing for position estimates using EMA
pub struct PositionSmoother {
    alpha: f32,
    smoothed_offset: Option<f32>,
    smoothed_width: Option<f32>,
}

impl PositionSmoother {
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha,
            smoothed_offset: None,
            smoothed_width: None,
        }
    }

    /// Apply temporal smoothing to vehicle state
    pub fn smooth(&mut self, state: VehicleState) -> VehicleState {
        // Smooth lateral offset
        let smoothed_offset = match self.smoothed_offset {
            None => {
                self.smoothed_offset = Some(state.lateral_offset);
                state.lateral_offset
            }
            Some(prev) => {
                let new_val = self.alpha * state.lateral_offset + (1.0 - self.alpha) * prev;
                self.smoothed_offset = Some(new_val);
                new_val
            }
        };

        // Smooth lane width
        let smoothed_width = if let Some(width) = state.lane_width {
            match self.smoothed_width {
                None => {
                    self.smoothed_width = Some(width);
                    Some(width)
                }
                Some(prev) => {
                    let new_val = self.alpha * width + (1.0 - self.alpha) * prev;
                    self.smoothed_width = Some(new_val);
                    Some(new_val)
                }
            }
        } else {
            self.smoothed_width
        };

        VehicleState {
            lateral_offset: smoothed_offset,
            lane_width: smoothed_width,
            heading_offset: state.heading_offset,
            frame_id: state.frame_id,
            timestamp_ms: state.timestamp_ms,
        }
    }

    /// Reset smoother state
    pub fn reset(&mut self) {
        self.smoothed_offset = None;
        self.smoothed_width = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Point;

    fn create_test_lane(lane_id: usize, x_positions: &[f32], y_start: f32, y_step: f32) -> Lane {
        let points: Vec<Point> = x_positions
            .iter()
            .enumerate()
            .map(|(i, &x)| Point::new(x, y_start + i as f32 * y_step))
            .collect();

        Lane {
            lane_id,
            points,
            confidence: 0.9,
            position: None,
        }
    }

    #[test]
    fn test_position_estimator_centered() {
        let estimator = PositionEstimator::new(0.8);

        // Create left and right lanes centered around frame center
        let left_lane = create_test_lane(0, &[400.0, 410.0, 420.0], 200.0, 50.0);
        let right_lane = create_test_lane(1, &[880.0, 870.0, 860.0], 200.0, 50.0);

        let lanes = vec![left_lane, right_lane];
        let state = estimator.estimate(&lanes, 1280, 720);

        assert!(state.is_valid());
        assert!(state.lateral_offset.abs() < 50.0); // Roughly centered
    }

    #[test]
    fn test_smoother_reduces_jitter() {
        let mut smoother = PositionSmoother::new(0.3);

        let states = vec![
            VehicleState {
                lateral_offset: 10.0,
                lane_width: Some(400.0),
                heading_offset: 0.0,
                frame_id: 0,
                timestamp_ms: 0.0,
            },
            VehicleState {
                lateral_offset: 50.0,
                lane_width: Some(400.0),
                heading_offset: 0.0,
                frame_id: 1,
                timestamp_ms: 33.3,
            }, // spike
            VehicleState {
                lateral_offset: 12.0,
                lane_width: Some(400.0),
                heading_offset: 0.0,
                frame_id: 2,
                timestamp_ms: 66.6,
            },
        ];

        let mut smoothed_values = Vec::new();
        for state in states {
            let smoothed = smoother.smooth(state);
            smoothed_values.push(smoothed.lateral_offset);
        }

        // The spike at 50.0 should be dampened
        assert!(smoothed_values[1] < 50.0);
        assert!(smoothed_values[1] > 10.0);
    }
}
