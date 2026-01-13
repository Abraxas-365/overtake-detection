// src/analysis/boundary_detector.rs

use std::collections::VecDeque;
use tracing::debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossingType {
    None,
    CrossedLeft,
    CrossedRight,
}

pub struct LaneBoundaryCrossingDetector {
    left_lane_x_history: VecDeque<f32>,
    right_lane_x_history: VecDeque<f32>,
    vehicle_x_history: VecDeque<f32>,
    history_size: usize,
    crossing_margin: f32,
}

impl LaneBoundaryCrossingDetector {
    pub fn new() -> Self {
        Self {
            left_lane_x_history: VecDeque::with_capacity(15),
            right_lane_x_history: VecDeque::with_capacity(15),
            vehicle_x_history: VecDeque::with_capacity(15),
            history_size: 15,
            crossing_margin: 20.0, // pixels - margin for noise tolerance
        }
    }

    pub fn detect_crossing(
        &mut self,
        left_x: Option<f32>,
        right_x: Option<f32>,
        vehicle_x: f32,
    ) -> CrossingType {
        // Need history to detect crossing
        if self.left_lane_x_history.is_empty() {
            self.update_history(left_x, right_x, vehicle_x);
            return CrossingType::None;
        }

        match (left_x, right_x) {
            (Some(lx), Some(rx)) => {
                // Get previous positions
                let prev_left = self.left_lane_x_history.back().copied();
                let prev_right = self.right_lane_x_history.back().copied();
                let prev_vehicle = self.vehicle_x_history.back().copied();

                if let (Some(prev_lx), Some(prev_rx), Some(prev_vx)) =
                    (prev_left, prev_right, prev_vehicle)
                {
                    // Check if vehicle WAS inside lane boundaries
                    let was_inside = prev_vx > (prev_lx + self.crossing_margin)
                        && prev_vx < (prev_rx - self.crossing_margin);

                    // Check if vehicle IS STILL inside lane boundaries
                    let is_inside = vehicle_x > (lx + self.crossing_margin)
                        && vehicle_x < (rx - self.crossing_margin);

                    // Crossing detected if vehicle went from inside to outside
                    if was_inside && !is_inside {
                        let crossing_type = if vehicle_x <= lx + self.crossing_margin {
                            CrossingType::CrossedLeft
                        } else if vehicle_x >= rx - self.crossing_margin {
                            CrossingType::CrossedRight
                        } else {
                            CrossingType::None
                        };

                        if crossing_type != CrossingType::None {
                            debug!(
                                "🚨 Boundary crossing detected: {:?} | Vehicle: {:.1} | Left: {:.1} | Right: {:.1}",
                                crossing_type, vehicle_x, lx, rx
                            );
                        }

                        self.update_history(left_x, right_x, vehicle_x);
                        return crossing_type;
                    }
                }

                self.update_history(left_x, right_x, vehicle_x);
                CrossingType::None
            }
            _ => {
                // Can't detect crossing without both lane boundaries
                self.update_history(left_x, right_x, vehicle_x);
                CrossingType::None
            }
        }
    }

    fn update_history(&mut self, left_x: Option<f32>, right_x: Option<f32>, vehicle_x: f32) {
        if let Some(lx) = left_x {
            self.left_lane_x_history.push_back(lx);
            if self.left_lane_x_history.len() > self.history_size {
                self.left_lane_x_history.pop_front();
            }
        }

        if let Some(rx) = right_x {
            self.right_lane_x_history.push_back(rx);
            if self.right_lane_x_history.len() > self.history_size {
                self.right_lane_x_history.pop_front();
            }
        }

        self.vehicle_x_history.push_back(vehicle_x);
        if self.vehicle_x_history.len() > self.history_size {
            self.vehicle_x_history.pop_front();
        }
    }

    pub fn reset(&mut self) {
        self.left_lane_x_history.clear();
        self.right_lane_x_history.clear();
        self.vehicle_x_history.clear();
    }
}
