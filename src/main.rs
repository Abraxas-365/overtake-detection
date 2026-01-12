// src/analysis/position_estimator.rs

use crate::types::{Lane, VehicleState};

pub struct PositionEstimator {
    pub reference_y_ratio: f32,
    pub min_lane_width: f32,
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

    pub fn estimate(&self, lanes: &[Lane], frame_width: u32, frame_height: u32) -> VehicleState {
        let vehicle_x = frame_width as f32 / 2.0;
        let reference_y = frame_height as f32 * self.reference_y_ratio;

        let left_lane = self.find_ego_lane(lanes, vehicle_x, true);
        let right_lane = self.find_ego_lane(lanes, vehicle_x, false);

        let left_x = left_lane.and_then(|l| l.get_x_at_y(reference_y));
        let right_x = right_lane.and_then(|l| l.get_x_at_y(reference_y));

        let mut lane_width: Option<f32> = None;
        let mut lateral_offset = 0.0f32;

        match (left_x, right_x) {
            (Some(lx), Some(rx)) => {
                let width = rx - lx;
                if width >= self.min_lane_width && width <= self.max_lane_width {
                    lane_width = Some(width);
                    let lane_center = (lx + rx) / 2.0;
                    lateral_offset = vehicle_x - lane_center;
                }
            }
            (Some(lx), None) => {
                lateral_offset = vehicle_x - lx;
            }
            (None, Some(rx)) => {
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

            if is_left && avg_x < vehicle_x {
                candidates.push((lane, vehicle_x - avg_x));
            } else if !is_left && avg_x > vehicle_x {
                candidates.push((lane, avg_x - vehicle_x));
            }
        }

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.first().map(|(lane, _)| *lane)
    }
}

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

    pub fn smooth(&mut self, state: VehicleState) -> VehicleState {
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

    pub fn reset(&mut self) {
        self.smoothed_offset = None;
        self.smoothed_width = None;
    }
}
