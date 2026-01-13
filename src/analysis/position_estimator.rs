// src/analysis/position_estimator.rs

use crate::types::{Lane, VehicleState};
use std::collections::VecDeque;
use tracing::debug;

/// Estimates vehicle position relative to lane boundaries
pub struct PositionEstimator {
    pub reference_y_ratio: f32,
    pub min_lane_width: f32,
    pub max_lane_width: f32,
    pub default_lane_width: f32,
    /// History of lane widths for stabilization
    lane_width_history: VecDeque<f32>,
    /// History of lateral offsets for outlier detection
    offset_history: VecDeque<f32>,
    /// Maximum history size
    history_size: usize,
    /// Last valid lane width (fallback)
    last_valid_width: Option<f32>,
}

impl PositionEstimator {
    pub fn new(reference_y_ratio: f32) -> Self {
        Self {
            reference_y_ratio,
            min_lane_width: 150.0,
            max_lane_width: 800.0,
            default_lane_width: 500.0,
            lane_width_history: VecDeque::with_capacity(15),
            offset_history: VecDeque::with_capacity(15),
            history_size: 15,
            last_valid_width: None,
        }
    }

    /// Get stable lane width using median of history
    fn get_stable_lane_width(&mut self, measured: f32) -> f32 {
        // Validate measurement
        if measured >= self.min_lane_width && measured <= self.max_lane_width {
            self.lane_width_history.push_back(measured);
            if self.lane_width_history.len() > self.history_size {
                self.lane_width_history.pop_front();
            }
            self.last_valid_width = Some(measured);
        }

        // Need at least 3 measurements for median
        if self.lane_width_history.len() < 3 {
            return self.last_valid_width.unwrap_or(self.default_lane_width);
        }

        // Calculate median
        let mut sorted: Vec<f32> = self.lane_width_history.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[sorted.len() / 2]
    }

    /// Check if an offset is an outlier based on history
    fn is_offset_outlier(&self, offset: f32) -> bool {
        if self.offset_history.len() < 5 {
            return false;
        }

        let mean: f32 = self.offset_history.iter().sum::<f32>() / self.offset_history.len() as f32;
        let variance: f32 = self
            .offset_history
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / self.offset_history.len() as f32;
        let std_dev = variance.sqrt();

        // Offset is outlier if more than 2.5 std deviations from mean
        (offset - mean).abs() > 2.5 * std_dev.max(10.0)
    }

    /// Update offset history
    fn update_offset_history(&mut self, offset: f32) {
        self.offset_history.push_back(offset);
        if self.offset_history.len() > self.history_size {
            self.offset_history.pop_front();
        }
    }

    pub fn estimate(
        &mut self,
        lanes: &[Lane],
        frame_width: u32,
        frame_height: u32,
    ) -> VehicleState {
        let vehicle_x = frame_width as f32 / 2.0;
        let reference_y = frame_height as f32 * self.reference_y_ratio;

        // Filter lanes by confidence first
        let confident_lanes: Vec<&Lane> = lanes
            .iter()
            .filter(|l| l.confidence > 0.4 && l.points.len() >= 5)
            .collect();

        let left_lane = self.find_ego_lane(&confident_lanes, vehicle_x, true);
        let right_lane = self.find_ego_lane(&confident_lanes, vehicle_x, false);

        let left_x = left_lane.and_then(|l| l.get_x_at_y(reference_y));
        let right_x = right_lane.and_then(|l| l.get_x_at_y(reference_y));

        // Calculate detection confidence
        let detection_confidence = match (&left_lane, &right_lane) {
            (Some(l), Some(r)) => (l.confidence + r.confidence) / 2.0,
            (Some(l), None) => l.confidence * 0.7,
            (None, Some(r)) => r.confidence * 0.7,
            (None, None) => 0.0,
        };

        let mut lane_width: Option<f32> = None;
        let mut lateral_offset = 0.0f32;
        let mut raw_offset = 0.0f32;

        match (left_x, right_x) {
            (Some(lx), Some(rx)) => {
                let measured_width = rx - lx;

                // Use stabilized lane width
                let stable_width = self.get_stable_lane_width(measured_width);
                lane_width = Some(stable_width);

                let lane_center = (lx + rx) / 2.0;
                raw_offset = vehicle_x - lane_center;
                lateral_offset = raw_offset;

                // Check for outlier and use smoothed value if needed
                if self.is_offset_outlier(raw_offset) {
                    debug!(
                        "Offset outlier detected: {:.1}px, using smoothed value",
                        raw_offset
                    );
                    // Use last known good offset if available
                    if let Some(&last_offset) = self.offset_history.back() {
                        lateral_offset = last_offset;
                    }
                }

                self.update_offset_history(lateral_offset);
            }
            (Some(lx), None) => {
                // Only left lane visible
                let stable_width = self.get_stable_lane_width(self.default_lane_width);
                let estimated_center = lx + (stable_width / 2.0);
                raw_offset = vehicle_x - estimated_center;
                lateral_offset = raw_offset;
                lane_width = Some(stable_width);

                if !self.is_offset_outlier(raw_offset) {
                    self.update_offset_history(lateral_offset);
                }
            }
            (None, Some(rx)) => {
                // Only right lane visible
                let stable_width = self.get_stable_lane_width(self.default_lane_width);
                let estimated_center = rx - (stable_width / 2.0);
                raw_offset = vehicle_x - estimated_center;
                lateral_offset = raw_offset;
                lane_width = Some(stable_width);

                if !self.is_offset_outlier(raw_offset) {
                    self.update_offset_history(lateral_offset);
                }
            }
            (None, None) => {
                // No lanes found - use last known values if available
                if let Some(width) = self.last_valid_width {
                    lane_width = Some(width);
                }
                if let Some(&last_offset) = self.offset_history.back() {
                    lateral_offset = last_offset;
                }
            }
        }

        VehicleState {
            lateral_offset,
            lane_width,
            heading_offset: 0.0,
            frame_id: 0,
            timestamp_ms: 0.0,
            raw_offset,
            detection_confidence,
        }
    }

    fn find_ego_lane<'a>(
        &self,
        lanes: &[&'a Lane],
        vehicle_x: f32,
        is_left: bool,
    ) -> Option<&'a Lane> {
        let mut candidates: Vec<(&Lane, f32)> = Vec::new();

        for lane in lanes {
            if lane.points.len() < 2 {
                continue;
            }

            // Use bottom_point (closest to car) for more accurate detection
            if let Some(p) = lane.bottom_point() {
                if is_left {
                    if p.x < vehicle_x {
                        candidates.push((lane, vehicle_x - p.x));
                    }
                } else {
                    if p.x > vehicle_x {
                        candidates.push((lane, p.x - vehicle_x));
                    }
                }
            }
        }

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.first().map(|(lane, _)| *lane)
    }

    /// Reset the estimator state (e.g., when video changes)
    pub fn reset(&mut self) {
        self.lane_width_history.clear();
        self.offset_history.clear();
        self.last_valid_width = None;
    }
}

/// Smooths vehicle position over time using exponential moving average
pub struct PositionSmoother {
    alpha: f32,
    smoothed_offset: Option<f32>,
    smoothed_width: Option<f32>,
    /// Velocity estimation for prediction
    last_offset: Option<f32>,
    velocity: f32,
}

impl PositionSmoother {
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.1, 0.5), // Ensure reasonable alpha
            smoothed_offset: None,
            smoothed_width: None,
            last_offset: None,
            velocity: 0.0,
        }
    }

    pub fn smooth(&mut self, state: VehicleState) -> VehicleState {
        // Update velocity estimate
        if let Some(last) = self.last_offset {
            let new_velocity = state.lateral_offset - last;
            self.velocity = self.alpha * new_velocity + (1.0 - self.alpha) * self.velocity;
        }
        self.last_offset = Some(state.lateral_offset);

        // Smooth offset with velocity-aware prediction
        let smoothed_offset = match self.smoothed_offset {
            None => {
                self.smoothed_offset = Some(state.lateral_offset);
                state.lateral_offset
            }
            Some(prev) => {
                // Predict based on velocity
                let predicted = prev + self.velocity;
                // Blend prediction with measurement
                let new_val = self.alpha * state.lateral_offset + (1.0 - self.alpha) * predicted;
                self.smoothed_offset = Some(new_val);
                new_val
            }
        };

        // Smooth width with less aggressive smoothing
        let width_alpha = self.alpha * 0.5; // More stable lane width
        let smoothed_width = if let Some(width) = state.lane_width {
            match self.smoothed_width {
                None => {
                    self.smoothed_width = Some(width);
                    Some(width)
                }
                Some(prev) => {
                    let new_val = width_alpha * width + (1.0 - width_alpha) * prev;
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
            raw_offset: state.raw_offset,
            detection_confidence: state.detection_confidence,
        }
    }

    pub fn reset(&mut self) {
        self.smoothed_offset = None;
        self.smoothed_width = None;
        self.last_offset = None;
        self.velocity = 0.0;
    }
}
