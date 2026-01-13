// src/analysis/curve_detector.rs

use crate::types::Lane;
use std::collections::VecDeque;
use tracing::debug;

pub struct CurveDetector {
    lane_angle_history: VecDeque<f32>,
    history_size: usize,
    curve_threshold: f32,
}

impl CurveDetector {
    pub fn new() -> Self {
        Self {
            lane_angle_history: VecDeque::with_capacity(30),
            history_size: 30,
            curve_threshold: 5.0, // degrees
        }
    }

    pub fn is_in_curve(&mut self, lanes: &[Lane]) -> bool {
        let angle = self.calculate_lane_angle(lanes);

        self.lane_angle_history.push_back(angle);
        if self.lane_angle_history.len() > self.history_size {
            self.lane_angle_history.pop_front();
        }

        if self.lane_angle_history.len() < 10 {
            return false;
        }

        // Calculate average absolute angle
        let avg_angle: f32 = self.lane_angle_history.iter().map(|a| a.abs()).sum::<f32>()
            / self.lane_angle_history.len() as f32;

        let is_curve = avg_angle > self.curve_threshold;

        if is_curve {
            debug!(
                "🌀 Curve detected: avg angle = {:.1}° (threshold: {:.1}°)",
                avg_angle, self.curve_threshold
            );
        }

        is_curve
    }

    fn calculate_lane_angle(&self, lanes: &[Lane]) -> f32 {
        // Find the most confident lane with enough points
        let best_lane = lanes.iter().filter(|l| l.points.len() >= 5).max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(lane) = best_lane {
            // Calculate angle between bottom and top points
            if lane.points.len() >= 2 {
                let bottom = &lane.points[0];
                let top = &lane.points[lane.points.len() - 1];

                let dx = top.x - bottom.x;
                let dy = top.y - bottom.y;

                if dy.abs() > 10.0 {
                    // Prevent division by near-zero
                    let angle_rad = (dx / dy).atan();
                    let angle_deg = angle_rad.to_degrees();
                    return angle_deg;
                }
            }
        }

        0.0 // No curve detected
    }

    pub fn reset(&mut self) {
        self.lane_angle_history.clear();
    }
}
