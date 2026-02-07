// src/analysis/curve_detector.rs

use crate::types::{CurveInfo, CurveType, Lane};
use std::collections::VecDeque;
use tracing::debug;

pub struct CurveDetector {
    curve_score_history: VecDeque<f32>,
    history_size: usize,
    // Thresholds for curvature (Angle difference between start and end of lane)
    moderate_threshold: f32,
    sharp_threshold: f32,
}

impl CurveDetector {
    pub fn new() -> Self {
        Self {
            curve_score_history: VecDeque::with_capacity(30),
            history_size: 20, // Keep slightly shorter history for responsiveness
            // 5 degrees of bend is noticeable
            moderate_threshold: 5.0,
            // 15 degrees of bend is a sharp turn
            sharp_threshold: 15.0,
        }
    }

    pub fn get_curve_info(&self) -> CurveInfo {
        if self.curve_score_history.is_empty() {
            return CurveInfo::none();
        }

        let avg_score: f32 =
            self.curve_score_history.iter().sum::<f32>() / self.curve_score_history.len() as f32;

        let abs_score = avg_score.abs();

        if abs_score < self.moderate_threshold {
            return CurveInfo::none();
        }

        let curve_type = if abs_score > self.sharp_threshold {
            CurveType::Sharp
        } else {
            CurveType::Moderate
        };

        CurveInfo {
            is_curve: true,
            angle_degrees: abs_score,
            confidence: 0.85,
            curve_type,
        }
    }

    pub fn is_in_curve(&mut self, lanes: &[Lane]) -> bool {
        let score = self.calculate_curve_score(lanes);

        self.curve_score_history.push_back(score);
        if self.curve_score_history.len() > self.history_size {
            self.curve_score_history.pop_front();
        }

        // We need some history to be sure
        if self.curve_score_history.len() < 5 {
            return false;
        }

        // Calculate weighted average (give more weight to recent frames)
        let mut total_score = 0.0;
        let mut total_weight = 0.0;

        for (i, &s) in self.curve_score_history.iter().enumerate() {
            let weight = (i + 1) as f32; // Older frames have less weight
            total_score += s * weight;
            total_weight += weight;
        }

        let avg_score = total_score / total_weight;

        let is_curve = avg_score.abs() > self.moderate_threshold;

        if is_curve {
            debug!(
                "🌀 Curve logic: raw={:.1}, avg={:.1} (thresh: {:.1})",
                score, avg_score, self.moderate_threshold
            );
        }

        is_curve
    }

    /// Calculates the "Bend" of the lanes.
    /// Returns 0.0 for straight roads (even if slanted by perspective).
    /// Returns +/- degrees for actual curves.
    fn calculate_curve_score(&self, lanes: &[Lane]) -> f32 {
        let mut total_bend = 0.0;
        let mut lane_count = 0;

        // Select the best lanes (long enough to measure curvature)
        let valid_lanes: Vec<&Lane> = lanes
            .iter()
            .filter(|l| l.points.len() >= 10 && l.confidence > 0.3)
            .collect();

        for lane in valid_lanes {
            // Strategy: Split lane into Bottom Half and Top Half
            // A straight road (perspective) has same slope in bottom and top.
            // A curved road has different slopes.

            let points = &lane.points;
            let n = points.len();
            let mid_idx = n / 2;

            // 1. Vector of Bottom Segment (Start -> Mid)
            let p_start = points[0];
            let p_mid = points[mid_idx];

            let dx1 = p_mid.x - p_start.x;
            let dy1 = p_mid.y - p_start.y; // dy is usually negative (going up)

            // 2. Vector of Top Segment (Mid -> End)
            let p_end = points[n - 1];

            let dx2 = p_end.x - p_mid.x;
            let dy2 = p_end.y - p_mid.y;

            // Avoid division by zero
            if dy1.abs() < 1.0 || dy2.abs() < 1.0 {
                continue;
            }

            // Calculate angles from vertical (in degrees)
            // atan(dx/dy) gives angle relative to vertical Y axis
            let angle1 = (dx1 / dy1).atan().to_degrees();
            let angle2 = (dx2 / dy2).atan().to_degrees();

            // 3. The "Bend" is the difference
            // If straight perspective: angle1 ≈ -40, angle2 ≈ -40. Diff ≈ 0.
            // If curve left: angle1 ≈ 0 (vertical), angle2 ≈ -30 (left). Diff ≈ 30.
            let bend = angle2 - angle1;

            total_bend += bend;
            lane_count += 1;
        }

        if lane_count == 0 {
            return 0.0;
        }

        // Return average bend
        total_bend / lane_count as f32
    }

    pub fn get_average_angle(&self) -> f32 {
        if self.curve_score_history.is_empty() {
            return 0.0;
        }
        self.curve_score_history.iter().sum::<f32>() / self.curve_score_history.len() as f32
    }

    pub fn reset(&mut self) {
        self.curve_score_history.clear();
    }
}
