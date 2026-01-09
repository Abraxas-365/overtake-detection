// src/smoother.rs

use crate::types::VehiclePosition;
use std::collections::{HashMap, VecDeque};

/// Temporal smoother for vehicle position using a sliding window
pub struct LanePositionSmoother {
    history: VecDeque<VehiclePosition>,
    window_size: usize,
}

impl LanePositionSmoother {
    /// Create a new smoother with specified window size
    ///
    /// # Arguments
    /// * `window_size` - Number of frames to use for smoothing (e.g., 10 frames)
    pub fn new(window_size: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Smooth the current position using temporal window
    ///
    /// Uses different strategies for different components:
    /// - Lane index: Mode (most common value)
    /// - Lateral offset: Median (resistant to outliers)
    /// - Confidence: Average
    pub fn smooth(&mut self, position: VehiclePosition) -> VehiclePosition {
        self.history.push_back(position);

        // Maintain window size
        if self.history.len() > self.window_size {
            self.history.pop_front();
        }

        // Need at least 3 frames for meaningful smoothing
        if self.history.len() < 3 {
            return position;
        }

        VehiclePosition {
            lane_index: self.smooth_lane_index(),
            lateral_offset: self.smooth_lateral_offset(),
            confidence: self.smooth_confidence(),
            timestamp: position.timestamp, // Keep current timestamp
        }
    }

    /// Get the most common lane index (mode)
    fn smooth_lane_index(&self) -> i32 {
        let mut counts: HashMap<i32, usize> = HashMap::new();

        for pos in &self.history {
            *counts.entry(pos.lane_index).or_insert(0) += 1;
        }

        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(lane, _)| lane)
            .unwrap_or(-1)
    }

    /// Get median lateral offset (resistant to outliers)
    fn smooth_lateral_offset(&self) -> f32 {
        let mut offsets: Vec<f32> = self.history.iter().map(|p| p.lateral_offset).collect();
        offsets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        offsets[offsets.len() / 2]
    }

    /// Get average confidence
    fn smooth_confidence(&self) -> f32 {
        let sum: f32 = self.history.iter().map(|p| p.confidence).sum();
        sum / self.history.len() as f32
    }

    /// Reset the smoother (e.g., when video changes)
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Get the number of frames currently in the history
    pub fn history_size(&self) -> usize {
        self.history.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoother_mode_for_lane_index() {
        let mut smoother = LanePositionSmoother::new(5);

        // Feed noisy lane detections: [1, 1, 2, 1, 1]
        let positions = vec![
            VehiclePosition {
                lane_index: 1,
                lateral_offset: 0.0,
                confidence: 0.8,
                timestamp: 0.0,
            },
            VehiclePosition {
                lane_index: 1,
                lateral_offset: 0.0,
                confidence: 0.8,
                timestamp: 0.033,
            },
            VehiclePosition {
                lane_index: 2, // noise
                lateral_offset: 0.0,
                confidence: 0.8,
                timestamp: 0.066,
            },
            VehiclePosition {
                lane_index: 1,
                lateral_offset: 0.0,
                confidence: 0.8,
                timestamp: 0.099,
            },
            VehiclePosition {
                lane_index: 1,
                lateral_offset: 0.0,
                confidence: 0.8,
                timestamp: 0.132,
            },
        ];

        for pos in positions {
            smoother.smooth(pos);
        }

        // Last smoothed position should have lane_index = 1 (mode)
        let last_pos = positions.last().unwrap();
        let smoothed = smoother.smooth(*last_pos);
        assert_eq!(smoothed.lane_index, 1);
    }

    #[test]
    fn test_smoother_median_for_offset() {
        let mut smoother = LanePositionSmoother::new(5);

        // Feed offsets with outlier: [-0.1, -0.05, 0.0, 0.05, 2.0 (outlier)]
        let offsets = vec![-0.1, -0.05, 0.0, 0.05, 2.0];

        for (i, offset) in offsets.iter().enumerate() {
            let pos = VehiclePosition {
                lane_index: 1,
                lateral_offset: *offset,
                confidence: 0.8,
                timestamp: i as f64 * 0.033,
            };
            smoother.smooth(pos);
        }

        let last_pos = VehiclePosition {
            lane_index: 1,
            lateral_offset: 2.0,
            confidence: 0.8,
            timestamp: 0.165,
        };
        let smoothed = smoother.smooth(last_pos);

        // Median should be 0.0 (middle value), not affected by 2.0 outlier
        assert_eq!(smoothed.lateral_offset, 0.0);
    }
}
