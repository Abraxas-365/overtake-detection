// src/detection/smoother.rs
use super::types::VehiclePosition;
use std::collections::{HashMap, VecDeque};

pub struct LanePositionSmoother {
    history: VecDeque<VehiclePosition>,
    window_size: usize,
}

impl LanePositionSmoother {
    pub fn new(window_size: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Smooth the position using temporal window
    pub fn smooth(&mut self, position: VehiclePosition) -> VehiclePosition {
        self.history.push_back(position);
        if self.history.len() > self.window_size {
            self.history.pop_front();
        }

        // Need at least 3 frames for smoothing
        if self.history.len() < 3 {
            return position;
        }

        VehiclePosition {
            lane_index: self.smooth_lane_index(),
            lateral_offset: self.smooth_lateral_offset(),
            confidence: self.smooth_confidence(),
            left_boundary: self.smooth_boundary(|p| p.left_boundary),
            right_boundary: self.smooth_boundary(|p| p.right_boundary),
            timestamp: position.timestamp, // Keep current timestamp
        }
    }

    /// Use mode (most common) for discrete lane index
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

    /// Use median for lateral offset (resistant to outliers)
    fn smooth_lateral_offset(&self) -> f32 {
        let mut offsets: Vec<f32> = self.history.iter().map(|p| p.lateral_offset).collect();
        offsets.sort_by(|a, b| a.partial_cmp(b).unwrap());
        offsets[offsets.len() / 2]
    }

    /// Average confidence
    fn smooth_confidence(&self) -> f32 {
        self.history.iter().map(|p| p.confidence).sum::<f32>() / self.history.len() as f32
    }

    /// Median for boundaries
    fn smooth_boundary<F>(&self, getter: F) -> f32
    where
        F: Fn(&VehiclePosition) -> f32,
    {
        let mut values: Vec<f32> = self.history.iter().map(&getter).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values[values.len() / 2]
    }

    /// Reset the smoother (e.g., after scene change)
    pub fn reset(&mut self) {
        self.history.clear();
    }
}
