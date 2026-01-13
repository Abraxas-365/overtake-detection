// src/analysis/velocity_tracker.rs

use std::collections::VecDeque;

pub struct LateralVelocityTracker {
    offset_history: VecDeque<(f32, f64)>, // (offset_px, timestamp_ms)
    history_size: usize,
}

impl LateralVelocityTracker {
    pub fn new() -> Self {
        Self {
            offset_history: VecDeque::with_capacity(20),
            history_size: 20,
        }
    }

    pub fn get_velocity(&mut self, offset_px: f32, timestamp_ms: f64) -> f32 {
        self.offset_history.push_back((offset_px, timestamp_ms));

        if self.offset_history.len() > self.history_size {
            self.offset_history.pop_front();
        }

        if self.offset_history.len() < 5 {
            return 0.0;
        }

        // Calculate velocity over the entire history window
        let first = self.offset_history.front().unwrap();
        let last = self.offset_history.back().unwrap();

        let delta_offset = last.0 - first.0;
        let delta_time = (last.1 - first.1) / 1000.0; // Convert to seconds

        if delta_time > 0.01 {
            // Avoid division by near-zero
            let velocity = delta_offset / delta_time as f32; // pixels per second
            velocity
        } else {
            0.0
        }
    }

    #[allow(dead_code)]
    pub fn is_moving_laterally(&self, min_velocity_px_per_sec: f32) -> bool {
        if self.offset_history.len() < 5 {
            return false;
        }

        let first = self.offset_history.front().unwrap();
        let last = self.offset_history.back().unwrap();

        let delta_offset = (last.0 - first.0).abs();
        let delta_time = (last.1 - first.1) / 1000.0;

        if delta_time > 0.01 {
            let velocity = delta_offset / delta_time as f32;
            velocity > min_velocity_px_per_sec
        } else {
            false
        }
    }

    pub fn reset(&mut self) {
        self.offset_history.clear();
    }
}
