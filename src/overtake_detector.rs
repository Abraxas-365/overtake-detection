// src/overtake_detector.rs

use crate::types::{Config, Direction, LaneChangeEvent, OvertakeEvent, VehiclePosition};
use std::collections::VecDeque;
use tracing::info;

pub struct DetectorResult {
    pub lane_change: Option<LaneChangeEvent>,
    pub overtake: Option<OvertakeEvent>,
}

pub struct OvertakeDetector {
    config: Config,
    lane_history: VecDeque<(i32, f64)>,
    last_stable_lane: Option<i32>,
    last_change_time: Option<f64>,
    recent_changes: Vec<LaneChangeEvent>,
    calibration_frames: Vec<i32>,
    baseline_lane: Option<i32>,
    is_calibrated: bool,
}

impl OvertakeDetector {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            lane_history: VecDeque::with_capacity(30),
            last_stable_lane: None,
            last_change_time: None,
            recent_changes: Vec::new(),
            calibration_frames: Vec::new(),
            baseline_lane: None,
            is_calibrated: false,
        }
    }

    /// Update with VehiclePosition (smoothed position)
    pub fn update_with_position(&mut self, position: VehiclePosition) -> DetectorResult {
        // Handle calibration phase
        if !self.is_calibrated {
            self.calibration_frames.push(position.lane_index);

            if self.calibration_frames.len() >= self.config.detection.calibration_frames {
                self.baseline_lane = Some(self.compute_baseline_lane());
                self.last_stable_lane = self.baseline_lane;
                self.is_calibrated = true;
                info!(
                    "✅ Calibration complete! Baseline lane: {}",
                    self.baseline_lane.unwrap()
                );
            }

            return DetectorResult {
                lane_change: None,
                overtake: None,
            };
        }

        // After calibration, detect lane changes and overtakes
        let overtake = self.update(
            position.lane_index,
            position.lateral_offset,
            position.timestamp,
        );

        // Extract the lane change event if one occurred
        let lane_change = if overtake.is_some() {
            self.recent_changes.last().cloned()
        } else {
            None
        };

        DetectorResult {
            lane_change,
            overtake,
        }
    }

    /// Check if calibration is complete
    pub fn is_calibrated(&self) -> bool {
        self.is_calibrated
    }

    /// Get the baseline lane from calibration
    pub fn get_baseline_lane(&self) -> Option<i32> {
        self.baseline_lane
    }

    /// Original update method (used internally)
    pub fn update(
        &mut self,
        current_lane: i32,
        _lateral_offset: f32,
        timestamp: f64,
    ) -> Option<OvertakeEvent> {
        // Add to history
        self.lane_history.push_back((current_lane, timestamp));
        if self.lane_history.len() > 30 {
            self.lane_history.pop_front();
        }

        // Get stable lane (mode of last 10 frames)
        let stable_lane = self.get_stable_lane();

        // Detect lane change
        if let Some(prev_stable) = self.last_stable_lane {
            if stable_lane != prev_stable {
                let event = LaneChangeEvent {
                    timestamp,
                    direction: if stable_lane > prev_stable {
                        Direction::Right
                    } else {
                        Direction::Left
                    },
                    from_lane: prev_stable,
                    to_lane: stable_lane,
                    confidence: 0.8, // You can calculate actual confidence
                };

                info!(
                    "Lane change detected: {:?} from {} to {}",
                    event.direction, event.from_lane, event.to_lane
                );

                // Check for overtake
                let overtake = self.check_overtake(&event);

                self.last_stable_lane = Some(stable_lane);
                self.last_change_time = Some(timestamp);

                return overtake;
            }
        } else {
            self.last_stable_lane = Some(stable_lane);
        }

        None
    }

    fn get_stable_lane(&self) -> i32 {
        if self.lane_history.is_empty() {
            return -1;
        }

        // Get mode of last 10 frames
        let recent: Vec<i32> = self
            .lane_history
            .iter()
            .rev()
            .take(10)
            .map(|(lane, _)| *lane)
            .collect();

        let mut counts = std::collections::HashMap::new();
        for &lane in &recent {
            *counts.entry(lane).or_insert(0) += 1;
        }

        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(lane, _)| lane)
            .unwrap_or(-1)
    }

    fn check_overtake(&mut self, current_event: &LaneChangeEvent) -> Option<OvertakeEvent> {
        // Add to recent changes
        self.recent_changes.push(current_event.clone());

        // Keep only events within time window
        self.recent_changes.retain(|e| {
            current_event.timestamp - e.timestamp < self.config.overtake.max_window_seconds
        });

        // Need at least 2 lane changes
        if self.recent_changes.len() < 2 {
            return None;
        }

        let prev = &self.recent_changes[self.recent_changes.len() - 2];
        let curr = current_event;

        let delta = curr.timestamp - prev.timestamp;

        // Check timing constraints
        if delta < self.config.overtake.min_interval_seconds
            || delta > self.config.overtake.max_window_seconds
        {
            return None;
        }

        // Check for opposite directions (overtake pattern)
        let is_complete = (prev.direction == Direction::Left && curr.direction == Direction::Right)
            || (prev.direction == Direction::Right && curr.direction == Direction::Left);

        if is_complete {
            info!("🚗 OVERTAKE DETECTED!");
        }

        Some(OvertakeEvent {
            start_timestamp: prev.timestamp,
            end_timestamp: curr.timestamp,
            first_direction: prev.direction,
            second_direction: curr.direction,
            start_lane: prev.from_lane,
            end_lane: curr.to_lane,
            is_complete,
            confidence: (prev.confidence + curr.confidence) / 2.0,
        })
    }

    /// Compute the baseline lane from calibration frames (mode)
    fn compute_baseline_lane(&self) -> i32 {
        let mut counts = std::collections::HashMap::new();
        for &lane in &self.calibration_frames {
            *counts.entry(lane).or_insert(0) += 1;
        }

        *counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(lane, _)| lane)
            .unwrap_or(&1) // Default to lane 1 if no data
    }

    /// Reset the detector (useful for new video or scene changes)
    pub fn reset(&mut self) {
        self.lane_history.clear();
        self.last_stable_lane = None;
        self.last_change_time = None;
        self.recent_changes.clear();
        self.calibration_frames.clear();
        self.baseline_lane = None;
        self.is_calibrated = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> Config {
        // You'll need to create a minimal valid config for testing
        // This is just a placeholder structure
        Config {
            model: crate::types::ModelConfig {
                path: "test.onnx".to_string(),
                input_width: 1600,
                input_height: 320,
                num_anchors: 72,
                num_lanes: 4,
                griding_num: 200,
            },
            inference: crate::types::InferenceConfig {
                use_tensorrt: false,
                use_fp16: false,
                enable_engine_cache: false,
                engine_cache_path: "".to_string(),
                num_threads: 4,
            },
            detection: crate::types::DetectionConfig {
                confidence_threshold: 0.5,
                min_points_per_lane: 5,
                smoother_window_size: 10,
                calibration_frames: 90,
                debounce_frames: 15,
                confirm_frames: 20,
                min_lane_confidence: 0.6,
                min_position_confidence: 0.5,
            },
            overtake: crate::types::OvertakeConfig {
                lane_change_offset_threshold: 0.7,
                debounce_frames: 15,
                confirm_frames: 20,
                max_window_seconds: 10.0,
                min_interval_seconds: 1.0,
            },
            video: crate::types::VideoConfig {
                input_dir: "".to_string(),
                output_dir: "".to_string(),
                source_width: 1920,
                source_height: 1080,
                target_fps: 30,
                save_annotated: true,
                save_events_only: false,
            },
            logging: crate::types::LoggingConfig {
                level: "info".to_string(),
            },
        }
    }

    #[test]
    fn test_calibration() {
        let config = create_test_config();
        let mut detector = OvertakeDetector::new(config);

        assert!(!detector.is_calibrated());
        assert_eq!(detector.get_baseline_lane(), None);

        // Simulate calibration with 90 frames in lane 1
        for i in 0..90 {
            let position = VehiclePosition {
                lane_index: 1,
                lateral_offset: 0.0,
                confidence: 0.8,
                timestamp: i as f64 * 0.033,
            };
            detector.update_with_position(position);
        }

        assert!(detector.is_calibrated());
        assert_eq!(detector.get_baseline_lane(), Some(1));
    }

    #[test]
    fn test_lane_change_detection() {
        let config = create_test_config();
        let mut detector = OvertakeDetector::new(config);

        // Complete calibration
        for i in 0..90 {
            let position = VehiclePosition {
                lane_index: 1,
                lateral_offset: 0.0,
                confidence: 0.8,
                timestamp: i as f64 * 0.033,
            };
            detector.update_with_position(position);
        }

        // Stay in lane 1 for a bit
        for i in 90..120 {
            let position = VehiclePosition {
                lane_index: 1,
                lateral_offset: 0.0,
                confidence: 0.8,
                timestamp: i as f64 * 0.033,
            };
            let result = detector.update_with_position(position);
            assert!(result.lane_change.is_none());
        }

        // Change to lane 2
        for i in 120..150 {
            let position = VehiclePosition {
                lane_index: 2,
                lateral_offset: 0.0,
                confidence: 0.8,
                timestamp: i as f64 * 0.033,
            };
            let result = detector.update_with_position(position);

            // Should detect lane change at some point
            if result.lane_change.is_some() {
                let event = result.lane_change.unwrap();
                assert_eq!(event.from_lane, 1);
                assert_eq!(event.to_lane, 2);
                assert_eq!(event.direction, Direction::Right);
                return; // Test passed
            }
        }
    }
}
