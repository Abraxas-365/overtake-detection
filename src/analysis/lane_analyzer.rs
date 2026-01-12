// src/analysis/lane_analyzer.rs

use crate::analysis::position_estimator::{PositionEstimator, PositionSmoother};
use crate::analysis::state_machine::LaneChangeStateMachine;
use crate::types::{Lane, LaneChangeConfig, LaneChangeEvent, VehicleState};
use tracing::debug;

/// High-level analyzer for lane change detection (matching Python's LaneChangeAnalyzer)
pub struct LaneChangeAnalyzer {
    position_estimator: PositionEstimator,
    smoother: PositionSmoother,
    state_machine: LaneChangeStateMachine,
    config: LaneChangeConfig,
    last_state: Option<VehicleState>,
}

impl LaneChangeAnalyzer {
    pub fn new(config: LaneChangeConfig) -> Self {
        let position_estimator = PositionEstimator::new(config.reference_y_ratio);
        let smoother = PositionSmoother::new(config.smoothing_alpha);
        let state_machine = LaneChangeStateMachine::new(config.clone());

        Self {
            position_estimator,
            smoother,
            state_machine,
            config,
            last_state: None,
        }
    }

    /// Analyze lanes for lane change events
    ///
    /// Performs the full analysis pipeline:
    /// 1. Estimate vehicle position from lanes
    /// 2. Apply temporal smoothing
    /// 3. Update state machine
    /// 4. Return event if lane change detected
    pub fn analyze(
        &mut self,
        lanes: &[Lane],
        frame_width: u32,
        frame_height: u32,
        frame_id: u64,
        timestamp_ms: f64,
    ) -> Option<LaneChangeEvent> {
        // Estimate raw vehicle position
        let mut raw_state = self
            .position_estimator
            .estimate(lanes, frame_width, frame_height);
        raw_state.frame_id = frame_id;
        raw_state.timestamp_ms = timestamp_ms;

        // Apply temporal smoothing
        let smoothed_state = self.smoother.smooth(raw_state);
        self.last_state = Some(smoothed_state);

        // Update state machine and check for lane change
        self.state_machine
            .update(&smoothed_state, frame_id, timestamp_ms)
    }

    /// Analyze pre-computed vehicle state for lane change
    pub fn analyze_with_state(
        &mut self,
        vehicle_state: &VehicleState,
        frame_id: u64,
        timestamp_ms: f64,
    ) -> Option<LaneChangeEvent> {
        self.state_machine
            .update(vehicle_state, frame_id, timestamp_ms)
    }

    /// Get current state machine state name
    pub fn current_state(&self) -> &str {
        self.state_machine.current_state()
    }

    /// Get last computed vehicle state
    pub fn last_vehicle_state(&self) -> Option<&VehicleState> {
        self.last_state.as_ref()
    }

    /// Reset analyzer for new video processing
    pub fn reset(&mut self) {
        self.state_machine.reset();
        self.smoother.reset();
        self.last_state = None;
    }

    /// Update source ID for new video
    pub fn set_source_id(&mut self, source_id: String) {
        self.state_machine.set_source_id(source_id);
    }

    /// Get the config
    pub fn config(&self) -> &LaneChangeConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Point;

    fn create_test_lanes(left_x: f32, right_x: f32) -> Vec<Lane> {
        vec![
            Lane {
                lane_id: 0,
                points: vec![
                    Point::new(left_x, 100.0),
                    Point::new(left_x + 10.0, 200.0),
                    Point::new(left_x + 20.0, 300.0),
                    Point::new(left_x + 30.0, 400.0),
                    Point::new(left_x + 40.0, 500.0),
                ],
                confidence: 0.9,
                position: None,
            },
            Lane {
                lane_id: 1,
                points: vec![
                    Point::new(right_x, 100.0),
                    Point::new(right_x - 10.0, 200.0),
                    Point::new(right_x - 20.0, 300.0),
                    Point::new(right_x - 30.0, 400.0),
                    Point::new(right_x - 40.0, 500.0),
                ],
                confidence: 0.9,
                position: None,
            },
        ]
    }

    #[test]
    fn test_analyzer_integration() {
        let config = LaneChangeConfig {
            min_frames_confirm: 2,
            cooldown_frames: 0,
            ..Default::default()
        };
        let mut analyzer = LaneChangeAnalyzer::new(config);

        let frame_width = 1280u32;
        let frame_height = 720u32;

        // Centered lanes (vehicle at x=640, lanes at 400 and 880)
        let centered_lanes = create_test_lanes(400.0, 880.0);

        // Process several centered frames
        for i in 0..5 {
            let event = analyzer.analyze(
                &centered_lanes,
                frame_width,
                frame_height,
                i,
                i as f64 * 33.3,
            );
            assert!(event.is_none());
        }

        assert_eq!(analyzer.current_state(), "CENTERED");

        // Check that we have a valid last state
        let last_state = analyzer.last_vehicle_state();
        assert!(last_state.is_some());
        assert!(last_state.unwrap().is_valid());
    }

    #[test]
    fn test_analyzer_reset() {
        let config = LaneChangeConfig::default();
        let mut analyzer = LaneChangeAnalyzer::new(config);

        // Do some processing
        let lanes = create_test_lanes(200.0, 680.0); // Off-center
        for i in 0..10 {
            analyzer.analyze(&lanes, 1280, 720, i, i as f64 * 33.3);
        }

        // Reset
        analyzer.reset();

        assert_eq!(analyzer.current_state(), "CENTERED");
        assert!(analyzer.last_vehicle_state().is_none());
    }
}
