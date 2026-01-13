// src/analysis/lane_analyzer.rs

use crate::analysis::position_estimator::{PositionEstimator, PositionSmoother};
use crate::analysis::state_machine::LaneChangeStateMachine;
use crate::types::{Lane, LaneChangeConfig, LaneChangeEvent, VehicleState};

pub struct LaneChangeAnalyzer {
    position_estimator: PositionEstimator,
    smoother: PositionSmoother,
    state_machine: LaneChangeStateMachine,
    config: LaneChangeConfig,
    last_state: Option<VehicleState>,
    /// Count of frames analyzed
    frame_count: u64,
    /// Count of valid position estimates
    valid_estimates: u64,
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
            frame_count: 0,
            valid_estimates: 0,
        }
    }

    pub fn analyze(
        &mut self,
        lanes: &[Lane],
        frame_width: u32,
        frame_height: u32,
        frame_id: u64,
        timestamp_ms: f64,
    ) -> Option<LaneChangeEvent> {
        self.frame_count += 1;

        // Get raw position estimate
        let mut raw_state = self
            .position_estimator
            .estimate(lanes, frame_width, frame_height);
        raw_state.frame_id = frame_id;
        raw_state.timestamp_ms = timestamp_ms;

        // Apply smoothing
        let smoothed_state = self.smoother.smooth(raw_state);

        if smoothed_state.is_valid() {
            self.valid_estimates += 1;
        }

        self.last_state = Some(smoothed_state);

        // Update state machine
        self.state_machine
            .update(&smoothed_state, frame_id, timestamp_ms)
    }

    pub fn current_state(&self) -> &str {
        self.state_machine.current_state()
    }

    pub fn last_vehicle_state(&self) -> Option<&VehicleState> {
        self.last_state.as_ref()
    }

    pub fn reset(&mut self) {
        self.state_machine.reset();
        self.smoother.reset();
        self.position_estimator.reset();
        self.last_state = None;
        self.frame_count = 0;
        self.valid_estimates = 0;
    }

    pub fn set_source_id(&mut self, source_id: String) {
        self.state_machine.set_source_id(source_id);
    }

    pub fn config(&self) -> &LaneChangeConfig {
        &self.config
    }

    /// Get statistics about detection quality
    pub fn get_stats(&self) -> (u64, u64, f32) {
        let valid_ratio = if self.frame_count > 0 {
            self.valid_estimates as f32 / self.frame_count as f32
        } else {
            0.0
        };
        (self.frame_count, self.valid_estimates, valid_ratio)
    }
}
