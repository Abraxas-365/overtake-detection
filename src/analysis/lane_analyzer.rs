// src/analysis/lane_analyzer.rs

use crate::analysis::boundary_detector::{CrossingType, LaneBoundaryCrossingDetector};
use crate::analysis::curve_detector::CurveDetector;
use crate::analysis::position_estimator::{PositionEstimator, PositionSmoother};
use crate::analysis::state_machine::LaneChangeStateMachine;
use crate::types::{CurveInfo, Lane, LaneChangeConfig, LaneChangeEvent, VehicleState};

pub struct LaneChangeAnalyzer {
    position_estimator: PositionEstimator,
    smoother: PositionSmoother,
    state_machine: LaneChangeStateMachine,
    boundary_detector: LaneBoundaryCrossingDetector,
    curve_detector: CurveDetector,
    config: LaneChangeConfig,
    last_state: Option<VehicleState>,
    frame_count: u64,
    valid_estimates: u64,
}

impl LaneChangeAnalyzer {
    pub fn new(config: LaneChangeConfig) -> Self {
        let position_estimator = PositionEstimator::new(config.reference_y_ratio);
        let smoother = PositionSmoother::new(config.smoothing_alpha);
        let state_machine = LaneChangeStateMachine::new(config.clone());
        let boundary_detector = LaneBoundaryCrossingDetector::new();
        let curve_detector = CurveDetector::new();

        Self {
            position_estimator,
            smoother,
            state_machine,
            boundary_detector,
            curve_detector,
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

        // Update curve detector with current lanes
        // Prefix with underscore to silence warning about unused variable
        let _is_in_curve = self.curve_detector.is_in_curve(lanes);

        // Pass curve info to state machine
        self.state_machine.update_curve_detector(lanes);

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

        // Detect lane boundary crossing
        let (left_x, right_x) = self.get_lane_boundaries(lanes, frame_height);
        let vehicle_x = frame_width as f32 / 2.0;

        let crossing_type = self
            .boundary_detector
            .detect_crossing(left_x, right_x, vehicle_x);

        self.last_state = Some(smoothed_state);

        // Update state machine with crossing info
        self.state_machine
            .update(&smoothed_state, frame_id, timestamp_ms, crossing_type)
    }

    // 🆕 Get curve information for API
    pub fn get_curve_info(&self) -> CurveInfo {
        self.curve_detector.get_curve_info()
    }

    // 🆕 Check if currently in curve
    pub fn is_in_curve(&self) -> bool {
        let info = self.get_curve_info();
        info.is_curve
    }

    fn get_lane_boundaries(&self, lanes: &[Lane], frame_height: u32) -> (Option<f32>, Option<f32>) {
        let reference_y = frame_height as f32 * self.config.reference_y_ratio;

        // Find left and right ego lanes
        let mut left_x = None;
        let mut right_x = None;

        let vehicle_x = 640.0; // Approximate center, adjust if needed

        for lane in lanes {
            if let Some(x) = lane.get_x_at_y(reference_y) {
                if x < vehicle_x && (left_x.is_none() || x > left_x.unwrap()) {
                    left_x = Some(x);
                } else if x > vehicle_x && (right_x.is_none() || x < right_x.unwrap()) {
                    right_x = Some(x);
                }
            }
        }

        (left_x, right_x)
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
        self.boundary_detector.reset();
        self.curve_detector.reset();
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

    pub fn get_stats(&self) -> (u64, u64, f32) {
        let valid_ratio = if self.frame_count > 0 {
            self.valid_estimates as f32 / self.frame_count as f32
        } else {
            0.0
        };
        (self.frame_count, self.valid_estimates, valid_ratio)
    }
}
