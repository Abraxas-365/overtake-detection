// src/analysis/lane_analyzer.rs

use crate::analysis::boundary_detector::{CrossingType, LaneBoundaryCrossingDetector};
use crate::analysis::fallback_detector::FallbackLaneChangeDetector; // ← ADD THIS
use crate::analysis::position_estimator::{PositionEstimator, PositionSmoother};
use crate::analysis::state_machine::LaneChangeStateMachine;
use crate::types::{Lane, LaneChangeConfig, LaneChangeEvent, VehicleState};
use tracing::info; // ← ADD THIS

pub struct LaneChangeAnalyzer {
    position_estimator: PositionEstimator,
    smoother: PositionSmoother,
    state_machine: LaneChangeStateMachine,
    boundary_detector: LaneBoundaryCrossingDetector,
    fallback_detector: FallbackLaneChangeDetector, // ← ADD THIS
    config: LaneChangeConfig,
    last_state: Option<VehicleState>,
    frame_count: u64,
    valid_estimates: u64,
    was_stuck_last_frame: bool, // ← ADD THIS
}

impl LaneChangeAnalyzer {
    pub fn new(config: LaneChangeConfig) -> Self {
        let position_estimator = PositionEstimator::new(config.reference_y_ratio);
        let smoother = PositionSmoother::new(config.smoothing_alpha);
        let state_machine = LaneChangeStateMachine::new(config.clone());
        let boundary_detector = LaneBoundaryCrossingDetector::new();
        let fallback_detector = FallbackLaneChangeDetector::new(); // ← ADD THIS

        Self {
            position_estimator,
            smoother,
            state_machine,
            boundary_detector,
            fallback_detector, // ← ADD THIS
            config,
            last_state: None,
            frame_count: 0,
            valid_estimates: 0,
            was_stuck_last_frame: false, // ← ADD THIS
        }
    }

    pub fn analyze(
        &mut self,
        lanes: &[Lane],
        frame_rgb: &[u8], // ← CHANGE SIGNATURE TO ACCEPT RAW FRAME
        frame_width: u32,
        frame_height: u32,
        frame_id: u64,
        timestamp_ms: f64,
    ) -> Option<LaneChangeEvent> {
        self.frame_count += 1;

        // Update curve detector with current lanes
        let _is_in_curve = self.state_machine.update_curve_detector(lanes);

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

        // ====================================================================
        // FALLBACK DETECTION: Check if position is stuck
        // ====================================================================
        let is_stuck = if let Some(normalized) = smoothed_state.normalized_offset() {
            self.fallback_detector.is_position_stuck(normalized)
        } else {
            false
        };

        if is_stuck {
            // Try optical flow fallback
            if let Some(fallback_detection) = self.fallback_detector.detect_with_optical_flow(
                frame_rgb,
                frame_width as usize,
                frame_height as usize,
                frame_id,
                timestamp_ms,
            ) {
                info!(
                    "🔄 FALLBACK LANE CHANGE: {:?} via {:?} (conf={:.2}, flow={:.1}px)",
                    fallback_detection.direction,
                    fallback_detection.method,
                    fallback_detection.confidence,
                    fallback_detection.accumulated_flow
                );

                // Create event from fallback
                let start_frame = self
                    .fallback_detector
                    .fallback_start_frame
                    .unwrap_or(frame_id);
                let event = fallback_detection.to_event(
                    timestamp_ms,
                    start_frame,
                    frame_id,
                    self.state_machine.source_id.clone(),
                );

                self.was_stuck_last_frame = true;
                return Some(event);
            }
        }

        // Check for position jump when detection "unsticks"
        if let Some(prev_position) = self.fallback_detector.get_last_valid_position() {
            if !is_stuck && self.was_stuck_last_frame {
                if let Some(current_normalized) = smoothed_state.normalized_offset() {
                    if let Some(jump_detection) = self
                        .fallback_detector
                        .detect_position_jump(prev_position, current_normalized)
                    {
                        info!(
                            "🔄 FALLBACK POSITION JUMP: {:?} (jump={:.1}%)",
                            jump_detection.direction,
                            jump_detection.accumulated_flow * 100.0
                        );

                        let event = jump_detection.to_event(
                            timestamp_ms,
                            frame_id.saturating_sub(5),
                            frame_id,
                            self.state_machine.source_id.clone(),
                        );
                        self.was_stuck_last_frame = false;
                        return Some(event);
                    }
                }
            }
        }

        self.was_stuck_last_frame = is_stuck;

        // ====================================================================
        // NORMAL DETECTION PATH (if not stuck)
        // ====================================================================

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

    fn get_lane_boundaries(&self, lanes: &[Lane], frame_height: u32) -> (Option<f32>, Option<f32>) {
        let reference_y = frame_height as f32 * self.config.reference_y_ratio;

        let mut left_x = None;
        let mut right_x = None;

        let vehicle_x = 640.0;

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
        self.fallback_detector.reset(); // ← ADD THIS
        self.last_state = None;
        self.frame_count = 0;
        self.valid_estimates = 0;
        self.was_stuck_last_frame = false; // ← ADD THIS
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

    pub fn is_stuck(&self) -> bool {
        self.fallback_detector.is_stuck()
    }
}
