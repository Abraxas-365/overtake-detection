// src/analysis/state_machine.rs

use crate::types::{Direction, LaneChangeConfig, LaneChangeEvent, LaneChangeState, VehicleState};
use std::collections::HashMap;
use tracing::{debug, info};

/// State machine for lane change detection (matching Python implementation)
pub struct LaneChangeStateMachine {
    config: LaneChangeConfig,
    source_id: String,

    // Current state
    state: LaneChangeState,
    frames_in_state: u32,

    // Pending state transition
    pending_state: Option<LaneChangeState>,
    pending_frames: u32,

    // Lane change tracking
    change_direction: Direction,
    change_start_frame: Option<u64>,
    change_start_time: Option<f64>,

    // Cooldown tracking
    cooldown_remaining: u32,
}

impl LaneChangeStateMachine {
    pub fn new(config: LaneChangeConfig) -> Self {
        Self {
            config,
            source_id: String::new(),
            state: LaneChangeState::Centered,
            frames_in_state: 0,
            pending_state: None,
            pending_frames: 0,
            change_direction: Direction::Unknown,
            change_start_frame: None,
            change_start_time: None,
            cooldown_remaining: 0,
        }
    }

    pub fn with_source_id(mut self, source_id: String) -> Self {
        self.source_id = source_id;
        self
    }

    /// Get current state name
    pub fn current_state(&self) -> &str {
        self.state.as_str()
    }

    /// Update state machine with new vehicle state
    pub fn update(
        &mut self,
        vehicle_state: &VehicleState,
        frame_id: u64,
        timestamp_ms: f64,
    ) -> Option<LaneChangeEvent> {
        // Handle cooldown period
        if self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
            if self.cooldown_remaining == 0 {
                // Reset to CENTERED after cooldown
                self.state = LaneChangeState::Centered;
                self.frames_in_state = 0;
            }
            return None;
        }

        // Check if we have valid lane information
        if !vehicle_state.is_valid() {
            return None;
        }

        // Calculate normalized offset (as fraction of lane width)
        let lane_width = vehicle_state.lane_width.unwrap(); // Safe due to is_valid check
        let normalized_offset = (vehicle_state.lateral_offset / lane_width).abs();
        let direction = Direction::from_offset(vehicle_state.lateral_offset);

        // Determine target state based on offset magnitude
        let target_state = self.determine_target_state(normalized_offset);

        // Check for state transition
        self.check_transition(target_state, direction, frame_id, timestamp_ms)
    }

    /// Determine target state based on lateral offset magnitude
    fn determine_target_state(&self, normalized_offset: f32) -> LaneChangeState {
        if normalized_offset >= self.config.crossing_threshold {
            LaneChangeState::Crossing
        } else if normalized_offset >= self.config.drift_threshold {
            LaneChangeState::Drifting
        } else {
            // Vehicle is centered
            if self.state == LaneChangeState::Crossing {
                // Was crossing, now centered = completed
                LaneChangeState::Completed
            } else {
                LaneChangeState::Centered
            }
        }
    }

    /// Check if state transition should occur
    fn check_transition(
        &mut self,
        target_state: LaneChangeState,
        direction: Direction,
        frame_id: u64,
        timestamp_ms: f64,
    ) -> Option<LaneChangeEvent> {
        // Same state as current - reset pending and increment counter
        if target_state == self.state {
            self.pending_state = None;
            self.pending_frames = 0;
            self.frames_in_state += 1;
            return None;
        }

        // Different target state - accumulate confirmation frames
        if self.pending_state == Some(target_state) {
            self.pending_frames += 1;
        } else {
            self.pending_state = Some(target_state);
            self.pending_frames = 1;
        }

        // Check if we have enough confirmation frames
        if self.pending_frames < self.config.min_frames_confirm {
            return None;
        }

        // Transition confirmed - execute and potentially emit event
        self.execute_transition(target_state, direction, frame_id, timestamp_ms)
    }

    /// Execute state transition and create event if applicable
    fn execute_transition(
        &mut self,
        target_state: LaneChangeState,
        direction: Direction,
        frame_id: u64,
        timestamp_ms: f64,
    ) -> Option<LaneChangeEvent> {
        let from_state = self.state;

        // Track lane change start
        if target_state == LaneChangeState::Drifting && from_state == LaneChangeState::Centered {
            self.change_direction = direction;
            self.change_start_frame = Some(frame_id);
            self.change_start_time = Some(timestamp_ms);
            debug!(
                "Lane change started: {} at frame {}",
                direction.as_str(),
                frame_id
            );
        }

        // Calculate duration for completed lane changes
        let duration_ms = if target_state == LaneChangeState::Completed {
            self.change_start_time.map(|start| timestamp_ms - start)
        } else {
            None
        };

        // Update state
        self.state = target_state;
        self.frames_in_state = 0;
        self.pending_state = None;
        self.pending_frames = 0;

        // Handle completed state - start cooldown and emit event
        if target_state == LaneChangeState::Completed {
            self.cooldown_remaining = self.config.cooldown_frames;

            let mut event = LaneChangeEvent::new(
                timestamp_ms,
                frame_id,
                self.change_direction,
                0.9, // Could be calculated from detection confidences
            );
            event.duration_ms = duration_ms;
            event.source_id = self.source_id.clone();
            event.metadata.insert(
                "start_frame".to_string(),
                serde_json::json!(self.change_start_frame),
            );
            event
                .metadata
                .insert("end_frame".to_string(), serde_json::json!(frame_id));

            info!(
                "Lane change completed: {} (duration: {:.0}ms) at frame {}",
                event.direction_name(),
                duration_ms.unwrap_or(0.0),
                frame_id
            );

            // Reset lane change tracking
            self.change_direction = Direction::Unknown;
            self.change_start_frame = None;
            self.change_start_time = None;

            return Some(event);
        }

        None
    }

    /// Reset state machine to initial state
    pub fn reset(&mut self) {
        self.state = LaneChangeState::Centered;
        self.frames_in_state = 0;
        self.pending_state = None;
        self.pending_frames = 0;
        self.change_direction = Direction::Unknown;
        self.change_start_frame = None;
        self.change_start_time = None;
        self.cooldown_remaining = 0;
    }

    /// Set source ID for events
    pub fn set_source_id(&mut self, source_id: String) {
        self.source_id = source_id;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_config() -> LaneChangeConfig {
        LaneChangeConfig {
            drift_threshold: 0.2,
            crossing_threshold: 0.4,
            min_frames_confirm: 3,
            cooldown_frames: 10,
            smoothing_alpha: 0.3,
            reference_y_ratio: 0.8,
        }
    }

    fn create_vehicle_state(lateral_offset: f32, lane_width: f32) -> VehicleState {
        VehicleState {
            lateral_offset,
            lane_width: Some(lane_width),
            heading_offset: 0.0,
            frame_id: 0,
            timestamp_ms: 0.0,
        }
    }

    #[test]
    fn test_initial_state_is_centered() {
        let fsm = LaneChangeStateMachine::new(create_config());
        assert_eq!(fsm.current_state(), "CENTERED");
    }

    #[test]
    fn test_no_event_when_centered() {
        let mut fsm = LaneChangeStateMachine::new(create_config());
        let centered = create_vehicle_state(0.0, 400.0);

        let event = fsm.update(&centered, 0, 0.0);

        assert!(event.is_none());
        assert_eq!(fsm.current_state(), "CENTERED");
    }

    #[test]
    fn test_drift_detection_requires_multiple_frames() {
        let mut fsm = LaneChangeStateMachine::new(create_config());

        // 25% offset = drifting
        let drifting = create_vehicle_state(100.0, 400.0);

        // First two frames shouldn't trigger transition
        let event1 = fsm.update(&drifting, 0, 0.0);
        let event2 = fsm.update(&drifting, 1, 33.3);

        assert!(event1.is_none());
        assert!(event2.is_none());
        assert_eq!(fsm.current_state(), "CENTERED");

        // Third frame should trigger transition to DRIFTING
        let event3 = fsm.update(&drifting, 2, 66.6);

        assert!(event3.is_none()); // No event for DRIFTING
        assert_eq!(fsm.current_state(), "DRIFTING");
    }

    #[test]
    fn test_complete_lane_change_emits_event() {
        let config = LaneChangeConfig {
            drift_threshold: 0.2,
            crossing_threshold: 0.4,
            min_frames_confirm: 2,
            cooldown_frames: 0,
            ..Default::default()
        };
        let mut fsm = LaneChangeStateMachine::new(config);

        let mut events = Vec::new();
        let mut frame_id = 0u64;

        // Phase 1: Start drifting (25% offset)
        let drifting = create_vehicle_state(100.0, 400.0);
        for _ in 0..3 {
            if let Some(e) = fsm.update(&drifting, frame_id, frame_id as f64 * 33.3) {
                events.push(e);
            }
            frame_id += 1;
        }
        assert_eq!(fsm.current_state(), "DRIFTING");

        // Phase 2: Cross boundary (50% offset)
        let crossing = create_vehicle_state(200.0, 400.0);
        for _ in 0..3 {
            if let Some(e) = fsm.update(&crossing, frame_id, frame_id as f64 * 33.3) {
                events.push(e);
            }
            frame_id += 1;
        }
        assert_eq!(fsm.current_state(), "CROSSING");

        // Phase 3: Re-center in new lane
        let centered = create_vehicle_state(10.0, 400.0);
        for _ in 0..3 {
            if let Some(e) = fsm.update(&centered, frame_id, frame_id as f64 * 33.3) {
                events.push(e);
            }
            frame_id += 1;
        }

        // Should have emitted a COMPLETED event
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].direction, Direction::Right);
        assert!(events[0].duration_ms.is_some());
        assert!(events[0].duration_ms.unwrap() > 0.0);
    }

    #[test]
    fn test_direction_tracking_left() {
        let config = LaneChangeConfig {
            drift_threshold: 0.2,
            crossing_threshold: 0.4,
            min_frames_confirm: 2,
            cooldown_frames: 0,
            ..Default::default()
        };
        let mut fsm = LaneChangeStateMachine::new(config);

        let mut event: Option<LaneChangeEvent> = None;
        let mut frame_id = 0u64;

        // Negative offset = drifting left
        let drifting_left = create_vehicle_state(-100.0, 400.0);
        for _ in 0..3 {
            if let Some(e) = fsm.update(&drifting_left, frame_id, frame_id as f64 * 33.3) {
                event = Some(e);
            }
            frame_id += 1;
        }

        let crossing_left = create_vehicle_state(-200.0, 400.0);
        for _ in 0..3 {
            if let Some(e) = fsm.update(&crossing_left, frame_id, frame_id as f64 * 33.3) {
                event = Some(e);
            }
            frame_id += 1;
        }

        let centered = create_vehicle_state(-10.0, 400.0);
        for _ in 0..3 {
            if let Some(e) = fsm.update(&centered, frame_id, frame_id as f64 * 33.3) {
                event = Some(e);
            }
            frame_id += 1;
        }

        assert!(event.is_some());
        let e = event.unwrap();
        assert_eq!(e.direction, Direction::Left);
        assert_eq!(e.direction_name(), "LEFT");
    }

    #[test]
    fn test_reset_clears_state() {
        let mut fsm = LaneChangeStateMachine::new(create_config());

        // Get into DRIFTING state
        let drifting = create_vehicle_state(100.0, 400.0);
        for i in 0..5 {
            fsm.update(&drifting, i, i as f64 * 33.3);
        }
        assert_eq!(fsm.current_state(), "DRIFTING");

        // Reset
        fsm.reset();
        assert_eq!(fsm.current_state(), "CENTERED");
    }
}
