// src/analysis/state_machine.rs

use crate::types::{Direction, LaneChangeConfig, LaneChangeEvent, LaneChangeState, VehicleState};
use tracing::{debug, info};

pub struct LaneChangeStateMachine {
    config: LaneChangeConfig,
    source_id: String,
    state: LaneChangeState,
    frames_in_state: u32,
    pending_state: Option<LaneChangeState>,
    pending_frames: u32,
    change_direction: Direction,
    change_start_frame: Option<u64>,
    change_start_time: Option<f64>,
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

    pub fn current_state(&self) -> &str {
        self.state.as_str()
    }

    pub fn update(
        &mut self,
        vehicle_state: &VehicleState,
        frame_id: u64,
        timestamp_ms: f64,
    ) -> Option<LaneChangeEvent> {
        if self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
            if self.cooldown_remaining == 0 {
                self.state = LaneChangeState::Centered;
                self.frames_in_state = 0;
            }
            return None;
        }

        if !vehicle_state.is_valid() {
            return None;
        }

        let lane_width = vehicle_state.lane_width.unwrap();
        let normalized_offset = (vehicle_state.lateral_offset / lane_width).abs();
        let direction = Direction::from_offset(vehicle_state.lateral_offset);

        let target_state = self.determine_target_state(normalized_offset);

        self.check_transition(target_state, direction, frame_id, timestamp_ms)
    }

    fn determine_target_state(&self, normalized_offset: f32) -> LaneChangeState {
        if normalized_offset >= self.config.crossing_threshold {
            LaneChangeState::Crossing
        } else if normalized_offset >= self.config.drift_threshold {
            LaneChangeState::Drifting
        } else {
            if self.state == LaneChangeState::Crossing {
                LaneChangeState::Completed
            } else {
                LaneChangeState::Centered
            }
        }
    }

    fn check_transition(
        &mut self,
        target_state: LaneChangeState,
        direction: Direction,
        frame_id: u64,
        timestamp_ms: f64,
    ) -> Option<LaneChangeEvent> {
        if target_state == self.state {
            self.pending_state = None;
            self.pending_frames = 0;
            self.frames_in_state += 1;
            return None;
        }

        if self.pending_state == Some(target_state) {
            self.pending_frames += 1;
        } else {
            self.pending_state = Some(target_state);
            self.pending_frames = 1;
        }

        if self.pending_frames < self.config.min_frames_confirm {
            return None;
        }

        self.execute_transition(target_state, direction, frame_id, timestamp_ms)
    }

    fn execute_transition(
        &mut self,
        target_state: LaneChangeState,
        direction: Direction,
        frame_id: u64,
        timestamp_ms: f64,
    ) -> Option<LaneChangeEvent> {
        let from_state = self.state;

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

        let duration_ms = if target_state == LaneChangeState::Completed {
            self.change_start_time.map(|start| timestamp_ms - start)
        } else {
            None
        };

        self.state = target_state;
        self.frames_in_state = 0;
        self.pending_state = None;
        self.pending_frames = 0;

        if target_state == LaneChangeState::Completed {
            self.cooldown_remaining = self.config.cooldown_frames;

            let mut event =
                LaneChangeEvent::new(timestamp_ms, frame_id, self.change_direction, 0.9);
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

            self.change_direction = Direction::Unknown;
            self.change_start_frame = None;
            self.change_start_time = None;

            return Some(event);
        }

        None
    }

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

    pub fn set_source_id(&mut self, source_id: String) {
        self.source_id = source_id;
    }
}
