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
    max_offset_in_change: f32,
    offset_history: Vec<f32>,
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
            max_offset_in_change: 0.0,
            offset_history: Vec::with_capacity(20),
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
        // Handle cooldown period
        if self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
            if self.cooldown_remaining == 0 {
                self.state = LaneChangeState::Centered;
                self.frames_in_state = 0;
                debug!("Cooldown ended, returning to CENTERED");
            }
            return None;
        }

        // Simple validity check - just need lane width
        if !vehicle_state.is_valid() {
            debug!("Frame {}: Invalid vehicle state, skipping", frame_id);
            return None;
        }

        let lane_width = vehicle_state.lane_width.unwrap();
        let normalized_offset = (vehicle_state.lateral_offset / lane_width).abs();
        let direction = Direction::from_offset(vehicle_state.lateral_offset);

        // Update offset history
        self.offset_history.push(normalized_offset);
        if self.offset_history.len() > 20 {
            self.offset_history.remove(0);
        }

        // Track max offset during lane change
        if self.state == LaneChangeState::Drifting || self.state == LaneChangeState::Crossing {
            if normalized_offset > self.max_offset_in_change {
                self.max_offset_in_change = normalized_offset;
            }
        }

        // Determine target state with hysteresis
        let target_state = self.determine_target_state_with_hysteresis(normalized_offset);

        debug!(
            "Frame {}: offset={:.3} ({:.1}px), width={:.0}, state={:?}→{:?}, pending={}",
            frame_id,
            normalized_offset,
            vehicle_state.lateral_offset,
            lane_width,
            self.state,
            target_state,
            self.pending_frames
        );

        self.check_transition(target_state, direction, frame_id, timestamp_ms)
    }

    fn determine_target_state_with_hysteresis(&self, normalized_offset: f32) -> LaneChangeState {
        match self.state {
            LaneChangeState::Centered => {
                if normalized_offset >= self.config.drift_threshold {
                    LaneChangeState::Drifting
                } else {
                    LaneChangeState::Centered
                }
            }
            LaneChangeState::Drifting => {
                if normalized_offset >= self.config.crossing_threshold {
                    LaneChangeState::Crossing
                } else if normalized_offset
                    < self.config.drift_threshold * self.config.hysteresis_factor
                {
                    // Only cancel if clearly back to center
                    if self.is_trending_to_center() {
                        LaneChangeState::Centered
                    } else {
                        LaneChangeState::Drifting
                    }
                } else {
                    LaneChangeState::Drifting
                }
            }
            LaneChangeState::Crossing => {
                // Complete when offset drops back (we've moved to new lane)
                if normalized_offset < self.config.drift_threshold {
                    LaneChangeState::Completed
                } else {
                    LaneChangeState::Crossing
                }
            }
            LaneChangeState::Completed => LaneChangeState::Centered,
        }
    }

    fn is_trending_to_center(&self) -> bool {
        if self.offset_history.len() < 6 {
            return false;
        }

        let recent: f32 = self.offset_history.iter().rev().take(3).sum::<f32>() / 3.0;
        let previous: f32 = self
            .offset_history
            .iter()
            .rev()
            .skip(3)
            .take(3)
            .sum::<f32>()
            / 3.0;

        recent < previous * 0.7
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

        info!(
            "State transition: {:?} → {:?} at frame {}",
            from_state, target_state, frame_id
        );

        // Starting a lane change
        if target_state == LaneChangeState::Drifting && from_state == LaneChangeState::Centered {
            self.change_direction = direction;
            self.change_start_frame = Some(frame_id);
            self.change_start_time = Some(timestamp_ms);
            self.max_offset_in_change = 0.0;
            info!(
                "🚗 Lane change started: {} at frame {}",
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

            let start_frame = self.change_start_frame.unwrap_or(frame_id);
            let confidence = self.calculate_confidence(duration_ms);

            let mut event = LaneChangeEvent::new(
                timestamp_ms,
                start_frame,
                frame_id,
                self.change_direction,
                confidence,
            );
            event.duration_ms = duration_ms;
            event.source_id = self.source_id.clone();

            info!(
                "✅ Lane change completed: {} (duration: {:.0}ms, confidence: {:.2}) at frame {}",
                event.direction_name(),
                duration_ms.unwrap_or(0.0),
                confidence,
                frame_id
            );

            // Reset
            self.change_direction = Direction::Unknown;
            self.change_start_frame = None;
            self.change_start_time = None;
            self.max_offset_in_change = 0.0;
            self.offset_history.clear();

            return Some(event);
        }

        None
    }

    fn calculate_confidence(&self, duration_ms: Option<f64>) -> f32 {
        let mut confidence: f32 = 0.7;

        if self.max_offset_in_change > self.config.crossing_threshold {
            confidence += 0.15;
        }

        if let Some(duration) = duration_ms {
            if duration > 500.0 && duration < 5000.0 {
                confidence += 0.1;
            }
        }

        confidence.min(0.95)
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
        self.max_offset_in_change = 0.0;
        self.offset_history.clear();
    }

    pub fn set_source_id(&mut self, source_id: String) {
        self.source_id = source_id;
    }
}
