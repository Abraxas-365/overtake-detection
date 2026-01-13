// src/analysis/state_machine.rs

use crate::types::{Direction, LaneChangeConfig, LaneChangeEvent, LaneChangeState, VehicleState};
use tracing::{debug, info, warn};

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
    total_frames_processed: u64,
    /// Count of consecutive frames with both lanes detected
    both_lanes_streak: u32,
    /// Count of frames with both lanes during lane change
    both_lanes_during_change: u32,
    /// Total frames during change
    frames_during_change: u32,
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
            offset_history: Vec::with_capacity(30),
            total_frames_processed: 0,
            both_lanes_streak: 0,
            both_lanes_during_change: 0,
            frames_during_change: 0,
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
        self.total_frames_processed += 1;

        // Skip initial frames
        if self.total_frames_processed < self.config.skip_initial_frames {
            return None;
        }

        // Track both lanes streak
        if vehicle_state.both_lanes_detected {
            self.both_lanes_streak += 1;
        } else {
            self.both_lanes_streak = 0;
        }

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

        // Check for timeout
        if self.state == LaneChangeState::Drifting || self.state == LaneChangeState::Crossing {
            if let Some(start_time) = self.change_start_time {
                let elapsed = timestamp_ms - start_time;
                if elapsed > self.config.max_duration_ms {
                    warn!(
                        "Lane change timeout after {:.0}ms, resetting to CENTERED",
                        elapsed
                    );
                    self.reset_to_centered();
                    self.cooldown_remaining = self.config.cooldown_frames / 2;
                    return None;
                }
            }
        }

        if !vehicle_state.is_valid() {
            return None;
        }

        // CRITICAL: If require_both_lanes is true, only process when both lanes detected
        if self.config.require_both_lanes && !vehicle_state.both_lanes_detected {
            // If we're in the middle of a lane change, track but don't update state
            if self.state == LaneChangeState::Drifting || self.state == LaneChangeState::Crossing {
                self.frames_during_change += 1;
                debug!(
                    "Frame {}: Only one lane detected during lane change, skipping state update",
                    frame_id
                );
            }
            return None;
        }

        // Track frames during lane change
        if self.state == LaneChangeState::Drifting || self.state == LaneChangeState::Crossing {
            self.frames_during_change += 1;
            if vehicle_state.both_lanes_detected {
                self.both_lanes_during_change += 1;
            }
        }

        let lane_width = vehicle_state.lane_width.unwrap();
        let normalized_offset = (vehicle_state.lateral_offset / lane_width).abs();
        let direction = Direction::from_offset(vehicle_state.lateral_offset);

        // Update offset history
        self.offset_history.push(normalized_offset);
        if self.offset_history.len() > 30 {
            self.offset_history.remove(0);
        }

        // Track max offset
        if self.state == LaneChangeState::Drifting || self.state == LaneChangeState::Crossing {
            if normalized_offset > self.max_offset_in_change {
                self.max_offset_in_change = normalized_offset;
            }
        }

        let target_state = self.determine_target_state_with_hysteresis(normalized_offset);

        debug!(
            "Frame {}: offset={:.3} ({:.1}px), width={:.0}, both_lanes={}, state={:?}→{:?}, pending={}",
            frame_id,
            normalized_offset,
            vehicle_state.lateral_offset,
            lane_width,
            vehicle_state.both_lanes_detected,
            self.state,
            target_state,
            self.pending_frames
        );

        self.check_transition(target_state, direction, frame_id, timestamp_ms)
    }

    fn reset_to_centered(&mut self) {
        self.state = LaneChangeState::Centered;
        self.frames_in_state = 0;
        self.pending_state = None;
        self.pending_frames = 0;
        self.change_direction = Direction::Unknown;
        self.change_start_frame = None;
        self.change_start_time = None;
        self.max_offset_in_change = 0.0;
        self.offset_history.clear();
        self.both_lanes_during_change = 0;
        self.frames_during_change = 0;
    }

    fn determine_target_state_with_hysteresis(&self, normalized_offset: f32) -> LaneChangeState {
        match self.state {
            LaneChangeState::Centered => {
                // Need stable high offset to start
                if normalized_offset >= self.config.drift_threshold {
                    // Check that offset has been consistently high
                    if self.offset_history.len() >= 5 {
                        let recent_avg: f32 = self.offset_history.iter().rev().take(5).sum::<f32>() / 5.0;
                        if recent_avg >= self.config.drift_threshold * 0.9 {
                            return LaneChangeState::Drifting;
                        }
                    } else {
                        return LaneChangeState::Drifting;
                    }
                }
                LaneChangeState::Centered
            }
            LaneChangeState::Drifting => {
                if normalized_offset >= self.config.crossing_threshold {
                    LaneChangeState::Crossing
                } else if normalized_offset < self.config.drift_threshold * self.config.hysteresis_factor {
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
                // Need offset to drop significantly to complete
                if normalized_offset < self.config.drift_threshold * 0.6 {
                    LaneChangeState::Completed
                } else {
                    LaneChangeState::Crossing
                }
            }
            LaneChangeState::Completed => {
                LaneChangeState::Centered
            }
        }
    }

    fn is_trending_to_center(&self) -> bool {
        if self.offset_history.len() < 8 {
            return false;
        }

        let recent: f32 = self.offset_history.iter().rev().take(4).sum::<f32>() / 4.0;
        let previous: f32 = self.offset_history.iter().rev().skip(4).take(4).sum::<f32>() / 4.0;

        recent < previous * 0.6
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

        if target_state == LaneChangeState::Drifting && from_state == LaneChangeState::Centered {
            self.change_direction = direction;
            self.change_start_frame = Some(frame_id);
            self.change_start_time = Some(timestamp_ms);
            self.max_offset_in_change = 0.0;
            self.both_lanes_during_change = 0;
            self.frames_during_change = 0;
            info!(
                "🚗 Lane change started: {} at frame {} ({:.2}s)",
                direction.as_str(),
                frame_id,
                timestamp_ms / 1000.0
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
            // VALIDATION 1: Check minimum duration
            if let Some(dur) = duration_ms {
                if dur < self.config.min_duration_ms {
                    warn!(
                        "❌ Lane change rejected: too short ({:.0}ms < {:.0}ms)",
                        dur, self.config.min_duration_ms
                    );
                    self.reset_to_centered();
                    self.cooldown_remaining = self.config.cooldown_frames / 2;
                    return None;
                }
            }

            // VALIDATION 2: Check that we actually crossed threshold
            if self.max_offset_in_change < self.config.crossing_threshold {
                warn!(
                    "❌ Lane change rejected: max offset {:.2} never reached crossing threshold {:.2}",
                    self.max_offset_in_change, self.config.crossing_threshold
                );
                self.reset_to_centered();
                self.cooldown_remaining = self.config.cooldown_frames / 2;
                return None;
            }

            // VALIDATION 3: Check both lanes detection ratio during change
            if self.config.require_both_lanes && self.frames_during_change > 0 {
                let both_lanes_ratio = self.both_lanes_during_change as f32 / self.frames_during_change as f32;
                if both_lanes_ratio < 0.5 {
                    warn!(
                        "❌ Lane change rejected: only {:.1}% of frames had both lanes detected",
                        both_lanes_ratio * 100.0
                    );
                    self.reset_to_centered();
                    self.cooldown_remaining = self.config.cooldown_frames / 2;
                    return None;
                }
            }

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
                "✅ Lane change CONFIRMED: {} (duration: {:.0}ms, confidence: {:.2}, max_offset: {:.2}, both_lanes: {:.0}%) at frame {} ({:.2}s)",
                event.direction_name(),
                duration_ms.unwrap_or(0.0),
                confidence,
                self.max_offset_in_change,
                if self.frames_during_change > 0 { 
                    (self.both_lanes_during_change as f32 / self.frames_during_change as f32) * 100.0 
                } else { 0.0 },
                frame_id,
                timestamp_ms / 1000.0
            );

            self.reset_to_centered();

            return Some(event);
        }

        None
    }

    fn calculate_confidence(&self, duration_ms: Option<f64>) -> f32 {
        let mut confidence: f32 = 0.5;

        // Higher offset = more confident
        if self.max_offset_in_change > self.config.crossing_threshold * 1.3 {
            confidence += 0.25;
        } else if self.max_offset_in_change > self.config.crossing_threshold * 1.1 {
            confidence += 0.15;
        } else {
            confidence += 0.05;
        }

        // Good duration range (2-4 seconds)
        if let Some(duration) = duration_ms {
            if duration > 2000.0 && duration < 4000.0 {
                confidence += 0.15;
            } else if duration > 1500.0 && duration < 5000.0 {
                confidence += 0.05;
            }
        }

        // Good both_lanes ratio
        if self.frames_during_change > 0 {
            let ratio = self.both_lanes_during_change as f32 / self.frames_during_change as f32;
            if ratio > 0.8 {
                confidence += 0.1;
            } else if ratio > 0.6 {
                confidence += 0.05;
            }
        }

        confidence.min(0.95)
    }

    pub fn reset(&mut self) {
        self.reset_to_centered();
        self.cooldown_remaining = 0;
        self.total_frames_processed = 0;
        self.both_lanes_streak = 0;
    }

    pub fn set_source_id(&mut self, source_id: String) {
        self.source_id = source_id;
    }
}

