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
    total_frames_processed: u64,

    // Offset tracking for pattern detection
    offset_history: Vec<f32>,
    baseline_offset: Option<f32>,
    baseline_samples: Vec<f32>,
    is_baseline_established: bool,
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
            total_frames_processed: 0,
            offset_history: Vec::with_capacity(50),
            baseline_offset: None,
            baseline_samples: Vec::with_capacity(60),
            is_baseline_established: false,
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

        // Handle cooldown
        if self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
            if self.cooldown_remaining == 0 {
                self.state = LaneChangeState::Centered;
                self.frames_in_state = 0;
                debug!("Cooldown ended");
            }
            return None;
        }

        // Check timeout
        if self.state == LaneChangeState::Drifting || self.state == LaneChangeState::Crossing {
            if let Some(start_time) = self.change_start_time {
                let elapsed = timestamp_ms - start_time;
                if elapsed > self.config.max_duration_ms {
                    warn!("Lane change timeout after {:.0}ms", elapsed);
                    self.reset_to_centered();
                    self.cooldown_remaining = 30;
                    return None;
                }
            }
        }

        if !vehicle_state.is_valid() {
            return None;
        }

        let lane_width = vehicle_state.lane_width.unwrap();
        let normalized_offset = vehicle_state.lateral_offset / lane_width;
        let abs_offset = normalized_offset.abs();

        // Update offset history
        self.offset_history.push(normalized_offset);
        if self.offset_history.len() > 50 {
            self.offset_history.remove(0);
        }

        // Establish baseline during centered state
        if self.state == LaneChangeState::Centered && !self.is_baseline_established {
            self.baseline_samples.push(normalized_offset);
            if self.baseline_samples.len() >= 30 {
                // Calculate median as baseline
                let mut sorted = self.baseline_samples.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                self.baseline_offset = Some(sorted[sorted.len() / 2]);
                self.is_baseline_established = true;
                info!("Baseline established: {:.3}", self.baseline_offset.unwrap());
            }
        }

        // Calculate offset relative to baseline
        let relative_offset = if let Some(baseline) = self.baseline_offset {
            (normalized_offset - baseline).abs()
        } else {
            abs_offset
        };

        // Track max offset during lane change
        if self.state == LaneChangeState::Drifting || self.state == LaneChangeState::Crossing {
            if relative_offset > self.max_offset_in_change {
                self.max_offset_in_change = relative_offset;
            }
        }

        let direction = Direction::from_offset(normalized_offset);
        let target_state = self.determine_target_state(relative_offset, abs_offset);

        debug!(
            "F{}: norm={:.3}, rel={:.3}, abs={:.3}, state={:?}→{:?}",
            frame_id, normalized_offset, relative_offset, abs_offset, self.state, target_state
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
        // Reset baseline to re-establish after lane change
        self.baseline_samples.clear();
        self.is_baseline_established = false;
    }

    fn determine_target_state(&self, relative_offset: f32, abs_offset: f32) -> LaneChangeState {
        // Use the higher of relative or absolute offset for detection
        let effective_offset = relative_offset.max(abs_offset * 0.8);

        match self.state {
            LaneChangeState::Centered => {
                if effective_offset >= self.config.drift_threshold {
                    // Verify it's sustained - check last few samples
                    if self.offset_history.len() >= 5 {
                        let recent_high = self
                            .offset_history
                            .iter()
                            .rev()
                            .take(5)
                            .filter(|o| o.abs() >= self.config.drift_threshold * 0.9)
                            .count();
                        if recent_high >= 3 {
                            return LaneChangeState::Drifting;
                        }
                    } else {
                        return LaneChangeState::Drifting;
                    }
                }
                LaneChangeState::Centered
            }
            LaneChangeState::Drifting => {
                if effective_offset >= self.config.crossing_threshold {
                    LaneChangeState::Crossing
                } else if effective_offset < self.config.drift_threshold * 0.5 {
                    // Check if returning to center
                    if self.is_returning_to_center() {
                        // If we never reached crossing threshold, just cancel
                        if self.max_offset_in_change < self.config.crossing_threshold * 0.9 {
                            LaneChangeState::Centered
                        } else {
                            // We did reach crossing, this is completion
                            LaneChangeState::Completed
                        }
                    } else {
                        LaneChangeState::Drifting
                    }
                } else {
                    LaneChangeState::Drifting
                }
            }
            LaneChangeState::Crossing => {
                // Complete when returning to center
                if effective_offset < self.config.drift_threshold * 0.6 {
                    LaneChangeState::Completed
                } else {
                    LaneChangeState::Crossing
                }
            }
            LaneChangeState::Completed => LaneChangeState::Centered,
        }
    }

    fn is_returning_to_center(&self) -> bool {
        if self.offset_history.len() < 8 {
            return false;
        }

        // Check if recent offsets are decreasing
        let recent: Vec<f32> = self
            .offset_history
            .iter()
            .rev()
            .take(4)
            .map(|x| x.abs())
            .collect();
        let previous: Vec<f32> = self
            .offset_history
            .iter()
            .rev()
            .skip(4)
            .take(4)
            .map(|x| x.abs())
            .collect();

        let recent_avg: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
        let previous_avg: f32 = previous.iter().sum::<f32>() / previous.len() as f32;

        recent_avg < previous_avg * 0.6
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
            "State: {:?} → {:?} at frame {} ({:.2}s)",
            from_state,
            target_state,
            frame_id,
            timestamp_ms / 1000.0
        );

        // Starting lane change
        if target_state == LaneChangeState::Drifting && from_state == LaneChangeState::Centered {
            self.change_direction = direction;
            self.change_start_frame = Some(frame_id);
            self.change_start_time = Some(timestamp_ms);
            self.max_offset_in_change = 0.0;
            info!(
                "🚗 Lane change started: {} at {:.2}s",
                direction.as_str(),
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

        // Handle completion
        if target_state == LaneChangeState::Completed {
            // Validation 1: Duration
            if let Some(dur) = duration_ms {
                if dur < self.config.min_duration_ms {
                    warn!("❌ Rejected: too short ({:.0}ms)", dur);
                    self.reset_to_centered();
                    self.cooldown_remaining = 30;
                    return None;
                }
            }

            // Validation 2: Max offset reached crossing threshold
            if self.max_offset_in_change < self.config.crossing_threshold * 0.85 {
                warn!(
                    "❌ Rejected: max offset {:.3} < threshold",
                    self.max_offset_in_change
                );
                self.reset_to_centered();
                self.cooldown_remaining = 30;
                return None;
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
                "✅ CONFIRMED: {} at {:.2}s (duration: {:.0}ms, max_offset: {:.2})",
                event.direction_name(),
                timestamp_ms / 1000.0,
                duration_ms.unwrap_or(0.0),
                self.max_offset_in_change
            );

            self.reset_to_centered();
            return Some(event);
        }

        None
    }

    fn calculate_confidence(&self, duration_ms: Option<f64>) -> f32 {
        let mut confidence: f32 = 0.6;

        // Higher max offset = more confident
        if self.max_offset_in_change > 0.6 {
            confidence += 0.2;
        } else if self.max_offset_in_change > 0.5 {
            confidence += 0.1;
        }

        // Good duration
        if let Some(dur) = duration_ms {
            if dur > 1500.0 && dur < 4000.0 {
                confidence += 0.15;
            }
        }

        confidence.min(0.95)
    }

    pub fn reset(&mut self) {
        self.reset_to_centered();
        self.cooldown_remaining = 0;
        self.total_frames_processed = 0;
        self.offset_history.clear();
        self.baseline_offset = None;
        self.baseline_samples.clear();
        self.is_baseline_established = false;
    }

    pub fn set_source_id(&mut self, source_id: String) {
        self.source_id = source_id;
    }
}
