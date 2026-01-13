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

    // Baseline tracking - MUST be established before detection
    offset_history: Vec<f32>,
    baseline_offset: f32,
    baseline_samples: Vec<f32>,
    is_baseline_established: bool,
    frames_since_baseline: u32,

    // Track if we're in a stable centered period
    stable_centered_frames: u32,
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
            offset_history: Vec::with_capacity(60),
            baseline_offset: 0.0,
            baseline_samples: Vec::with_capacity(90),
            is_baseline_established: false,
            frames_since_baseline: 0,
            stable_centered_frames: 0,
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

        // Skip initial frames completely
        if self.total_frames_processed < self.config.skip_initial_frames {
            return None;
        }

        // Handle cooldown
        if self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
            if self.cooldown_remaining == 0 {
                self.state = LaneChangeState::Centered;
                self.frames_in_state = 0;
            }
            return None;
        }

        // Check timeout for ongoing lane change
        if self.state == LaneChangeState::Drifting || self.state == LaneChangeState::Crossing {
            if let Some(start_time) = self.change_start_time {
                let elapsed = timestamp_ms - start_time;
                if elapsed > self.config.max_duration_ms {
                    warn!("⏰ Lane change timeout after {:.0}ms", elapsed);
                    self.reset_lane_change();
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
        if self.offset_history.len() > 60 {
            self.offset_history.remove(0);
        }

        // PHASE 1: Establish baseline (need stable centered driving)
        if !self.is_baseline_established {
            // Only collect samples when offset is relatively small (< 15%)
            if abs_offset < 0.15 {
                self.baseline_samples.push(normalized_offset);
                self.stable_centered_frames += 1;
            } else {
                // Large offset during baseline collection - could be curve, reset
                if self.stable_centered_frames < 30 {
                    self.baseline_samples.clear();
                    self.stable_centered_frames = 0;
                }
            }

            // Need 60 stable frames to establish baseline
            if self.baseline_samples.len() >= 60 && self.stable_centered_frames >= 60 {
                let mut sorted = self.baseline_samples.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                self.baseline_offset = sorted[sorted.len() / 2];
                self.is_baseline_established = true;
                self.frames_since_baseline = 0;
                info!(
                    "✅ Baseline established: {:.3} ({:.1}%) at frame {} ({:.1}s)",
                    self.baseline_offset,
                    self.baseline_offset * 100.0,
                    frame_id,
                    timestamp_ms / 1000.0
                );
            }
            return None;
        }

        // Count frames since baseline
        self.frames_since_baseline += 1;

        // Need some frames after baseline before detecting
        if self.frames_since_baseline < 30 {
            return None;
        }

        // Calculate deviation from baseline
        let deviation = (normalized_offset - self.baseline_offset).abs();

        // Track max offset during lane change
        if self.state == LaneChangeState::Drifting || self.state == LaneChangeState::Crossing {
            if deviation > self.max_offset_in_change {
                self.max_offset_in_change = deviation;
            }
        }

        // Track stable centered for curve rejection
        if deviation < 0.10 {
            self.stable_centered_frames += 1;
        } else {
            self.stable_centered_frames = 0;
        }

        let direction = Direction::from_offset(normalized_offset - self.baseline_offset);
        let target_state = self.determine_target_state(deviation);

        debug!(
            "F{}: offset={:.1}%, base={:.1}%, dev={:.1}%, state={:?}→{:?}",
            frame_id,
            normalized_offset * 100.0,
            self.baseline_offset * 100.0,
            deviation * 100.0,
            self.state,
            target_state
        );

        self.check_transition(target_state, direction, frame_id, timestamp_ms)
    }

    fn reset_lane_change(&mut self) {
        self.state = LaneChangeState::Centered;
        self.frames_in_state = 0;
        self.pending_state = None;
        self.pending_frames = 0;
        self.change_direction = Direction::Unknown;
        self.change_start_frame = None;
        self.change_start_time = None;
        self.max_offset_in_change = 0.0;
    }

    fn determine_target_state(&self, deviation: f32) -> LaneChangeState {
        match self.state {
            LaneChangeState::Centered => {
                // Need sustained high deviation to start
                if deviation >= self.config.drift_threshold {
                    if self.is_deviation_sustained(self.config.drift_threshold * 0.9) {
                        return LaneChangeState::Drifting;
                    }
                }
                LaneChangeState::Centered
            }
            LaneChangeState::Drifting => {
                if deviation >= self.config.crossing_threshold {
                    LaneChangeState::Crossing
                } else if deviation < self.config.drift_threshold * 0.4 {
                    // Returned to center without crossing - probably a curve or swerve
                    if self.max_offset_in_change < self.config.crossing_threshold * 0.85 {
                        // Never reached crossing threshold, cancel
                        LaneChangeState::Centered
                    } else {
                        // Did reach threshold, this is completion
                        LaneChangeState::Completed
                    }
                } else {
                    LaneChangeState::Drifting
                }
            }
            LaneChangeState::Crossing => {
                if deviation < self.config.drift_threshold * 0.5 {
                    LaneChangeState::Completed
                } else {
                    LaneChangeState::Crossing
                }
            }
            LaneChangeState::Completed => LaneChangeState::Centered,
        }
    }

    fn is_deviation_sustained(&self, threshold: f32) -> bool {
        if self.offset_history.len() < 8 {
            return false;
        }

        // Check last 6 frames all have high deviation from baseline
        let high_count = self
            .offset_history
            .iter()
            .rev()
            .take(6)
            .filter(|o| (*o - self.baseline_offset).abs() >= threshold)
            .count();

        high_count >= 5
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

        // Starting a lane change
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

        // Cancellation (Drifting back to Centered without completing)
        if target_state == LaneChangeState::Centered && from_state == LaneChangeState::Drifting {
            info!("↩️ Lane change cancelled (returned to center)");
            self.reset_lane_change();
            self.cooldown_remaining = 30;
            return None;
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
                    warn!(
                        "❌ Rejected: too short ({:.0}ms < {:.0}ms)",
                        dur, self.config.min_duration_ms
                    );
                    self.reset_lane_change();
                    self.cooldown_remaining = 60;
                    return None;
                }
            }

            // Validation 2: Max offset must have crossed threshold
            if self.max_offset_in_change < self.config.crossing_threshold {
                warn!(
                    "❌ Rejected: max deviation {:.1}% < crossing threshold {:.1}%",
                    self.max_offset_in_change * 100.0,
                    self.config.crossing_threshold * 100.0
                );
                self.reset_lane_change();
                self.cooldown_remaining = 60;
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
                "✅ LANE CHANGE CONFIRMED: {} at {:.2}s (duration: {:.0}ms, max_dev: {:.1}%)",
                event.direction_name(),
                timestamp_ms / 1000.0,
                duration_ms.unwrap_or(0.0),
                self.max_offset_in_change * 100.0
            );

            // Update baseline after lane change (we're now in a new lane)
            self.baseline_offset = 0.0; // Will be re-established
            self.baseline_samples.clear();
            self.is_baseline_established = false;
            self.stable_centered_frames = 0;
            self.frames_since_baseline = 0;

            self.reset_lane_change();
            return Some(event);
        }

        None
    }

    fn calculate_confidence(&self, duration_ms: Option<f64>) -> f32 {
        let mut confidence: f32 = 0.5;

        if self.max_offset_in_change > 0.60 {
            confidence += 0.25;
        } else if self.max_offset_in_change > 0.50 {
            confidence += 0.15;
        } else {
            confidence += 0.05;
        }

        if let Some(dur) = duration_ms {
            if dur > 1500.0 && dur < 4000.0 {
                confidence += 0.15;
            } else if dur > 1200.0 && dur < 6000.0 {
                confidence += 0.05;
            }
        }

        confidence.min(0.95)
    }

    pub fn reset(&mut self) {
        self.reset_lane_change();
        self.cooldown_remaining = 0;
        self.total_frames_processed = 0;
        self.offset_history.clear();
        self.baseline_offset = 0.0;
        self.baseline_samples.clear();
        self.is_baseline_established = false;
        self.frames_since_baseline = 0;
        self.stable_centered_frames = 0;
    }

    pub fn set_source_id(&mut self, source_id: String) {
        self.source_id = source_id;
    }
}
