// src/overtake_tracker.rs — Production version

use crate::types::{Direction, LaneChangeEvent};
use tracing::{info, warn};

#[derive(Debug, Clone, PartialEq)]
enum OvertakeState {
    Idle,
    InProgress {
        start_event: LaneChangeEvent,
        start_frame: u64,
        direction: Direction,
    },
}

/// Configuration for dynamic timeout behavior
#[derive(Debug, Clone)]
pub struct OvertakeTimeoutConfig {
    /// Base timeout in frames (default: 30s * fps)
    pub base_timeout_frames: u64,
    /// Minimum timeout — never go below this (e.g., 10s)
    pub min_timeout_frames: u64,
    /// Maximum timeout — never exceed this (e.g., 60s)
    pub max_timeout_frames: u64,
    /// If we see vehicles being overtaken, extend by this factor
    pub active_overtake_extension: f64,
    /// If shadow detected, reduce timeout (dangerous situation)
    pub shadow_reduction_factor: f64,
}

impl OvertakeTimeoutConfig {
    pub fn new(base_timeout_seconds: f64, fps: f64) -> Self {
        let base = (base_timeout_seconds * fps) as u64;
        Self {
            base_timeout_frames: base,
            min_timeout_frames: (10.0 * fps) as u64,
            max_timeout_frames: (60.0 * fps) as u64,
            active_overtake_extension: 1.5,
            shadow_reduction_factor: 0.5,
        }
    }
}

pub struct OvertakeTracker {
    state: OvertakeState,
    timeout_config: OvertakeTimeoutConfig,
    /// Dynamic timeout for the current overtake
    effective_timeout: u64,
    /// Whether vehicles are actively being passed
    vehicles_being_passed: bool,
    /// Whether shadow overtake is active
    shadow_active: bool,
}

#[derive(Debug, Clone)]
pub enum OvertakeResult {
    Complete {
        start_event: LaneChangeEvent,
        end_event: LaneChangeEvent,
        total_duration_ms: f64,
        vehicles_overtaken: Vec<String>,
        is_legal_crossing: Option<bool>,
        line_type: Option<String>,
    },
    Incomplete {
        start_event: LaneChangeEvent,
        reason: String,
    },
    SimpleLaneChange {
        event: LaneChangeEvent,
    },
}

impl OvertakeTracker {
    pub fn new(timeout_seconds: f64, fps: f64) -> Self {
        let config = OvertakeTimeoutConfig::new(timeout_seconds, fps);
        let effective_timeout = config.base_timeout_frames;
        Self {
            state: OvertakeState::Idle,
            timeout_config: config,
            effective_timeout,
            vehicles_being_passed: false,
            shadow_active: false,
        }
    }

    pub fn get_direction(&self) -> Direction {
        match &self.state {
            OvertakeState::InProgress { direction, .. } => *direction,
            _ => Direction::Unknown,
        }
    }

    /// Call this when YOLO detects vehicles are being passed
    pub fn set_vehicles_being_passed(&mut self, active: bool) {
        self.vehicles_being_passed = active;
        self.recalculate_timeout();
    }

    /// Call this when shadow detector finds a blocker
    pub fn set_shadow_active(&mut self, active: bool) {
        self.shadow_active = active;
        self.recalculate_timeout();
    }

    fn recalculate_timeout(&mut self) {
        let mut timeout = self.timeout_config.base_timeout_frames as f64;

        if self.vehicles_being_passed {
            timeout *= self.timeout_config.active_overtake_extension;
        }

        if self.shadow_active {
            // Dangerous — cut short, don't wait forever
            timeout *= self.timeout_config.shadow_reduction_factor;
        }

        self.effective_timeout = (timeout as u64)
            .max(self.timeout_config.min_timeout_frames)
            .min(self.timeout_config.max_timeout_frames);
    }

    pub fn process_lane_change(
        &mut self,
        event: LaneChangeEvent,
        current_frame: u64,
    ) -> Option<OvertakeResult> {
        match &self.state {
            OvertakeState::Idle => {
                info!(
                    "🟡 Overtake initiated: {} at {:.2}s",
                    event.direction_name(),
                    event.video_timestamp_ms / 1000.0
                );
                self.state = OvertakeState::InProgress {
                    start_event: event.clone(),
                    start_frame: current_frame,
                    direction: event.direction,
                };
                None
            }

            OvertakeState::InProgress {
                start_event,
                direction,
                ..
            } => {
                // Check if this is a RETURN maneuver (opposite of the start)
                let is_return = match (*direction, event.direction) {
                    (Direction::Left, Direction::Right) => true,
                    (Direction::Right, Direction::Left) => true,
                    _ => false,
                };

                if is_return {
                    let total_duration_ms =
                        event.video_timestamp_ms - start_event.video_timestamp_ms;
                    let result = OvertakeResult::Complete {
                        start_event: start_event.clone(),
                        end_event: event,
                        total_duration_ms,
                        vehicles_overtaken: vec![], // Populated by analyzer later
                        is_legal_crossing: None,
                        line_type: None,
                    };
                    self.state = OvertakeState::Idle;
                    Some(result)
                } else {
                    // PRODUCTION FIX:
                    // If we get another 'Right' while we are already in 'Right' state,
                    // DO NOT fail. Just ignore it and keep the original start.
                    warn!(
                        "Ignoring redundant {} trigger during active maneuver.",
                        event.direction_name()
                    );
                    None
                }
            }
        }
    }

    pub fn check_timeout(
        &mut self,
        current_frame: u64,
        lanes_visible: bool,
    ) -> Option<OvertakeResult> {
        if let OvertakeState::InProgress {
            start_event,
            start_frame,
            ..
        } = &self.state
        {
            // 🚀 PRODUCTION FIX: If lanes are NOT visible, we pause the timeout.
            // We assume the vehicle is still overtaking until proven otherwise.
            // This prevents "Incomplete" status during desert road 'blind spots'.
            if !lanes_visible {
                return None;
            }

            let elapsed = current_frame - start_frame;

            if elapsed > self.effective_timeout {
                let reason = if self.shadow_active {
                    format!(
                        "Shadow overtake timeout ({} frames) — dangerous situation cut short",
                        self.effective_timeout
                    )
                } else if self.vehicles_being_passed {
                    format!(
                        "Extended timeout expired ({} frames) despite active passing",
                        self.effective_timeout
                    )
                } else {
                    format!(
                        "No return to original lane within {} frames (Lanes were visible)",
                        self.effective_timeout
                    )
                };

                warn!("⏰ Overtake timeout: {}", reason);

                let incomplete = OvertakeResult::Incomplete {
                    start_event: start_event.clone(),
                    reason,
                };

                self.state = OvertakeState::Idle;
                return Some(incomplete);
            }
        }
        None
    }

    pub fn is_tracking(&self) -> bool {
        !matches!(self.state, OvertakeState::Idle)
    }

    pub fn frames_elapsed(&self, current_frame: u64) -> Option<u64> {
        if let OvertakeState::InProgress { start_frame, .. } = &self.state {
            Some(current_frame - start_frame)
        } else {
            None
        }
    }

    pub fn effective_timeout_frames(&self) -> u64 {
        self.effective_timeout
    }
}
