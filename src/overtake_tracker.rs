// src/overtake_tracker.rs

use crate::types::{Direction, LaneChangeEvent};
use std::time::Duration;
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

pub struct OvertakeTracker {
    state: OvertakeState,
    timeout_frames: u64, // If no return within this, consider incomplete
}

#[derive(Debug, Clone)]
pub enum OvertakeResult {
    /// Complete overtaking maneuver (left → right or right → left)
    Complete {
        start_event: LaneChangeEvent,
        end_event: LaneChangeEvent,
        total_duration_ms: f64,
        vehicles_overtaken: Vec<String>, // Will be filled later
    },
    /// Incomplete overtake (changed lane but didn't return)
    Incomplete {
        start_event: LaneChangeEvent,
        reason: String,
    },
    /// Not an overtake pattern, just a lane change
    SimpleLaneChange { event: LaneChangeEvent },
}

impl OvertakeTracker {
    pub fn new(timeout_seconds: f64, fps: f64) -> Self {
        let timeout_frames = (timeout_seconds * fps) as u64;
        Self {
            state: OvertakeState::Idle,
            timeout_frames,
        }
    }

    /// Process a lane change event and determine if it's part of an overtake
    pub fn process_lane_change(
        &mut self,
        event: LaneChangeEvent,
        current_frame: u64,
    ) -> Option<OvertakeResult> {
        match &self.state {
            OvertakeState::Idle => {
                // Start tracking a potential overtake
                info!(
                    "🟡 Overtake initiated: {} at {:.2}s",
                    event.direction_name(),
                    event.video_timestamp_ms / 1000.0
                );

                self.state = OvertakeState::InProgress {
                    start_event: event.clone(),
                    start_frame: event.end_frame_id,
                    direction: event.direction,
                };

                // Don't emit event yet - wait for return
                None
            }

            OvertakeState::InProgress {
                start_event,
                start_frame,
                direction,
            } => {
                // Check if this is the return lane change
                let is_return = match (direction, event.direction) {
                    (Direction::Left, Direction::Right) => true,
                    (Direction::Right, Direction::Left) => true,
                    _ => false,
                };

                if is_return {
                    // Complete overtake!
                    let total_duration_ms =
                        event.video_timestamp_ms - start_event.video_timestamp_ms;

                    info!(
                        "✅ Overtake completed: {:.2}s → {:.2}s (duration: {:.1}s)",
                        start_event.video_timestamp_ms / 1000.0,
                        event.video_timestamp_ms / 1000.0,
                        total_duration_ms / 1000.0
                    );

                    let result = OvertakeResult::Complete {
                        start_event: start_event.clone(),
                        end_event: event,
                        total_duration_ms,
                        vehicles_overtaken: vec![], // Will be filled by overtake_analyzer
                    };

                    // Reset state
                    self.state = OvertakeState::Idle;

                    Some(result)
                } else {
                    // Same direction again? This is unusual - might be weaving
                    warn!(
                        "⚠️  Multiple {} lane changes without returning",
                        event.direction_name()
                    );

                    // Treat previous as incomplete, start tracking new one
                    let incomplete = OvertakeResult::Incomplete {
                        start_event: start_event.clone(),
                        reason: "Driver didn't return to original lane".to_string(),
                    };

                    // Start tracking this new lane change
                    self.state = OvertakeState::InProgress {
                        start_event: event,
                        start_frame: current_frame,
                        direction: event.direction,
                    };

                    Some(incomplete)
                }
            }
        }
    }

    /// Check for timeout (incomplete overtake)
    pub fn check_timeout(&mut self, current_frame: u64) -> Option<OvertakeResult> {
        if let OvertakeState::InProgress {
            start_event,
            start_frame,
            ..
        } = &self.state
        {
            if current_frame - start_frame > self.timeout_frames {
                warn!("⏰ Overtake timeout: Driver stayed in overtaking lane for too long");

                let incomplete = OvertakeResult::Incomplete {
                    start_event: start_event.clone(),
                    reason: format!(
                        "No return to original lane within {} frames",
                        self.timeout_frames
                    ),
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
}
