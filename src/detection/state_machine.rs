// src/detection/state_machine.rs
use super::types::{Direction, LaneChangeEvent, VehiclePosition};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PipelineState {
    Calibrating,
    StableInLane,
    ChangingLane,
    LaneChanged,
}

pub struct StateMachine {
    config: Config,
    state: PipelineState,
    current_position: VehiclePosition,
    previous_position: VehiclePosition,

    // Calibration
    calibration_frames: Vec<i32>,
    baseline_lane: Option<i32>,

    // Lane change tracking
    frames_above_threshold: u32,
    frames_in_new_lane: u32,
    lane_change_start_time: Option<Instant>,
    pending_direction: Direction,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub offset_threshold: f32,
    pub debounce_frames: u32,
    pub confirm_frames: u32,
    pub timeout_ms: u64,
    pub calibration_frames: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            offset_threshold: 0.85, // Stricter than 0.7
            debounce_frames: 15,    // More than 5
            confirm_frames: 20,     // More than 10
            timeout_ms: 8000,
            calibration_frames: 90, // 3 seconds @ 30fps
        }
    }
}

impl StateMachine {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            state: PipelineState::Calibrating,
            current_position: VehiclePosition::invalid(),
            previous_position: VehiclePosition::invalid(),
            calibration_frames: Vec::new(),
            baseline_lane: None,
            frames_above_threshold: 0,
            frames_in_new_lane: 0,
            lane_change_start_time: None,
            pending_direction: Direction::None,
        }
    }

    pub fn update(&mut self, position: VehiclePosition) -> Option<LaneChangeEvent> {
        self.previous_position = self.current_position;
        self.current_position = position;

        // Skip invalid positions
        if !position.is_valid() {
            return None;
        }

        match self.state {
            PipelineState::Calibrating => {
                self.calibration_frames.push(position.lane_index);

                if self.calibration_frames.len() >= self.config.calibration_frames {
                    self.baseline_lane = Some(self.compute_baseline_lane());
                    self.state = PipelineState::StableInLane;
                    println!(
                        "✓ Calibration complete. Baseline lane: {}",
                        self.baseline_lane.unwrap()
                    );
                }
                None
            }

            PipelineState::StableInLane => {
                if position.lateral_offset.abs() > self.config.offset_threshold {
                    self.frames_above_threshold += 1;

                    if self.frames_above_threshold >= self.config.debounce_frames {
                        self.state = PipelineState::ChangingLane;
                        self.lane_change_start_time = Some(Instant::now());
                        self.pending_direction = if position.lateral_offset < 0.0 {
                            Direction::Left
                        } else {
                            Direction::Right
                        };
                        println!("→ Lane change started: {:?}", self.pending_direction);
                    }
                } else {
                    self.frames_above_threshold = 0;
                }
                None
            }

            PipelineState::ChangingLane => {
                // Check timeout
                if let Some(start_time) = self.lane_change_start_time {
                    if start_time.elapsed() > Duration::from_millis(self.config.timeout_ms) {
                        println!("✗ Lane change timeout");
                        self.reset_to_stable();
                        return None;
                    }
                }

                // Check if lane index changed
                if position.lane_index != self.previous_position.lane_index
                    && position.lane_index >= 0
                    && self.previous_position.lane_index >= 0
                {
                    self.state = PipelineState::LaneChanged;
                    self.frames_in_new_lane = 1;
                    println!(
                        "→ Lane index changed: {} → {}",
                        self.previous_position.lane_index, position.lane_index
                    );
                }
                None
            }

            PipelineState::LaneChanged => {
                self.frames_in_new_lane += 1;

                if self.frames_in_new_lane >= self.config.confirm_frames {
                    println!("✓ Lane change confirmed");

                    let event = LaneChangeEvent {
                        timestamp: Instant::now(),
                        direction: self.pending_direction,
                        from_lane: self.previous_position.lane_index,
                        to_lane: position.lane_index,
                        confidence: position.confidence,
                    };

                    self.reset_to_stable();
                    Some(event)
                } else {
                    None
                }
            }
        }
    }

    fn reset_to_stable(&mut self) {
        self.state = PipelineState::StableInLane;
        self.frames_above_threshold = 0;
        self.frames_in_new_lane = 0;
        self.lane_change_start_time = None;
    }

    fn compute_baseline_lane(&self) -> i32 {
        let mut counts = std::collections::HashMap::new();
        for &lane in &self.calibration_frames {
            *counts.entry(lane).or_insert(0) += 1;
        }
        *counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(lane, _)| lane)
            .unwrap_or(&1) // Default to lane 1
    }

    pub fn is_calibrated(&self) -> bool {
        self.baseline_lane.is_some()
    }
}
