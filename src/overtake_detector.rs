use crate::types::{Config, Direction, LaneChangeEvent, OvertakeEvent};
use std::collections::VecDeque;
use tracing::info;

pub struct OvertakeDetector {
    config: Config,
    lane_history: VecDeque<(i32, f64)>,
    last_stable_lane: Option<i32>,
    last_change_time: Option<f64>,
    recent_changes: Vec<LaneChangeEvent>,
}

impl OvertakeDetector {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            lane_history: VecDeque::with_capacity(30),
            last_stable_lane: None,
            last_change_time: None,
            recent_changes: Vec::new(),
        }
    }

    pub fn update(
        &mut self,
        current_lane: i32,
        lateral_offset: f32,
        timestamp: f64,
    ) -> Option<OvertakeEvent> {
        // Add to history
        self.lane_history.push_back((current_lane, timestamp));
        if self.lane_history.len() > 30 {
            self.lane_history.pop_front();
        }

        // Get stable lane (mode of last 10 frames)
        let stable_lane = self.get_stable_lane();

        // Detect lane change
        if let Some(prev_stable) = self.last_stable_lane {
            if stable_lane != prev_stable {
                let event = LaneChangeEvent {
                    timestamp,
                    direction: if stable_lane > prev_stable {
                        Direction::Right
                    } else {
                        Direction::Left
                    },
                    from_lane: prev_stable,
                    to_lane: stable_lane,
                    confidence: 0.8, // You can calculate actual confidence
                };

                info!(
                    "Lane change detected: {:?} from {} to {}",
                    event.direction, event.from_lane, event.to_lane
                );

                // Check for overtake
                let overtake = self.check_overtake(&event);

                self.last_stable_lane = Some(stable_lane);
                self.last_change_time = Some(timestamp);

                return overtake;
            }
        } else {
            self.last_stable_lane = Some(stable_lane);
        }

        None
    }

    fn get_stable_lane(&self) -> i32 {
        if self.lane_history.is_empty() {
            return -1;
        }

        // Get mode of last 10 frames
        let recent: Vec<i32> = self
            .lane_history
            .iter()
            .rev()
            .take(10)
            .map(|(lane, _)| *lane)
            .collect();

        let mut counts = std::collections::HashMap::new();
        for &lane in &recent {
            *counts.entry(lane).or_insert(0) += 1;
        }

        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(lane, _)| lane)
            .unwrap_or(-1)
    }

    fn check_overtake(&mut self, current_event: &LaneChangeEvent) -> Option<OvertakeEvent> {
        // Add to recent changes
        self.recent_changes.push(current_event.clone());

        // Keep only events within time window
        self.recent_changes.retain(|e| {
            current_event.timestamp - e.timestamp < self.config.overtake.max_window_seconds
        });

        // Need at least 2 lane changes
        if self.recent_changes.len() < 2 {
            return None;
        }

        let prev = &self.recent_changes[self.recent_changes.len() - 2];
        let curr = current_event;

        let delta = curr.timestamp - prev.timestamp;

        // Check timing constraints
        if delta < self.config.overtake.min_interval_seconds
            || delta > self.config.overtake.max_window_seconds
        {
            return None;
        }

        // Check for opposite directions (overtake pattern)
        let is_complete = (prev.direction == Direction::Left && curr.direction == Direction::Right)
            || (prev.direction == Direction::Right && curr.direction == Direction::Left);

        if is_complete {
            info!("🚗 OVERTAKE DETECTED!");
        }

        Some(OvertakeEvent {
            start_timestamp: prev.timestamp,
            end_timestamp: curr.timestamp,
            first_direction: prev.direction,
            second_direction: curr.direction,
            start_lane: prev.from_lane,
            end_lane: curr.to_lane,
            is_complete,
            confidence: (prev.confidence + curr.confidence) / 2.0,
        })
    }
}
