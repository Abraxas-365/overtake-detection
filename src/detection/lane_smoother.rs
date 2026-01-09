use std::collections::VecDeque;

pub struct LanePositionSmoother {
    history: VecDeque<VehiclePosition>,
    window_size: usize,
}

impl LanePositionSmoother {
    pub fn new(window_size: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    pub fn smooth(&mut self, position: VehiclePosition) -> VehiclePosition {
        self.history.push_back(position);
        if self.history.len() > self.window_size {
            self.history.pop_front();
        }

        // Mode for lane_index (most common in window)
        let lane_index = self.most_common_lane();

        // Median for lateral_offset (reduce noise)
        let mut offsets: Vec<f32> = self.history.iter().map(|p| p.lateral_offset).collect();
        offsets.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lateral_offset = offsets[offsets.len() / 2];

        // Average confidence
        let confidence =
            self.history.iter().map(|p| p.confidence).sum::<f32>() / self.history.len() as f32;

        VehiclePosition {
            lane_index,
            lateral_offset,
            confidence,
            ..position
        }
    }

    fn most_common_lane(&self) -> i32 {
        let mut counts = std::collections::HashMap::new();
        for pos in &self.history {
            *counts.entry(pos.lane_index).or_insert(0) += 1;
        }
        *counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(lane, _)| lane)
            .unwrap_or(&-1)
    }
}
