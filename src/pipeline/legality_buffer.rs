// src/pipeline/legality_buffer.rs
//
// Ring buffer of legality results indexed by frame_id.
// When an overtake completes, we look up the legality result
// closest to when the vehicle actually crossed the line,
// NOT whatever was cached from the last inference run.

use crate::lane_legality::{FusedLegalityResult, LegalityResult, LineLegality};
use std::collections::VecDeque;
use tracing::debug;

const DEFAULT_CAPACITY: usize = 300; // ~10s at 30fps

pub struct LegalityRingBuffer {
    entries: VecDeque<LegalityEntry>,
    capacity: usize,
}

#[derive(Debug, Clone)]
struct LegalityEntry {
    frame_id: u64,
    timestamp_ms: f64,
    result: FusedLegalityResult,
}

impl LegalityRingBuffer {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, frame_id: u64, timestamp_ms: f64, result: FusedLegalityResult) {
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(LegalityEntry {
            frame_id,
            timestamp_ms,
            result,
        });
    }

    /// Find the most severe legality result in a frame range.
    /// This is what you attach to an overtake event.
    pub fn worst_in_range(&self, start_frame: u64, end_frame: u64) -> Option<FusedLegalityResult> {
        let mut worst: Option<&LegalityEntry> = None;

        for entry in &self.entries {
            if entry.frame_id < start_frame || entry.frame_id > end_frame {
                continue;
            }
            if !entry.result.crossing_confirmed_by_lane_model {
                continue;
            }

            match worst {
                None => worst = Some(entry),
                Some(current_worst) => {
                    if severity_rank(entry.result.verdict)
                        > severity_rank(current_worst.result.verdict)
                    {
                        worst = Some(entry);
                    }
                }
            }
        }

        worst.map(|e| e.result.clone())
    }

    /// Find the closest legality result to a specific frame.
    pub fn closest_to_frame(&self, target_frame: u64) -> Option<FusedLegalityResult> {
        self.entries
            .iter()
            .min_by_key(|e| (e.frame_id as i64 - target_frame as i64).unsigned_abs())
            .map(|e| e.result.clone())
    }

    /// Get the latest result (backward compat).
    pub fn latest(&self) -> Option<&FusedLegalityResult> {
        self.entries.back().map(|e| &e.result)
    }

    /// Convert latest to LegalityResult for video overlay.
    pub fn latest_as_legality_result(&self) -> Option<LegalityResult> {
        self.entries.back().map(|e| LegalityResult {
            verdict: e.result.verdict,
            intersecting_line: e.result.line_type_from_seg_model.clone(),
            all_markings: e.result.all_markings.clone(),
            ego_intersects_marking: e.result.ego_intersects_marking,
            frame_id: e.frame_id,
        })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

fn severity_rank(v: LineLegality) -> u8 {
    match v {
        LineLegality::Unknown => 0,
        LineLegality::Legal => 1,
        LineLegality::Caution => 2,
        LineLegality::Illegal => 3,
        LineLegality::CriticalIllegal => 4,
    }
}
