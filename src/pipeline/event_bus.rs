// src/pipeline/event_bus.rs
//
// Decoupled event system. Subsystems publish events instead of
// reaching into each other's state.

use crate::lane_legality::{FusedLegalityResult, LineLegality};
use crate::overtake_analyzer::OvertakeEvent;
use crate::shadow_overtake::ShadowOvertakeEvent;
use crate::types::{CurveInfo, LaneChangeEvent};
use std::collections::VecDeque;
use tracing::{info, warn};

#[derive(Debug, Clone)]
pub enum PipelineEvent {
    LaneChangeDetected(LaneChangeEvent),

    OvertakeStarted {
        event: LaneChangeEvent,
        frame_id: u64,
    },

    OvertakeCompleted {
        start_event: LaneChangeEvent,
        end_event: LaneChangeEvent,
        duration_ms: f64,
        vehicles_overtaken: Vec<OvertakeEvent>,
        legality_at_crossing: Option<FusedLegalityResult>,
        shadow_events: Vec<ShadowOvertakeEvent>,
        curve_info: CurveInfo,
    },

    OvertakeIncomplete {
        start_event: LaneChangeEvent,
        reason: String,
        shadow_events: Vec<ShadowOvertakeEvent>,
    },

    LegalityViolation {
        frame_id: u64,
        timestamp_ms: f64,
        verdict: LineLegality,
        line_type: String,
        confidence: f32,
    },

    ShadowOvertakeDetected(ShadowOvertakeEvent),
}

pub struct EventBus {
    events: VecDeque<PipelineEvent>,
    max_pending: usize,
}

impl EventBus {
    pub fn new(max_pending: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(max_pending),
            max_pending,
        }
    }

    pub fn publish(&mut self, event: PipelineEvent) {
        if self.events.len() >= self.max_pending {
            warn!(
                "Event bus full ({} events), dropping oldest",
                self.max_pending
            );
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    pub fn drain(&mut self) -> Vec<PipelineEvent> {
        self.events.drain(..).collect()
    }

    pub fn pending_count(&self) -> usize {
        self.events.len()
    }
}
