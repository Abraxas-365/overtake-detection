// src/pipeline/frame_context.rs
//
// Single source of truth for all detections on a given frame.
// Solves the temporal alignment problem — every subsystem reads
// from the same context instead of stale cached values.

use crate::lane_legality::{FusedLegalityResult, LineLegality};
use crate::overtake_analyzer::TrackedVehicle;
use crate::types::{DetectedLane, Frame, VehicleState};
use crate::vehicle_detection::Detection;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct FrameContext {
    pub frame_id: u64,
    pub timestamp_ms: f64,
    pub frame: Frame,

    // Lane detection
    pub detected_lanes: Vec<DetectedLane>,
    pub vehicle_state: Option<VehicleState>,
    pub lane_change_state: String,
    pub left_lane_x: Option<f32>,
    pub right_lane_x: Option<f32>,

    // Vehicle detection
    pub vehicle_detections: Vec<Detection>,
    pub tracked_vehicles: HashMap<u32, TrackedVehicle>,

    // Legality
    pub legality_result: Option<FusedLegalityResult>,
}

impl FrameContext {
    pub fn new(frame_id: u64, frame: Frame) -> Self {
        let timestamp_ms = frame.timestamp_ms;
        Self {
            frame_id,
            timestamp_ms,
            frame,
            detected_lanes: Vec::new(),
            vehicle_state: None,
            lane_change_state: "CENTERED".to_string(),
            left_lane_x: None,
            right_lane_x: None,
            vehicle_detections: Vec::new(),
            tracked_vehicles: HashMap::new(),
            legality_result: None,
        }
    }

    /// Is the ego vehicle currently crossing a lane boundary?
    pub fn is_crossing(&self) -> bool {
        matches!(self.lane_change_state.as_str(), "CROSSING" | "DRIFTING")
    }

    /// Get the legality verdict, defaulting to Unknown
    pub fn legality_verdict(&self) -> LineLegality {
        self.legality_result
            .as_ref()
            .map(|r| r.verdict)
            .unwrap_or(LineLegality::Unknown)
    }
}
