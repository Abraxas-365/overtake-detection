// src/analysis/maneuver_classifier.rs
//
// Fusion layer: combines vehicle tracking, lateral shift detection,
// and ego-motion estimation to classify maneuvers.

use super::ego_motion::EgoMotionEstimate;
use super::lateral_detector::{LateralShiftEvent, ShiftDirection};
use super::pass_detector::{PassDirection, PassEvent, PassSide};
use crate::lane_legality::{FusedLegalityResult, LineLegality};
use crate::pipeline::legality_buffer::LegalityRingBuffer;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::info;

// ============================================================================
// CONFIGURATION
// ============================================================================

#[derive(Debug, Clone)]
pub struct ClassifierConfig {
    pub max_correlation_gap_ms: f64,
    pub min_single_source_confidence: f32,
    pub min_combined_confidence: f32,
    pub ego_motion_threshold: f32,
    pub weight_pass: f32,
    pub weight_lateral: f32,
    pub weight_ego_motion: f32,
    pub correlation_window_ms: f64,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            max_correlation_gap_ms: 50000.0,    // ✅ Wider window
            min_single_source_confidence: 0.25, // ✅ LOWERED from 0.65
            min_combined_confidence: 0.25,      // ✅ LOWERED from 0.45
            ego_motion_threshold: 2.0,
            weight_pass: 0.60, // ✅ Higher weight for passes
            weight_lateral: 0.25,
            weight_ego_motion: 0.15,
            correlation_window_ms: 60000.0, // ✅ Wider retention
        }
    }
}

// ============================================================================
// TYPES
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ManeuverType {
    Overtake,
    LaneChange,
    BeingOvertaken,
}

impl ManeuverType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Overtake => "OVERTAKE",
            Self::LaneChange => "LANE_CHANGE",
            Self::BeingOvertaken => "BEING_OVERTAKEN",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ManeuverSide {
    Left,
    Right,
}

impl ManeuverSide {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Left => "LEFT",
            Self::Right => "RIGHT",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DetectionSources {
    pub vehicle_tracking: bool,
    pub lane_detection: bool,
    pub ego_motion: bool,
}

impl DetectionSources {
    pub fn count(&self) -> u32 {
        self.vehicle_tracking as u32 + self.lane_detection as u32 + self.ego_motion as u32
    }

    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if self.vehicle_tracking {
            parts.push("VEH_TRACK");
        }
        if self.lane_detection {
            parts.push("LANE_DET");
        }
        if self.ego_motion {
            parts.push("EGO_FLOW");
        }
        parts.join("+")
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MarkingSnapshot {
    pub left_name: Option<String>,
    pub right_name: Option<String>,
    pub frame_id: u64,
}

/// Final output event - uses serde(skip) for non-serializable nested types
#[derive(Debug, Clone, Serialize)]
pub struct ManeuverEvent {
    pub maneuver_type: ManeuverType,
    pub side: ManeuverSide,
    pub legality: LineLegality,
    #[serde(skip)]
    pub legality_at_crossing: Option<FusedLegalityResult>,
    pub confidence: f32,
    pub sources: DetectionSources,
    pub start_ms: f64,
    pub end_ms: f64,
    pub start_frame: u64,
    pub end_frame: u64,
    pub duration_ms: f64,
    pub passed_vehicle_id: Option<u32>,
    pub passed_vehicle_class: Option<u32>,
    #[serde(skip)]
    pub pass_event: Option<PassEvent>,
    #[serde(skip)]
    pub lateral_event: Option<LateralShiftEvent>,
    pub marking_context: Option<MarkingSnapshot>,
}

// ============================================================================
// INTERNAL BUFFER ENTRIES
// ============================================================================

#[derive(Debug, Clone)]
struct BufferedPass {
    event: PassEvent,
    correlated: bool,
}

#[derive(Debug, Clone)]
struct BufferedShift {
    event: LateralShiftEvent,
    correlated: bool,
}

// ============================================================================
// CLASSIFIER
// ============================================================================

pub struct ManeuverClassifier {
    config: ClassifierConfig,
    pass_buffer: VecDeque<BufferedPass>,
    shift_buffer: VecDeque<BufferedShift>,
    current_ego_motion: EgoMotionEstimate,
    ego_motion_during_window: VecDeque<f32>,
    recent_events: Vec<ManeuverEvent>,
    total_maneuvers: u64,
    latest_markings: MarkingSnapshot,
}

impl ManeuverClassifier {
    pub fn new(config: ClassifierConfig) -> Self {
        Self {
            config,
            pass_buffer: VecDeque::with_capacity(20),
            shift_buffer: VecDeque::with_capacity(20),
            current_ego_motion: EgoMotionEstimate::none(),
            ego_motion_during_window: VecDeque::with_capacity(150),
            recent_events: Vec::new(),
            total_maneuvers: 0,
            latest_markings: MarkingSnapshot::default(),
        }
    }

    pub fn feed_pass(&mut self, event: PassEvent) {
        self.pass_buffer.push_back(BufferedPass {
            event,
            correlated: false,
        });
    }

    pub fn feed_shift(&mut self, event: LateralShiftEvent) {
        self.shift_buffer.push_back(BufferedShift {
            event,
            correlated: false,
        });
    }

    pub fn feed_ego_motion(&mut self, estimate: EgoMotionEstimate) {
        self.current_ego_motion = estimate;
        self.ego_motion_during_window
            .push_back(estimate.lateral_velocity_px);
        if self.ego_motion_during_window.len() > 150 {
            self.ego_motion_during_window.pop_front();
        }
    }

    pub fn update_markings(&mut self, snapshot: MarkingSnapshot) {
        self.latest_markings = snapshot;
    }

    pub fn classify(
        &mut self,
        timestamp_ms: f64,
        _frame_id: u64,
        legality_buffer: Option<&LegalityRingBuffer>,
    ) -> Vec<ManeuverEvent> {
        self.recent_events.clear();

        let window = self.config.correlation_window_ms;
        self.pass_buffer
            .retain(|p| timestamp_ms - p.event.beside_end_ms < window);
        self.shift_buffer
            .retain(|s| timestamp_ms - s.event.end_ms < window);

        // ── CORRELATE PASS + SHIFT → OVERTAKE ───────────────────
        let gap = self.config.max_correlation_gap_ms;

        for pass_idx in 0..self.pass_buffer.len() {
            if self.pass_buffer[pass_idx].correlated {
                continue;
            }
            if self.pass_buffer[pass_idx].event.direction == PassDirection::VehicleOvertookEgo {
                continue;
            }

            let mut best_shift_idx: Option<usize> = None;
            let mut best_time_gap = f64::MAX;

            for (si, shift) in self.shift_buffer.iter().enumerate() {
                if shift.correlated {
                    continue;
                }

                let time_gap = temporal_gap(
                    self.pass_buffer[pass_idx].event.beside_start_ms,
                    self.pass_buffer[pass_idx].event.beside_end_ms,
                    shift.event.start_ms,
                    shift.event.end_ms,
                );

                if time_gap < gap && time_gap < best_time_gap {
                    if directions_agree(&self.pass_buffer[pass_idx].event, &shift.event) {
                        best_shift_idx = Some(si);
                        best_time_gap = time_gap;
                    }
                }
            }

            if let Some(si) = best_shift_idx {
                let pass_event = self.pass_buffer[pass_idx].event.clone();
                let shift_event = self.shift_buffer[si].event.clone();

                let maneuver =
                    self.build_overtake(&pass_event, Some(&shift_event), legality_buffer);

                info!(
                    "🚗 OVERTAKE: {} | conf={:.2} | sources={} | legality={:?}",
                    maneuver.side.as_str(),
                    maneuver.confidence,
                    maneuver.sources.summary(),
                    maneuver.legality,
                );

                self.recent_events.push(maneuver);
                self.total_maneuvers += 1;

                self.pass_buffer[pass_idx].correlated = true;
                self.shift_buffer[si].correlated = true;
            }
        }

        // ── UNCORRELATED HIGH-CONFIDENCE PASSES → OVERTAKE ──────
        for pass_idx in 0..self.pass_buffer.len() {
            if self.pass_buffer[pass_idx].correlated {
                continue;
            }
            if self.pass_buffer[pass_idx].event.direction == PassDirection::VehicleOvertookEgo {
                continue;
            }

            let pass_conf = self.pass_buffer[pass_idx].event.confidence;

            // ✅ ADD THIS LOG
            info!(
                "🔍 Checking single-source pass: track={} side={:?} conf={:.2} (min={:.2})",
                self.pass_buffer[pass_idx].event.vehicle_track_id,
                self.pass_buffer[pass_idx].event.side,
                pass_conf,
                self.config.min_single_source_confidence
            );

            if pass_conf >= self.config.min_single_source_confidence {
                let pass_event = self.pass_buffer[pass_idx].event.clone();
                let maneuver = self.build_overtake(&pass_event, None, legality_buffer);

                // ✅ ADD THIS LOG
                info!(
                    "  → Built overtake: conf={:.2} (min_combined={:.2}) sources={}",
                    maneuver.confidence,
                    self.config.min_combined_confidence,
                    maneuver.sources.summary()
                );

                if maneuver.confidence >= self.config.min_combined_confidence {
                    info!(
                        "🚗 OVERTAKE (tracking-only): {} | conf={:.2} | ego={}",
                        maneuver.side.as_str(),
                        maneuver.confidence,
                        self.ego_motion_confirms_lateral(),
                    );

                    self.recent_events.push(maneuver);
                    self.total_maneuvers += 1;
                    self.pass_buffer[pass_idx].correlated = true;
                } else {
                    // ✅ ADD THIS LOG
                    info!(
                        "  ❌ Rejected: conf={:.2} < min_combined={:.2}",
                        maneuver.confidence, self.config.min_combined_confidence
                    );
                }
            } else {
                // ✅ ADD THIS LOG
                info!(
                    "  ❌ Pass conf too low: {:.2} < {:.2}",
                    pass_conf, self.config.min_single_source_confidence
                );
            }
        }

        // ── UNCORRELATED SHIFTS → LANE CHANGE ───────────────────
        for shift_idx in 0..self.shift_buffer.len() {
            if self.shift_buffer[shift_idx].correlated {
                continue;
            }
            if !self.shift_buffer[shift_idx].event.confirmed {
                continue;
            }

            let shift_event = self.shift_buffer[shift_idx].event.clone();
            let maneuver = self.build_lane_change(&shift_event, legality_buffer);

            if maneuver.confidence >= self.config.min_combined_confidence {
                info!(
                    "🔀 LANE CHANGE: {} | conf={:.2} | legality={:?}",
                    maneuver.side.as_str(),
                    maneuver.confidence,
                    maneuver.legality,
                );

                self.recent_events.push(maneuver);
                self.total_maneuvers += 1;
                self.shift_buffer[shift_idx].correlated = true;
            }
        }

        // ── BEING OVERTAKEN ─────────────────────────────────────
        for pass_idx in 0..self.pass_buffer.len() {
            if self.pass_buffer[pass_idx].correlated {
                continue;
            }
            if self.pass_buffer[pass_idx].event.direction == PassDirection::VehicleOvertookEgo {
                let pass_event = self.pass_buffer[pass_idx].event.clone();
                let maneuver = self.build_being_overtaken(&pass_event);

                info!(
                    "⚠️  BEING OVERTAKEN: track {} on {} | conf={:.2}",
                    pass_event.vehicle_track_id,
                    maneuver.side.as_str(),
                    maneuver.confidence,
                );

                self.recent_events.push(maneuver);
                self.total_maneuvers += 1;
                self.pass_buffer[pass_idx].correlated = true;
            }
        }

        self.recent_events.clone()
    }

    // ── EVENT BUILDERS ──────────────────────────────────────────

    fn build_overtake(
        &self,
        pass: &PassEvent,
        shift: Option<&LateralShiftEvent>,
        legality_buffer: Option<&LegalityRingBuffer>,
    ) -> ManeuverEvent {
        let side = match pass.side {
            PassSide::Left => ManeuverSide::Left,
            PassSide::Right => ManeuverSide::Right,
        };

        let (start_frame, end_frame) = self.compute_frame_range(pass, shift);

        let legality_at_crossing =
            legality_buffer.and_then(|buf| buf.worst_in_range(start_frame, end_frame));
        let legality = legality_at_crossing
            .as_ref()
            .map(|r| r.verdict)
            .unwrap_or(LineLegality::Unknown);

        let ego_confirms = self.ego_motion_confirms_lateral();

        // ✅ BOOST pass weight when it's the only source
        let pass_weight = if shift.is_none() && !ego_confirms {
            0.70 // Single-source: give passes more weight
        } else {
            self.config.weight_pass
        };

        let pass_conf = pass.confidence * pass_weight; // ✅ Now: 0.70 * 0.70 = 0.49
        let shift_conf = shift
            .map(|s| s.confidence * self.config.weight_lateral)
            .unwrap_or(0.0);
        let ego_conf = if ego_confirms {
            self.config.weight_ego_motion * 0.8
        } else {
            0.0
        };

        let sources = DetectionSources {
            vehicle_tracking: true,
            lane_detection: shift.is_some(),
            ego_motion: ego_confirms,
        };

        let source_bonus = match sources.count() {
            3 => 0.15,
            2 => 0.08,
            1 => 0.0, // No bonus for single source (but high pass weight compensates)
            _ => 0.0,
        };

        let confidence = (pass_conf + shift_conf + ego_conf + source_bonus).min(0.98);
        let (start_ms, end_ms) = if let Some(s) = shift {
            (
                pass.ahead_start_ms.min(s.start_ms),
                pass.beside_end_ms.max(s.end_ms),
            )
        } else {
            (pass.ahead_start_ms, pass.beside_end_ms)
        };

        ManeuverEvent {
            maneuver_type: ManeuverType::Overtake,
            side,
            legality,
            legality_at_crossing,
            confidence,
            sources,
            start_ms,
            end_ms,
            start_frame,
            end_frame,
            duration_ms: end_ms - start_ms,
            passed_vehicle_id: Some(pass.vehicle_track_id),
            passed_vehicle_class: Some(pass.vehicle_class_id),
            pass_event: Some(pass.clone()),
            lateral_event: shift.cloned(),
            marking_context: Some(self.latest_markings.clone()),
        }
    }

    fn build_lane_change(
        &self,
        shift: &LateralShiftEvent,
        legality_buffer: Option<&LegalityRingBuffer>,
    ) -> ManeuverEvent {
        let side = match shift.direction {
            ShiftDirection::Left => ManeuverSide::Left,
            ShiftDirection::Right => ManeuverSide::Right,
        };

        let legality_at_crossing =
            legality_buffer.and_then(|buf| buf.worst_in_range(shift.start_frame, shift.end_frame));
        let legality = legality_at_crossing
            .as_ref()
            .map(|r| r.verdict)
            .unwrap_or(LineLegality::Unknown);

        let ego_confirms = self.ego_motion_confirms_lateral();

        let shift_conf = shift.confidence * (self.config.weight_pass + self.config.weight_lateral);
        let ego_conf = if ego_confirms {
            self.config.weight_ego_motion * 0.8
        } else {
            0.0
        };

        let sources = DetectionSources {
            vehicle_tracking: false,
            lane_detection: true,
            ego_motion: ego_confirms,
        };

        let confidence =
            (shift_conf + ego_conf + if sources.count() >= 2 { 0.05 } else { 0.0 }).min(0.95);

        ManeuverEvent {
            maneuver_type: ManeuverType::LaneChange,
            side,
            legality,
            legality_at_crossing,
            confidence,
            sources,
            start_ms: shift.start_ms,
            end_ms: shift.end_ms,
            start_frame: shift.start_frame,
            end_frame: shift.end_frame,
            duration_ms: shift.duration_ms,
            passed_vehicle_id: None,
            passed_vehicle_class: None,
            pass_event: None,
            lateral_event: Some(shift.clone()),
            marking_context: Some(self.latest_markings.clone()),
        }
    }

    fn build_being_overtaken(&self, pass: &PassEvent) -> ManeuverEvent {
        let side = match pass.side {
            PassSide::Left => ManeuverSide::Left,
            PassSide::Right => ManeuverSide::Right,
        };

        ManeuverEvent {
            maneuver_type: ManeuverType::BeingOvertaken,
            side,
            legality: LineLegality::Unknown,
            legality_at_crossing: None,
            confidence: pass.confidence * 0.85,
            sources: DetectionSources {
                vehicle_tracking: true,
                lane_detection: false,
                ego_motion: false,
            },
            start_ms: pass.beside_start_ms,
            end_ms: pass.beside_end_ms,
            start_frame: pass
                .frame_id
                .saturating_sub((pass.duration_ms / 33.3) as u64),
            end_frame: pass.frame_id,
            duration_ms: pass.duration_ms,
            passed_vehicle_id: Some(pass.vehicle_track_id),
            passed_vehicle_class: Some(pass.vehicle_class_id),
            pass_event: Some(pass.clone()),
            lateral_event: None,
            marking_context: None,
        }
    }

    // ── HELPERS ──────────────────────────────────────────────────

    fn compute_frame_range(
        &self,
        pass: &PassEvent,
        shift: Option<&LateralShiftEvent>,
    ) -> (u64, u64) {
        if let Some(s) = shift {
            (
                s.start_frame.min(pass.frame_id.saturating_sub(100)),
                pass.frame_id.max(s.end_frame),
            )
        } else {
            (
                pass.frame_id
                    .saturating_sub((pass.duration_ms / 33.3) as u64),
                pass.frame_id,
            )
        }
    }

    fn ego_motion_confirms_lateral(&self) -> bool {
        if self.ego_motion_during_window.len() < 5 {
            return false;
        }
        let peak = self
            .ego_motion_during_window
            .iter()
            .rev()
            .take(30)
            .map(|v| v.abs())
            .fold(0.0f32, |a, b| a.max(b));

        peak >= self.config.ego_motion_threshold
    }

    pub fn recent_events(&self) -> &[ManeuverEvent] {
        &self.recent_events
    }

    pub fn total_maneuvers(&self) -> u64 {
        self.total_maneuvers
    }

    pub fn reset(&mut self) {
        self.pass_buffer.clear();
        self.shift_buffer.clear();
        self.ego_motion_during_window.clear();
        self.current_ego_motion = EgoMotionEstimate::none();
        self.recent_events.clear();
        self.total_maneuvers = 0;
        self.latest_markings = MarkingSnapshot::default();
    }
}

// ============================================================================
// UTILITY
// ============================================================================

fn temporal_gap(start_a: f64, end_a: f64, start_b: f64, end_b: f64) -> f64 {
    if end_a < start_b {
        start_b - end_a
    } else if end_b < start_a {
        start_a - end_b
    } else {
        0.0
    }
}

fn directions_agree(pass: &PassEvent, shift: &LateralShiftEvent) -> bool {
    // For mining routes with poor tracking, accept opposite directions too
    // because the vehicle might disappear and reappear, causing zone confusion

    // Strict matching (preferred)
    let strict_match = matches!(
        (pass.side, shift.direction),
        (PassSide::Left, ShiftDirection::Left) | (PassSide::Right, ShiftDirection::Right)
    );

    // Lenient: if it's a valid pass and shift within time window, likely related
    // This handles cases where vehicle tracking drops and zones get confused
    strict_match || pass.confidence > 0.6 // Accept high-confidence passes with any shift
}
