// src/analysis/maneuver_classifier.rs
//
// Fusion layer: combines vehicle tracking, lateral shift detection,
// and ego-motion estimation to classify maneuvers.
//
// v4.3 FIX: Ego-motion gate for single-source (tracking-only) passes.
//           A real overtake REQUIRES the ego vehicle to move laterally.
//           If there's no lateral shift and no ego motion, the "pass"
//           is almost certainly a zone misclassification.
//
// v4.8 FIX: Raised single-source confidence thresholds for mining.
//           Added pass quality gate (min_pass_confidence_single_source).
//
// v4.10 FIX: Ego-motion-induced pass rejection (INVERSE ego-motion gate).
//            When the ego vehicle is actively changing lanes, vehicles in
//            the original lane appear to shift through zones (AHEAD →
//            BESIDE → BEHIND) WITHOUT any actual longitudinal passing.
//            This creates false EgoOvertook events.
//
//            The fix detects when ego lateral velocity is strong enough
//            in the direction that would produce the observed pass side
//            as an apparent artifact, and rejects the pass. Applied to
//            both correlated (pass+shift) and uncorrelated paths.
//
//            Direction mapping:
//              Ego moves LEFT  → vehicles appear to shift RIGHT
//                → AHEAD→BESIDE_RIGHT → PassSide::Left (false EgoOvertook)
//              Ego moves RIGHT → vehicles appear to shift LEFT
//                → AHEAD→BESIDE_LEFT → PassSide::Right (false EgoOvertook)

// v4.12 FIX: Extended ego-motion-induced pass rejection to VehicleOvertookEgo.
// v4.13 FIX (Bug 1): Drain consumed (correlated) buffer entries after classification.
// v4.13 FIX (Bug 2): Curve ego cross-validation gate in lateral_detector.

use super::ego_motion::EgoMotionEstimate;
use super::lateral_detector::{LateralShiftEvent, ShiftConfirmedNotification, ShiftDirection};
use super::pass_detector::{PassDirection, PassEvent, PassSide};
use crate::lane_legality::{FusedLegalityResult, LineLegality};
use crate::pipeline::legality_buffer::LegalityRingBuffer;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::{debug, info, warn};

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
    /// v4.3: Minimum peak ego lateral velocity (px/frame) required for
    /// single-source (tracking-only) passes. Below this, the pass is
    /// rejected as a zone misclassification.
    pub min_ego_motion_for_single_source: f32,
    /// v4.8: Minimum pass event confidence for single-source overtakes.
    pub min_pass_confidence_single_source: f32,
    /// v4.10: Minimum sustained ego lateral velocity (px/frame) above
    /// which an EgoOvertook pass is rejected as ego-motion-induced.
    /// When ego moves at this speed for enough frames, apparent zone
    /// transitions are artifacts of the lane change, not real overtakes.
    pub ego_induced_pass_velocity_threshold: f32,
    /// v4.10: Minimum number of frames at or above the velocity threshold
    /// to consider the ego lateral motion "sustained" (not just noise).
    pub ego_induced_pass_min_sustained_frames: usize,
    // ── v7.0: Lane change detection ─────────────────────────────
    /// Minimum confidence to emit a standalone lane change event.
    pub min_lane_change_confidence: f32,
    /// Minimum geometric override score to allow lane change through
    /// curve suppression. When boundaries diverge strongly enough,
    /// a lane change is real even on a curve.
    pub geometric_override_min_score: f32,
    /// Minimum polynomial tracker signal confidence to consider
    /// geometric override. Prevents stale/predict-only signals
    /// from causing false overrides.
    pub geometric_min_signal_confidence: f32,
    /// Weight for geometric signal contribution to lane change confidence.
    pub weight_geometric: f32,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            max_correlation_gap_ms: 50000.0,
            min_single_source_confidence: 0.35,
            min_combined_confidence: 0.30,
            ego_motion_threshold: 2.0,
            weight_pass: 0.60,
            weight_lateral: 0.25,
            weight_ego_motion: 0.15,
            correlation_window_ms: 60000.0,
            min_ego_motion_for_single_source: 1.0,
            min_pass_confidence_single_source: 0.55,
            ego_induced_pass_velocity_threshold: 3.0, // v4.10: 3 px/frame
            ego_induced_pass_min_sustained_frames: 8, // v4.10: ~267ms at 30fps
            // v7.0: Lane change detection
            min_lane_change_confidence: 0.35,
            geometric_override_min_score: 0.50,
            geometric_min_signal_confidence: 0.4,
            weight_geometric: 0.20,
        }
    }
}

impl ClassifierConfig {
    /// Mining environment — more conservative thresholds.
    pub fn mining() -> Self {
        Self {
            max_correlation_gap_ms: 50000.0,
            min_single_source_confidence: 0.40,
            min_combined_confidence: 0.35,
            ego_motion_threshold: 2.0,
            weight_pass: 0.60,
            weight_lateral: 0.25,
            weight_ego_motion: 0.15,
            correlation_window_ms: 60000.0,
            min_ego_motion_for_single_source: 1.2,
            min_pass_confidence_single_source: 0.60,
            ego_induced_pass_velocity_threshold: 2.5, // lower threshold in mining (dusty, noisy flow)
            ego_induced_pass_min_sustained_frames: 6,
            // v7.0: Lane change detection (slightly higher thresholds for mining)
            min_lane_change_confidence: 0.40,
            geometric_override_min_score: 0.55,
            geometric_min_signal_confidence: 0.45,
            weight_geometric: 0.20,
        }
    }
}

// ============================================================================
// TYPES
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ManeuverType {
    Overtake,
    ShadowOvertake,
    /// v7.0: Standalone lane change (no vehicle pass involved).
    LaneChange,
}

impl ManeuverType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Overtake => "OVERTAKE",
            Self::ShadowOvertake => "SHADOW_OVERTAKE",
            Self::LaneChange => "LANE_CHANGE",
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

/// v8.0: Snapshot of the road classification at maneuver time.
/// Captures the RoadClassifier's temporal consensus so legality can be
/// determined even when no crossing event fires (e.g., dashed line gaps).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadClassificationSnapshot {
    /// The temporal-consensus line type name (e.g. "mixed_double_yellow_dashed_right")
    pub center_line_class: Option<String>,
    /// The passing legality from RoadClassifier
    pub passing_legality: String,
    /// Whether the road classifier considers passing legal
    pub is_passing_legal: bool,
    /// Confidence of the road classification
    pub confidence: f32,
}

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

    /// v6.1: The actual line marking class that was crossed during this maneuver.
    /// Set by main.rs after correlating with LineCrossingDetector events.
    pub crossed_line_class: Option<String>,
    /// v6.1: The class_id of the crossed line marking (4=solid_white, 5=solid_yellow, etc.)
    pub crossed_line_class_id: Option<usize>,

    /// v8.0: Whether the maneuver was executed on a curve.
    /// True when curvature or boundary coherence indicates the road is curving
    /// at the time of the lane change / overtake.
    pub is_on_curve: bool,
    /// v8.0: Snapshot of the road classification at the time of the maneuver.
    /// Captures the RoadClassifier's temporal consensus (line type, passing legality)
    /// so we know what the center line was when the overtake started.
    pub road_classification_at_maneuver: Option<RoadClassificationSnapshot>,
}

// ============================================================================
// INTERNAL BUFFER ENTRIES
// ============================================================================

#[derive(Debug, Clone)]
struct BufferedPass {
    event: PassEvent,
    correlated: bool,
    /// v4.13b: VehicleOvertookEgo passes suppressed by the curve gate are
    /// deferred rather than consumed. This allows later re-interpretation
    /// as an OVERTAKE if a corroborating ego lateral shift appears.
    /// (Ego's lane change caused the zone transition, not the vehicle overtaking.)
    curve_deferred: bool,
}

#[derive(Debug, Clone)]
struct BufferedShift {
    event: LateralShiftEvent,
    correlated: bool,
    /// v7.0: An early LANE_CHANGE was already emitted for this shift via
    /// ShiftConfirmedNotification. Don't emit another in the LC phase.
    early_lc_emitted: bool,
}

// ============================================================================
// CLASSIFIER
// ============================================================================

pub struct ManeuverClassifier {
    config: ClassifierConfig,
    pass_buffer: VecDeque<BufferedPass>,
    shift_buffer: VecDeque<BufferedShift>,
    current_ego_motion: EgoMotionEstimate,
    /// v4.10: Timestamped ego motion samples (timestamp_ms, velocity_px).
    /// Used for time-window queries during pass event validation.
    ego_motion_during_window: VecDeque<(f64, f32)>,
    recent_events: Vec<ManeuverEvent>,
    total_maneuvers: u64,
    latest_markings: MarkingSnapshot,
    // v4.13b: Curve mode awareness for zone-oscillation suppression.
    // On curves, BESIDE→AHEAD zone flips are unreliable (perspective distortion).
    // VehicleOvertookEgo events require additional validation when active.
    curve_mode_active: bool,
    frames_since_curve_mode: u32,
    /// v7.0: Pending early shift confirmed notifications to process in classify().
    pending_confirmed: Vec<ShiftConfirmedNotification>,
    /// v7.0: Tracks (start_frame, direction) of shifts that already had an early
    /// LANE_CHANGE emitted, to prevent duplicate LC when the completed shift arrives.
    early_lc_records: Vec<(u64, ShiftDirection)>,
    /// v8.0: Current road classification snapshot (from RoadClassifier temporal consensus).
    /// Updated each frame by the pipeline. Used as fallback legality when no crossing event fires.
    latest_road_classification: Option<RoadClassificationSnapshot>,
}

impl ManeuverClassifier {
    pub fn new(config: ClassifierConfig) -> Self {
        Self {
            config,
            pass_buffer: VecDeque::with_capacity(20),
            shift_buffer: VecDeque::with_capacity(20),
            current_ego_motion: EgoMotionEstimate::none(),
            ego_motion_during_window: VecDeque::with_capacity(900), // ~30s at 30fps
            recent_events: Vec::new(),
            total_maneuvers: 0,
            latest_markings: MarkingSnapshot::default(),
            curve_mode_active: false,
            frames_since_curve_mode: u32::MAX,
            pending_confirmed: Vec::new(),
            early_lc_records: Vec::new(),
            latest_road_classification: None,
        }
    }

    pub fn feed_pass(&mut self, event: PassEvent) {
        self.pass_buffer.push_back(BufferedPass {
            event,
            correlated: false,
            curve_deferred: false,
        });
    }

    pub fn feed_shift(&mut self, event: LateralShiftEvent) {
        // v7.0: Check if an early LC was already emitted for this shift.
        // Match by start_frame and direction — a completed shift with the same
        // start as a confirmed notification is the same shift.
        let early_lc = self.early_lc_records.iter().position(|(frame, dir)| {
            *frame == event.start_frame && *dir == event.direction
        });
        let early_lc_emitted = if let Some(idx) = early_lc {
            self.early_lc_records.swap_remove(idx);
            true
        } else {
            false
        };
        self.shift_buffer.push_back(BufferedShift {
            event,
            correlated: false,
            early_lc_emitted,
        });
    }

    /// v7.0: Accept an early "shift confirmed" notification for immediate
    /// LANE_CHANGE emission. Processed in the next classify() call.
    pub fn feed_shift_confirmed(&mut self, notification: ShiftConfirmedNotification) {
        self.pending_confirmed.push(notification);
    }

    pub fn feed_ego_motion(&mut self, estimate: EgoMotionEstimate, timestamp_ms: f64) {
        self.current_ego_motion = estimate;
        self.ego_motion_during_window
            .push_back((timestamp_ms, estimate.lateral_velocity_px));
        if self.ego_motion_during_window.len() > 900 {
            self.ego_motion_during_window.pop_front();
        }
    }

    pub fn update_markings(&mut self, snapshot: MarkingSnapshot) {
        self.latest_markings = snapshot;
    }

    /// v8.0: Update road classification snapshot from the RoadClassifier.
    /// Called by the pipeline each frame with the latest temporal consensus.
    pub fn update_road_classification(&mut self, snapshot: RoadClassificationSnapshot) {
        self.latest_road_classification = Some(snapshot);
    }

    /// v4.13b: Update curve mode state from the lateral detector.
    /// Called by the pipeline each frame BEFORE classify().
    /// Uses a cooldown to handle curve mode chattering — if curve was
    /// active anytime in the last ~1.5s, zone-based events are suspect.
    pub fn set_curve_mode(&mut self, active: bool) {
        self.curve_mode_active = active;
        if active {
            self.frames_since_curve_mode = 0;
        } else {
            self.frames_since_curve_mode = self.frames_since_curve_mode.saturating_add(1);
        }
    }

    /// v4.13b: Whether curve conditions make zone-based VehicleOvertookEgo unreliable.
    /// True when in curve mode OR within cooldown after it (curve mode chatters).
    fn curve_suppresses_zone_events(&self) -> bool {
        // 45 frames ≈ 1.5s at 30fps — generous cooldown for curve chatter
        self.curve_mode_active || self.frames_since_curve_mode <= 45
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
        // v7.0: Cap early_lc_records to prevent unbounded growth.
        // Records older than 60s are no longer relevant.
        if self.early_lc_records.len() > 50 {
            self.early_lc_records.drain(..self.early_lc_records.len() - 20);
        }

        debug!(
            "═══ CLASSIFIER CALLED ═══ ts={:.1}s | {} passes | {} shifts",
            timestamp_ms / 1000.0,
            self.pass_buffer.len(),
            self.shift_buffer.len()
        );

        for (i, p) in self.pass_buffer.iter().enumerate() {
            debug!(
                "  Pass[{}]: track={} side={:?} direction={:?} conf={:.2} correlated={} beside_end={:.1}s",
                i, p.event.vehicle_track_id, p.event.side, p.event.direction,
                p.event.confidence, p.correlated, p.event.beside_end_ms / 1000.0
            );
        }

        for (i, s) in self.shift_buffer.iter().enumerate() {
            debug!(
                "  Shift[{}]: dir={:?} conf={:.2} correlated={} end={:.1}s",
                i,
                s.event.direction,
                s.event.confidence,
                s.correlated,
                s.event.end_ms / 1000.0
            );
        }

        // ── CORRELATE PASS + SHIFT → OVERTAKE ───────────────────
        let gap = self.config.max_correlation_gap_ms;

        for pass_idx in 0..self.pass_buffer.len() {
            if self.pass_buffer[pass_idx].correlated {
                continue;
            }
            if self.pass_buffer[pass_idx].event.direction == PassDirection::VehicleOvertookEgo {
                continue;
            }

            // ══════════════════════════════════════════════════════
            // v4.10: REJECT EGO-MOTION-INDUCED PASSES (pre-check)
            //
            // If ego's own lane change explains the apparent zone
            // transition, this pass is an artifact — skip correlation
            // so the shift event becomes a LANE_CHANGE instead.
            // ══════════════════════════════════════════════════════
            if self.ego_motion_explains_pass(&self.pass_buffer[pass_idx].event) {
                warn!(
                    "  ❌ REJECTED pass (track {}, side={:?}): ego lateral motion \
                     explains apparent zone transition (ego lane change artifact)",
                    self.pass_buffer[pass_idx].event.vehicle_track_id,
                    self.pass_buffer[pass_idx].event.side,
                );
                self.pass_buffer[pass_idx].correlated = true; // consume it
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

        // ══════════════════════════════════════════════════════════════════
        // UNCORRELATED HIGH-CONFIDENCE PASSES → OVERTAKE
        //
        // v4.3: Ego-motion gate (reject if NO ego motion)
        // v4.8: Pass quality gate (min_pass_confidence_single_source)
        // v4.10: Ego-induced gate already applied above (correlated check).
        //        Re-check here for passes that weren't consumed by the
        //        correlated path.
        // ══════════════════════════════════════════════════════════════════
        for pass_idx in 0..self.pass_buffer.len() {
            if self.pass_buffer[pass_idx].correlated {
                continue;
            }
            if self.pass_buffer[pass_idx].event.direction == PassDirection::VehicleOvertookEgo {
                continue;
            }

            let pass_conf = self.pass_buffer[pass_idx].event.confidence;

            info!(
                "🔍 Checking single-source pass: track={} side={:?} conf={:.2} \
                 (min_single={:.2}, min_pass_conf={:.2})",
                self.pass_buffer[pass_idx].event.vehicle_track_id,
                self.pass_buffer[pass_idx].event.side,
                pass_conf,
                self.config.min_single_source_confidence,
                self.config.min_pass_confidence_single_source,
            );

            // ── v4.10: Ego-motion-induced pass gate (redundant safety check) ──
            // The pre-check above should have caught this, but if timing
            // or buffer order caused it to slip through, catch it here.
            if self.ego_motion_explains_pass(&self.pass_buffer[pass_idx].event) {
                warn!(
                    "  ❌ REJECTED single-source pass (track {}): ego lateral motion \
                     explains apparent zone transition",
                    self.pass_buffer[pass_idx].event.vehicle_track_id,
                );
                self.pass_buffer[pass_idx].correlated = true;
                continue;
            }

            // ── v4.8: Pass event quality gate ──
            if pass_conf < self.config.min_pass_confidence_single_source {
                info!(
                    "  ❌ Pass conf too low for single-source: {:.2} < {:.2}",
                    pass_conf, self.config.min_pass_confidence_single_source
                );
                continue;
            }

            if pass_conf >= self.config.min_single_source_confidence {
                // ── v4.3 EGO-MOTION GATE (absence check) ──
                let ego_confirms = self.ego_motion_confirms_lateral();
                let had_any_lateral_shift = self.shift_buffer.iter().any(|s| !s.correlated);

                if !ego_confirms && !had_any_lateral_shift {
                    // v4.10: Check ego motion during the MANEUVER's time window,
                    // not just "the last 60 frames". The ego lane change may have
                    // happened seconds before the pass event fires (the pass fires
                    // when the vehicle disappears, which can be long after the ego
                    // completed its lane change).
                    //
                    // Window: from ahead_start_ms (or beside_start - 10s) to now.
                    // This captures the lane change that preceded the alongside phase.
                    let pass_event = &self.pass_buffer[pass_idx].event;
                    let window_start = pass_event
                        .ahead_start_ms
                        .min(pass_event.beside_start_ms - 10000.0)
                        .max(0.0);
                    let window_end = timestamp_ms;

                    let peak_ego = self.peak_ego_motion_in_window(window_start, window_end);

                    if peak_ego < self.config.min_ego_motion_for_single_source {
                        warn!(
                            "  ❌ REJECTED single-source pass (track {}): no lateral motion \
                             in maneuver window [{:.1}s..{:.1}s] \
                             (peak_ego={:.2} < {:.2}, ego_confirms={}, shifts={})",
                            self.pass_buffer[pass_idx].event.vehicle_track_id,
                            window_start / 1000.0,
                            window_end / 1000.0,
                            peak_ego,
                            self.config.min_ego_motion_for_single_source,
                            ego_confirms,
                            had_any_lateral_shift
                        );
                        self.pass_buffer[pass_idx].correlated = true;
                        continue;
                    }
                }

                let pass_event = self.pass_buffer[pass_idx].event.clone();
                let maneuver = self.build_overtake(&pass_event, None, legality_buffer);

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
                    info!(
                        "  ❌ Rejected: conf={:.2} < min_combined={:.2}",
                        maneuver.confidence, self.config.min_combined_confidence
                    );
                }
            } else {
                info!(
                    "  ❌ Pass conf too low: {:.2} < {:.2}",
                    pass_conf, self.config.min_single_source_confidence
                );
            }
        }

        // ══════════════════════════════════════════════════════════════════
        // v4.13b: CURVE-DEFERRED VehicleOvertookEgo + SHIFT → OVERTAKE
        //
        // IMPORTANT: This MUST run BEFORE the lane change emission below.
        // Otherwise the shift gets emitted as a standalone LANE_CHANGE first
        // (marked correlated), and the reinterpretation loop can't claim it.
        //
        // When the curve gate defers a VehicleOvertookEgo (no corroborating
        // shift at the time), a subsequent ego lateral shift may reveal that
        // the zone transition was actually caused by the ego overtaking the
        // vehicle, not the vehicle overtaking the ego.
        //
        // Pattern: ego LEFT shift + vehicle on RIGHT side (BESIDE_R→AHEAD)
        //   = ego moved to opposite lane to pass the vehicle
        //
        // The VehicleOvertookEgo zone sequence (BESIDE→AHEAD) is actually
        // the perspective effect of the ego's own lane change — the vehicle
        // appeared to shift zones because the camera moved, not the vehicle.
        // ══════════════════════════════════════════════════════════════════
        for pass_idx in 0..self.pass_buffer.len() {
            if self.pass_buffer[pass_idx].correlated {
                continue;
            }
            if !self.pass_buffer[pass_idx].curve_deferred {
                continue;
            }

            let pass = &self.pass_buffer[pass_idx].event;

            let mut best_shift_idx: Option<usize> = None;
            let mut best_time_gap = f64::MAX;

            for (si, shift) in self.shift_buffer.iter().enumerate() {
                if shift.correlated {
                    continue;
                }

                // Ego moved OPPOSITE to the vehicle's side:
                //   vehicle on RIGHT + ego shifted LEFT = ego overtook
                //   vehicle on LEFT  + ego shifted RIGHT = ego overtook
                let ego_overtook = matches!(
                    (pass.side, shift.event.direction),
                    (PassSide::Right, ShiftDirection::Left)
                        | (PassSide::Left, ShiftDirection::Right)
                );

                if !ego_overtook {
                    continue;
                }

                // Use extended window: pass's beside phase through a generous
                // post-ahead window, since the ego shift often starts well after
                // the vehicle's zone transitions due to tracking delays.
                let pass_window_end =
                    pass.beside_end_ms.max(pass.ahead_start_ms) + self.config.correlation_window_ms;

                let time_gap = temporal_gap(
                    pass.beside_start_ms,
                    pass_window_end,
                    shift.event.start_ms,
                    shift.event.end_ms,
                );

                if time_gap < gap && time_gap < best_time_gap {
                    best_shift_idx = Some(si);
                    best_time_gap = time_gap;
                }
            }

            if let Some(si) = best_shift_idx {
                let pass_event = self.pass_buffer[pass_idx].event.clone();
                let shift_event = self.shift_buffer[si].event.clone();

                let maneuver =
                    self.build_overtake(&pass_event, Some(&shift_event), legality_buffer);

                if maneuver.confidence >= self.config.min_combined_confidence {
                    info!(
                        "🚗 OVERTAKE (ego passed vehicle): Track {} re-interpreted \
                         VehicleOvertookEgo + {} shift | conf={:.2} | legality={:?}",
                        pass_event.vehicle_track_id,
                        shift_event.direction.as_str(),
                        maneuver.confidence,
                        maneuver.legality,
                    );

                    self.recent_events.push(maneuver);
                    self.total_maneuvers += 1;

                    self.pass_buffer[pass_idx].correlated = true;
                    self.shift_buffer[si].correlated = true;
                }
            }
        }

        // ── SHADOW OVERTAKE (vehicle overtakes ego) ──────────────
        for pass_idx in 0..self.pass_buffer.len() {
            if self.pass_buffer[pass_idx].correlated {
                continue;
            }
            if self.pass_buffer[pass_idx].event.direction == PassDirection::VehicleOvertookEgo {
                // v4.13b: Skip passes deferred by the curve gate — they're
                // waiting for potential re-interpretation as ego-overtakes.
                // If no corroborating shift appears, they'll expire from the
                // buffer naturally (not emitted as false shadow_overtake).
                if self.pass_buffer[pass_idx].curve_deferred {
                    continue;
                }

                // v4.12: Ego-motion-induced VehicleOvertookEgo gate
                if self.ego_motion_explains_pass(&self.pass_buffer[pass_idx].event) {
                    warn!(
                        "  ❌ REJECTED VehicleOvertookEgo (track {}): ego lateral motion \
                         explains apparent BESIDE→AHEAD zone transition \
                         (ego lane change / curve artifact, not being overtaken)",
                        self.pass_buffer[pass_idx].event.vehicle_track_id,
                    );
                    self.pass_buffer[pass_idx].correlated = true;
                    continue;
                }

                // v4.13b: Curve-induced zone oscillation gate.
                //
                // On curves, perspective distortion shifts bbox positions,
                // causing BESIDE→AHEAD zone flips that aren't real overtakes.
                // The truck goes BESIDE→AHEAD→BESIDE (zone oscillation).
                //
                // Single-source VehicleOvertookEgo (no lateral shift corroboration)
                // is unreliable during curve mode because it relies entirely on
                // zone position, which is the signal most distorted by curves.
                //
                // Gate: when curve conditions are active or recent, require that
                // VehicleOvertookEgo has corroboration from a lateral shift in the
                // shift_buffer (i.e., the ego was actually pushed aside). If no
                // corroborating shift exists, suppress as zone oscillation artifact.
                if self.curve_suppresses_zone_events() {
                    let has_corroborating_shift = self.shift_buffer.iter().any(|s| {
                        !s.correlated
                            && temporal_gap(
                                self.pass_buffer[pass_idx].event.beside_start_ms,
                                self.pass_buffer[pass_idx].event.beside_end_ms,
                                s.event.start_ms,
                                s.event.end_ms,
                            ) < self.config.correlation_window_ms
                    });

                    if !has_corroborating_shift {
                        warn!(
                            "  ⏳ DEFERRED VehicleOvertookEgo (track {}): curve-induced \
                             zone oscillation suspected (BESIDE→AHEAD flip on curve, no lateral \
                             shift corroboration YET) | curve_active={} frames_since={}",
                            self.pass_buffer[pass_idx].event.vehicle_track_id,
                            self.curve_mode_active,
                            self.frames_since_curve_mode,
                        );
                        // v4.13b: Defer instead of consuming. If a lateral shift
                        // appears later, this can be re-interpreted as an OVERTAKE
                        // (ego's lane change caused the zone transition).
                        self.pass_buffer[pass_idx].curve_deferred = true;
                        continue;
                    }
                }

                let pass_event = self.pass_buffer[pass_idx].event.clone();
                let maneuver = self.build_shadow_overtake(&pass_event);

                info!(
                    "⚠️  SHADOW OVERTAKE: track {} on {} | conf={:.2}",
                    pass_event.vehicle_track_id,
                    maneuver.side.as_str(),
                    maneuver.confidence,
                );

                self.recent_events.push(maneuver);
                self.total_maneuvers += 1;
                self.pass_buffer[pass_idx].correlated = true;
            }
        }

        // ══════════════════════════════════════════════════════════════
        // v7.0: EARLY LANE CHANGE (from in-progress shift confirmation)
        //
        // When a shift is confirmed while still in progress, we emit an
        // immediate LANE_CHANGE so the entry LC of an overtake fires in
        // near-real-time instead of waiting for the shift to complete.
        // ══════════════════════════════════════════════════════════════
        let pending: Vec<_> = self.pending_confirmed.drain(..).collect();
        for confirmed in pending {
            let side = match confirmed.direction {
                ShiftDirection::Left => ManeuverSide::Left,
                ShiftDirection::Right => ManeuverSide::Right,
            };

            // Geometric override check for curve suppression
            let (geo_override, geo_score) = if let Some(signals) = &confirmed.geometric_signals {
                if signals.confidence >= self.config.geometric_min_signal_confidence
                    && signals.suggests_lane_change
                {
                    let div_strength = (signals.boundary_velocity_divergence.abs() / 3.0).min(1.0);
                    let width_strength = (signals.lane_width_rate.abs() / 2.0).min(1.0);
                    let nf_strength = (signals.near_far_offset_divergence / 0.08).min(1.0);
                    let score = (div_strength * 0.5 + width_strength * 0.3 + nf_strength * 0.2)
                        * signals.confidence;
                    (score >= self.config.geometric_override_min_score, score)
                } else {
                    (false, 0.0)
                }
            } else {
                (false, 0.0)
            };

            // Curve suppression gate
            if confirmed.curve_mode && !geo_override {
                debug!(
                    "  Early LC suppressed: shift {} on curve without geometric override",
                    confirmed.direction.as_str(),
                );
                continue;
            }

            // Confidence
            let base_conf = confirmed.confidence * self.config.weight_lateral;
            let ego_conf = if self.ego_motion_confirms_lateral() {
                self.config.weight_ego_motion * 0.8
            } else {
                0.0
            };
            let geo_conf = geo_score * self.config.weight_geometric;
            let confidence = (base_conf + ego_conf + geo_conf).clamp(0.0, 0.95);

            if confidence < self.config.min_lane_change_confidence {
                debug!(
                    "  Early LC below threshold: {} conf={:.2} < {:.2}",
                    confirmed.direction.as_str(),
                    confidence,
                    self.config.min_lane_change_confidence,
                );
                continue;
            }

            let ego_confirms = self.ego_motion_confirms_lateral();

            let maneuver = ManeuverEvent {
                maneuver_type: ManeuverType::LaneChange,
                side,
                legality: LineLegality::Unknown,
                legality_at_crossing: None,
                confidence,
                sources: DetectionSources {
                    vehicle_tracking: false,
                    lane_detection: true,
                    ego_motion: ego_confirms,
                },
                start_ms: confirmed.start_ms,
                end_ms: confirmed.confirmed_ms,
                start_frame: confirmed.start_frame,
                end_frame: confirmed.confirmed_frame,
                duration_ms: confirmed.confirmed_ms - confirmed.start_ms,
                passed_vehicle_id: None,
                passed_vehicle_class: None,
                pass_event: None,
                lateral_event: None, // No completed shift yet
                marking_context: Some(self.latest_markings.clone()),
                crossed_line_class: None,
                crossed_line_class_id: None,
                is_on_curve: confirmed.curve_mode,
                road_classification_at_maneuver: self.latest_road_classification.clone(),
            };

            info!(
                "🔀 EARLY LANE_CHANGE: {} | conf={:.2} | geo_override={} | curve={}",
                maneuver.side.as_str(),
                maneuver.confidence,
                geo_override,
                confirmed.curve_mode,
            );

            self.recent_events.push(maneuver);
            self.total_maneuvers += 1;

            // Record so the completed shift won't produce a duplicate LC
            self.early_lc_records.push((confirmed.start_frame, confirmed.direction));
        }

        // ══════════════════════════════════════════════════════════════
        // v7.0: STANDALONE LANE CHANGE DETECTION
        //
        // Lateral shifts not correlated with any pass event are candidates
        // for standalone lane change events. On Peru's narrow curvy mountain
        // roads, the key challenge is distinguishing real lane changes from
        // curve-induced perspective artifacts.
        //
        // The polynomial tracker's geometric signals discriminate:
        //   - Curve: boundaries move TOGETHER (coherent) → suppress
        //   - Lane change: boundaries DIVERGE → allow (even on curves)
        // ══════════════════════════════════════════════════════════════
        for shift_idx in 0..self.shift_buffer.len() {
            if self.shift_buffer[shift_idx].correlated {
                continue;
            }

            // v7.0: Skip shifts that already had an early LC emitted.
            // These shifts are still available for overtake correlation (above),
            // but shouldn't produce a duplicate standalone LANE_CHANGE.
            if self.shift_buffer[shift_idx].early_lc_emitted {
                self.shift_buffer[shift_idx].correlated = true;
                continue;
            }

            let shift = &self.shift_buffer[shift_idx].event;

            // Step 1: Evaluate geometric override
            let (geo_override, geo_score) = self.evaluate_geometric_override(shift);

            // Step 2: Curve suppression gate
            // On curves without geometric override → suppress (perspective artifact)
            if shift.curve_mode && !geo_override {
                debug!(
                    "  LC suppressed: shift {} on curve without geometric override | peak={:.1}%",
                    shift.direction.as_str(),
                    shift.peak_offset * 100.0,
                );
                self.shift_buffer[shift_idx].correlated = true;
                continue;
            }

            // Step 3: Compute confidence
            let base_conf = shift.confidence * self.config.weight_lateral;
            let ego_conf = if self.ego_motion_confirms_lateral() {
                self.config.weight_ego_motion * 0.8
            } else {
                0.0
            };
            let geo_conf = geo_score * self.config.weight_geometric;

            let confidence = (base_conf + ego_conf + geo_conf).clamp(0.0, 0.95);

            if confidence < self.config.min_lane_change_confidence {
                debug!(
                    "  LC below threshold: {} conf={:.2} < {:.2}",
                    shift.direction.as_str(),
                    confidence,
                    self.config.min_lane_change_confidence,
                );
                continue; // Leave in buffer; may correlate later with a pass
            }

            // Step 4: Build and emit
            let ego_confirms = self.ego_motion_confirms_lateral();
            let maneuver = self.build_lane_change(shift, confidence, ego_confirms);

            info!(
                "🔀 LANE_CHANGE: {} | conf={:.2} | geo_override={} | geo_score={:.2} | curve_mode={}",
                maneuver.side.as_str(),
                maneuver.confidence,
                geo_override,
                geo_score,
                shift.curve_mode,
            );

            self.recent_events.push(maneuver);
            self.total_maneuvers += 1;
            self.shift_buffer[shift_idx].correlated = true;
        }

        // v4.13 FIX (Bug 1): Drain consumed entries immediately.
        // Prevents stale correlated passes from lingering in the buffer
        // for the full correlation_window_ms, producing log spam every frame.
        self.pass_buffer.retain(|p| !p.correlated);
        self.shift_buffer.retain(|s| !s.correlated);

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

        let pass_weight = if shift.is_none() && !ego_confirms {
            0.70
        } else {
            self.config.weight_pass
        };

        let pass_conf = pass.confidence * pass_weight;
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
            1 => 0.0,
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
            crossed_line_class: None,
            crossed_line_class_id: None,
            is_on_curve: self.curve_mode_active,
            road_classification_at_maneuver: self.latest_road_classification.clone(),
        }
    }

    fn build_shadow_overtake(&self, pass: &PassEvent) -> ManeuverEvent {
        let side = match pass.side {
            PassSide::Left => ManeuverSide::Left,
            PassSide::Right => ManeuverSide::Right,
        };

        ManeuverEvent {
            maneuver_type: ManeuverType::ShadowOvertake,
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
            crossed_line_class: None,
            crossed_line_class_id: None,
            is_on_curve: self.curve_mode_active,
            road_classification_at_maneuver: self.latest_road_classification.clone(),
        }
    }

    // ── v7.0: LANE CHANGE HELPERS ────────────────────────────────

    /// Evaluate whether geometric signals from the polynomial tracker
    /// indicate a real lane change, overriding curve suppression.
    ///
    /// Returns (should_override, confidence_score).
    ///
    /// On curves, both boundaries move together → divergence ≈ 0.
    /// On lane changes (even on curves), boundaries diverge.
    /// When divergence is strong enough, we allow the lane change through.
    fn evaluate_geometric_override(
        &self,
        shift: &LateralShiftEvent,
    ) -> (bool, f32) {
        let signals = match &shift.geometric_signals {
            Some(s) => s,
            None => return (false, 0.0),
        };

        // Need minimum signal confidence (both boundaries actively tracked)
        if signals.confidence < self.config.geometric_min_signal_confidence {
            return (false, 0.0);
        }

        // The polynomial tracker already computes suggests_lane_change using
        // a 2-of-3 vote on boundary_velocity_divergence, lane_width_rate,
        // and near_far_offset_divergence.
        if !signals.suggests_lane_change {
            return (false, 0.0);
        }

        // Compute graded confidence based on signal strengths.
        // Thresholds from polynomial_tracker: div>3.0, width_rate>2.0, nf_div>0.08
        let div_strength = (signals.boundary_velocity_divergence.abs() / 3.0).min(1.0);
        let width_strength = (signals.lane_width_rate.abs() / 2.0).min(1.0);
        let nf_strength = (signals.near_far_offset_divergence / 0.08).min(1.0);

        let geo_score = div_strength * 0.5 + width_strength * 0.3 + nf_strength * 0.2;
        let boosted_score = geo_score * signals.confidence;

        let should_override = boosted_score >= self.config.geometric_override_min_score;

        (should_override, boosted_score)
    }

    /// Build a ManeuverEvent for a standalone lane change.
    fn build_lane_change(
        &self,
        shift: &LateralShiftEvent,
        confidence: f32,
        ego_confirms: bool,
    ) -> ManeuverEvent {
        let side = match shift.direction {
            ShiftDirection::Left => ManeuverSide::Left,
            ShiftDirection::Right => ManeuverSide::Right,
        };

        ManeuverEvent {
            maneuver_type: ManeuverType::LaneChange,
            side,
            legality: LineLegality::Unknown, // Set later by crossing correlation
            legality_at_crossing: None,
            confidence,
            sources: DetectionSources {
                vehicle_tracking: false,
                lane_detection: true,
                ego_motion: ego_confirms,
            },
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
            crossed_line_class: None,
            crossed_line_class_id: None,
            is_on_curve: self.curve_mode_active,
            road_classification_at_maneuver: self.latest_road_classification.clone(),
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

    /// v4.10: Query peak ego lateral velocity within a time window.
    /// Returns peak absolute velocity during [start_ms, end_ms].
    /// Falls back to the full buffer if the window is empty.
    fn peak_ego_motion_in_window(&self, start_ms: f64, end_ms: f64) -> f32 {
        let in_window = self
            .ego_motion_during_window
            .iter()
            .filter(|(ts, _)| *ts >= start_ms && *ts <= end_ms)
            .map(|(_, v)| v.abs())
            .fold(0.0f32, |a, b| a.max(b));

        if in_window > 0.0 {
            return in_window;
        }

        // Fallback: window empty (timestamps not aligned), check full buffer
        self.ego_motion_during_window
            .iter()
            .map(|(_, v)| v.abs())
            .fold(0.0f32, |a, b| a.max(b))
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
            .map(|(_, v)| v.abs())
            .fold(0.0f32, |a, b| a.max(b));

        peak >= self.config.ego_motion_threshold
    }

    /// v4.10: Determine if ego's own lateral motion could create the
    /// apparent zone transition that generated this EgoOvertook pass event.
    ///
    /// When the ego vehicle changes lanes, vehicles that were AHEAD in
    /// the original lane appear to shift laterally through zones:
    ///
    ///   Ego moves LEFT (negative velocity):
    ///     Ahead vehicle appears to shift RIGHT
    ///     → AHEAD → BESIDE_RIGHT → BEHIND
    ///     → PassSide::Left (ego "passed on left") ← false positive
    ///
    ///   Ego moves RIGHT (positive velocity):
    ///     Ahead vehicle appears to shift LEFT
    ///     → AHEAD → BESIDE_LEFT → BEHIND
    ///     → PassSide::Right (ego "passed on right") ← false positive
    ///
    /// Returns true if ego lateral motion is sustained and strong enough
    /// in the direction that would produce the observed pass side.
    fn ego_motion_explains_pass(&self, pass: &PassEvent) -> bool {
        // v4.10/v4.12: Only EgoOvertook and VehicleOvertookEgo can be ego-induced.
        if pass.direction != PassDirection::EgoOvertook
            && pass.direction != PassDirection::VehicleOvertookEgo
        {
            return false;
        }

        if self.ego_motion_during_window.len() < 15 {
            return false;
        }

        let threshold = self.config.ego_induced_pass_velocity_threshold;
        let min_sustained = self.config.ego_induced_pass_min_sustained_frames;

        // Examine ego motion over a window covering the pass.
        // Use the last 90 samples (~3s at 30fps) to capture the lane
        // change that produced the apparent zone transition.
        let samples: Vec<f32> = self
            .ego_motion_during_window
            .iter()
            .rev()
            .take(90)
            .map(|(_, v)| *v)
            .collect();

        // v4.12: Direction-aware ego motion check.
        // EgoOvertook: same-direction ego creates artifact.
        // VehicleOvertookEgo: opposite-direction ego creates artifact.
        let (check_negative, check_positive) = match (pass.direction, pass.side) {
            (PassDirection::EgoOvertook, PassSide::Left) => (true, false),
            (PassDirection::EgoOvertook, PassSide::Right) => (false, true),
            (PassDirection::VehicleOvertookEgo, PassSide::Right) => (true, false),
            (PassDirection::VehicleOvertookEgo, PassSide::Left) => (false, true),
        };

        if check_negative {
            let sustained_left = samples.iter().filter(|&&v| v < -threshold).count();
            let peak_left = samples.iter().fold(0.0f32, |a, &b| a.min(b));
            if sustained_left >= min_sustained {
                info!(
                    "  🔍 ego_motion_explains_pass: track={} dir={:?} side={:?} | \
                     ego moving LEFT: peak={:.2} px/f, sustained={}/{} frames → EXPLAINS pass",
                    pass.vehicle_track_id,
                    pass.direction,
                    pass.side,
                    peak_left,
                    sustained_left,
                    min_sustained
                );
                return true;
            }
        }

        if check_positive {
            let sustained_right = samples.iter().filter(|&&v| v > threshold).count();
            let peak_right = samples.iter().fold(0.0f32, |a, &b| a.max(b));
            if sustained_right >= min_sustained {
                info!(
                    "  🔍 ego_motion_explains_pass: track={} dir={:?} side={:?} | \
                     ego moving RIGHT: peak={:.2} px/f, sustained={}/{} frames → EXPLAINS pass",
                    pass.vehicle_track_id,
                    pass.direction,
                    pass.side,
                    peak_right,
                    sustained_right,
                    min_sustained
                );
                return true;
            }
        }

        false
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
        self.curve_mode_active = false;
        self.frames_since_curve_mode = u32::MAX;
        self.latest_road_classification = None;
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
    let strict_match = matches!(
        (pass.side, shift.direction),
        (PassSide::Left, ShiftDirection::Left) | (PassSide::Right, ShiftDirection::Right)
    );

    strict_match || (pass.confidence > 0.75 && shift.confirmed)
}
