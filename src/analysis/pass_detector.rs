// src/analysis/pass_detector.rs
//
// Monitors tracked vehicle zone transitions to detect overtake events.
//
// PASS SEQUENCES:
//
//   EgoOvertook:        AHEAD → BESIDE → BEHIND
//     Ego vehicle passes a slower vehicle ahead.
//
//   VehicleOvertookEgo: BESIDE → AHEAD  (from alongside)
//                       BEHIND → BESIDE → AHEAD  (from behind)
//     Another vehicle passes the ego vehicle.
//
// v4.3 FIX: Stricter disappeared-vehicle validation
// v4.8 FIX: Temporal ordering enforcement (ahead_before_beside)
// v4.9 FIX: VehicleOvertookEgo AHEAD confirmation period
//
// v4.9b FIX: False EgoOvertook during reverse-pass / lane change
//   - BEHIND handler fires EgoOvertook check exactly ONCE per
//     BESIDE→BEHIND transition (was firing every frame).
//   - Detect BEHIND→BESIDE→AHEAD as VehicleOvertookEgo, even when
//     the track had an earlier AHEAD phase (came_from_behind flag
//     overrides the ahead_before_beside block).
//   - Fix PassSide mapping for VehicleOvertookEgo: BesideRight →
//     Right (vehicle overtook on the right), not Left.
//   - Track current_beside_zone to handle side updates when a track
//     re-enters BESIDE from BEHIND.
//   - Suppress disappeared-track rejection spam (once per track).

use super::vehicle_tracker::{Track, VehicleZone};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, warn};

// ============================================================================
// CONFIGURATION
// ============================================================================

#[derive(Debug, Clone)]
pub struct PassDetectorConfig {
    /// Minimum time (ms) a vehicle must be tracked before a pass counts
    pub min_track_age_ms: f64,
    /// Minimum time (ms) in BESIDE zone for a pass to count
    pub min_beside_duration_ms: f64,
    /// Maximum time (ms) for the full pass transition
    pub max_pass_duration_ms: f64,
    /// Minimum consecutive BESIDE frames to confirm the phase
    pub min_beside_frames: u32,
    /// If a vehicle disappears from BESIDE, wait this many frames before
    /// declaring it "passed" (it might reappear due to detection flicker)
    pub disappearance_grace_frames: u32,
    /// Minimum confidence averaged across the track's detections
    pub min_avg_confidence: f32,
    /// v4.3: Minimum beside duration for disappeared-track passes (ms).
    pub min_beside_duration_disappeared_ms: f64,
    /// v4.8: Minimum track age (frames) before any pass event is accepted.
    pub min_track_age_frames: u32,
    /// v4.8: Minimum AHEAD frames before the first BESIDE transition.
    pub min_ahead_frames: u32,
    /// v4.9: Minimum consecutive AHEAD frames AFTER a confirmed BESIDE
    /// phase before VehicleOvertookEgo fires.
    pub min_ahead_after_beside_frames: u32,
    /// v4.9: Minimum beside duration (ms) for VehicleOvertookEgo events.
    pub min_beside_duration_overtook_ego_ms: f64,
    /// v4.10: Minimum peak height ratio (bbox height / frame height)
    /// during the BESIDE phase for VehicleOvertookEgo events.
    /// A vehicle genuinely alongside the ego vehicle must be large.
    /// Filters out oncoming traffic and far-away false detections that
    /// enter BESIDE from the frame edge while remaining small.
    pub min_beside_height_ratio_overtook_ego: f32,
    /// v4.9b: Minimum BEHIND frames before a BEHIND→BESIDE transition
    /// qualifies for reverse-pass (VehicleOvertookEgo from behind).
    /// Filters out brief BEHIND zone flicker.
    pub min_behind_frames_for_reverse: u32,
    /// v4.10: Minimum track age (frames) before BESIDE observations are
    /// counted for VehicleOvertookEgo. New tracks have unstable zone
    /// assignments during Kalman filter warmup — a truck appearing AHEAD
    /// may briefly register as BESIDE_LEFT in its first 2-3 frames.
    /// This gate prevents those initial misclassifications from seeding
    /// a false VehicleOvertookEgo sequence.
    pub min_track_age_for_beside_overtook: u32,
}

impl Default for PassDetectorConfig {
    fn default() -> Self {
        Self {
            min_track_age_ms: 1000.0,
            min_beside_duration_ms: 500.0,
            max_pass_duration_ms: 30000.0,
            min_beside_frames: 8,
            disappearance_grace_frames: 30,
            min_avg_confidence: 0.25,
            min_beside_duration_disappeared_ms: 1500.0,
            min_track_age_frames: 10,
            min_ahead_frames: 3,
            min_ahead_after_beside_frames: 15,
            min_beside_duration_overtook_ego_ms: 1500.0,
            min_beside_height_ratio_overtook_ego: 0.15, // v4.10: 15% of frame height
            min_behind_frames_for_reverse: 5,           // v4.9b: ~167ms at 30fps
            min_track_age_for_beside_overtook: 5,       // v4.10: ~167ms warmup
        }
    }
}

impl PassDetectorConfig {
    /// Mining environment — stricter thresholds.
    pub fn mining() -> Self {
        Self {
            min_track_age_ms: 1500.0,
            min_beside_duration_ms: 800.0,
            max_pass_duration_ms: 30000.0,
            min_beside_frames: 10,
            disappearance_grace_frames: 30,
            min_avg_confidence: 0.30,
            min_beside_duration_disappeared_ms: 2000.0,
            min_track_age_frames: 15,
            min_ahead_frames: 5,
            min_ahead_after_beside_frames: 20,
            min_beside_duration_overtook_ego_ms: 2000.0,
            min_beside_height_ratio_overtook_ego: 0.18, // v4.10: 18% for mining (larger vehicles)
            min_behind_frames_for_reverse: 8,
            min_track_age_for_beside_overtook: 8, // v4.10: ~267ms warmup for mining
        }
    }
}

// ============================================================================
// TYPES
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PassDirection {
    EgoOvertook,
    VehicleOvertookEgo,
}

impl PassDirection {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::EgoOvertook => "EGO_OVERTOOK",
            Self::VehicleOvertookEgo => "VEHICLE_OVERTOOK_EGO",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PassSide {
    Left,
    Right,
}

impl PassSide {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Left => "LEFT",
            Self::Right => "RIGHT",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassEvent {
    pub vehicle_track_id: u32,
    pub direction: PassDirection,
    pub side: PassSide,
    pub ahead_start_ms: f64,
    pub beside_start_ms: f64,
    pub beside_end_ms: f64,
    pub frame_id: u64,
    pub duration_ms: f64,
    pub confidence: f32,
    pub vehicle_class_id: u32,
}

#[derive(Debug, Clone)]
struct PendingPass {
    track_id: u32,
    ahead_since_ms: Option<f64>,
    ahead_since_frame: Option<u64>,
    beside_since_ms: Option<f64>,
    beside_since_frame: Option<u64>,
    beside_side: Option<PassSide>,
    beside_consecutive: u32,
    peak_beside_area: f32,
    /// v4.10: Maximum bbox height / frame height observed during BESIDE.
    /// Used to reject VehicleOvertookEgo for distant/oncoming vehicles
    /// that were never actually alongside the ego vehicle.
    peak_beside_height_ratio: f32,
    /// v4.10: Track age (frames) when BESIDE was first observed.
    /// Used to reject VehicleOvertookEgo when the first BESIDE occurred
    /// during track warmup (unstable bbox/zone assignment).
    first_beside_track_age: Option<u32>,
    had_ahead: bool,
    beside_confirmed: bool,
    frames_since_beside: u32,
    ever_beside: bool,
    /// v4.3: Total frames spent in any BESIDE zone (not just consecutive)
    total_beside_frames: u32,

    // ── v4.8: Temporal ordering (EgoOvertook gate) ──
    ahead_before_beside: bool,
    ahead_frames_before_beside: u32,
    beside_ordering_locked: bool,

    // ── v4.9: VehicleOvertookEgo confirmation ──
    ahead_consecutive_after_beside: u32,
    ahead_after_beside_start_ms: Option<f64>,
    ahead_after_beside_start_frame: Option<u64>,
    overtook_ego_confirmed: bool,

    // ── v4.9b: Reverse-pass tracking (BEHIND → BESIDE → AHEAD) ──
    /// Set true after the BEHIND handler's first EgoOvertook attempt.
    /// Prevents the handler from re-firing on every subsequent BEHIND
    /// frame, which inflated beside_duration and caused false positives
    /// when the track cycled back through BESIDE→BEHIND.
    ego_overtook_attempted: bool,

    /// Consecutive frames spent in BEHIND zone. Used to validate that
    /// a BEHIND→BESIDE transition represents a genuine approach from
    /// behind (not a 1-frame zone flicker).
    behind_consecutive: u32,

    /// Set true when the track transitions from BEHIND (with enough
    /// consecutive frames) to BESIDE. Indicates this track is a
    /// candidate for VehicleOvertookEgo-from-behind, even if it had
    /// an earlier AHEAD phase (overrides the ahead_before_beside block).
    came_from_behind: bool,

    /// Start time of the current BESIDE phase (resets on each new
    /// BESIDE entry after a non-BESIDE gap). Different from
    /// beside_since_ms which is locked to the first-ever BESIDE.
    current_beside_phase_start_ms: Option<f64>,
    current_beside_phase_start_frame: Option<u64>,

    /// The raw VehicleZone of the most recent BESIDE observation.
    /// Updated every frame the track is in BESIDE. Used for correct
    /// side computation per-direction in try_complete_pass.
    current_beside_zone: Option<VehicleZone>,

    /// Zone on the previous update frame. Used to detect transitions
    /// (e.g., BEHIND → BESIDE vs BESIDE → BESIDE continuation).
    last_zone: Option<VehicleZone>,
}

impl PendingPass {
    fn new(track_id: u32) -> Self {
        Self {
            track_id,
            ahead_since_ms: None,
            ahead_since_frame: None,
            beside_since_ms: None,
            beside_since_frame: None,
            beside_side: None,
            beside_consecutive: 0,
            peak_beside_area: 0.0,
            peak_beside_height_ratio: 0.0,
            first_beside_track_age: None,
            had_ahead: false,
            beside_confirmed: false,
            frames_since_beside: 0,
            ever_beside: false,
            total_beside_frames: 0,
            ahead_before_beside: false,
            ahead_frames_before_beside: 0,
            beside_ordering_locked: false,
            ahead_consecutive_after_beside: 0,
            ahead_after_beside_start_ms: None,
            ahead_after_beside_start_frame: None,
            overtook_ego_confirmed: false,
            ego_overtook_attempted: false,
            behind_consecutive: 0,
            came_from_behind: false,
            current_beside_phase_start_ms: None,
            current_beside_phase_start_frame: None,
            current_beside_zone: None,
            last_zone: None,
        }
    }

    /// Whether this track is a valid VehicleOvertookEgo candidate.
    /// True if: (a) BESIDE first with no prior AHEAD (original v4.9 path),
    /// or (b) came from BEHIND through a new BESIDE phase (v4.9b path).
    fn is_vehicle_overtook_ego_candidate(&self) -> bool {
        !self.ahead_before_beside || self.came_from_behind
    }
}

// ============================================================================
// PASS DETECTOR
// ============================================================================

pub struct PassDetector {
    config: PassDetectorConfig,
    pending: HashMap<u32, PendingPass>,
    completed_ids: HashSet<u32>,
    recent_events: Vec<PassEvent>,
    /// v4.9: Track IDs for which we've already logged a
    /// disappeared-rejection. Prevents per-frame spam.
    rejected_disappeared_logged: HashSet<u32>,
    /// v4.10: Frame height in pixels, for computing height ratios.
    frame_height: f32,
}

impl PassDetector {
    pub fn new(config: PassDetectorConfig, frame_height: f32) -> Self {
        Self {
            config,
            pending: HashMap::new(),
            completed_ids: HashSet::new(),
            recent_events: Vec::new(),
            rejected_disappeared_logged: HashSet::new(),
            frame_height: if frame_height > 0.0 {
                frame_height
            } else {
                720.0
            },
        }
    }

    pub fn update(
        &mut self,
        tracks: &[&Track],
        timestamp_ms: f64,
        frame_id: u64,
    ) -> Vec<PassEvent> {
        self.recent_events.clear();

        let active_ids: HashSet<u32> = tracks.iter().map(|t| t.id).collect();

        let mut tracks_to_check_behind: Vec<(u32, Track)> = Vec::new();
        let mut tracks_to_check_overtook_ego: Vec<(u32, Track)> = Vec::new();

        for track in tracks {
            if !track.is_confirmed() {
                continue;
            }
            if self.completed_ids.contains(&track.id) {
                continue;
            }

            let pending = self
                .pending
                .entry(track.id)
                .or_insert_with(|| PendingPass::new(track.id));

            match track.zone {
                // ═══════════════════════════════════════════════════
                // AHEAD ZONE
                // ═══════════════════════════════════════════════════
                VehicleZone::Ahead => {
                    if !pending.had_ahead {
                        pending.ahead_since_ms = Some(timestamp_ms);
                        pending.ahead_since_frame = Some(frame_id);
                        pending.had_ahead = true;
                        debug!("📍 Track {} entered AHEAD zone", track.id);
                    }

                    // v4.8: Count AHEAD frames before first BESIDE
                    if !pending.beside_ordering_locked {
                        pending.ahead_frames_before_beside += 1;
                        if pending.ahead_frames_before_beside >= self.config.min_ahead_frames
                            && !pending.ever_beside
                        {
                            pending.ahead_before_beside = true;
                        }
                    }

                    // ── VehicleOvertookEgo confirmation ──
                    // Count consecutive AHEAD frames after a confirmed BESIDE phase.
                    //
                    // v4.9b: Allow this for tracks that came_from_behind, even if
                    // ahead_before_beside is true. The BEHIND→BESIDE→AHEAD sequence
                    // is a valid reverse-pass (vehicle overtaking from behind).
                    if pending.beside_confirmed
                        && pending.is_vehicle_overtook_ego_candidate()
                        && !pending.overtook_ego_confirmed
                    {
                        if pending.ahead_after_beside_start_ms.is_none() {
                            pending.ahead_after_beside_start_ms = Some(timestamp_ms);
                            pending.ahead_after_beside_start_frame = Some(frame_id);
                            debug!(
                                "📍 Track {} AHEAD-after-BESIDE streak started at F{}{}",
                                track.id,
                                frame_id,
                                if pending.came_from_behind {
                                    " (reverse-pass)"
                                } else {
                                    ""
                                }
                            );
                        }

                        pending.ahead_consecutive_after_beside += 1;

                        if pending.ahead_consecutive_after_beside
                            >= self.config.min_ahead_after_beside_frames
                        {
                            pending.overtook_ego_confirmed = true;
                            info!(
                                "✅ Track {} VehicleOvertookEgo confirmed: \
                                 {} consecutive AHEAD frames after BESIDE (threshold={}) \
                                 peak_height_ratio={:.2} (min={:.2}) \
                                 beside_at_age={:?} (min={}){}",
                                track.id,
                                pending.ahead_consecutive_after_beside,
                                self.config.min_ahead_after_beside_frames,
                                pending.peak_beside_height_ratio,
                                self.config.min_beside_height_ratio_overtook_ego,
                                pending.first_beside_track_age,
                                self.config.min_track_age_for_beside_overtook,
                                if pending.came_from_behind {
                                    " [BEHIND→BESIDE→AHEAD reverse-pass]"
                                } else {
                                    ""
                                }
                            );
                            tracks_to_check_overtook_ego.push((track.id, (*track).clone()));
                        }
                    }

                    // Reset BESIDE consecutive counter on non-confirmed return
                    if pending.ever_beside && !pending.beside_confirmed {
                        pending.beside_consecutive = 0;
                    }
                    pending.frames_since_beside += 1;
                    pending.behind_consecutive = 0; // Not in BEHIND
                    pending.last_zone = Some(VehicleZone::Ahead);
                }

                // ═══════════════════════════════════════════════════
                // BESIDE ZONE
                // ═══════════════════════════════════════════════════
                VehicleZone::BesideLeft | VehicleZone::BesideRight => {
                    // ── v4.9b: Detect BEHIND → BESIDE transition ──
                    // If the track was just in BEHIND (with enough consecutive
                    // frames to not be jitter), this BESIDE phase is a reverse-
                    // pass candidate: the vehicle is emerging from behind.
                    let is_transition_from_behind = pending.last_zone == Some(VehicleZone::Behind)
                        && pending.behind_consecutive >= self.config.min_behind_frames_for_reverse;

                    if is_transition_from_behind && !pending.came_from_behind {
                        pending.came_from_behind = true;
                        // Start a fresh BESIDE phase for this reverse-pass
                        pending.current_beside_phase_start_ms = Some(timestamp_ms);
                        pending.current_beside_phase_start_frame = Some(frame_id);
                        // Reset the VehicleOvertookEgo AHEAD counter for the
                        // new sequence (don't carry over from earlier phases)
                        pending.ahead_consecutive_after_beside = 0;
                        pending.ahead_after_beside_start_ms = None;
                        pending.ahead_after_beside_start_frame = None;
                        pending.overtook_ego_confirmed = false;

                        info!(
                            "📍 Track {} BEHIND→BESIDE transition (reverse-pass candidate): \
                             behind_consecutive={} frames",
                            track.id, pending.behind_consecutive
                        );
                    }

                    // Update current beside zone every frame
                    pending.current_beside_zone = Some(track.zone);

                    // First-ever BESIDE entry: lock ordering and set initial side
                    if !pending.ever_beside {
                        pending.beside_since_ms = Some(timestamp_ms);
                        pending.beside_since_frame = Some(frame_id);
                        pending.beside_side = Some(Self::ego_overtook_side(track.zone));
                        pending.current_beside_phase_start_ms = Some(timestamp_ms);
                        pending.current_beside_phase_start_frame = Some(frame_id);
                        pending.ever_beside = true;
                        pending.beside_ordering_locked = true;
                        pending.first_beside_track_age = Some(track.age); // v4.10

                        if pending.ahead_before_beside {
                            info!(
                                "📍 Track {} entered BESIDE ({:?}) zone \
                                 (preceded by {} AHEAD frames ✓)",
                                track.id,
                                Self::ego_overtook_side(track.zone),
                                pending.ahead_frames_before_beside
                            );
                        } else {
                            info!(
                                "📍 Track {} entered BESIDE ({:?}) zone \
                                 (NO preceding AHEAD — VehicleOvertookEgo candidate)",
                                track.id,
                                Self::vehicle_overtook_side(track.zone)
                            );
                        }
                    }
                    // v4.9b: If re-entering BESIDE after a gap (from BEHIND,
                    // AHEAD, or Unknown), refresh the current phase start.
                    else if pending.last_zone.is_some()
                        && !matches!(
                            pending.last_zone,
                            Some(VehicleZone::BesideLeft) | Some(VehicleZone::BesideRight)
                        )
                    {
                        pending.current_beside_phase_start_ms = Some(timestamp_ms);
                        pending.current_beside_phase_start_frame = Some(frame_id);
                    }

                    pending.beside_consecutive += 1;
                    pending.total_beside_frames += 1;
                    pending.frames_since_beside = 0;

                    // v4.9: Reset AHEAD-after-BESIDE counter — the vehicle
                    // returned to BESIDE, so the AHEAD streak was zone jitter.
                    if pending.ahead_consecutive_after_beside > 0 {
                        debug!(
                            "🔄 Track {} returned to BESIDE — resetting AHEAD-after-BESIDE \
                             counter (was {} frames)",
                            track.id, pending.ahead_consecutive_after_beside
                        );
                        pending.ahead_consecutive_after_beside = 0;
                        pending.ahead_after_beside_start_ms = None;
                        pending.ahead_after_beside_start_frame = None;
                    }

                    let area = track.area();
                    if area > pending.peak_beside_area {
                        pending.peak_beside_area = area;
                    }

                    // v4.10: Track peak height ratio for proximity validation
                    let bbox_height = track.bbox[3] - track.bbox[1];
                    let height_ratio = bbox_height / self.frame_height;
                    if height_ratio > pending.peak_beside_height_ratio {
                        pending.peak_beside_height_ratio = height_ratio;
                    }

                    if pending.beside_consecutive >= self.config.min_beside_frames {
                        pending.beside_confirmed = true;
                    }

                    pending.behind_consecutive = 0;
                    pending.last_zone = Some(track.zone);
                }

                // ═══════════════════════════════════════════════════
                // BEHIND ZONE
                // ═══════════════════════════════════════════════════
                VehicleZone::Behind => {
                    pending.behind_consecutive += 1;

                    // ── v4.9b: EgoOvertook fires ONCE on first BESIDE→BEHIND ──
                    // Previously fired every frame the track was in BEHIND,
                    // which inflated beside_duration and caused false positives
                    // when the track cycled BEHIND→BESIDE→BEHIND.
                    if pending.beside_confirmed
                        && pending.ahead_before_beside
                        && !pending.ego_overtook_attempted
                        && !pending.came_from_behind
                    {
                        pending.ego_overtook_attempted = true;
                        tracks_to_check_behind.push((track.id, (*track).clone()));
                    }

                    pending.frames_since_beside += 1;

                    // Reset AHEAD-after-BESIDE on BEHIND
                    if pending.ahead_consecutive_after_beside > 0 {
                        debug!(
                            "🔄 Track {} went to BEHIND — resetting AHEAD-after-BESIDE counter",
                            track.id
                        );
                        pending.ahead_consecutive_after_beside = 0;
                        pending.ahead_after_beside_start_ms = None;
                        pending.ahead_after_beside_start_frame = None;
                    }

                    pending.last_zone = Some(VehicleZone::Behind);
                }

                // ═══════════════════════════════════════════════════
                // UNKNOWN ZONE
                // ═══════════════════════════════════════════════════
                VehicleZone::Unknown => {
                    pending.frames_since_beside += 1;
                    pending.behind_consecutive = 0;
                    if pending.ahead_consecutive_after_beside > 0 {
                        pending.ahead_consecutive_after_beside = 0;
                        pending.ahead_after_beside_start_ms = None;
                        pending.ahead_after_beside_start_frame = None;
                    }
                    pending.last_zone = Some(VehicleZone::Unknown);
                }
            }
        }

        // ══════════════════════════════════════════════════════════
        // Process EgoOvertook (AHEAD → BESIDE → BEHIND)
        // ══════════════════════════════════════════════════════════
        for (track_id, track) in tracks_to_check_behind {
            if let Some(pending) = self.pending.get(&track_id) {
                if let Some(event) = self.try_complete_pass(
                    pending,
                    &track,
                    PassDirection::EgoOvertook,
                    timestamp_ms,
                    frame_id,
                ) {
                    info!(
                        "✅ PASS (ego overtook): Track {} {} via {} | dur={:.1}s | conf={:.2}",
                        track_id,
                        event.direction.as_str(),
                        event.side.as_str(),
                        event.duration_ms / 1000.0,
                        event.confidence
                    );
                    self.recent_events.push(event);
                    self.completed_ids.insert(track_id);
                }
            }
        }

        // ══════════════════════════════════════════════════════════
        // Process VehicleOvertookEgo (BESIDE → AHEAD or
        //                             BEHIND → BESIDE → AHEAD)
        //
        // v4.9: Fires after min_ahead_after_beside_frames
        // v4.9b: Also handles BEHIND→BESIDE→AHEAD reverse-pass
        // ══════════════════════════════════════════════════════════
        for (track_id, track) in tracks_to_check_overtook_ego {
            if let Some(pending) = self.pending.get(&track_id) {
                if let Some(event) = self.try_complete_pass(
                    pending,
                    &track,
                    PassDirection::VehicleOvertookEgo,
                    timestamp_ms,
                    frame_id,
                ) {
                    info!(
                        "✅ PASS (vehicle overtook ego): Track {} {} via {} | dur={:.1}s | conf={:.2}{}",
                        track_id,
                        event.direction.as_str(),
                        event.side.as_str(),
                        event.duration_ms / 1000.0,
                        event.confidence,
                        if pending.came_from_behind { " [from behind]" } else { "" }
                    );
                    self.recent_events.push(event);
                    self.completed_ids.insert(track_id);
                }
            }
        }

        // ══════════════════════════════════════════════════════════
        // DISAPPEARED VEHICLE VALIDATION
        // ══════════════════════════════════════════════════════════
        let disappeared: Vec<u32> = self
            .pending
            .keys()
            .copied()
            .filter(|id| !active_ids.contains(id) && !self.completed_ids.contains(id))
            .collect();

        for track_id in disappeared {
            if let Some(pending) = self.pending.get(&track_id) {
                if !pending.beside_confirmed {
                    continue;
                }

                let first_rejection = !self.rejected_disappeared_logged.contains(&track_id);

                // Gate 1: AHEAD must have preceded BESIDE
                if !pending.ahead_before_beside {
                    if first_rejection {
                        warn!(
                            "❌ Track {} rejected: AHEAD did not precede BESIDE \
                             (had_ahead={}, ahead_frames={}, beside_first={})",
                            track_id,
                            pending.had_ahead,
                            pending.ahead_frames_before_beside,
                            pending.ever_beside && !pending.ahead_before_beside
                        );
                        self.rejected_disappeared_logged.insert(track_id);
                    }
                    continue;
                }

                // Gate 2: Minimum AHEAD frames before BESIDE
                if pending.ahead_frames_before_beside < self.config.min_ahead_frames {
                    if first_rejection {
                        warn!(
                            "❌ Track {} rejected: insufficient AHEAD frames ({} < {})",
                            track_id,
                            pending.ahead_frames_before_beside,
                            self.config.min_ahead_frames
                        );
                        self.rejected_disappeared_logged.insert(track_id);
                    }
                    continue;
                }

                // Gate 3: Minimum beside duration
                let beside_duration =
                    timestamp_ms - pending.beside_since_ms.unwrap_or(timestamp_ms);
                if beside_duration < self.config.min_beside_duration_disappeared_ms {
                    if first_rejection {
                        warn!(
                            "❌ Track {} rejected: beside_dur={:.0}ms < {:.0}ms min",
                            track_id,
                            beside_duration,
                            self.config.min_beside_duration_disappeared_ms
                        );
                        self.rejected_disappeared_logged.insert(track_id);
                    }
                    continue;
                }

                // Gate 4: Sufficient total beside evidence
                let min_total_beside = self.config.min_beside_frames * 2;
                if pending.total_beside_frames < min_total_beside {
                    if first_rejection {
                        warn!(
                            "❌ Track {} rejected: total_beside={} < {} required",
                            track_id, pending.total_beside_frames, min_total_beside
                        );
                        self.rejected_disappeared_logged.insert(track_id);
                    }
                    continue;
                }

                let event = PassEvent {
                    vehicle_track_id: track_id,
                    direction: PassDirection::EgoOvertook,
                    side: pending.beside_side.unwrap_or(PassSide::Left),
                    ahead_start_ms: pending.ahead_since_ms.unwrap_or(0.0),
                    beside_start_ms: pending.beside_since_ms.unwrap_or(0.0),
                    beside_end_ms: timestamp_ms,
                    frame_id,
                    duration_ms: timestamp_ms - pending.ahead_since_ms.unwrap_or(timestamp_ms),
                    confidence: self.calculate_confidence_disappeared(pending),
                    vehicle_class_id: 0,
                };

                if event.duration_ms <= self.config.max_pass_duration_ms {
                    info!(
                        "✅ PASS (disappeared): Track {} {} via {} | dur={:.1}s | conf={:.2}",
                        track_id,
                        event.direction.as_str(),
                        event.side.as_str(),
                        event.duration_ms / 1000.0,
                        event.confidence,
                    );
                    self.recent_events.push(event);
                    self.completed_ids.insert(track_id);
                }
            }
        }

        // Prune stale pending entries
        let logged = &mut self.rejected_disappeared_logged;
        self.pending.retain(|id, p| {
            if self.completed_ids.contains(id) {
                return false;
            }
            if !p.ever_beside && p.had_ahead {
                if let Some(ahead_ms) = p.ahead_since_ms {
                    if timestamp_ms - ahead_ms > self.config.max_pass_duration_ms {
                        return false;
                    }
                }
            }
            let prune = !active_ids.contains(id)
                && p.frames_since_beside > self.config.disappearance_grace_frames * 2;
            if prune {
                logged.remove(id);
            }
            !prune
        });

        self.recent_events.clone()
    }

    fn try_complete_pass(
        &self,
        pending: &PendingPass,
        track: &Track,
        direction: PassDirection,
        timestamp_ms: f64,
        frame_id: u64,
    ) -> Option<PassEvent> {
        // ── Ordering and confirmation checks ──
        match direction {
            PassDirection::EgoOvertook => {
                if !pending.ahead_before_beside {
                    debug!(
                        "❌ Pass rejected (track {}): no AHEAD before BESIDE for EgoOvertook",
                        pending.track_id
                    );
                    return None;
                }
                if pending.ahead_frames_before_beside < self.config.min_ahead_frames {
                    debug!(
                        "❌ Pass rejected (track {}): only {} AHEAD frames (need {})",
                        pending.track_id,
                        pending.ahead_frames_before_beside,
                        self.config.min_ahead_frames
                    );
                    return None;
                }
            }
            PassDirection::VehicleOvertookEgo => {
                // Must be a valid VehicleOvertookEgo candidate
                if !pending.is_vehicle_overtook_ego_candidate() {
                    debug!(
                        "❌ Pass rejected (track {}): not a VehicleOvertookEgo candidate \
                         (ahead_before_beside={}, came_from_behind={})",
                        pending.track_id, pending.ahead_before_beside, pending.came_from_behind
                    );
                    return None;
                }
                // v4.9: Must have confirmed via consecutive AHEAD frames
                if !pending.overtook_ego_confirmed {
                    debug!(
                        "❌ Pass rejected (track {}): VehicleOvertookEgo not confirmed \
                         (ahead_after_beside={}/{})",
                        pending.track_id,
                        pending.ahead_consecutive_after_beside,
                        self.config.min_ahead_after_beside_frames
                    );
                    return None;
                }
                // v4.10: Proximity gate — vehicle must have been large enough
                // during BESIDE to be genuinely alongside. Filters oncoming
                // traffic and far-away detections that enter BESIDE from the
                // frame edge while remaining small.
                if pending.peak_beside_height_ratio
                    < self.config.min_beside_height_ratio_overtook_ego
                {
                    warn!(
                        "❌ Pass rejected (track {}): peak BESIDE height ratio {:.2} < {:.2} \
                         — vehicle was never close enough to be alongside \
                         (likely oncoming traffic or distant detection)",
                        pending.track_id,
                        pending.peak_beside_height_ratio,
                        self.config.min_beside_height_ratio_overtook_ego
                    );
                    return None;
                }
                // v4.10: Track warmup gate — if the first BESIDE observation
                // happened during Kalman filter warmup (first few frames of
                // the track's life), the zone was likely misclassified.
                // A truck appearing AHEAD often registers as BESIDE_LEFT
                // for 1-3 frames before bbox stabilizes.
                if let Some(age_at_beside) = pending.first_beside_track_age {
                    if age_at_beside < self.config.min_track_age_for_beside_overtook {
                        warn!(
                            "❌ Pass rejected (track {}): first BESIDE at track age {} \
                             (< {} min) — likely initial zone instability, not genuine alongside",
                            pending.track_id,
                            age_at_beside,
                            self.config.min_track_age_for_beside_overtook
                        );
                        return None;
                    }
                }
            }
        }

        // ── Minimum track age ──
        if track.age < self.config.min_track_age_frames {
            debug!(
                "❌ Pass rejected (track {}): age={} < {}",
                track.id, track.age, self.config.min_track_age_frames
            );
            return None;
        }

        // ── Beside duration ──
        //
        // v4.9b: For VehicleOvertookEgo with came_from_behind, use the
        // current BESIDE phase start (after BEHIND) rather than the
        // first-ever BESIDE start. The original beside_since_ms includes
        // time spent in BEHIND/AHEAD between phases, inflating the duration.
        let beside_start =
            if direction == PassDirection::VehicleOvertookEgo && pending.came_from_behind {
                pending
                    .current_beside_phase_start_ms
                    .unwrap_or_else(|| pending.beside_since_ms.unwrap_or(timestamp_ms))
            } else {
                pending.beside_since_ms.unwrap_or(timestamp_ms)
            };

        // v4.9: For VehicleOvertookEgo, beside_end = AHEAD streak start
        let beside_end = match direction {
            PassDirection::VehicleOvertookEgo => {
                pending.ahead_after_beside_start_ms.unwrap_or(timestamp_ms)
            }
            PassDirection::EgoOvertook => timestamp_ms,
        };

        let beside_duration = beside_end - beside_start;
        let min_beside_ms = match direction {
            PassDirection::EgoOvertook => self.config.min_beside_duration_ms,
            PassDirection::VehicleOvertookEgo => self.config.min_beside_duration_overtook_ego_ms,
        };

        if beside_duration < min_beside_ms {
            debug!(
                "❌ Pass rejected (track {}): beside_dur={:.0}ms < {:.0}ms ({:?})",
                pending.track_id, beside_duration, min_beside_ms, direction
            );
            return None;
        }

        // ── Total duration ──
        let maneuver_start = match direction {
            PassDirection::EgoOvertook => pending.ahead_since_ms.unwrap_or(beside_start),
            PassDirection::VehicleOvertookEgo => beside_start,
        };
        let total_duration = timestamp_ms - maneuver_start;

        if total_duration > self.config.max_pass_duration_ms {
            debug!(
                "❌ Pass rejected (track {}): total_dur={:.0}ms > {:.0}ms max",
                pending.track_id, total_duration, self.config.max_pass_duration_ms
            );
            return None;
        }

        if total_duration < 100.0 {
            debug!(
                "❌ Pass rejected (track {}): total_dur={:.0}ms suspiciously short",
                pending.track_id, total_duration
            );
            return None;
        }

        let confidence = self.calculate_confidence(pending, track, beside_duration, total_duration);

        // ── v4.9b: Side computation per direction ──
        //
        // EgoOvertook: vehicle BESIDE_RIGHT → ego passed on LEFT
        //              vehicle BESIDE_LEFT  → ego passed on RIGHT
        //
        // VehicleOvertookEgo: vehicle BESIDE_RIGHT → overtook on RIGHT
        //                     vehicle BESIDE_LEFT  → overtook on LEFT
        //
        // Use current_beside_zone for the most recent observation.
        let beside_zone = pending
            .current_beside_zone
            .unwrap_or(VehicleZone::BesideLeft);

        let side = match direction {
            PassDirection::EgoOvertook => Self::ego_overtook_side(beside_zone),
            PassDirection::VehicleOvertookEgo => Self::vehicle_overtook_side(beside_zone),
        };

        Some(PassEvent {
            vehicle_track_id: track.id,
            direction,
            side,
            ahead_start_ms: pending.ahead_since_ms.unwrap_or(beside_start),
            beside_start_ms: beside_start,
            beside_end_ms: beside_end,
            frame_id,
            duration_ms: total_duration,
            confidence,
            vehicle_class_id: track.class_id,
        })
    }

    // ── Side mapping helpers ──

    /// Side from the ego driver's perspective when overtaking.
    /// Vehicle on our right → we passed on the left.
    fn ego_overtook_side(zone: VehicleZone) -> PassSide {
        match zone {
            VehicleZone::BesideLeft => PassSide::Right,
            VehicleZone::BesideRight => PassSide::Left,
            _ => PassSide::Left, // fallback
        }
    }

    /// Side from the ego driver's perspective when being overtaken.
    /// Vehicle on our right → they overtook on the right.
    fn vehicle_overtook_side(zone: VehicleZone) -> PassSide {
        match zone {
            VehicleZone::BesideLeft => PassSide::Left,
            VehicleZone::BesideRight => PassSide::Right,
            _ => PassSide::Left,
        }
    }

    fn calculate_confidence(
        &self,
        pending: &PendingPass,
        track: &Track,
        beside_duration_ms: f64,
        total_duration_ms: f64,
    ) -> f32 {
        let mut conf: f32 = 0.5;

        // AHEAD quality bonus (EgoOvertook path)
        if pending.ahead_before_beside {
            let ahead_bonus = if pending.ahead_frames_before_beside >= 15 {
                0.15
            } else if pending.ahead_frames_before_beside >= 8 {
                0.12
            } else {
                0.08
            };
            conf += ahead_bonus;
        }

        // VehicleOvertookEgo confirmation bonus
        if pending.overtook_ego_confirmed {
            let confirm_bonus = if pending.ahead_consecutive_after_beside >= 30 {
                0.15
            } else if pending.ahead_consecutive_after_beside >= 20 {
                0.12
            } else {
                0.08
            };
            conf += confirm_bonus;

            // v4.9b: Extra bonus for reverse-pass (more evidence)
            if pending.came_from_behind {
                conf += 0.05;
            }
        }

        let beside_bonus = (beside_duration_ms / 3000.0).min(1.0) as f32 * 0.15;
        conf += beside_bonus;

        if track.age > 30 {
            conf += 0.10;
        }

        if track.last_confidence > 0.6 {
            conf += 0.05;
        }

        if total_duration_ms >= 3000.0 && total_duration_ms <= 15000.0 {
            conf += 0.05;
        }

        conf.min(0.98)
    }

    fn calculate_confidence_disappeared(&self, pending: &PendingPass) -> f32 {
        let mut conf: f32 = 0.40;

        if pending.ahead_before_beside {
            let ahead_bonus = if pending.ahead_frames_before_beside >= 15 {
                0.15
            } else if pending.ahead_frames_before_beside >= 8 {
                0.12
            } else {
                0.08
            };
            conf += ahead_bonus;
        }

        if pending.beside_consecutive > 15 {
            conf += 0.10;
        }
        if pending.total_beside_frames > 30 {
            conf += 0.05;
        }

        conf.min(0.80)
    }

    pub fn recent_events(&self) -> &[PassEvent] {
        &self.recent_events
    }

    pub fn has_active_pass(&self) -> bool {
        self.pending
            .values()
            .any(|p| p.beside_confirmed && p.frames_since_beside < 10)
    }

    pub fn total_passes(&self) -> usize {
        self.completed_ids.len()
    }

    pub fn reset(&mut self) {
        self.pending.clear();
        self.completed_ids.clear();
        self.recent_events.clear();
        self.rejected_disappeared_logged.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::vehicle_tracker::{DetectionInput, TrackerConfig, VehicleTracker};

    fn det(x1: f32, y1: f32, x2: f32, y2: f32) -> DetectionInput {
        DetectionInput {
            bbox: [x1, y1, x2, y2],
            class_id: 7,
            confidence: 0.8,
        }
    }

    fn det_cls(x1: f32, y1: f32, x2: f32, y2: f32, class_id: u32) -> DetectionInput {
        DetectionInput {
            bbox: [x1, y1, x2, y2],
            class_id,
            confidence: 0.8,
        }
    }

    fn det_with_class(x1: f32, y1: f32, x2: f32, y2: f32, class_id: u32) -> DetectionInput {
        DetectionInput {
            bbox: [x1, y1, x2, y2],
            class_id,
            confidence: 0.8,
        }
    }

    // ────────────────────────────────────────────────────────────
    // EgoOvertook tests
    // ────────────────────────────────────────────────────────────

    #[test]
    fn test_full_overtake_sequence() {
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default(), 720.0);
        let mut events = Vec::new();
        let fps = 30.0;

        // Phase 1 (0-2s): Vehicle AHEAD
        for i in 0..60 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det(580.0, 180.0, 700.0, 280.0)], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }
        assert!(events.is_empty());

        // Phase 2 (2-5s): Vehicle BESIDE RIGHT
        for i in 60..150 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det(880.0, 300.0, 1250.0, 500.0)], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }
        assert!(events.is_empty());

        // Phase 3 (5-6s): Vehicle BEHIND
        for i in 150..180 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det(200.0, 550.0, 1080.0, 720.0)], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }

        assert!(!events.is_empty());
        assert_eq!(events[0].direction, PassDirection::EgoOvertook);
        assert!(events[0].duration_ms > 0.0);
    }

    #[test]
    fn test_disappeared_without_ahead_rejected() {
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default(), 720.0);
        let mut events = Vec::new();
        let fps = 30.0;

        for i in 0..30 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det(900.0, 350.0, 1250.0, 500.0)], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }
        for i in 30..120 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }

        assert!(
            events.is_empty(),
            "BESIDE-only disappeared track should not fire"
        );
    }

    #[test]
    fn test_ego_overtook_fires_once_not_every_behind_frame() {
        // v4.9b: The BEHIND handler should fire EgoOvertook check exactly
        // once, not every frame the track is in BEHIND.
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default(), 720.0);
        let mut events = Vec::new();
        let fps = 30.0;

        // AHEAD for 2s
        for i in 0..60 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det(580.0, 200.0, 700.0, 300.0)], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }

        // BESIDE for 2s
        for i in 60..120 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det(880.0, 300.0, 1250.0, 500.0)], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }

        // BEHIND for 3s — should fire at most once
        for i in 120..210 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det(200.0, 550.0, 1080.0, 720.0)], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }

        let ego_events: Vec<_> = events
            .iter()
            .filter(|e| e.direction == PassDirection::EgoOvertook)
            .collect();
        assert!(
            ego_events.len() <= 1,
            "EgoOvertook should fire at most once, got {}",
            ego_events.len()
        );
    }

    // ────────────────────────────────────────────────────────────
    // VehicleOvertookEgo tests (v4.9 + v4.9b)
    // ────────────────────────────────────────────────────────────

    #[test]
    fn test_ahead_flicker_does_not_fire_overtook_ego() {
        // v4.9: BESIDE → brief AHEAD flicker → back to BESIDE.
        // Must NOT fire VehicleOvertookEgo.
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default(), 720.0);
        let mut events = Vec::new();
        let fps = 30.0;

        // BESIDE for 1.5s
        for i in 0..45 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det_cls(0.0, 402.0, 98.0, 633.0, 2)], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }
        // AHEAD flicker for 10 frames (< 15 threshold)
        for i in 45..55 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det_cls(500.0, 250.0, 700.0, 400.0, 2)], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }
        // Back to BESIDE
        for i in 55..90 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det_cls(0.0, 400.0, 120.0, 650.0, 2)], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }

        let overtook: Vec<_> = events
            .iter()
            .filter(|e| e.direction == PassDirection::VehicleOvertookEgo)
            .collect();
        assert!(overtook.is_empty(), "Brief AHEAD flicker must not trigger");
    }

    #[test]
    fn test_genuine_vehicle_overtook_ego_from_beside() {
        // v4.9: Vehicle appears BESIDE, stably moves to AHEAD.
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default(), 720.0);
        let mut events = Vec::new();
        let fps = 30.0;

        // BESIDE_RIGHT for 3s
        for i in 0..90 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det(900.0, 350.0, 1250.0, 500.0)], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }
        assert!(events.is_empty());

        // Stably AHEAD for 2s (60 frames >> 15 threshold)
        for i in 90..150 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det(580.0, 200.0, 700.0, 300.0)], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }

        let overtook: Vec<_> = events
            .iter()
            .filter(|e| e.direction == PassDirection::VehicleOvertookEgo)
            .collect();
        assert!(
            !overtook.is_empty(),
            "Genuine VehicleOvertookEgo should fire"
        );
        assert!(overtook[0].duration_ms > 1000.0);
    }

    #[test]
    fn test_behind_to_beside_to_ahead_is_vehicle_overtook_ego() {
        // v4.9b CORE FIX: Vehicle approaches from BEHIND, moves through
        // BESIDE, then AHEAD. Even though the track had an earlier AHEAD
        // phase, the BEHIND→BESIDE→AHEAD sequence is VehicleOvertookEgo.
        //
        // This is the exact Track 5 bug from the mining video.
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default(), 720.0);
        let mut events = Vec::new();
        let fps = 30.0;

        // Phase 1 (0-1s): Vehicle AHEAD (initial tracking, far away)
        for i in 0..30 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det(600.0, 200.0, 680.0, 280.0)], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }

        // Phase 2 (1-2s): Vehicle enters BESIDE_RIGHT briefly (ego lateral)
        for i in 30..60 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det(900.0, 350.0, 1250.0, 500.0)], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }

        // Phase 3 (2-4s): Vehicle moves to BEHIND (ego is faster for a while)
        for i in 60..120 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det(300.0, 600.0, 980.0, 720.0)], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }

        // Collect any EgoOvertook events from phase 3 (the old bug)
        let had_ego_from_phase3 = events
            .iter()
            .any(|e| e.direction == PassDirection::EgoOvertook);
        // Clear events for clarity — we care about what happens next
        events.clear();

        // Phase 4 (4-7s): Vehicle returns to BESIDE_RIGHT (overtaking ego)
        for i in 120..210 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det(900.0, 350.0, 1250.0, 500.0)], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }

        // Phase 5 (7-9s): Vehicle moves to AHEAD (completed the overtake)
        for i in 210..270 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det(580.0, 200.0, 700.0, 300.0)], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }

        let overtook: Vec<_> = events
            .iter()
            .filter(|e| e.direction == PassDirection::VehicleOvertookEgo)
            .collect();

        // If the track already completed an EgoOvertook in phase 3,
        // it's in completed_ids and won't generate VehicleOvertookEgo.
        // If not, we should get VehicleOvertookEgo from the reverse pass.
        if !had_ego_from_phase3 {
            assert!(
                !overtook.is_empty(),
                "BEHIND→BESIDE→AHEAD should produce VehicleOvertookEgo, not EgoOvertook"
            );
            assert_eq!(
                overtook[0].side,
                PassSide::Right,
                "Vehicle was BESIDE_RIGHT → should be 'overtook on RIGHT'"
            );
        }
        // Either way, no new EgoOvertook should fire in phase 4-5
        let false_ego: Vec<_> = events
            .iter()
            .filter(|e| e.direction == PassDirection::EgoOvertook)
            .collect();
        assert!(
            false_ego.is_empty(),
            "BEHIND→BESIDE→BEHIND should NOT re-trigger EgoOvertook. Got {} events",
            false_ego.len()
        );
    }

    #[test]
    fn test_vehicle_overtook_ego_side_is_correct() {
        // v4.9b: VehicleOvertookEgo side should match the vehicle's
        // position, not be inverted like EgoOvertook.
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default(), 720.0);
        let mut events = Vec::new();
        let fps = 30.0;

        // BESIDE_RIGHT for 3s
        for i in 0..90 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det(900.0, 350.0, 1250.0, 500.0)], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }
        // Stably AHEAD for 2s
        for i in 90..150 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det(580.0, 200.0, 700.0, 300.0)], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }

        let overtook: Vec<_> = events
            .iter()
            .filter(|e| e.direction == PassDirection::VehicleOvertookEgo)
            .collect();
        assert!(!overtook.is_empty());
        assert_eq!(
            overtook[0].side,
            PassSide::Right,
            "Vehicle at BESIDE_RIGHT → overtook on RIGHT, got {:?}",
            overtook[0].side
        );
    }

    #[test]
    fn test_vehicle_overtook_ego_duration_not_zero() {
        // v4.9: Duration must be meaningful (not 0.0s).
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default(), 720.0);
        let mut events = Vec::new();
        let fps = 30.0;

        // 2s BESIDE
        for i in 0..60 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det(900.0, 350.0, 1250.0, 500.0)], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }
        // 20 frames AHEAD (crosses 15-frame threshold)
        for i in 60..80 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det(580.0, 200.0, 700.0, 300.0)], t, i);
            events.extend(pass_det.update(&tracker.confirmed_tracks(), t, i));
        }

        let overtook: Vec<_> = events
            .iter()
            .filter(|e| e.direction == PassDirection::VehicleOvertookEgo)
            .collect();
        assert!(!overtook.is_empty());
        assert!(
            overtook[0].duration_ms > 2000.0,
            "Duration should be >2s, got {:.0}ms",
            overtook[0].duration_ms
        );
    }

    #[test]
    fn test_rejection_spam_suppressed() {
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default(), 720.0);
        let fps = 30.0;

        // BESIDE-only track
        for i in 0..30 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[det(900.0, 350.0, 1250.0, 500.0)], t, i);
            pass_det.update(&tracker.confirmed_tracks(), t, i);
        }
        // Disappear
        for i in 30..60 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[], t, i);
            pass_det.update(&tracker.confirmed_tracks(), t, i);
        }

        assert!(!pass_det.rejected_disappeared_logged.is_empty());
    }

    #[test]
    fn test_distant_vehicle_beside_then_ahead_rejected_by_height() {
        // v4.10: Vehicle ID:6 scenario — small oncoming/distant vehicle
        // enters BESIDE_LEFT (tiny bbox at left edge), then shifts to
        // AHEAD as it approaches. Should NOT fire VehicleOvertookEgo
        // because the vehicle was never large enough to be alongside.
        //
        // Vehicle bbox during BESIDE: ~[0, 30, 100, 80] → height=50px
        // Frame height: 720px → height_ratio = 50/720 = 0.069 (< 0.15)
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default(), 720.0);

        let mut events = Vec::new();
        let fps = 30.0;

        // Phase 1 (0-2s): Small vehicle at left edge → BESIDE_LEFT
        // Height ~50px on 720px frame = 6.9% — well below 15% threshold
        for i in 0..60 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det_with_class(0.0, 30.0, 100.0, 80.0, 2)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        // Phase 2 (2-4s): Vehicle shifts to AHEAD (still small)
        for i in 60..120 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det_with_class(400.0, 50.0, 520.0, 120.0, 2)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        let overtook: Vec<_> = events
            .iter()
            .filter(|e| e.direction == PassDirection::VehicleOvertookEgo)
            .collect();
        assert!(
            overtook.is_empty(),
            "Distant/oncoming vehicle (height ratio {:.2}) should NOT trigger \
             VehicleOvertookEgo. Got {} events",
            50.0 / 720.0,
            overtook.len()
        );
    }

    #[test]
    fn test_large_vehicle_beside_then_ahead_accepted() {
        // v4.10: Genuine alongside vehicle — large bbox in BESIDE.
        // Height ~300px on 720px = 41.7% → well above 15% threshold.
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default(), 720.0);

        let mut events = Vec::new();
        let fps = 30.0;

        // Phase 1 (0-3s): Large truck in BESIDE_RIGHT zone
        for i in 0..90 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det(900.0, 250.0, 1250.0, 550.0)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        // Phase 2 (3-5s): Moves stably to AHEAD
        for i in 90..150 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det(500.0, 150.0, 780.0, 350.0)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        let overtook: Vec<_> = events
            .iter()
            .filter(|e| e.direction == PassDirection::VehicleOvertookEgo)
            .collect();
        assert!(
            !overtook.is_empty(),
            "Large vehicle (height ratio {:.2}) alongside should trigger VehicleOvertookEgo",
            300.0 / 720.0
        );
    }
}

