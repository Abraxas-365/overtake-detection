// src/analysis/pass_detector.rs
//
// Monitors tracked vehicle zone transitions to detect overtake events.
//
// An OVERTAKE (EgoOvertook) occurs when a tracked vehicle transitions:
//   AHEAD → BESIDE → BEHIND (or disappeared from BESIDE)
//
// A BEING_OVERTAKEN (VehicleOvertookEgo) occurs when a vehicle transitions:
//   BESIDE → AHEAD (with confirmation period)
//
// v4.3 FIX: Stricter disappeared-vehicle validation
//   - Require had_ahead phase for disappeared tracks
//   - Require minimum 1s beside duration for disappeared tracks
//   - Log rejected disappeared passes for debugging
//
// v4.8 FIX: Temporal ordering enforcement
//   - Track AHEAD-before-BESIDE ordering, not just boolean had_ahead
//   - Vehicles entering BESIDE first are excluded from EgoOvertook
//   - try_complete_pass requires ahead_before_beside for EgoOvertook
//   - Minimum track age on all pass paths
//
// v4.9 FIX: VehicleOvertookEgo confirmation period
//   - Require N consecutive AHEAD frames after confirmed BESIDE
//     before firing VehicleOvertookEgo. Prevents single-frame zone
//     flicker from triggering false BEING_OVERTAKEN events.
//   - Reset AHEAD-after-BESIDE counter when track returns to
//     BESIDE/BEHIND/Unknown (the AHEAD streak was just jitter).
//   - Fix duration calculation: beside_end = AHEAD streak start,
//     total_duration = beside_start → now. Was 0.0s because
//     ahead_since_ms was set same frame as event generation.
//   - Minimum beside duration for VehicleOvertookEgo (1.5s default).
//   - Suppress per-frame disappeared-track rejection spam:
//     warn once per track, not once per frame.

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
    /// Maximum time (ms) for the full AHEAD→BESIDE→BEHIND transition
    pub max_pass_duration_ms: f64,
    /// Minimum consecutive BESIDE frames to confirm the phase
    pub min_beside_frames: u32,
    /// If a vehicle disappears from BESIDE, wait this many frames before
    /// declaring it "passed" (it might reappear due to detection flicker)
    pub disappearance_grace_frames: u32,
    /// Minimum confidence averaged across the track's detections
    pub min_avg_confidence: f32,
    /// v4.3: Minimum beside duration for disappeared-track passes (ms).
    /// Stricter than min_beside_duration_ms because we have less evidence.
    pub min_beside_duration_disappeared_ms: f64,
    /// v4.8: Minimum track age (frames) before any pass event is accepted.
    /// Filters out ephemeral tracks that confirm then immediately pass.
    pub min_track_age_frames: u32,
    /// v4.8: Minimum AHEAD frames before the first BESIDE transition.
    /// Ensures the vehicle was genuinely tracked ahead, not just a
    /// single-frame zone flicker.
    pub min_ahead_frames: u32,
    /// v4.9: Minimum consecutive AHEAD frames AFTER a confirmed BESIDE
    /// phase before VehicleOvertookEgo fires. Prevents zone-flicker false
    /// positives where a BESIDE vehicle briefly gets classified AHEAD.
    pub min_ahead_after_beside_frames: u32,
    /// v4.9: Minimum beside duration (ms) for VehicleOvertookEgo events.
    /// A vehicle overtaking ego should be alongside for a meaningful period.
    pub min_beside_duration_overtook_ego_ms: f64,
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
            min_track_age_frames: 10,                    // ~333ms at 30fps
            min_ahead_frames: 3,                         // must see 3+ AHEAD before BESIDE
            min_ahead_after_beside_frames: 15,           // v4.9: ~500ms at 30fps
            min_beside_duration_overtook_ego_ms: 1500.0, // v4.9: 1.5s
        }
    }
}

impl PassDetectorConfig {
    /// Mining environment configuration — stricter thresholds due to
    /// dust, detection flicker, and large vehicles.
    pub fn mining() -> Self {
        Self {
            min_track_age_ms: 1500.0,
            min_beside_duration_ms: 800.0,
            max_pass_duration_ms: 30000.0,
            min_beside_frames: 10,
            disappearance_grace_frames: 30,
            min_avg_confidence: 0.30,
            min_beside_duration_disappeared_ms: 2000.0,
            min_track_age_frames: 15,                    // 500ms at 30fps
            min_ahead_frames: 5,                         // 5+ solid AHEAD frames
            min_ahead_after_beside_frames: 20,           // ~667ms at 30fps
            min_beside_duration_overtook_ego_ms: 2000.0, // 2s
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
    had_ahead: bool,
    beside_confirmed: bool,
    frames_since_beside: u32,
    ever_beside: bool,
    /// v4.3: Total frames spent in any BESIDE zone (not just consecutive)
    total_beside_frames: u32,

    // ── v4.8: Temporal ordering (EgoOvertook gate) ──
    /// True only if AHEAD was observed BEFORE any BESIDE zone.
    /// Permanently locked once BESIDE is first observed.
    ahead_before_beside: bool,
    /// AHEAD frames accumulated before the first BESIDE transition.
    ahead_frames_before_beside: u32,
    /// Lock: once BESIDE is first seen, no more AHEAD counting for
    /// the ordering check.
    beside_ordering_locked: bool,

    // ── v4.9: VehicleOvertookEgo confirmation ──
    /// Consecutive AHEAD frames accumulated AFTER beside_confirmed.
    /// Resets to 0 whenever the track returns to any non-AHEAD zone.
    ahead_consecutive_after_beside: u32,
    /// Timestamp when the current post-BESIDE AHEAD streak began.
    /// Used as beside_end_ms for the pass event (the moment the
    /// vehicle left the BESIDE zone and moved AHEAD).
    ahead_after_beside_start_ms: Option<f64>,
    /// Frame when the current post-BESIDE AHEAD streak began.
    ahead_after_beside_start_frame: Option<u64>,
    /// Once true, the pass event fires on the next update cycle.
    /// Set when ahead_consecutive_after_beside reaches threshold.
    overtook_ego_confirmed: bool,
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
        }
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
    /// disappeared-rejection reason. Prevents 60+ per-frame spam.
    rejected_disappeared_logged: HashSet<u32>,
}

impl PassDetector {
    pub fn new(config: PassDetectorConfig) -> Self {
        Self {
            config,
            pending: HashMap::new(),
            completed_ids: HashSet::new(),
            recent_events: Vec::new(),
            rejected_disappeared_logged: HashSet::new(),
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
                VehicleZone::Ahead => {
                    if !pending.had_ahead {
                        pending.ahead_since_ms = Some(timestamp_ms);
                        pending.ahead_since_frame = Some(frame_id);
                        pending.had_ahead = true;
                        debug!("📍 Track {} entered AHEAD zone", track.id);
                    }

                    // ── v4.8: Count AHEAD frames before first BESIDE ──
                    if !pending.beside_ordering_locked {
                        pending.ahead_frames_before_beside += 1;
                        if pending.ahead_frames_before_beside >= self.config.min_ahead_frames
                            && !pending.ever_beside
                        {
                            pending.ahead_before_beside = true;
                        }
                    }

                    // ── v4.9: Count consecutive AHEAD frames AFTER confirmed BESIDE ──
                    // This is the VehicleOvertookEgo confirmation counter.
                    // Only accumulate if:
                    //   (a) BESIDE was confirmed (real alongside phase)
                    //   (b) AHEAD did NOT precede BESIDE (otherwise it's
                    //       EgoOvertook, not VehicleOvertookEgo)
                    //   (c) Not already confirmed (don't re-trigger)
                    if pending.beside_confirmed
                        && !pending.ahead_before_beside
                        && !pending.overtook_ego_confirmed
                    {
                        // Record when this AHEAD streak started
                        if pending.ahead_after_beside_start_ms.is_none() {
                            pending.ahead_after_beside_start_ms = Some(timestamp_ms);
                            pending.ahead_after_beside_start_frame = Some(frame_id);
                            debug!(
                                "📍 Track {} AHEAD-after-BESIDE streak started at F{}",
                                track.id, frame_id
                            );
                        }

                        pending.ahead_consecutive_after_beside += 1;

                        if pending.ahead_consecutive_after_beside
                            >= self.config.min_ahead_after_beside_frames
                        {
                            pending.overtook_ego_confirmed = true;
                            info!(
                                "✅ Track {} VehicleOvertookEgo confirmed: \
                                 {} consecutive AHEAD frames after BESIDE (threshold={})",
                                track.id,
                                pending.ahead_consecutive_after_beside,
                                self.config.min_ahead_after_beside_frames
                            );
                            tracks_to_check_overtook_ego.push((track.id, (*track).clone()));
                        }
                    }

                    // Reset BESIDE consecutive counter if returning to AHEAD
                    // from a non-confirmed BESIDE streak
                    if pending.ever_beside && !pending.beside_confirmed {
                        pending.beside_consecutive = 0;
                    }
                    pending.frames_since_beside += 1;
                }

                VehicleZone::BesideLeft | VehicleZone::BesideRight => {
                    let side = if track.zone == VehicleZone::BesideLeft {
                        PassSide::Right
                    } else {
                        PassSide::Left
                    };

                    if !pending.ever_beside {
                        pending.beside_since_ms = Some(timestamp_ms);
                        pending.beside_since_frame = Some(frame_id);
                        pending.beside_side = Some(side);
                        pending.ever_beside = true;
                        // v4.8: Lock ordering — no further AHEAD frames
                        // can change ahead_before_beside.
                        pending.beside_ordering_locked = true;

                        if pending.ahead_before_beside {
                            info!(
                                "📍 Track {} entered BESIDE ({:?}) zone \
                                 (preceded by {} AHEAD frames ✓)",
                                track.id, side, pending.ahead_frames_before_beside
                            );
                        } else {
                            info!(
                                "📍 Track {} entered BESIDE ({:?}) zone \
                                 (NO preceding AHEAD phase — will not qualify for EgoOvertook)",
                                track.id, side
                            );
                        }
                    }

                    pending.beside_consecutive += 1;
                    pending.total_beside_frames += 1;
                    pending.frames_since_beside = 0;

                    // v4.9: Reset AHEAD-after-BESIDE counter — the vehicle
                    // returned to BESIDE, so the previous AHEAD streak was
                    // zone flicker, not a genuine forward transition.
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

                    if pending.beside_consecutive >= self.config.min_beside_frames {
                        pending.beside_confirmed = true;
                    }
                }

                VehicleZone::Behind => {
                    // ── v4.8: Require ahead_before_beside for EgoOvertook ──
                    if pending.beside_confirmed && pending.ahead_before_beside {
                        tracks_to_check_behind.push((track.id, (*track).clone()));
                    } else if pending.beside_confirmed && !pending.ahead_before_beside {
                        debug!(
                            "⏭️  Track {} in BEHIND but no AHEAD-before-BESIDE \
                             — skipping EgoOvertook check",
                            track.id
                        );
                    }
                    pending.frames_since_beside += 1;

                    // v4.9: Reset AHEAD-after-BESIDE on BEHIND.
                    // BESIDE→BEHIND is EgoOvertook direction, not VehicleOvertookEgo.
                    if pending.ahead_consecutive_after_beside > 0 {
                        debug!(
                            "🔄 Track {} went to BEHIND — resetting AHEAD-after-BESIDE counter",
                            track.id
                        );
                        pending.ahead_consecutive_after_beside = 0;
                        pending.ahead_after_beside_start_ms = None;
                        pending.ahead_after_beside_start_frame = None;
                    }
                }

                VehicleZone::Unknown => {
                    pending.frames_since_beside += 1;
                    // v4.9: Reset AHEAD-after-BESIDE on Unknown
                    if pending.ahead_consecutive_after_beside > 0 {
                        pending.ahead_consecutive_after_beside = 0;
                        pending.ahead_after_beside_start_ms = None;
                        pending.ahead_after_beside_start_frame = None;
                    }
                }
            }
        }

        // ══════════════════════════════════════════════════════════
        // Process tracks that moved to BEHIND (EgoOvertook path)
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
        // Process tracks that overtook ego (VehicleOvertookEgo)
        //
        // v4.9: Only fires after min_ahead_after_beside_frames
        // consecutive AHEAD frames following a confirmed BESIDE
        // phase. The confirmation check happens in the AHEAD zone
        // handler above — we only get here when overtook_ego_confirmed
        // transitions from false → true.
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
                        "✅ PASS (vehicle overtook ego): Track {} {} via {} | dur={:.1}s | conf={:.2}",
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
        // DISAPPEARED VEHICLE VALIDATION
        //
        // v4.3: Require had_ahead, minimum beside duration, total frames
        // v4.8: Require ahead_before_beside (temporal ordering)
        // v4.9: Warn once per track, not once per frame
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

                // v4.9: Only log the rejection reason once per track
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
                            "❌ Track {} rejected: insufficient AHEAD frames before BESIDE \
                             ({} < {} required)",
                            track_id,
                            pending.ahead_frames_before_beside,
                            self.config.min_ahead_frames
                        );
                        self.rejected_disappeared_logged.insert(track_id);
                    }
                    continue;
                }

                // Gate 3: Minimum beside duration (stricter for disappeared)
                let beside_duration =
                    timestamp_ms - pending.beside_since_ms.unwrap_or(timestamp_ms);
                if beside_duration < self.config.min_beside_duration_disappeared_ms {
                    if first_rejection {
                        warn!(
                            "❌ Track {} rejected: beside_dur={:.0}ms < {:.0}ms min for disappeared",
                            track_id, beside_duration, self.config.min_beside_duration_disappeared_ms
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
                        "✅ PASS (disappeared): Track {} {} via {} | dur={:.1}s | conf={:.2} | \
                         beside_total={} | ahead_frames={}",
                        track_id,
                        event.direction.as_str(),
                        event.side.as_str(),
                        event.duration_ms / 1000.0,
                        event.confidence,
                        pending.total_beside_frames,
                        pending.ahead_frames_before_beside
                    );
                    self.recent_events.push(event);
                    self.completed_ids.insert(track_id);
                }
            }
        }

        // Prune stale pending entries
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
                // v4.9: Clean up spam-suppression set
                self.rejected_disappeared_logged.remove(id);
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
        // ── v4.8+v4.9: Enforce temporal ordering and confirmation ──
        match direction {
            PassDirection::EgoOvertook => {
                if !pending.ahead_before_beside {
                    debug!(
                        "❌ Pass rejected (track {}): AHEAD did not precede BESIDE for EgoOvertook",
                        pending.track_id
                    );
                    return None;
                }
                if pending.ahead_frames_before_beside < self.config.min_ahead_frames {
                    debug!(
                        "❌ Pass rejected (track {}): only {} AHEAD frames before BESIDE (need {})",
                        pending.track_id,
                        pending.ahead_frames_before_beside,
                        self.config.min_ahead_frames
                    );
                    return None;
                }
            }
            PassDirection::VehicleOvertookEgo => {
                // Must NOT have had AHEAD before BESIDE
                if pending.ahead_before_beside {
                    debug!(
                        "❌ Pass rejected (track {}): AHEAD preceded BESIDE, not VehicleOvertookEgo",
                        pending.track_id
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
            }
        }

        // ── v4.8: Minimum track age ──
        if track.age < self.config.min_track_age_frames {
            debug!(
                "❌ Pass rejected (track {}): age={} < {} min frames",
                track.id, track.age, self.config.min_track_age_frames
            );
            return None;
        }

        let beside_start = pending.beside_since_ms?;

        // ── v4.9: Compute beside_end and min duration per direction ──
        //
        // For VehicleOvertookEgo: beside_end = when the stable AHEAD
        //   streak started (the vehicle left the BESIDE zone). This
        //   gives a meaningful beside duration instead of 0.0s.
        // For EgoOvertook: beside_end = now (track just entered BEHIND).
        let (beside_end, min_beside_ms) = match direction {
            PassDirection::VehicleOvertookEgo => {
                let end = pending.ahead_after_beside_start_ms.unwrap_or(timestamp_ms);
                (end, self.config.min_beside_duration_overtook_ego_ms)
            }
            PassDirection::EgoOvertook => (timestamp_ms, self.config.min_beside_duration_ms),
        };

        let beside_duration = beside_end - beside_start;
        if beside_duration < min_beside_ms {
            debug!(
                "❌ Pass rejected (track {}): beside duration {:.0}ms < {:.0}ms min ({:?})",
                pending.track_id, beside_duration, min_beside_ms, direction
            );
            return None;
        }

        // ── Total maneuver duration ──
        // For EgoOvertook: AHEAD start → now
        // For VehicleOvertookEgo: BESIDE start → now
        //   (there's no preceding AHEAD phase for this direction)
        let maneuver_start = match direction {
            PassDirection::EgoOvertook => pending.ahead_since_ms.unwrap_or(beside_start),
            PassDirection::VehicleOvertookEgo => beside_start,
        };
        let total_duration = timestamp_ms - maneuver_start;

        if total_duration > self.config.max_pass_duration_ms {
            debug!(
                "❌ Pass rejected (track {}): total duration {:.0}ms > {:.0}ms max",
                pending.track_id, total_duration, self.config.max_pass_duration_ms
            );
            return None;
        }

        // Sanity: total duration must be positive and meaningful
        if total_duration < 100.0 {
            debug!(
                "❌ Pass rejected (track {}): total duration {:.0}ms suspiciously short",
                pending.track_id, total_duration
            );
            return None;
        }

        let confidence = self.calculate_confidence(pending, track, beside_duration, total_duration);

        Some(PassEvent {
            vehicle_track_id: track.id,
            direction,
            side: pending.beside_side.unwrap_or(PassSide::Left),
            ahead_start_ms: pending.ahead_since_ms.unwrap_or(beside_start),
            beside_start_ms: beside_start,
            beside_end_ms: beside_end,
            frame_id,
            duration_ms: total_duration,
            confidence,
            vehicle_class_id: track.class_id,
        })
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

        // v4.9: AHEAD-after-BESIDE quality bonus (VehicleOvertookEgo path)
        if pending.overtook_ego_confirmed {
            let confirm_bonus = if pending.ahead_consecutive_after_beside >= 30 {
                0.15
            } else if pending.ahead_consecutive_after_beside >= 20 {
                0.12
            } else {
                0.08
            };
            conf += confirm_bonus;
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
        let mut pass_det = PassDetector::new(PassDetectorConfig::default());

        let mut events = Vec::new();
        let fps = 30.0;

        // Phase 1 (0-2s): Vehicle AHEAD
        for i in 0..60 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det(580.0, 180.0, 700.0, 280.0)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }
        assert!(events.is_empty());

        // Phase 2 (2-5s): Vehicle BESIDE RIGHT
        for i in 60..150 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det(880.0, 300.0, 1250.0, 500.0)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }
        assert!(events.is_empty());

        // Phase 3 (5-6s): Vehicle BEHIND
        for i in 150..180 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det(200.0, 550.0, 1080.0, 720.0)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        assert!(!events.is_empty());
        assert_eq!(events[0].direction, PassDirection::EgoOvertook);
        assert!(
            events[0].duration_ms > 0.0,
            "Duration should be positive, got {}",
            events[0].duration_ms
        );
    }

    #[test]
    fn test_disappeared_without_ahead_rejected() {
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default());

        let mut events = Vec::new();
        let fps = 30.0;

        // Vehicle appears directly in BESIDE zone
        for i in 0..30 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det(900.0, 350.0, 1250.0, 500.0)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        // Vehicle disappears
        for i in 30..120 {
            let t = i as f64 * (1000.0 / fps);
            let dets: Vec<DetectionInput> = vec![];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        assert!(
            events.is_empty(),
            "Disappeared track without AHEAD phase should not trigger pass"
        );
    }

    #[test]
    fn test_genuine_ego_overtook_with_sufficient_ahead() {
        // AHEAD → BESIDE → disappeared (genuine ego overtake)
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default());

        let mut events = Vec::new();
        let fps = 30.0;

        // Phase 1 (0-3s): Vehicle clearly AHEAD
        for i in 0..90 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det(580.0, 200.0, 700.0, 300.0)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }
        assert!(events.is_empty());

        // Phase 2 (3-6s): Vehicle moves to BESIDE RIGHT
        for i in 90..180 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det(900.0, 350.0, 1250.0, 500.0)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        // Phase 3 (6-9s): Vehicle disappears
        for i in 180..270 {
            let t = i as f64 * (1000.0 / fps);
            let dets: Vec<DetectionInput> = vec![];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        assert!(!events.is_empty(), "Genuine EgoOvertook should fire");
        assert_eq!(events[0].direction, PassDirection::EgoOvertook);
    }

    #[test]
    fn test_beside_then_ahead_flicker_rejected_for_ego_overtook() {
        // v4.8: BESIDE first, then AHEAD flicker → no EgoOvertook
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default());

        let mut events = Vec::new();
        let fps = 30.0;

        // Phase 1: Vehicle at left edge → BESIDE_LEFT
        for i in 0..30 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det_with_class(0.0, 402.0, 98.0, 633.0, 2)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        // Phase 2: Detection shifts center → might get AHEAD
        for i in 30..45 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det_with_class(400.0, 300.0, 600.0, 500.0, 2)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        // Phase 3: Disappears
        for i in 45..150 {
            let t = i as f64 * (1000.0 / fps);
            let dets: Vec<DetectionInput> = vec![];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        let ego_overtook: Vec<_> = events
            .iter()
            .filter(|e| e.direction == PassDirection::EgoOvertook)
            .collect();
        assert!(
            ego_overtook.is_empty(),
            "BESIDE-first track should never generate EgoOvertook"
        );
    }

    // ────────────────────────────────────────────────────────────
    // VehicleOvertookEgo tests (v4.9)
    // ────────────────────────────────────────────────────────────

    #[test]
    fn test_single_frame_ahead_flicker_no_overtook_ego() {
        // v4.9 CORE FIX: Track 6 bug scenario.
        // Vehicle enters BESIDE, zone flickers to AHEAD for a few frames,
        // returns to BESIDE or goes BEHIND. Must NOT fire VehicleOvertookEgo.
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default());

        let mut events = Vec::new();
        let fps = 30.0;

        // Phase 1 (0-1.5s): Vehicle in BESIDE zone
        for i in 0..45 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det_with_class(0.0, 402.0, 98.0, 633.0, 2)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        // Phase 2 (1.5-1.8s): Zone flickers to AHEAD for ~10 frames
        // (below the 15-frame threshold)
        for i in 45..55 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det_with_class(500.0, 250.0, 700.0, 400.0, 2)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        // Phase 3 (1.8-3s): Back to BESIDE
        for i in 55..90 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det_with_class(0.0, 400.0, 120.0, 650.0, 2)];
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
            "Brief AHEAD flicker ({} frames) after BESIDE must NOT trigger VehicleOvertookEgo. \
             Got {} events",
            10,
            overtook.len()
        );
    }

    #[test]
    fn test_genuine_vehicle_overtook_ego() {
        // v4.9: Genuine "being overtaken" — vehicle enters BESIDE for
        // a meaningful period, then stably transitions to AHEAD and stays.
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default());

        let mut events = Vec::new();
        let fps = 30.0;

        // Phase 1 (0-3s): Vehicle in BESIDE_RIGHT
        for i in 0..90 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det(900.0, 350.0, 1250.0, 500.0)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }
        assert!(events.is_empty(), "No events during BESIDE phase");

        // Phase 2 (3-5s): Vehicle moves stably to AHEAD
        // 60 frames >> 15-frame threshold → should confirm
        for i in 90..150 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det(580.0, 200.0, 700.0, 300.0)];
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
            "Genuine VehicleOvertookEgo should fire after {} consecutive AHEAD frames",
            60
        );
        assert!(
            overtook[0].duration_ms > 1000.0,
            "Duration should be meaningful (beside + ahead), got {:.0}ms",
            overtook[0].duration_ms
        );
        assert!(
            overtook[0].beside_end_ms < overtook[0].beside_start_ms + overtook[0].duration_ms,
            "beside_end_ms should be when AHEAD streak started, not same as event timestamp"
        );
    }

    #[test]
    fn test_vehicle_overtook_ego_duration_not_zero() {
        // v4.9: Verifies the dur=0.0s bug is fixed.
        // Duration must reflect the full BESIDE→AHEAD maneuver time.
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default());

        let mut events = Vec::new();
        let fps = 30.0;

        // 2s BESIDE + enough AHEAD for confirmation
        for i in 0..60 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det(900.0, 350.0, 1250.0, 500.0)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        // Move to AHEAD for 20 frames (crosses 15-frame threshold)
        for i in 60..80 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det(580.0, 200.0, 700.0, 300.0)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        let overtook: Vec<_> = events
            .iter()
            .filter(|e| e.direction == PassDirection::VehicleOvertookEgo)
            .collect();
        assert!(!overtook.is_empty(), "Should have fired VehicleOvertookEgo");
        // Duration should be approximately 2s beside + 0.5s AHEAD ≈ 2.5s
        assert!(
            overtook[0].duration_ms > 2000.0,
            "Duration should be >2s (beside + AHEAD confirmation), got {:.0}ms",
            overtook[0].duration_ms
        );
    }

    #[test]
    fn test_beside_too_short_for_overtook_ego() {
        // v4.9: If BESIDE duration is below min_beside_duration_overtook_ego_ms,
        // VehicleOvertookEgo should not fire even with stable AHEAD.
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let config = PassDetectorConfig {
            min_beside_duration_overtook_ego_ms: 3000.0, // 3s minimum
            ..PassDetectorConfig::default()
        };
        let mut pass_det = PassDetector::new(config);

        let mut events = Vec::new();
        let fps = 30.0;

        // 0.5s BESIDE only (15 frames, below 3s threshold)
        for i in 0..15 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det(900.0, 350.0, 1250.0, 500.0)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        // Stable AHEAD for 60 frames
        for i in 15..75 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det(580.0, 200.0, 700.0, 300.0)];
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
            "BESIDE too short (0.5s < 3s) should reject VehicleOvertookEgo"
        );
    }

    // ────────────────────────────────────────────────────────────
    // Spam suppression tests (v4.9)
    // ────────────────────────────────────────────────────────────

    #[test]
    fn test_rejection_spam_suppressed() {
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default());

        let fps = 30.0;

        // Create a BESIDE-only track
        for i in 0..30 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det(900.0, 350.0, 1250.0, 500.0)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            pass_det.update(&tracks, t, i);
        }

        // Disappear it — rejection logged once, not 60+ times
        for i in 30..60 {
            let t = i as f64 * (1000.0 / fps);
            tracker.update(&[], t, i);
            let tracks = tracker.confirmed_tracks();
            pass_det.update(&tracks, t, i);
        }

        assert!(
            !pass_det.rejected_disappeared_logged.is_empty(),
            "Should have recorded the rejected track ID"
        );
    }
}
