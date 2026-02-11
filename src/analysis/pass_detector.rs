// src/analysis/pass_detector.rs
//
// Monitors tracked vehicle zone transitions to detect overtake events.
//
// An OVERTAKE occurs when a tracked vehicle transitions:
//   AHEAD → BESIDE → BEHIND (or disappeared from BESIDE)
//
// v4.3 FIX: Stricter disappeared-vehicle validation
//   - Require had_ahead phase for disappeared tracks
//   - Require minimum 1s beside duration for disappeared tracks
//   - Log rejected disappeared passes for debugging
//
// v4.8 FIX: Temporal ordering enforcement
//   - Track AHEAD-before-BESIDE ordering, not just boolean had_ahead
//   - Vehicles entering BESIDE first (oncoming, parked, adjacent lane)
//     are categorically rejected from EgoOvertook passes
//   - try_complete_pass now requires ahead_before_beside for EgoOvertook
//   - Minimum track age enforced on all pass paths
//   - BEHIND handler gated by ahead_before_beside

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
            min_track_age_frames: 10, // v4.8: ~333ms at 30fps
            min_ahead_frames: 3,      // v4.8: must have 3+ AHEAD frames before BESIDE
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
            min_track_age_frames: 15, // 500ms at 30fps
            min_ahead_frames: 5,      // Must have 5+ solid AHEAD frames
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

    // ── v4.8: Temporal ordering ──
    /// True only if AHEAD was observed BEFORE any BESIDE zone.
    /// A vehicle that enters BESIDE first (oncoming, parked, adjacent lane)
    /// gets ahead_before_beside=false and is permanently excluded from
    /// EgoOvertook pass events, even if it later flickers into AHEAD.
    ahead_before_beside: bool,
    /// Number of frames the track has spent in AHEAD zone before the
    /// first BESIDE transition. Must exceed min_ahead_frames to qualify.
    ahead_frames_before_beside: u32,
    /// Once the track first enters BESIDE, this is locked to prevent
    /// subsequent AHEAD flickers from resetting the ordering.
    beside_ordering_locked: bool,
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
}

impl PassDetector {
    pub fn new(config: PassDetectorConfig) -> Self {
        Self {
            config,
            pending: HashMap::new(),
            completed_ids: HashSet::new(),
            recent_events: Vec::new(),
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

                    // ── v4.8: Track AHEAD frames before first BESIDE ──
                    // Only count AHEAD frames while ordering is not yet locked
                    // (i.e., before the first BESIDE observation).
                    if !pending.beside_ordering_locked {
                        pending.ahead_frames_before_beside += 1;
                        // Mark ahead_before_beside once we have enough AHEAD frames
                        if pending.ahead_frames_before_beside >= self.config.min_ahead_frames
                            && !pending.ever_beside
                        {
                            pending.ahead_before_beside = true;
                        }
                    }

                    if pending.ever_beside && !pending.beside_confirmed {
                        pending.beside_consecutive = 0;
                    }
                    pending.frames_since_beside += 1;

                    // VehicleOvertookEgo: BESIDE → AHEAD (vehicle passed us)
                    // This requires beside_confirmed AND track was NOT ahead
                    // before beside (otherwise it's ego overtaking, not vehicle).
                    if pending.beside_confirmed && !pending.ahead_before_beside {
                        tracks_to_check_overtook_ego.push((track.id, (*track).clone()));
                    }
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
                        // v4.8: Lock the ordering — no further AHEAD frames
                        // can change ahead_before_beside after this point.
                        pending.beside_ordering_locked = true;

                        if pending.ahead_before_beside {
                            info!(
                                "📍 Track {} entered BESIDE ({:?}) zone (preceded by {} AHEAD frames ✓)",
                                track.id, side, pending.ahead_frames_before_beside
                            );
                        } else {
                            info!(
                                "📍 Track {} entered BESIDE ({:?}) zone (NO preceding AHEAD phase — \
                                 will not qualify for EgoOvertook)",
                                track.id, side
                            );
                        }
                    }

                    pending.beside_consecutive += 1;
                    pending.total_beside_frames += 1;
                    pending.frames_since_beside = 0;

                    let area = track.area();
                    if area > pending.peak_beside_area {
                        pending.peak_beside_area = area;
                    }

                    if pending.beside_consecutive >= self.config.min_beside_frames {
                        pending.beside_confirmed = true;
                    }
                }

                VehicleZone::Behind => {
                    // ── v4.8 FIX: Require ahead_before_beside for EgoOvertook ──
                    // Previously only checked beside_confirmed. A track that
                    // entered BESIDE first (oncoming/parked) and somehow got
                    // BEHIND would incorrectly fire as a pass.
                    if pending.beside_confirmed && pending.ahead_before_beside {
                        tracks_to_check_behind.push((track.id, (*track).clone()));
                    } else if pending.beside_confirmed && !pending.ahead_before_beside {
                        debug!(
                            "⏭️  Track {} in BEHIND but no AHEAD-before-BESIDE — skipping pass check",
                            track.id
                        );
                    }
                    pending.frames_since_beside += 1;
                }

                VehicleZone::Unknown => {
                    pending.frames_since_beside += 1;
                }
            }
        }

        // Process tracks that moved to BEHIND
        for (track_id, track) in tracks_to_check_behind {
            if let Some(pending) = self.pending.get(&track_id) {
                if let Some(event) = self.try_complete_pass(
                    pending,
                    &track,
                    PassDirection::EgoOvertook,
                    timestamp_ms,
                    frame_id,
                ) {
                    self.recent_events.push(event);
                    self.completed_ids.insert(track_id);
                }
            }
        }

        // Process tracks that overtook ego (BESIDE → AHEAD, no prior AHEAD)
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
                        "✅ PASS (vehicle overtook ego): Track {} | dur={:.1}s",
                        track_id,
                        event.duration_ms / 1000.0
                    );
                    self.recent_events.push(event);
                    self.completed_ids.insert(track_id);
                }
            }
        }

        // ══════════════════════════════════════════════════════════════════
        // DISAPPEARED VEHICLE VALIDATION
        //
        // v4.3: Require had_ahead, minimum beside duration, total beside frames
        // v4.8: Upgraded to require ahead_before_beside (temporal ordering)
        //       and minimum ahead_frames_before_beside
        // ══════════════════════════════════════════════════════════════════
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

                // ── v4.8 Gate 1: AHEAD must have preceded BESIDE ──
                // This catches vehicles that entered BESIDE first (oncoming,
                // parked, adjacent lane) and later flickered into AHEAD.
                // The boolean had_ahead alone is insufficient because it
                // doesn't encode temporal ordering.
                if !pending.ahead_before_beside {
                    warn!(
                        "❌ Track {} rejected: AHEAD did not precede BESIDE \
                         (had_ahead={}, ahead_frames={}, ever_beside_first={})",
                        track_id,
                        pending.had_ahead,
                        pending.ahead_frames_before_beside,
                        pending.ever_beside && !pending.ahead_before_beside
                    );
                    continue;
                }

                // v4.8 Gate 2: Minimum AHEAD frames before BESIDE
                if pending.ahead_frames_before_beside < self.config.min_ahead_frames {
                    warn!(
                        "❌ Track {} rejected: insufficient AHEAD frames before BESIDE \
                         ({} < {} required)",
                        track_id, pending.ahead_frames_before_beside, self.config.min_ahead_frames
                    );
                    continue;
                }

                // v4.3 Gate 3: Minimum beside duration (stricter for disappeared)
                let beside_duration =
                    timestamp_ms - pending.beside_since_ms.unwrap_or(timestamp_ms);
                if beside_duration < self.config.min_beside_duration_disappeared_ms {
                    warn!(
                        "❌ Track {} rejected: beside_dur={:.0}ms < {:.0}ms min for disappeared",
                        track_id, beside_duration, self.config.min_beside_duration_disappeared_ms
                    );
                    continue;
                }

                // v4.3 Gate 4: Sufficient total beside evidence
                let min_total_beside = self.config.min_beside_frames * 2;
                if pending.total_beside_frames < min_total_beside {
                    warn!(
                        "❌ Track {} rejected: total_beside={} < {} required",
                        track_id, pending.total_beside_frames, min_total_beside
                    );
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
            if !active_ids.contains(id)
                && p.frames_since_beside > self.config.disappearance_grace_frames * 2
            {
                return false;
            }
            true
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
        // ── v4.8 FIX: Enforce temporal ordering for EgoOvertook ──
        // For EgoOvertook: require AHEAD preceded BESIDE with enough frames.
        // For VehicleOvertookEgo: require BESIDE preceded AHEAD (inverse).
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
                // For VehicleOvertookEgo, BESIDE should come first — which means
                // ahead_before_beside should be FALSE. If ahead_before_beside is
                // true, this is actually EgoOvertook, not VehicleOvertookEgo.
                if pending.ahead_before_beside {
                    debug!(
                        "❌ Pass rejected (track {}): AHEAD preceded BESIDE, not VehicleOvertookEgo",
                        pending.track_id
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
        let beside_duration = timestamp_ms - beside_start;

        if beside_duration < self.config.min_beside_duration_ms {
            debug!(
                "❌ Pass rejected (track {}): beside duration {:.0}ms < {:.0}ms min",
                pending.track_id, beside_duration, self.config.min_beside_duration_ms
            );
            return None;
        }

        let total_duration = timestamp_ms - pending.ahead_since_ms.unwrap_or(beside_start);
        if total_duration > self.config.max_pass_duration_ms {
            debug!(
                "❌ Pass rejected (track {}): total duration {:.0}ms > {:.0}ms max",
                pending.track_id, total_duration, self.config.max_pass_duration_ms
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
            beside_end_ms: timestamp_ms,
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

        // v4.8: Scale AHEAD bonus by number of AHEAD frames (stronger evidence)
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
    }

    #[test]
    fn test_disappeared_without_ahead_rejected() {
        // v4.3: A track that only ever appeared as BESIDE (no AHEAD phase)
        // and then disappeared should NOT fire as a pass
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default());

        let mut events = Vec::new();
        let fps = 30.0;

        // Vehicle appears directly in BESIDE zone (no AHEAD phase)
        for i in 0..30 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det(900.0, 350.0, 1250.0, 500.0)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        // Vehicle disappears (no detections for 60+ frames)
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
    fn test_beside_then_ahead_flicker_rejected() {
        // v4.8: A track that enters BESIDE first, then flickers to AHEAD,
        // should NOT fire as an EgoOvertook pass.
        // This is the exact Track 6 scenario from the mining video.
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default());

        let mut events = Vec::new();
        let fps = 30.0;

        // Phase 1 (0-1s): Vehicle appears at left edge → BESIDE_LEFT
        for i in 0..30 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det_with_class(0.0, 402.0, 98.0, 633.0, 2)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        // Phase 2 (1-1.5s): Detection shifts slightly, might get AHEAD
        // (simulates bbox jitter)
        for i in 30..45 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det_with_class(400.0, 300.0, 600.0, 500.0, 2)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        // Phase 3 (1.5-3s): Back to BESIDE or disappears
        for i in 45..90 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det_with_class(0.0, 400.0, 120.0, 650.0, 2)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        // Phase 4: Disappears
        for i in 90..180 {
            let t = i as f64 * (1000.0 / fps);
            let dets: Vec<DetectionInput> = vec![];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }

        // Should NOT have any EgoOvertook events
        let ego_overtook: Vec<_> = events
            .iter()
            .filter(|e| e.direction == PassDirection::EgoOvertook)
            .collect();
        assert!(
            ego_overtook.is_empty(),
            "Track that entered BESIDE before AHEAD should never generate EgoOvertook. Got {} events",
            ego_overtook.len()
        );
    }

    #[test]
    fn test_genuine_overtake_with_sufficient_ahead() {
        // v4.8: A genuine overtake should still work — vehicle appears AHEAD
        // for many frames, then moves to BESIDE, then disappears.
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default());

        let mut events = Vec::new();
        let fps = 30.0;

        // Phase 1 (0-3s): Vehicle clearly AHEAD (centered, small)
        for i in 0..90 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det(580.0, 200.0, 700.0, 300.0)];
            tracker.update(&dets, t, i);
            let tracks = tracker.confirmed_tracks();
            let e = pass_det.update(&tracks, t, i);
            events.extend(e);
        }
        assert!(events.is_empty(), "No events during AHEAD phase");

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

        // Should have detected the pass (genuine AHEAD → BESIDE → disappeared)
        assert!(
            !events.is_empty(),
            "Genuine overtake with long AHEAD phase should fire"
        );
        assert_eq!(events[0].direction, PassDirection::EgoOvertook);
    }
}

