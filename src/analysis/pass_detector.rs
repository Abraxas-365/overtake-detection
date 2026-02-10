// src/analysis/pass_detector.rs
//
// Monitors tracked vehicle zone transitions to detect overtake events.
//
// An OVERTAKE occurs when a tracked vehicle transitions:
//   AHEAD → BESIDE → BEHIND (or disappeared from BESIDE)
//
// This means the ego vehicle passed the tracked vehicle.
//
// A vehicle OVERTAKING US (being overtaken) would show:
//   appeared-as-BESIDE → AHEAD
//
// Key design: this module is completely independent of lane markings.
// It works in dust, at night, in rain — whenever vehicles are detectable.

use super::vehicle_tracker::{Track, VehicleZone};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info};

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
}

impl Default for PassDetectorConfig {
    fn default() -> Self {
        Self {
            min_track_age_ms: 1000.0,      // Track must exist ≥1s
            min_beside_duration_ms: 500.0, // Beside phase ≥0.5s
            max_pass_duration_ms: 30000.0, // Full sequence ≤30s
            min_beside_frames: 8,
            disappearance_grace_frames: 30, // 1s grace
            min_avg_confidence: 0.25,
        }
    }
}

// ============================================================================
// TYPES
// ============================================================================

/// Direction of the pass from the ego vehicle's perspective
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassDirection {
    /// Ego overtook the vehicle (vehicle went AHEAD → BESIDE → BEHIND)
    EgoOvertook,
    /// Vehicle overtook ego (vehicle appeared BESIDE → moved AHEAD)
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

/// Which side the overtake happened on (from ego's perspective)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassSide {
    Left,  // Ego passed on the left (vehicle was on right during BESIDE)
    Right, // Ego passed on the right
}

impl PassSide {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Left => "LEFT",
            Self::Right => "RIGHT",
        }
    }
}

/// A completed pass event
#[derive(Debug, Clone)]
pub struct PassEvent {
    /// ID of the tracked vehicle that was passed
    pub vehicle_track_id: u32,
    /// Direction of the pass
    pub direction: PassDirection,
    /// Which side the pass occurred on
    pub side: PassSide,
    /// Timestamp when the vehicle first entered AHEAD zone (or was first seen)
    pub ahead_start_ms: f64,
    /// Timestamp when the vehicle entered BESIDE zone
    pub beside_start_ms: f64,
    /// Timestamp when the vehicle exited BESIDE (entered BEHIND or disappeared)
    pub beside_end_ms: f64,
    /// Frame ID at completion
    pub frame_id: u64,
    /// Total duration of the pass sequence
    pub duration_ms: f64,
    /// Confidence in this detection [0, 1]
    pub confidence: f32,
    /// YOLO class ID of the passed vehicle
    pub vehicle_class_id: u32,
}

/// Internal state for tracking a potential pass in progress
#[derive(Debug, Clone)]
struct PendingPass {
    track_id: u32,
    /// When did we first see this vehicle AHEAD?
    ahead_since_ms: Option<f64>,
    ahead_since_frame: Option<u64>,
    /// When did it enter BESIDE?
    beside_since_ms: Option<f64>,
    beside_since_frame: Option<u64>,
    /// Which side is it BESIDE on?
    beside_side: Option<PassSide>,
    /// How many consecutive BESIDE frames?
    beside_consecutive: u32,
    /// Peak area ratio while beside (proxy for how close it got)
    peak_beside_area: f32,
    /// Has the AHEAD phase been confirmed?
    had_ahead: bool,
    /// Has the BESIDE phase been confirmed (enough consecutive frames)?
    beside_confirmed: bool,
    /// Frames since the vehicle was last in BESIDE (for disappearance detection)
    frames_since_beside: u32,
    /// Was the vehicle ever in BESIDE zone? (even if not yet confirmed)
    ever_beside: bool,
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
        }
    }
}

// ============================================================================
// PASS DETECTOR
// ============================================================================

pub struct PassDetector {
    config: PassDetectorConfig,
    /// Pending passes indexed by track_id
    pending: HashMap<u32, PendingPass>,
    /// Track IDs that have already been reported as passed (avoid duplicates)
    completed_ids: HashSet<u32>,
    /// Recently emitted events (for the fusion layer to consume)
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

    /// Process current frame's confirmed tracks and emit any completed pass events.
    /// Call this every frame after VehicleTracker::update().
    pub fn update(&mut self, tracks: &[Track], timestamp_ms: f64, frame_id: u64) -> Vec<PassEvent> {
        self.recent_events.clear();

        let active_ids: HashSet<u32> = tracks.iter().map(|t| t.id).collect();

        // Collect tracks that need pass completion checking
        let mut tracks_to_check_behind: Vec<(u32, Track)> = Vec::new();
        let mut tracks_to_check_overtook_ego: Vec<(u32, Track)> = Vec::new();

        // Update pending passes for each active confirmed track
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
                    // If it was beside and now back to ahead, reset beside
                    if pending.ever_beside && !pending.beside_confirmed {
                        pending.beside_consecutive = 0;
                    }
                    pending.frames_since_beside += 1;

                    // Check if vehicle overtook us
                    if pending.beside_confirmed && !pending.had_ahead {
                        tracks_to_check_overtook_ego.push((track.id, track.clone()));
                    }
                }

                VehicleZone::BesideLeft | VehicleZone::BesideRight => {
                    let side = if track.zone == VehicleZone::BesideLeft {
                        // Vehicle is on our left → we passed on the right
                        PassSide::Right
                    } else {
                        // Vehicle is on our right → we passed on the left
                        PassSide::Left
                    };

                    if !pending.ever_beside {
                        pending.beside_since_ms = Some(timestamp_ms);
                        pending.beside_since_frame = Some(frame_id);
                        pending.beside_side = Some(side);
                        pending.ever_beside = true;
                        info!("📍 Track {} entered BESIDE ({:?}) zone", track.id, side);
                    }

                    pending.beside_consecutive += 1;
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
                    // Mark for checking if pass is complete
                    if pending.beside_confirmed {
                        tracks_to_check_behind.push((track.id, track.clone()));
                    }
                    pending.frames_since_beside += 1;
                }

                VehicleZone::Unknown => {
                    pending.frames_since_beside += 1;
                }
            }
        }

        // Now process tracks that moved to BEHIND (borrow of pending is released)
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

        // Process tracks that overtook ego
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

        // Check for vehicles that disappeared while in BESIDE (passed and gone)
        let disappeared: Vec<u32> = self
            .pending
            .keys()
            .copied()
            .filter(|id| !active_ids.contains(id) && !self.completed_ids.contains(id))
            .collect();

        for track_id in disappeared {
            if let Some(pending) = self.pending.get(&track_id) {
                if pending.beside_confirmed {
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
                        vehicle_class_id: 0, // Unknown — track is gone
                    };

                    if event.duration_ms <= self.config.max_pass_duration_ms {
                        info!(
                            "✅ PASS (disappeared): Track {} {} via {} | dur={:.1}s | conf={:.2}",
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
        }

        // Prune stale pending entries
        self.pending.retain(|id, p| {
            if self.completed_ids.contains(id) {
                return false;
            }
            // Remove if beside was never reached and track is very old
            if !p.ever_beside && p.had_ahead {
                if let Some(ahead_ms) = p.ahead_since_ms {
                    if timestamp_ms - ahead_ms > self.config.max_pass_duration_ms {
                        return false;
                    }
                }
            }
            // Remove if disappeared too long ago
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
        let beside_start = pending.beside_since_ms?;
        let beside_duration = timestamp_ms - beside_start;

        if beside_duration < self.config.min_beside_duration_ms {
            debug!(
                "❌ Pass rejected: beside duration {:.0}ms < {:.0}ms min",
                beside_duration, self.config.min_beside_duration_ms
            );
            return None;
        }

        let total_duration = timestamp_ms - pending.ahead_since_ms.unwrap_or(beside_start);
        if total_duration > self.config.max_pass_duration_ms {
            debug!(
                "❌ Pass rejected: total duration {:.0}ms > {:.0}ms max",
                total_duration, self.config.max_pass_duration_ms
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

        // Had a clear AHEAD phase → more confident
        if pending.had_ahead {
            conf += 0.15;
        }

        // Longer beside duration → more confident (up to +0.15)
        let beside_bonus = (beside_duration_ms / 3000.0).min(1.0) as f32 * 0.15;
        conf += beside_bonus;

        // Track age and stability
        if track.age > 30 {
            conf += 0.10;
        }

        // Detection confidence of the track
        if track.last_confidence > 0.6 {
            conf += 0.05;
        }

        // Reasonable total duration (3-15s is typical)
        if total_duration_ms >= 3000.0 && total_duration_ms <= 15000.0 {
            conf += 0.05;
        }

        conf.min(0.98)
    }

    fn calculate_confidence_disappeared(&self, pending: &PendingPass) -> f32 {
        let mut conf: f32 = 0.45; // Lower base — we lost the track

        if pending.had_ahead {
            conf += 0.15;
        }
        if pending.beside_consecutive > 15 {
            conf += 0.10;
        }

        conf.min(0.85) // Cap lower than normal since track disappeared
    }

    /// Get the most recent pass events (from last update call)
    pub fn recent_events(&self) -> &[PassEvent] {
        &self.recent_events
    }

    /// Check if any vehicle is currently in BESIDE zone (pass in progress)
    pub fn has_active_pass(&self) -> bool {
        self.pending
            .values()
            .any(|p| p.beside_confirmed && p.frames_since_beside < 10)
    }

    /// How many vehicles have been passed so far
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

    #[test]
    fn test_full_overtake_sequence() {
        let mut tracker = VehicleTracker::new(TrackerConfig::default(), 1280.0, 720.0);
        let mut pass_det = PassDetector::new(PassDetectorConfig::default());

        let mut events = Vec::new();
        let fps = 30.0;

        // Phase 1 (0-2s): Vehicle AHEAD — small, centered, upper frame
        for i in 0..60 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det(580.0, 180.0, 700.0, 280.0)];
            let tracks = tracker.update(&dets, t, i);
            let e = pass_det.update(tracks, t, i);
            events.extend(e);
        }
        assert!(events.is_empty(), "No pass should be detected yet");

        // Phase 2 (2-5s): Vehicle BESIDE RIGHT — large, offset right
        for i in 60..150 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det(880.0, 300.0, 1250.0, 680.0)];
            let tracks = tracker.update(&dets, t, i);
            let e = pass_det.update(tracks, t, i);
            events.extend(e);
        }
        assert!(events.is_empty(), "Pass not complete yet (still beside)");

        // Phase 3 (5-6s): Vehicle BEHIND — very large at bottom
        for i in 150..180 {
            let t = i as f64 * (1000.0 / fps);
            let dets = vec![det(200.0, 550.0, 1080.0, 720.0)];
            let tracks = tracker.update(&dets, t, i);
            let e = pass_det.update(tracks, t, i);
            events.extend(e);
        }

        assert!(!events.is_empty(), "Pass should be detected");
        assert_eq!(events[0].direction, PassDirection::EgoOvertook);
        assert_eq!(events[0].side, PassSide::Left); // Vehicle was on right = ego passed on left
    }
}
