// src/analysis/vehicle_tracker.rs
//
// IoU-based multi-object tracker for dashcam vehicle tracking.
// Assigns spatial zones (AHEAD / BESIDE / BEHIND) relative to the ego vehicle
// and tracks zone transitions over time for overtake detection.
//
// Design:
//   - Greedy IoU matching (sufficient for <20 objects per frame)
//   - Tracks coast through brief detection gaps (dust, glare)
//   - Zone assignment from bbox geometry — no lane markings needed
//   - Each track stores zone history for transition analysis
//
// v4.3 FIX: Zone hysteresis to prevent large AHEAD vehicles from
//           flickering into BESIDE due to bbox jitter or camera offset.
// v4.5 FIX: Hybrid matching (IoU + centroid distance) to maintain track
//           continuity during extreme geometric transformations (overtakes).

use std::collections::VecDeque;
use tracing::{debug, info, warn};

// ============================================================================
// CONFIGURATION
// ============================================================================

#[derive(Debug, Clone)]
pub struct TrackerConfig {
    /// Minimum IoU to match a detection to an existing track
    pub min_iou: f32,
    /// Frames a track survives without a detection before deletion
    pub max_coast_frames: u32,
    /// Consecutive hits required to promote Tentative → Confirmed
    pub min_hits_to_confirm: u32,
    /// Minimum detection confidence to accept
    pub min_confidence: f32,
    /// YOLO class IDs treated as vehicles (COCO: 2=car, 3=motorcycle, 5=bus, 7=truck)
    pub vehicle_class_ids: Vec<u32>,
    /// Zone geometry — ratio thresholds relative to frame dimensions
    pub zone: ZoneConfig,
    /// Maximum centroid distance (as fraction of frame width) for fallback matching
    pub max_centroid_distance_ratio: f32,
    /// Maximum frames since last hit for centroid fallback to apply
    pub centroid_fallback_max_coast: u32,
}

#[derive(Debug, Clone)]
pub struct ZoneConfig {
    /// Max Y ratio (center_y / frame_h) for AHEAD zone
    pub ahead_y_max: f32,
    /// Min lateral offset ratio (|cx - center_x| / half_width) for BESIDE
    pub beside_lateral_min: f32,
    /// Min area ratio (bbox_area / frame_area) to reinforce BESIDE
    pub beside_area_min: f32,
    /// Min bottom-Y ratio (bbox_y2 / frame_h) for BEHIND
    pub behind_bottom_y_min: f32,
    /// Min area ratio for BEHIND (very large = right behind us)
    pub behind_area_min: f32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            min_iou: 0.12,
            max_coast_frames: 45, // 1.5s at 30fps — survives dust bursts
            min_hits_to_confirm: 3,
            min_confidence: 0.20,
            vehicle_class_ids: vec![2, 3, 5, 7],
            zone: ZoneConfig::default(),
            max_centroid_distance_ratio: 0.20, // 20% of frame width (generous for mining)
            centroid_fallback_max_coast: 10,   // Only rescue recently-lost tracks
        }
    }
}

impl Default for ZoneConfig {
    fn default() -> Self {
        Self {
            ahead_y_max: 0.60,
            beside_lateral_min: 0.22,
            beside_area_min: 0.015,
            behind_bottom_y_min: 0.88,
            behind_area_min: 0.06,
        }
    }
}

// ============================================================================
// TYPES
// ============================================================================

/// Input detection — adapt from your YOLO output
#[derive(Debug, Clone)]
pub struct DetectionInput {
    pub bbox: [f32; 4], // [x1, y1, x2, y2] pixels
    pub class_id: u32,
    pub confidence: f32,
}

impl DetectionInput {
    pub fn center(&self) -> (f32, f32) {
        (
            (self.bbox[0] + self.bbox[2]) * 0.5,
            (self.bbox[1] + self.bbox[3]) * 0.5,
        )
    }
    pub fn area(&self) -> f32 {
        (self.bbox[2] - self.bbox[0]).max(0.0) * (self.bbox[3] - self.bbox[1]).max(0.0)
    }
    pub fn bottom_y(&self) -> f32 {
        self.bbox[3]
    }
}

/// Spatial zone relative to the ego vehicle's forward-facing camera
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VehicleZone {
    Ahead,
    BesideLeft,
    BesideRight,
    Behind,
    Unknown,
}

impl VehicleZone {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ahead => "AHEAD",
            Self::BesideLeft => "BESIDE_L",
            Self::BesideRight => "BESIDE_R",
            Self::Behind => "BEHIND",
            Self::Unknown => "UNKNOWN",
        }
    }
    pub fn is_beside(&self) -> bool {
        matches!(self, Self::BesideLeft | Self::BesideRight)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    Tentative,
    Confirmed,
    Lost,
}

#[derive(Debug, Clone, Copy)]
pub struct ZoneObservation {
    pub zone: VehicleZone,
    pub timestamp_ms: f64,
    pub frame_id: u64,
}

/// A single tracked vehicle with zone history
#[derive(Debug, Clone)]
pub struct Track {
    pub id: u32,
    pub bbox: [f32; 4],
    pub state: TrackState,
    pub zone: VehicleZone,
    pub zone_history: VecDeque<ZoneObservation>,
    pub class_id: u32,
    pub consecutive_hits: u32,
    pub age: u32,
    pub frames_since_hit: u32,
    pub peak_area: f32,
    pub initial_area: f32,
    pub initial_cx: f32,
    pub last_confidence: f32,
    prev_cx: f32,
    prev_cy: f32,

    // ── v4.3 FIX: Zone hysteresis ──
    /// Consecutive frames the raw zone computation returned BESIDE
    /// while the confirmed zone is still AHEAD.
    consecutive_beside_raw: u32,
    /// The last zone that was actually committed (after hysteresis).
    /// Used to detect AHEAD→BESIDE transitions that need confirmation.
    last_stable_zone: VehicleZone,
}

/// Minimum consecutive raw-BESIDE frames before we allow transition from AHEAD.
/// At 30fps this means ~170ms — enough to filter out single-frame bbox jitter
/// but fast enough to catch real beside events within 200ms.
const BESIDE_HYSTERESIS_FRAMES: u32 = 5;

impl Track {
    fn new(id: u32, det: &DetectionInput, timestamp_ms: f64, frame_id: u64) -> Self {
        let (cx, cy) = det.center();
        let area = det.area();
        Self {
            id,
            bbox: det.bbox,
            state: TrackState::Tentative,
            zone: VehicleZone::Unknown,
            zone_history: VecDeque::with_capacity(300), // 10s at 30fps
            class_id: det.class_id,
            consecutive_hits: 1,
            age: 1,
            frames_since_hit: 0,
            peak_area: area,
            initial_area: area,
            initial_cx: cx,
            last_confidence: det.confidence,
            prev_cx: cx,
            prev_cy: cy,
            consecutive_beside_raw: 0,
            last_stable_zone: VehicleZone::Unknown,
        }
    }

    pub fn center(&self) -> (f32, f32) {
        (
            (self.bbox[0] + self.bbox[2]) * 0.5,
            (self.bbox[1] + self.bbox[3]) * 0.5,
        )
    }

    pub fn area(&self) -> f32 {
        (self.bbox[2] - self.bbox[0]).max(0.0) * (self.bbox[3] - self.bbox[1]).max(0.0)
    }

    /// Lateral shift in pixels since first detection (positive = moved right)
    pub fn lateral_shift(&self) -> f32 {
        self.center().0 - self.initial_cx
    }

    /// Area growth ratio relative to initial observation (>1 = approaching)
    pub fn area_growth(&self) -> f32 {
        if self.initial_area > 1.0 {
            self.area() / self.initial_area
        } else {
            1.0
        }
    }

    pub fn is_confirmed(&self) -> bool {
        self.state == TrackState::Confirmed
    }

    /// Deduplicated zone transition sequence (collapses consecutive same-zone entries)
    pub fn zone_transitions(&self) -> Vec<VehicleZone> {
        let mut result: Vec<VehicleZone> = Vec::new();
        for obs in &self.zone_history {
            if result.last() != Some(&obs.zone) {
                result.push(obs.zone);
            }
        }
        result
    }

    /// Check if this track has completed a specific zone sequence
    /// within its entire history. Order matters, but non-matching zones
    /// between sequence elements are ignored (subsequence match).
    pub fn has_zone_sequence(&self, sequence: &[VehicleZone]) -> bool {
        let transitions = self.zone_transitions();
        if transitions.len() < sequence.len() {
            return false;
        }
        let mut seq_idx = 0;
        for zone in &transitions {
            if *zone == sequence[seq_idx] {
                seq_idx += 1;
                if seq_idx == sequence.len() {
                    return true;
                }
            }
        }
        false
    }

    /// Time span of zone history in milliseconds
    pub fn history_duration_ms(&self) -> f64 {
        if self.zone_history.len() < 2 {
            return 0.0;
        }
        let first = self.zone_history.front().unwrap().timestamp_ms;
        let last = self.zone_history.back().unwrap().timestamp_ms;
        last - first
    }

    fn update_with_detection(&mut self, det: &DetectionInput) {
        let (new_cx, new_cy) = det.center();
        self.prev_cx = self.center().0;
        self.prev_cy = self.center().1;
        self.bbox = det.bbox;
        self.class_id = det.class_id;
        self.last_confidence = det.confidence;
        self.consecutive_hits += 1;
        self.frames_since_hit = 0;
        self.age += 1;

        let area = det.area();
        if area > self.peak_area {
            self.peak_area = area;
        }

        if self.state == TrackState::Tentative && self.consecutive_hits >= 3 {
            self.state = TrackState::Confirmed;
        }
        if self.state == TrackState::Lost {
            self.state = TrackState::Confirmed;
            self.consecutive_hits = 1;
        }
    }

    fn mark_missed(&mut self) {
        self.frames_since_hit += 1;
        self.consecutive_hits = 0;
        self.age += 1;
        if self.state == TrackState::Confirmed && self.frames_since_hit > 5 {
            self.state = TrackState::Lost;
        }
    }

    fn assign_zone(
        &mut self,
        frame_w: f32,
        frame_h: f32,
        cfg: &ZoneConfig,
        timestamp_ms: f64,
        frame_id: u64,
    ) {
        let (cx, cy) = self.center();
        let frame_area = frame_w * frame_h;
        let area_ratio = self.area() / frame_area;
        let half_w = frame_w * 0.5;
        let lateral_offset = (cx - half_w).abs() / half_w; // 0=center, 1=edge
        let y_ratio = cy / frame_h;
        let bottom_ratio = self.bbox[3] / frame_h;

        let bbox_height = self.bbox[3] - self.bbox[1];
        let height_ratio = bbox_height / frame_h;

        // ══════════════════════════════════════════════════════════════════════
        // SIZE-BASED OVERRIDE: Large vehicles are AHEAD
        //
        // v4.3 FIX: Lowered thresholds from 0.35/0.20 to 0.28/0.12 to catch
        // mining trucks that vary frame-to-frame and whose bboxes can be
        // slightly below the old thresholds while clearly being ahead.
        // ══════════════════════════════════════════════════════════════════════
        let raw_zone = if height_ratio > 0.28 || area_ratio > 0.12 {
            VehicleZone::Ahead
        }
        // ══════════════════════════════════════════════════════════════════════
        // BEHIND: Very large bbox at bottom = just passed us or tailgating
        // ══════════════════════════════════════════════════════════════════════
        else if bottom_ratio > cfg.behind_bottom_y_min && area_ratio > cfg.behind_area_min {
            VehicleZone::Behind
        }
        // ══════════════════════════════════════════════════════════════════════
        // BESIDE: Laterally offset + (medium area OR lower in frame)
        // ══════════════════════════════════════════════════════════════════════
        else if lateral_offset > cfg.beside_lateral_min
            && (area_ratio > cfg.beside_area_min || bottom_ratio > 0.55)
            && height_ratio < 0.25
        // v4.3: tightened from 0.30
        {
            if cx < half_w {
                VehicleZone::BesideLeft
            } else {
                VehicleZone::BesideRight
            }
        }
        // ══════════════════════════════════════════════════════════════════════
        // AHEAD: Upper frame, roughly centered
        // ══════════════════════════════════════════════════════════════════════
        else if y_ratio < cfg.ahead_y_max && lateral_offset < 0.45 {
            VehicleZone::Ahead
        }
        // ══════════════════════════════════════════════════════════════════════
        // TRANSITIONAL: Lower half but not clearly beside/behind
        // ══════════════════════════════════════════════════════════════════════
        else if y_ratio >= cfg.ahead_y_max {
            if lateral_offset > cfg.beside_lateral_min * 1.2 && height_ratio < 0.22 {
                if cx < half_w {
                    VehicleZone::BesideLeft
                } else {
                    VehicleZone::BesideRight
                }
            } else {
                VehicleZone::Ahead
            }
        } else {
            VehicleZone::Unknown
        };

        // ══════════════════════════════════════════════════════════════════════
        // v4.3 FIX: ZONE HYSTERESIS
        //
        // Prevent AHEAD→BESIDE flickering. When a vehicle has been stably
        // AHEAD, require BESIDE_HYSTERESIS_FRAMES consecutive raw-BESIDE
        // readings before committing the transition. This filters out
        // single-frame bbox jitter from lateral camera offset or dust.
        //
        // BEHIND and AHEAD transitions are NOT gated — they're less
        // prone to false positives and need to be responsive.
        // ══════════════════════════════════════════════════════════════════════
        let zone = if self.last_stable_zone == VehicleZone::Ahead && raw_zone.is_beside() {
            self.consecutive_beside_raw += 1;
            if self.consecutive_beside_raw >= BESIDE_HYSTERESIS_FRAMES {
                // Confirmed transition: vehicle genuinely moved to BESIDE
                debug!(
                    "✅ Track {} AHEAD→{} confirmed after {} frames",
                    self.id,
                    raw_zone.as_str(),
                    self.consecutive_beside_raw
                );
                self.consecutive_beside_raw = 0;
                raw_zone
            } else {
                debug!(
                    "⏳ Track {} raw={} but holding AHEAD ({}/{} hysteresis)",
                    self.id,
                    raw_zone.as_str(),
                    self.consecutive_beside_raw,
                    BESIDE_HYSTERESIS_FRAMES
                );
                VehicleZone::Ahead // Hold at AHEAD during hysteresis
            }
        } else {
            // Not an AHEAD→BESIDE transition — apply raw zone immediately
            if !raw_zone.is_beside() {
                self.consecutive_beside_raw = 0;
            }
            raw_zone
        };

        // Update stable zone tracking
        self.last_stable_zone = zone;
        self.zone = zone;

        self.zone_history.push_back(ZoneObservation {
            zone,
            timestamp_ms,
            frame_id,
        });

        // Cap history at 10s
        while self.zone_history.len() > 300 {
            self.zone_history.pop_front();
        }
    }
}

// ============================================================================
// IoU COMPUTATION
// ============================================================================

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);

    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    if inter <= 0.0 {
        return 0.0;
    }

    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;

    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

// ============================================================================
// MAIN TRACKER
// ============================================================================

pub struct VehicleTracker {
    pub config: TrackerConfig,
    tracks: Vec<Track>,
    next_id: u32,
    frame_w: f32,
    frame_h: f32,
}

impl VehicleTracker {
    pub fn new(config: TrackerConfig, frame_w: f32, frame_h: f32) -> Self {
        Self {
            config,
            tracks: Vec::with_capacity(32),
            next_id: 1,
            frame_w,
            frame_h,
        }
    }

    /// Process one frame of detections. Returns the current set of confirmed tracks.
    pub fn update(
        &mut self,
        detections: &[DetectionInput],
        timestamp_ms: f64,
        frame_id: u64,
    ) -> &[Track] {
        // Filter to vehicle classes with sufficient confidence
        let valid: Vec<&DetectionInput> = detections
            .iter()
            .filter(|d| {
                d.confidence >= self.config.min_confidence
                    && self.config.vehicle_class_ids.contains(&d.class_id)
            })
            .collect();

        // ══════════════════════════════════════════════════════════════════════
        // PHASE 1: IoU-BASED MATCHING (PRIMARY)
        // ══════════════════════════════════════════════════════════════════════
        let mut matched_track_indices: Vec<bool> = vec![false; self.tracks.len()];
        let mut matched_det_indices: Vec<bool> = vec![false; valid.len()];

        let mut iou_pairs: Vec<(usize, usize, f32)> = Vec::new();
        for (ti, track) in self.tracks.iter().enumerate() {
            for (di, det) in valid.iter().enumerate() {
                let score = iou(&track.bbox, &det.bbox);
                if score >= self.config.min_iou {
                    iou_pairs.push((ti, di, score));
                }
            }
        }
        iou_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        for (ti, di, _score) in &iou_pairs {
            if matched_track_indices[*ti] || matched_det_indices[*di] {
                continue;
            }
            matched_track_indices[*ti] = true;
            matched_det_indices[*di] = true;
            self.tracks[*ti].update_with_detection(valid[*di]);
        }

        // ══════════════════════════════════════════════════════════════════════
        // PHASE 2: CENTROID-DISTANCE FALLBACK (v4.5 FIX)
        //
        // When IoU fails due to extreme geometric transformations (e.g., mining
        // truck bbox growing 6x as it approaches), use spatial proximity as a
        // fallback for confirmed/recently-lost tracks.
        //
        // This prevents track churn during overtakes where the same vehicle is
        // continuously detected but IoU drops below threshold, causing the old
        // track to die and new tentative tracks to spawn repeatedly.
        // ══════════════════════════════════════════════════════════════════════
        let max_dist_px = self.frame_w * self.config.max_centroid_distance_ratio;
        let max_dist_sq = max_dist_px * max_dist_px;

        let mut centroid_pairs: Vec<(usize, usize, f32)> = Vec::new();
        for (ti, track) in self.tracks.iter().enumerate() {
            if matched_track_indices[ti] {
                continue; // Already matched via IoU
            }

            // Only rescue confirmed or recently-lost tracks (not tentative)
            // Tentative tracks should naturally expire if IoU fails — they haven't
            // proven themselves yet
            if track.state == TrackState::Tentative {
                continue;
            }

            // Only rescue tracks lost in the last N frames (prevent stale matches)
            if track.frames_since_hit > self.config.centroid_fallback_max_coast {
                continue;
            }

            let (tcx, tcy) = track.center();

            for (di, det) in valid.iter().enumerate() {
                if matched_det_indices[di] {
                    continue; // Detection already matched
                }

                // Class ID must match (car can't become truck)
                if track.class_id != det.class_id {
                    continue;
                }

                let (dcx, dcy) = det.center();
                let dist_sq = (tcx - dcx).powi(2) + (tcy - dcy).powi(2);

                if dist_sq < max_dist_sq {
                    centroid_pairs.push((ti, di, dist_sq));
                }
            }
        }

        // Sort by distance (nearest first)
        centroid_pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Greedy matching: assign nearest pairs first
        for (ti, di, dist_sq) in &centroid_pairs {
            if matched_track_indices[*ti] || matched_det_indices[*di] {
                continue;
            }
            matched_track_indices[*ti] = true;
            matched_det_indices[*di] = true;

            debug!(
                "🔗 Centroid rescue: Track {} ↔ det (dist={:.0}px, class={}, IoU was below threshold)",
                self.tracks[*ti].id,
                dist_sq.sqrt(),
                self.tracks[*ti].class_id
            );

            self.tracks[*ti].update_with_detection(valid[*di]);
        }

        // ══════════════════════════════════════════════════════════════════════
        // UNMATCHED TRACKS → COAST
        // ══════════════════════════════════════════════════════════════════════
        for (ti, matched) in matched_track_indices.iter().enumerate() {
            if !matched {
                self.tracks[ti].mark_missed();
            }
        }

        // ══════════════════════════════════════════════════════════════════════
        // UNMATCHED DETECTIONS → NEW TRACKS
        // ══════════════════════════════════════════════════════════════════════
        for (di, matched) in matched_det_indices.iter().enumerate() {
            if !matched {
                let track = Track::new(self.next_id, valid[di], timestamp_ms, frame_id);
                debug!(
                    "🆕 New track T{} created: class={}, bbox=[{:.0},{:.0},{:.0},{:.0}]",
                    self.next_id,
                    track.class_id,
                    track.bbox[0],
                    track.bbox[1],
                    track.bbox[2],
                    track.bbox[3]
                );
                self.next_id += 1;
                self.tracks.push(track);
            }
        }

        // ══════════════════════════════════════════════════════════════════════
        // ZONE ASSIGNMENT
        // ══════════════════════════════════════════════════════════════════════
        let zone_cfg = self.config.zone.clone();
        let (fw, fh) = (self.frame_w, self.frame_h);
        for track in &mut self.tracks {
            if track.frames_since_hit == 0 {
                track.assign_zone(fw, fh, &zone_cfg, timestamp_ms, frame_id);
            }
        }

        // ══════════════════════════════════════════════════════════════════════
        // PRUNE DEAD TRACKS
        // ══════════════════════════════════════════════════════════════════════
        let max_coast = self.config.max_coast_frames;
        let min_hits = self.config.min_hits_to_confirm;
        self.tracks.retain(|t| {
            if t.frames_since_hit > max_coast {
                debug!(
                    "🗑️  Track {} pruned (coasted {} frames)",
                    t.id, t.frames_since_hit
                );
                return false;
            }
            if t.state == TrackState::Tentative && t.age > min_hits * 3 {
                debug!(
                    "🗑️  Track {} pruned (tentative too long: age={})",
                    t.id, t.age
                );
                return false;
            }
            true
        });

        &self.tracks
    }

    pub fn confirmed_tracks(&self) -> Vec<&Track> {
        self.tracks.iter().filter(|t| t.is_confirmed()).collect()
    }

    pub fn all_tracks(&self) -> &[Track] {
        &self.tracks
    }

    pub fn get_track(&self, id: u32) -> Option<&Track> {
        self.tracks.iter().find(|t| t.id == id)
    }

    pub fn confirmed_count(&self) -> usize {
        self.tracks.iter().filter(|t| t.is_confirmed()).count()
    }

    pub fn reset(&mut self) {
        self.tracks.clear();
        self.next_id = 1;
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x1: f32, y1: f32, x2: f32, y2: f32) -> DetectionInput {
        DetectionInput {
            bbox: [x1, y1, x2, y2],
            class_id: 7, // truck
            confidence: 0.8,
        }
    }

    #[test]
    fn test_iou_overlap() {
        let a = [0.0, 0.0, 100.0, 100.0];
        let b = [50.0, 50.0, 150.0, 150.0];
        let score = iou(&a, &b);
        assert!((score - 2500.0 / 17500.0).abs() < 0.01);
    }

    #[test]
    fn test_iou_no_overlap() {
        let a = [0.0, 0.0, 50.0, 50.0];
        let b = [100.0, 100.0, 200.0, 200.0];
        assert_eq!(iou(&a, &b), 0.0);
    }

    #[test]
    fn test_track_creation_and_confirmation() {
        let cfg = TrackerConfig::default();
        let mut tracker = VehicleTracker::new(cfg, 1280.0, 720.0);

        let dets = vec![det(500.0, 200.0, 600.0, 300.0)];
        tracker.update(&dets, 0.0, 1);
        assert_eq!(tracker.all_tracks().len(), 1);
        assert_eq!(tracker.all_tracks()[0].state, TrackState::Tentative);

        tracker.update(&dets, 33.3, 2);
        tracker.update(&dets, 66.6, 3);
        assert_eq!(tracker.all_tracks()[0].state, TrackState::Confirmed);
    }

    #[test]
    fn test_centroid_fallback_rescues_track() {
        // v4.5: Track should survive extreme bbox transformation via centroid fallback
        let cfg = TrackerConfig::default();
        let mut tracker = VehicleTracker::new(cfg, 1280.0, 720.0);

        // Frame 1-3: Normal progression, track confirms
        for i in 1..=3 {
            let dets = vec![det(500.0, 200.0, 600.0, 300.0)];
            tracker.update(&dets, i as f64 * 33.3, i);
        }
        assert_eq!(tracker.confirmed_count(), 1);
        let track_id = tracker.all_tracks()[0].id;

        // Frame 4: Extreme bbox change (simulating close approach)
        // IoU will be very low, but centroid is close (~100px)
        let dets = vec![det(550.0, 250.0, 1100.0, 600.0)];
        tracker.update(&dets, 4.0 * 33.3, 4);

        // Track should still exist with same ID (rescued by centroid fallback)
        assert_eq!(
            tracker.confirmed_count(),
            1,
            "Track should survive via centroid fallback"
        );
        assert_eq!(
            tracker.all_tracks()[0].id,
            track_id,
            "Track ID should not change"
        );
    }

    #[test]
    fn test_zone_assignment_ahead() {
        let cfg = TrackerConfig::default();
        let mut tracker = VehicleTracker::new(cfg, 1280.0, 720.0);

        let dets = vec![det(580.0, 200.0, 700.0, 300.0)];
        for i in 0..4 {
            tracker.update(&dets, i as f64 * 33.3, i as u64);
        }
        assert_eq!(tracker.all_tracks()[0].zone, VehicleZone::Ahead);
    }

    #[test]
    fn test_zone_hysteresis_prevents_flicker() {
        let cfg = TrackerConfig::default();
        let mut tracker = VehicleTracker::new(cfg, 1280.0, 720.0);

        // Phase 1: clearly AHEAD (centered, upper)
        for i in 0..10 {
            let dets = vec![det(500.0, 200.0, 780.0, 400.0)];
            tracker.update(&dets, i as f64 * 33.3, i as u64);
        }
        assert_eq!(tracker.all_tracks()[0].zone, VehicleZone::Ahead);

        // Phase 2: bbox shifts slightly right (camera jitter) for 3 frames
        // This should NOT trigger BESIDE due to hysteresis
        for i in 10..13 {
            let dets = vec![det(700.0, 300.0, 1100.0, 550.0)];
            tracker.update(&dets, i as f64 * 33.3, i as u64);
        }
        // Should still be AHEAD (hysteresis requires 5 frames)
        assert_eq!(
            tracker.all_tracks()[0].zone,
            VehicleZone::Ahead,
            "Should remain AHEAD during hysteresis period"
        );
    }

    #[test]
    fn test_zone_assignment_beside_right_after_hysteresis() {
        let cfg = TrackerConfig::default();
        let mut tracker = VehicleTracker::new(cfg, 1280.0, 720.0);

        // Phase 1: clearly AHEAD
        for i in 0..10 {
            let dets = vec![det(580.0, 200.0, 700.0, 300.0)];
            tracker.update(&dets, i as f64 * 33.3, i as u64);
        }

        // Phase 2: genuinely BESIDE for enough frames to pass hysteresis
        for i in 10..20 {
            let dets = vec![det(900.0, 300.0, 1250.0, 500.0)];
            tracker.update(&dets, i as f64 * 33.3, i as u64);
        }
        assert_eq!(tracker.all_tracks()[0].zone, VehicleZone::BesideRight);
    }

    #[test]
    fn test_zone_sequence_detection() {
        let cfg = TrackerConfig::default();
        let mut tracker = VehicleTracker::new(cfg, 1280.0, 720.0);

        // Phase 1: ahead
        for i in 0..10 {
            let dets = vec![det(580.0, 200.0, 700.0, 300.0)];
            tracker.update(&dets, i as f64 * 33.3, i as u64);
        }

        // Phase 2: beside right (long enough for hysteresis)
        for i in 10..25 {
            let dets = vec![det(900.0, 350.0, 1250.0, 500.0)];
            tracker.update(&dets, i as f64 * 33.3, i as u64);
        }

        let track = &tracker.all_tracks()[0];
        assert!(track.has_zone_sequence(&[VehicleZone::Ahead, VehicleZone::BesideRight]));
    }

    #[test]
    fn test_large_truck_stays_ahead() {
        let cfg = TrackerConfig::default();
        let mut tracker = VehicleTracker::new(cfg, 1280.0, 720.0);

        // Truck: 800x216 bbox (height_ratio = 216/720 = 0.30, area_ratio = 0.187)
        let dets = vec![det(300.0, 250.0, 1100.0, 466.0)];
        for i in 0..10 {
            tracker.update(&dets, i as f64 * 33.3, i as u64);
        }
        assert_eq!(
            tracker.all_tracks()[0].zone,
            VehicleZone::Ahead,
            "Large truck with 30% height ratio should be AHEAD"
        );
    }
}
