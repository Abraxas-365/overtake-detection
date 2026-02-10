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
}

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

        let zone = if lateral_offset > cfg.beside_lateral_min
            && (area_ratio > cfg.beside_area_min || bottom_ratio > 0.55)
        {
            // Large or low bbox that's laterally offset = BESIDE
            if cx < half_w {
                VehicleZone::BesideLeft
            } else {
                VehicleZone::BesideRight
            }
        } else if bottom_ratio > cfg.behind_bottom_y_min && area_ratio > cfg.behind_area_min {
            // Very large bbox filling the bottom of frame = BEHIND (tailgating or just passed)
            VehicleZone::Behind
        } else if y_ratio < cfg.ahead_y_max && lateral_offset < 0.40 {
            // Upper frame, roughly centered = AHEAD
            VehicleZone::Ahead
        } else if y_ratio >= cfg.ahead_y_max {
            // Lower half but not clearly beside or behind — transitional
            // Use lateral offset to disambiguate
            if lateral_offset > cfg.beside_lateral_min * 0.8 {
                if cx < half_w {
                    VehicleZone::BesideLeft
                } else {
                    VehicleZone::BesideRight
                }
            } else {
                VehicleZone::Ahead // Still ahead but closer
            }
        } else {
            VehicleZone::Unknown
        };

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
    config: TrackerConfig,
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

        // ── MATCHING ────────────────────────────────────────────
        // Greedy IoU matching: for each detection, find best matching track
        let mut matched_track_indices: Vec<bool> = vec![false; self.tracks.len()];
        let mut matched_det_indices: Vec<bool> = vec![false; valid.len()];

        // Build IoU matrix and sort by descending IoU for greedy assignment
        let mut pairs: Vec<(usize, usize, f32)> = Vec::new();
        for (ti, track) in self.tracks.iter().enumerate() {
            for (di, det) in valid.iter().enumerate() {
                let score = iou(&track.bbox, &det.bbox);
                if score >= self.config.min_iou {
                    pairs.push((ti, di, score));
                }
            }
        }
        pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        for (ti, di, _score) in &pairs {
            if matched_track_indices[*ti] || matched_det_indices[*di] {
                continue;
            }
            matched_track_indices[*ti] = true;
            matched_det_indices[*di] = true;

            self.tracks[*ti].update_with_detection(valid[*di]);
        }

        // ── UNMATCHED TRACKS → COAST ────────────────────────────
        for (ti, matched) in matched_track_indices.iter().enumerate() {
            if !matched {
                self.tracks[ti].mark_missed();
            }
        }

        // ── UNMATCHED DETECTIONS → NEW TRACKS ───────────────────
        for (di, matched) in matched_det_indices.iter().enumerate() {
            if !matched {
                let track = Track::new(self.next_id, valid[di], timestamp_ms, frame_id);
                self.next_id += 1;
                self.tracks.push(track);
            }
        }

        // ── ZONE ASSIGNMENT ─────────────────────────────────────
        let zone_cfg = self.config.zone.clone();
        let (fw, fh) = (self.frame_w, self.frame_h);
        for track in &mut self.tracks {
            if track.frames_since_hit == 0 {
                // Only update zone when we have a fresh detection
                track.assign_zone(fw, fh, &zone_cfg, timestamp_ms, frame_id);
            }
        }

        // ── PRUNE DEAD TRACKS ───────────────────────────────────
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
            // Remove tentative tracks that never confirmed
            if t.state == TrackState::Tentative && t.age > min_hits * 3 {
                return false;
            }
            true
        });

        &self.tracks
    }

    /// Get all currently confirmed tracks
    pub fn confirmed_tracks(&self) -> Vec<&Track> {
        self.tracks.iter().filter(|t| t.is_confirmed()).collect()
    }

    /// Get all tracks (including tentative/lost) — useful for debugging
    pub fn all_tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// Get a specific track by ID
    pub fn get_track(&self, id: u32) -> Option<&Track> {
        self.tracks.iter().find(|t| t.id == id)
    }

    /// Number of confirmed tracks
    pub fn confirmed_count(&self) -> usize {
        self.tracks.iter().filter(|t| t.is_confirmed()).count()
    }

    pub fn reset(&mut self) {
        self.tracks.clear();
        self.next_id = 1;
    }
}

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
        // Intersection = 50*50 = 2500, Union = 10000 + 10000 - 2500 = 17500
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

        // Frame 1: detection appears
        let dets = vec![det(500.0, 200.0, 600.0, 300.0)];
        tracker.update(&dets, 0.0, 1);
        assert_eq!(tracker.all_tracks().len(), 1);
        assert_eq!(tracker.all_tracks()[0].state, TrackState::Tentative);

        // Frame 2-3: same area, track confirms
        tracker.update(&dets, 33.3, 2);
        tracker.update(&dets, 66.6, 3);
        assert_eq!(tracker.all_tracks()[0].state, TrackState::Confirmed);
    }

    #[test]
    fn test_zone_assignment_ahead() {
        let cfg = TrackerConfig::default();
        let mut tracker = VehicleTracker::new(cfg, 1280.0, 720.0);

        // Small bbox in upper center = AHEAD
        let dets = vec![det(580.0, 200.0, 700.0, 300.0)];
        for i in 0..4 {
            tracker.update(&dets, i as f64 * 33.3, i as u64);
        }
        assert_eq!(tracker.all_tracks()[0].zone, VehicleZone::Ahead);
    }

    #[test]
    fn test_zone_assignment_beside_right() {
        let cfg = TrackerConfig::default();
        let mut tracker = VehicleTracker::new(cfg, 1280.0, 720.0);

        // Large bbox on right side = BESIDE_RIGHT
        let dets = vec![det(900.0, 300.0, 1250.0, 650.0)];
        for i in 0..4 {
            tracker.update(&dets, i as f64 * 33.3, i as u64);
        }
        assert_eq!(tracker.all_tracks()[0].zone, VehicleZone::BesideRight);
    }

    #[test]
    fn test_zone_sequence_detection() {
        let cfg = TrackerConfig::default();
        let mut tracker = VehicleTracker::new(cfg, 1280.0, 720.0);

        // Simulate a vehicle moving from AHEAD → BESIDE_RIGHT
        // Phase 1: ahead (small, centered, upper)
        for i in 0..10 {
            let dets = vec![det(580.0, 200.0, 700.0, 300.0)];
            tracker.update(&dets, i as f64 * 33.3, i as u64);
        }

        // Phase 2: beside right (large, offset right, lower)
        for i in 10..20 {
            let dets = vec![det(900.0, 350.0, 1250.0, 680.0)];
            tracker.update(&dets, i as f64 * 33.3, i as u64);
        }

        let track = &tracker.all_tracks()[0];
        assert!(track.has_zone_sequence(&[VehicleZone::Ahead, VehicleZone::BesideRight]));
    }
}
