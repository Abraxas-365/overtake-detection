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
// v4.8 FIX: Lateral-aware size override — large bboxes at frame edges are
//           BESIDE, not AHEAD. Class stability after track confirmation.
//           Tightened centroid rescue to prevent ghost associations.

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
    /// Maximum centroid distance (as fraction of frame width) for confirmed/lost tracks
    pub max_centroid_distance_ratio: f32,
    /// Maximum centroid distance (as fraction of frame width) for tentative tracks
    pub max_centroid_distance_ratio_tentative: f32,
    /// Maximum frames since last hit for centroid fallback to apply (confirmed/lost)
    pub centroid_fallback_max_coast: u32,
    /// Maximum frames since last hit for centroid fallback to apply (tentative)
    pub centroid_fallback_max_coast_tentative: u32,
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
    /// v4.8: Max lateral offset ratio (|cx - frame_center| / frame_w) for
    /// the size-based AHEAD override to apply. Vehicles farther from center
    /// than this are candidates for BESIDE even if large.
    pub size_override_max_lateral: f32,
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
            max_centroid_distance_ratio: 0.20, // 20% of frame width
            max_centroid_distance_ratio_tentative: 0.15, // 15% for tentative (stricter)
            centroid_fallback_max_coast: 10,
            centroid_fallback_max_coast_tentative: 5,
        }
    }
}

impl TrackerConfig {
    /// Mining-optimized configuration — more generous coasting and centroid
    /// rescue for dust/glare, but still strict enough to prevent ghost tracks.
    pub fn mining() -> Self {
        Self {
            min_iou: 0.12,
            max_coast_frames: 45,
            min_hits_to_confirm: 3,
            min_confidence: 0.20,
            vehicle_class_ids: vec![2, 3, 5, 7],
            zone: ZoneConfig::mining(),
            max_centroid_distance_ratio: 0.22, // Slightly generous for dust
            max_centroid_distance_ratio_tentative: 0.15,
            centroid_fallback_max_coast: 12,
            centroid_fallback_max_coast_tentative: 5,
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
            size_override_max_lateral: 0.25, // v4.8: only override when within 25% of center
        }
    }
}

impl ZoneConfig {
    pub fn mining() -> Self {
        Self {
            ahead_y_max: 0.60,
            beside_lateral_min: 0.22,
            beside_area_min: 0.015,
            behind_bottom_y_min: 0.88,
            behind_area_min: 0.06,
            size_override_max_lateral: 0.25,
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

    // ── v4.8 FIX: Class stability ──
    /// The class_id that was assigned when the track was confirmed.
    /// Once set, IoU and centroid matches with a different class_id
    /// are penalized (IoU) or rejected (centroid).
    confirmed_class_id: Option<u32>,
}

/// Minimum consecutive raw-BESIDE frames before we allow transition from AHEAD.
/// At 30fps this means ~170ms — enough to filter out single-frame bbox jitter
/// but fast enough to catch real beside events within 200ms.
const BESIDE_HYSTERESIS_FRAMES: u32 = 5;

/// v4.8: IoU penalty multiplier when detection class differs from confirmed
/// track class. 0.5 = halve the IoU score, making same-class matches win
/// the greedy assignment when both are available.
const CROSS_CLASS_IOU_PENALTY: f32 = 0.5;

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
            confirmed_class_id: None,
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
        self.prev_cx = self.center().0;
        self.prev_cy = self.center().1;
        self.bbox = det.bbox;
        self.last_confidence = det.confidence;
        self.consecutive_hits += 1;
        self.frames_since_hit = 0;
        self.age += 1;

        // ── v4.8 FIX: Class stability ──
        // Only update class_id for tentative tracks (identity not yet locked).
        // Once confirmed, keep the original class to prevent identity drift
        // where a track hops between car→bus→truck across frames.
        if self.confirmed_class_id.is_none() {
            self.class_id = det.class_id;
        }
        // else: keep self.class_id as confirmed_class_id

        let area = det.area();
        if area > self.peak_area {
            self.peak_area = area;
        }

        if self.state == TrackState::Tentative && self.consecutive_hits >= 3 {
            self.state = TrackState::Confirmed;
            self.confirmed_class_id = Some(self.class_id);
            debug!(
                "✅ Track {} confirmed with class={}",
                self.id, self.class_id
            );
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
        // v4.8 FIX: LATERAL-AWARE SIZE OVERRIDE
        //
        // The size-based AHEAD override was catching vehicles that are BESIDE
        // during overtakes. A vehicle being passed is close → large bbox, but
        // displaced to one side. We now only apply the size override when the
        // bbox center is roughly centered in the frame.
        //
        // lateral_offset_ratio: |cx - frame_center| / frame_width
        //   0.0 = perfectly centered, 0.5 = at frame edge
        //
        // Example from the bug:
        //   T6 bbox=[0,402,98,633] → cx=49, lateral_offset_ratio=0.46
        //   height_ratio=0.32 (exceeds 0.28) but laterally it's at the edge
        //   → should NOT be forced AHEAD, should fall through to BESIDE logic
        // ══════════════════════════════════════════════════════════════════════
        let lateral_offset_ratio = (cx - half_w).abs() / frame_w; // 0=center, 0.5=edge
        let is_laterally_centered = lateral_offset_ratio < cfg.size_override_max_lateral;

        let raw_zone = if is_laterally_centered && (height_ratio > 0.28 || area_ratio > 0.12) {
            // Large, centered vehicle — genuinely AHEAD
            VehicleZone::Ahead
        }
        // ══════════════════════════════════════════════════════════════════════
        // BEHIND: Very large bbox at bottom = just passed us or tailgating
        // ══════════════════════════════════════════════════════════════════════
        else if bottom_ratio > cfg.behind_bottom_y_min && area_ratio > cfg.behind_area_min {
            VehicleZone::Behind
        }
        // ══════════════════════════════════════════════════════════════════════
        // v4.8: BESIDE for large lateral vehicles (previously swallowed by
        // the size override). These are vehicles close to the ego vehicle
        // during an overtake — large bbox but displaced to one side.
        // ══════════════════════════════════════════════════════════════════════
        else if !is_laterally_centered
            && (height_ratio > 0.20 || area_ratio > 0.08)
            && lateral_offset > cfg.beside_lateral_min
        {
            if cx < half_w {
                VehicleZone::BesideLeft
            } else {
                VehicleZone::BesideRight
            }
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
        //
        // v4.8: Cross-class IoU penalty. When a confirmed track's locked
        // class_id differs from the detection's class_id, the effective IoU
        // is halved. This makes same-class matches win the greedy assignment
        // when both are available, while still allowing cross-class matches
        // as a last resort (YOLO can flicker between car/truck for the same
        // physical vehicle, especially with mining equipment).
        // ══════════════════════════════════════════════════════════════════════
        let mut matched_track_indices: Vec<bool> = vec![false; self.tracks.len()];
        let mut matched_det_indices: Vec<bool> = vec![false; valid.len()];

        let mut iou_pairs: Vec<(usize, usize, f32)> = Vec::new();
        for (ti, track) in self.tracks.iter().enumerate() {
            for (di, det) in valid.iter().enumerate() {
                let raw_iou = iou(&track.bbox, &det.bbox);
                if raw_iou < self.config.min_iou {
                    continue;
                }

                // v4.8: Penalize cross-class matches for confirmed tracks
                let effective_iou = if let Some(locked_class) = track.confirmed_class_id {
                    if locked_class != det.class_id {
                        raw_iou * CROSS_CLASS_IOU_PENALTY
                    } else {
                        raw_iou
                    }
                } else {
                    raw_iou
                };

                // Only include if the penalized IoU still meets the threshold
                if effective_iou >= self.config.min_iou {
                    iou_pairs.push((ti, di, effective_iou));
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
        // PHASE 2: CENTROID-DISTANCE FALLBACK
        //
        // v4.8: Tightened from v4.7's extremely generous thresholds.
        //   - Strict class matching: confirmed tracks ONLY match same class
        //   - Reduced distances: 20%/15% (was 30%/25%)
        //   - Separate coast limits for confirmed vs tentative
        //   - Prevents ghost associations that were corrupting track identity
        //
        // The v4.7 thresholds (30% = 384px) allowed matching completely
        // different vehicles at opposite sides of the frame. The new limits
        // are generous enough for dust/glare gaps but strict enough to
        // prevent cross-vehicle associations.
        // ══════════════════════════════════════════════════════════════════════
        let max_dist_confirmed_px = self.frame_w * self.config.max_centroid_distance_ratio;
        let max_dist_confirmed_sq = max_dist_confirmed_px * max_dist_confirmed_px;

        let max_dist_tentative_px =
            self.frame_w * self.config.max_centroid_distance_ratio_tentative;
        let max_dist_tentative_sq = max_dist_tentative_px * max_dist_tentative_px;

        let mut centroid_pairs: Vec<(usize, usize, f32)> = Vec::new();
        for (ti, track) in self.tracks.iter().enumerate() {
            if matched_track_indices[ti] {
                continue;
            }

            let (max_allowed_dist_sq, max_coast_frames) = match track.state {
                TrackState::Confirmed | TrackState::Lost => (
                    max_dist_confirmed_sq,
                    self.config.centroid_fallback_max_coast,
                ),
                TrackState::Tentative => (
                    max_dist_tentative_sq,
                    self.config.centroid_fallback_max_coast_tentative,
                ),
            };

            if track.frames_since_hit > max_coast_frames {
                continue;
            }

            let (tcx, tcy) = track.center();

            // v4.8: Determine which class to match against.
            // Confirmed tracks use their locked class; tentative use current.
            let required_class = track.confirmed_class_id.unwrap_or(track.class_id);

            for (di, det) in valid.iter().enumerate() {
                if matched_det_indices[di] {
                    continue;
                }

                // v4.8: Strict class matching for centroid rescue.
                // Unlike IoU (which penalizes but allows cross-class), centroid
                // rescue has no geometric overlap to validate the match — class
                // agreement is the only identity signal.
                if required_class != det.class_id {
                    continue;
                }

                let (dcx, dcy) = det.center();
                let dist_sq = (tcx - dcx).powi(2) + (tcy - dcy).powi(2);

                if dist_sq < max_allowed_dist_sq {
                    centroid_pairs.push((ti, di, dist_sq));
                }
            }
        }

        centroid_pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        for (ti, di, dist_sq) in &centroid_pairs {
            if matched_track_indices[*ti] || matched_det_indices[*di] {
                continue;
            }
            matched_track_indices[*ti] = true;
            matched_det_indices[*di] = true;

            let track_state = self.tracks[*ti].state;
            info!(
                "🔗 Centroid rescue: Track {} ({:?}) ↔ det (dist={:.0}px, class={}, IoU < {:.2})",
                self.tracks[*ti].id,
                track_state,
                dist_sq.sqrt(),
                valid[*di].class_id,
                self.config.min_iou
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
                info!(
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
                info!(
                    "🗑️  Track {} pruned (coasted {} frames)",
                    t.id, t.frames_since_hit
                );
                return false;
            }
            if t.state == TrackState::Tentative && t.age > min_hits * 3 {
                info!(
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

    fn det_with_class(x1: f32, y1: f32, x2: f32, y2: f32, class_id: u32) -> DetectionInput {
        DetectionInput {
            bbox: [x1, y1, x2, y2],
            class_id,
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
    fn test_confirmed_class_locked() {
        // v4.8: After confirmation, class_id should not change even if
        // matched with a different class via IoU
        let cfg = TrackerConfig::default();
        let mut tracker = VehicleTracker::new(cfg, 1280.0, 720.0);

        // Confirm as class=7 (truck)
        let dets = vec![det(500.0, 200.0, 600.0, 300.0)];
        for i in 0..4 {
            tracker.update(&dets, i as f64 * 33.3, i as u64);
        }
        assert_eq!(tracker.all_tracks()[0].state, TrackState::Confirmed);
        assert_eq!(tracker.all_tracks()[0].class_id, 7);
        assert_eq!(tracker.all_tracks()[0].confirmed_class_id, Some(7));

        // Now send same bbox with class=2 (car) — should still keep class=7
        let dets = vec![det_with_class(500.0, 200.0, 600.0, 300.0, 2)];
        tracker.update(&dets, 5.0 * 33.3, 5);
        assert_eq!(
            tracker.all_tracks()[0].class_id,
            7,
            "Confirmed track should keep locked class_id"
        );
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

        // Frame 4: Extreme bbox change (simulating close approach) — SAME CLASS
        // IoU will be very low, but centroid is close (~100px)
        let dets = vec![det(550.0, 250.0, 800.0, 450.0)];
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
    fn test_centroid_rejects_cross_class() {
        // v4.8: Centroid rescue should NOT match across different classes
        let cfg = TrackerConfig::default();
        let mut tracker = VehicleTracker::new(cfg, 1280.0, 720.0);

        // Confirm track as class=7 (truck)
        for i in 1..=3 {
            let dets = vec![det(500.0, 200.0, 600.0, 300.0)];
            tracker.update(&dets, i as f64 * 33.3, i);
        }
        assert_eq!(tracker.confirmed_count(), 1);
        let original_id = tracker.all_tracks()[0].id;

        // Coast for a few frames (no detections)
        for i in 4..=6 {
            tracker.update(&[], i as f64 * 33.3, i);
        }

        // Now detection appears nearby but as class=2 (car)
        // Centroid distance is ~0 (same position) but class differs
        let dets = vec![det_with_class(500.0, 200.0, 600.0, 300.0, 2)];
        tracker.update(&dets, 7.0 * 33.3, 7);

        // Should create a NEW track, not rescue the old one
        let tracks: Vec<_> = tracker.all_tracks().iter().collect();
        let new_track = tracks.iter().find(|t| t.class_id == 2);
        assert!(
            new_track.is_some(),
            "Cross-class detection should create new track"
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
    fn test_large_centered_truck_stays_ahead() {
        let cfg = TrackerConfig::default();
        let mut tracker = VehicleTracker::new(cfg, 1280.0, 720.0);

        // Truck: large bbox but CENTERED in frame — should be AHEAD
        // cx=700, lateral_offset_ratio = |700-640|/1280 = 0.047 (well under 0.25)
        let dets = vec![det(300.0, 250.0, 1100.0, 466.0)];
        for i in 0..10 {
            tracker.update(&dets, i as f64 * 33.3, i as u64);
        }
        assert_eq!(
            tracker.all_tracks()[0].zone,
            VehicleZone::Ahead,
            "Large centered truck should be AHEAD"
        );
    }

    #[test]
    fn test_large_lateral_vehicle_is_beside() {
        // v4.8: The critical regression test — this was the overtake detection bug.
        // A vehicle at the far left edge with a large bbox should be BESIDE_LEFT,
        // not forced to AHEAD by the size override.
        let cfg = TrackerConfig::default();
        let mut tracker = VehicleTracker::new(cfg, 1280.0, 720.0);

        // Reproduce the exact scenario from the logs:
        // T6 bbox=[0,402,98,633] → cx=49, height=231, height_ratio=0.32
        // lateral_offset_ratio = |49 - 640| / 1280 = 0.46 >> 0.25
        // → should NOT be forced AHEAD, should be BESIDE_LEFT
        let dets = vec![det_with_class(0.0, 402.0, 98.0, 633.0, 2)];
        for i in 0..10 {
            tracker.update(&dets, i as f64 * 33.3, i as u64);
        }
        assert_eq!(
            tracker.all_tracks()[0].zone,
            VehicleZone::BesideLeft,
            "Large vehicle at far left edge should be BESIDE_LEFT, not AHEAD"
        );
    }

    #[test]
    fn test_overtake_zone_sequence_with_lateral_vehicle() {
        // v4.8: End-to-end test for the overtake zone sequence that was failing.
        // Simulates a vehicle that starts AHEAD, moves to BESIDE_LEFT as we pass it.
        let cfg = TrackerConfig::default();
        let mut tracker = VehicleTracker::new(cfg, 1280.0, 720.0);

        // Phase 1: Vehicle ahead (centered, small-medium)
        for i in 0..15 {
            let dets = vec![det_with_class(550.0, 250.0, 720.0, 380.0, 2)];
            tracker.update(&dets, i as f64 * 33.3, i as u64);
        }
        assert_eq!(tracker.all_tracks()[0].zone, VehicleZone::Ahead);

        // Phase 2: Vehicle moves to left side of frame as we overtake it
        // Gets larger (closer) and displaced left
        for i in 15..30 {
            let dets = vec![det_with_class(0.0, 300.0, 200.0, 600.0, 2)];
            tracker.update(&dets, i as f64 * 33.3, i as u64);
        }

        let track = &tracker.all_tracks()[0];
        assert!(
            track.has_zone_sequence(&[VehicleZone::Ahead, VehicleZone::BesideLeft]),
            "Should detect AHEAD→BESIDE_LEFT sequence during overtake. Got: {:?}",
            track.zone_transitions()
        );
    }
}
