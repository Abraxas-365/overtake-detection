// src/lane_crossing.rs
//
// v5.2: Direct YOLO-based lane crossing detection + detection frame cache.
// v6.1: Min penetration filter, per-side debounce, increased cooldown.
//       Crossings are now conservative — only meaningful overlap fires.
//
// ════════════════════════════════════════════════════════════════════════════
// PART 1: LINE CROSSING DETECTOR
// ════════════════════════════════════════════════════════════════════════════
//
// Instead of inferring lane changes from lateral offset drift (which is
// indirect and delayed), this directly checks whether the ego vehicle's
// bottom-center position overlaps with a detected lane marking's bbox/mask.
//
// When the vehicle physically drives over a line, we get:
//   - WHICH line was crossed (left boundary, right boundary, center line)
//   - The marking type (solid, dashed, mixed, double)
//   - The passing legality of that specific marking
//   - Direction of crossing (leftward or rightward)
//
// This is a much more direct signal than offset-based detection and can be
// fused with the existing LateralShiftDetector as corroborating evidence.
//
// v6.1 additions:
//   - min_penetration_ratio: ignore grazing contact (pen < 0.12)
//   - clear_frames_required: per-side debounce to prevent re-fire spam
//   - Increased cooldown from 30 → 50 frames
//   - Per-side fire tracking to allow first fire freely
//
// ════════════════════════════════════════════════════════════════════════════
// PART 2: YOLO DETECTION FRAME CACHE
// ════════════════════════════════════════════════════════════════════════════
//
// Peru-specific problem: roads can transition instantly from paved carretera
// with full lane markings to unpaved "trocha" (dirt road) with zero markings.
//
// The cache:
//   - Stores the last N frames of valid lane boundary detections
//   - When a frame has no fresh YOLO detections, returns ego-motion-compensated
//     cached positions with decaying confidence
//   - Expires after `max_stale_frames` (default: 15 frames ≈ 500ms at 30fps)
//   - After expiry → "unmarked road" mode (trocha), no phantom lanes
//   - Tracks cache health metrics for diagnostic overlay
//
// The key insight: 500ms of cache bridges momentary occlusions (passing trucks,
// shadows, worn paint) but is short enough that entering a trocha (where you'd
// go many seconds without markings) naturally expires the cache.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::{debug, info, warn};

use crate::road_classification::{MarkingInfo, PassingLegality};

// ============================================================================
// TYPES
// ============================================================================

/// A detected line-crossing event from a single frame.
#[derive(Debug, Clone)]
pub struct LineCrossingEvent {
    /// Which line the vehicle is crossing
    pub line_role: LineRole,
    /// The marking's class name (e.g. "solid_single_yellow", "dashed_single_white")
    pub marking_class: String,
    /// The marking's class_id
    pub marking_class_id: usize,
    /// How illegal is this crossing
    pub passing_legality: PassingLegality,
    /// Direction of crossing relative to ego (leftward = vehicle moving left)
    pub crossing_direction: CrossingDirection,
    /// Confidence of the crossing detection [0, 1]
    pub confidence: f32,
    /// How deep the vehicle center is into the marking's bbox (0 = edge, 1 = center)
    pub penetration_ratio: f32,
    /// Frame ID when detected
    pub frame_id: u64,
    /// Timestamp in ms
    pub timestamp_ms: f64,
}

/// The role of the line being crossed relative to the ego lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineRole {
    /// Left boundary of ego lane
    LeftBoundary,
    /// Right boundary of ego lane
    RightBoundary,
    /// Center/dividing line (in a two-way road context)
    CenterLine,
    /// Unknown or cannot determine role
    Unknown,
}

impl LineRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::LeftBoundary => "LEFT_BOUNDARY",
            Self::RightBoundary => "RIGHT_BOUNDARY",
            Self::CenterLine => "CENTER_LINE",
            Self::Unknown => "UNKNOWN",
        }
    }
}

/// Direction the vehicle is moving when crossing a line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrossingDirection {
    Leftward,
    Rightward,
    Unknown,
}

impl CrossingDirection {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Leftward => "LEFT",
            Self::Rightward => "RIGHT",
            Self::Unknown => "UNKNOWN",
        }
    }
}

/// State of crossing for temporal tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OverlapState {
    /// Not overlapping any line
    Clear,
    /// Currently overlapping a line (building up evidence)
    Overlapping,
    /// Confirmed crossing (overlap persisted long enough)
    Confirmed,
}

// ============================================================================
// LINE CROSSING DETECTOR
// ============================================================================

/// Configuration for the crossing detector.
#[derive(Debug, Clone)]
pub struct CrossingDetectorConfig {
    /// Horizontal margin (px) around the vehicle center to consider as "ego zone".
    /// The vehicle's front is wider than a single pixel — this approximates the
    /// vehicle width at the reference Y line. Default: 40px for a dashcam at 1280w.
    pub ego_half_width_px: f32,
    /// Reference Y ratio (0.0=top, 1.0=bottom). Where to sample the ego position.
    /// Default: 0.82 (close to bottom, where the vehicle "touches" the road).
    pub reference_y_ratio: f32,
    /// Minimum frames of overlap before confirming a crossing event.
    /// Prevents false positives from single-frame noise. Default: 3.
    pub min_overlap_frames: u32,
    /// Cooldown frames after a confirmed crossing before detecting another.
    /// Prevents double-counting the same physical crossing. Default: 50 (~2s).
    pub cooldown_frames: u32,
    /// Minimum marking confidence to consider for crossing detection.
    pub min_marking_confidence: f32,
    /// v6.1: Minimum penetration ratio to emit a crossing event.
    /// Below this, the overlap is considered grazing contact / noise.
    /// Default: 0.12 (12% of marking width).
    pub min_penetration_ratio: f32,
    /// v6.1: After a crossing fires on a given side (Left/Right/Center),
    /// require this many frames of Clear state before the SAME side can fire again.
    /// Prevents repeated crossings when driving near a line. Default: 20.
    pub clear_frames_required: u32,
}

impl Default for CrossingDetectorConfig {
    fn default() -> Self {
        Self {
            ego_half_width_px: 40.0,
            reference_y_ratio: 0.82,
            min_overlap_frames: 3,
            cooldown_frames: 50,
            min_marking_confidence: 0.25,
            min_penetration_ratio: 0.12,
            clear_frames_required: 20,
        }
    }
}

pub struct LineCrossingDetector {
    config: CrossingDetectorConfig,
    frame_width: f32,
    frame_height: f32,

    // Per-line overlap tracking
    left_overlap_state: OverlapState,
    left_overlap_frames: u32,
    right_overlap_state: OverlapState,
    right_overlap_frames: u32,
    center_overlap_state: OverlapState,
    center_overlap_frames: u32,

    // Cooldown
    cooldown_remaining: u32,

    // Previous frame's ego center for direction estimation
    prev_ego_x: Option<f32>,

    // History for external consumers
    recent_crossings: VecDeque<LineCrossingEvent>,

    // v6.1: Per-side debounce — frames since each side was last Clear.
    // A side can only re-fire after being Clear for >= clear_frames_required.
    left_clear_counter: u32,
    right_clear_counter: u32,
    center_clear_counter: u32,
    // v6.1: Track which sides have fired at least once (to allow the first fire freely)
    left_has_fired: bool,
    right_has_fired: bool,
    center_has_fired: bool,
}

impl LineCrossingDetector {
    pub fn new(frame_width: f32, frame_height: f32, config: CrossingDetectorConfig) -> Self {
        Self {
            config,
            frame_width,
            frame_height,
            left_overlap_state: OverlapState::Clear,
            left_overlap_frames: 0,
            right_overlap_state: OverlapState::Clear,
            right_overlap_frames: 0,
            center_overlap_state: OverlapState::Clear,
            center_overlap_frames: 0,
            cooldown_remaining: 0,
            prev_ego_x: None,
            recent_crossings: VecDeque::with_capacity(20),
            left_clear_counter: u32::MAX, // allow first fire
            right_clear_counter: u32::MAX,
            center_clear_counter: u32::MAX,
            left_has_fired: false,
            right_has_fired: false,
            center_has_fired: false,
        }
    }

    /// Process one frame's worth of YOLO-seg detections.
    ///
    /// `markings` — all lane-line MarkingInfo from the current frame
    /// `ego_left_x` — left boundary x of ego lane (from YOLO boundary estimation)
    /// `ego_right_x` — right boundary x of ego lane
    /// `frame_id` — current frame number
    /// `timestamp_ms` — current video timestamp
    ///
    /// Returns a crossing event if the vehicle is confirmed to be crossing a line.
    pub fn update(
        &mut self,
        markings: &[MarkingInfo],
        ego_left_x: Option<f32>,
        ego_right_x: Option<f32>,
        frame_id: u64,
        timestamp_ms: f64,
    ) -> Option<LineCrossingEvent> {
        // Tick cooldown
        if self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
            return None;
        }

        let ego_center_x = self.frame_width / 2.0;
        let reference_y = self.frame_height * self.config.reference_y_ratio;
        let ego_zone_left = ego_center_x - self.config.ego_half_width_px;
        let ego_zone_right = ego_center_x + self.config.ego_half_width_px;

        // Determine crossing direction from frame-to-frame ego movement
        let crossing_dir = if let Some(prev_x) = self.prev_ego_x {
            let dx = ego_center_x - prev_x;
            if dx < -2.0 {
                CrossingDirection::Leftward
            } else if dx > 2.0 {
                CrossingDirection::Rightward
            } else {
                CrossingDirection::Unknown
            }
        } else {
            CrossingDirection::Unknown
        };
        self.prev_ego_x = Some(ego_center_x);

        // Classify each marking's role relative to ego lane
        let lane_lines: Vec<&MarkingInfo> = markings
            .iter()
            .filter(|m| {
                is_lane_line_class(m.class_id) && m.confidence >= self.config.min_marking_confidence
            })
            .collect();

        // Check each marking for overlap with ego zone
        let mut left_overlap = false;
        let mut right_overlap = false;
        let mut center_overlap = false;
        let mut best_crossing: Option<(LineRole, &MarkingInfo, f32)> = None;

        for marking in &lane_lines {
            let m_left = marking.bbox[0];
            let m_right = marking.bbox[2];
            let m_top = marking.bbox[1];
            let m_bottom = marking.bbox[3];

            // Check vertical: marking must span the reference Y line
            if reference_y < m_top || reference_y > m_bottom {
                continue;
            }

            // Check horizontal overlap with ego zone
            let overlap_left = ego_zone_left.max(m_left);
            let overlap_right = ego_zone_right.min(m_right);
            let horizontal_overlap = overlap_right - overlap_left;

            if horizontal_overlap <= 0.0 {
                continue;
            }

            // Compute penetration ratio (how centered is ego on the marking)
            let marking_width = (m_right - m_left).max(1.0);
            let penetration = horizontal_overlap / marking_width;

            // Determine the marking's role
            let role = classify_line_role(
                marking.center_x,
                ego_center_x,
                ego_left_x,
                ego_right_x,
                self.frame_width,
            );

            match role {
                LineRole::LeftBoundary => left_overlap = true,
                LineRole::RightBoundary => right_overlap = true,
                LineRole::CenterLine => center_overlap = true,
                _ => {}
            }

            // Track the best (deepest penetration) crossing candidate
            if let Some((_, _, best_pen)) = &best_crossing {
                if penetration > *best_pen {
                    best_crossing = Some((role, marking, penetration));
                }
            } else {
                best_crossing = Some((role, marking, penetration));
            }
        }

        // v6.1c: Adaptive min_overlap for dashed vs solid markings.
        // Dashed markings (class 9, 10) have short intermittent segments —
        // ego zone overlaps 1-2 frames max. Use min_overlap=1 for dashed.
        let adaptive_min_overlap = best_crossing
            .as_ref()
            .map(|(_, m, _)| {
                if is_dashed_marking(m.class_id) {
                    1u32
                } else {
                    self.config.min_overlap_frames
                }
            })
            .unwrap_or(self.config.min_overlap_frames);

        // Update overlap state machines
        self.left_overlap_state = if left_overlap {
            match self.left_overlap_state {
                OverlapState::Clear => {
                    self.left_overlap_frames = 1;
                    OverlapState::Overlapping
                }
                OverlapState::Overlapping => {
                    self.left_overlap_frames += 1;
                    if self.left_overlap_frames >= adaptive_min_overlap {
                        OverlapState::Confirmed
                    } else {
                        OverlapState::Overlapping
                    }
                }
                OverlapState::Confirmed => {
                    self.left_overlap_frames += 1;
                    OverlapState::Confirmed
                }
            }
        } else {
            self.left_overlap_frames = 0;
            OverlapState::Clear
        };

        self.right_overlap_state = if right_overlap {
            match self.right_overlap_state {
                OverlapState::Clear => {
                    self.right_overlap_frames = 1;
                    OverlapState::Overlapping
                }
                OverlapState::Overlapping => {
                    self.right_overlap_frames += 1;
                    if self.right_overlap_frames >= adaptive_min_overlap {
                        OverlapState::Confirmed
                    } else {
                        OverlapState::Overlapping
                    }
                }
                OverlapState::Confirmed => {
                    self.right_overlap_frames += 1;
                    OverlapState::Confirmed
                }
            }
        } else {
            self.right_overlap_frames = 0;
            OverlapState::Clear
        };

        self.center_overlap_state = if center_overlap {
            match self.center_overlap_state {
                OverlapState::Clear => {
                    self.center_overlap_frames = 1;
                    OverlapState::Overlapping
                }
                OverlapState::Overlapping => {
                    self.center_overlap_frames += 1;
                    if self.center_overlap_frames >= adaptive_min_overlap {
                        OverlapState::Confirmed
                    } else {
                        OverlapState::Overlapping
                    }
                }
                OverlapState::Confirmed => {
                    self.center_overlap_frames += 1;
                    OverlapState::Confirmed
                }
            }
        } else {
            self.center_overlap_frames = 0;
            OverlapState::Clear
        };

        // v6.1: Update per-side clear counters for debounce
        if self.left_overlap_state == OverlapState::Clear {
            self.left_clear_counter = self.left_clear_counter.saturating_add(1);
        } else {
            self.left_clear_counter = 0;
        }
        if self.right_overlap_state == OverlapState::Clear {
            self.right_clear_counter = self.right_clear_counter.saturating_add(1);
        } else {
            self.right_clear_counter = 0;
        }
        if self.center_overlap_state == OverlapState::Clear {
            self.center_clear_counter = self.center_clear_counter.saturating_add(1);
        } else {
            self.center_clear_counter = 0;
        }

        // Emit event on first frame of Confirmed state (with v6.1 gates)
        if let Some((role, marking, penetration)) = best_crossing {
            // v6.1 GATE 1: Minimum penetration — ignore grazing contact
            if penetration < self.config.min_penetration_ratio {
                debug!(
                    "🚫 Crossing suppressed (pen={:.2} < {:.2}): {} {}",
                    penetration,
                    self.config.min_penetration_ratio,
                    role.as_str(),
                    marking.class_name,
                );
                return None;
            }

            // v6.1c: Adaptive overlap threshold — dashed markings confirm on first frame
            let effective_min_overlap = if is_dashed_marking(marking.class_id) {
                1u32
            } else {
                self.config.min_overlap_frames
            };

            let is_newly_confirmed = match role {
                LineRole::LeftBoundary => {
                    self.left_overlap_state == OverlapState::Confirmed
                        && self.left_overlap_frames == effective_min_overlap
                }
                LineRole::RightBoundary => {
                    self.right_overlap_state == OverlapState::Confirmed
                        && self.right_overlap_frames == effective_min_overlap
                }
                LineRole::CenterLine => {
                    self.center_overlap_state == OverlapState::Confirmed
                        && self.center_overlap_frames == effective_min_overlap
                }
                _ => false,
            };

            if is_newly_confirmed {
                // v6.1 GATE 2: Per-side debounce — require enough Clear frames before re-fire
                let (has_fired, clear_count) = match role {
                    LineRole::LeftBoundary => (self.left_has_fired, self.left_clear_counter),
                    LineRole::RightBoundary => (self.right_has_fired, self.right_clear_counter),
                    LineRole::CenterLine => (self.center_has_fired, self.center_clear_counter),
                    _ => (false, u32::MAX),
                };

                // First fire on this side is always allowed; subsequent fires require debounce
                if has_fired && clear_count < self.config.clear_frames_required {
                    debug!(
                        "🚫 Crossing debounced (side={} clear={}/{}): {} pen={:.2}",
                        role.as_str(),
                        clear_count,
                        self.config.clear_frames_required,
                        marking.class_name,
                        penetration,
                    );
                    return None;
                }

                let legality = marking_to_passing_legality(marking);

                let event = LineCrossingEvent {
                    line_role: role,
                    marking_class: marking.class_name.clone(),
                    marking_class_id: marking.class_id,
                    passing_legality: legality,
                    crossing_direction: crossing_dir,
                    confidence: marking.confidence,
                    penetration_ratio: penetration,
                    frame_id,
                    timestamp_ms,
                };

                info!(
                    "🚨 LINE CROSSING: {} {} ({}) → legality={} dir={} pen={:.2}",
                    role.as_str(),
                    marking.class_name,
                    marking.class_id,
                    legality.as_str(),
                    crossing_dir.as_str(),
                    penetration,
                );

                // v6.1: Mark this side as having fired (for debounce on next fire)
                match role {
                    LineRole::LeftBoundary => self.left_has_fired = true,
                    LineRole::RightBoundary => self.right_has_fired = true,
                    LineRole::CenterLine => self.center_has_fired = true,
                    _ => {}
                }

                self.cooldown_remaining = self.config.cooldown_frames;
                self.recent_crossings.push_back(event.clone());
                if self.recent_crossings.len() > 20 {
                    self.recent_crossings.pop_front();
                }

                return Some(event);
            }
        }

        None
    }

    /// Dummy helper — state tracking is done inline above.
    fn update_overlap_state(
        &self,
        _state: &mut OverlapState,
        _frames: &mut u32,
        _overlapping: bool,
    ) -> bool {
        false
    }

    /// Get recent crossing events (for diagnostics / fusion).
    pub fn recent_crossings(&self) -> &VecDeque<LineCrossingEvent> {
        &self.recent_crossings
    }

    /// Check if the ego vehicle is currently overlapping any line.
    pub fn is_on_line(&self) -> bool {
        self.left_overlap_state != OverlapState::Clear
            || self.right_overlap_state != OverlapState::Clear
            || self.center_overlap_state != OverlapState::Clear
    }

    pub fn reset(&mut self) {
        self.left_overlap_state = OverlapState::Clear;
        self.left_overlap_frames = 0;
        self.right_overlap_state = OverlapState::Clear;
        self.right_overlap_frames = 0;
        self.center_overlap_state = OverlapState::Clear;
        self.center_overlap_frames = 0;
        self.cooldown_remaining = 0;
        self.prev_ego_x = None;
        self.recent_crossings.clear();
        self.left_clear_counter = u32::MAX;
        self.right_clear_counter = u32::MAX;
        self.center_clear_counter = u32::MAX;
        self.left_has_fired = false;
        self.right_has_fired = false;
        self.center_has_fired = false;
    }
}

// ============================================================================
// PART 2: YOLO DETECTION FRAME CACHE
// ============================================================================

/// Cached lane boundary state from YOLO-seg detections.
#[derive(Debug, Clone)]
pub struct CachedLaneBoundaries {
    /// Left boundary x position (in original image coordinates)
    pub left_x: f32,
    /// Right boundary x position
    pub right_x: f32,
    /// Detection confidence when captured
    pub original_confidence: f32,
    /// Whether both boundaries were detected (vs single + estimated width)
    pub both_detected: bool,
    /// All lane-line markings from this frame (for crossing detector, road classifier, etc.)
    pub markings: Vec<MarkingInfo>,
    /// Frame ID when this was captured
    pub captured_frame_id: u64,
    /// Timestamp when captured
    pub captured_timestamp_ms: f64,
}

/// Current state of the detection cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheState {
    /// Fresh detection from the current frame
    Fresh,
    /// Using cached data, compensated by ego motion
    Cached,
    /// Cache expired — no lanes available (trocha / unmarked road)
    Expired,
    /// Never had a detection (startup)
    Empty,
}

impl CacheState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fresh => "FRESH",
            Self::Cached => "CACHED",
            Self::Expired => "EXPIRED",
            Self::Empty => "EMPTY",
        }
    }

    pub fn has_lanes(&self) -> bool {
        matches!(self, Self::Fresh | Self::Cached)
    }
}

impl std::fmt::Display for CacheState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Configuration for the detection cache.
#[derive(Debug, Clone)]
pub struct DetectionCacheConfig {
    /// Maximum frames to keep using cached lane positions after last fresh detection.
    /// After this many frames without fresh data, the cache expires.
    ///
    /// Tuning guide:
    ///   - Too low (< 5): Normal occlusions (shadows, trucks) break lane tracking
    ///   - Too high (> 30): Trochas/unmarked roads get phantom lanes
    ///   - Sweet spot: 10-15 frames (330-500ms at 30fps)
    pub max_stale_frames: u32,

    /// Confidence decay multiplier per cached frame.
    /// Applied multiplicatively each frame the cache is stale.
    /// 0.85^10 ≈ 0.20, so after 10 stale frames confidence is ~20% of original.
    pub confidence_decay_per_frame: f32,

    /// Minimum confidence below which cached data is treated as expired
    /// even if max_stale_frames hasn't been reached.
    pub min_cache_confidence: f32,

    /// Whether to apply ego-motion compensation to cached positions.
    /// When true, cached left_x/right_x are adjusted by the accumulated
    /// ego lateral displacement since capture. This keeps the cached
    /// boundaries aligned with the vehicle's actual position.
    pub ego_compensate: bool,
}

impl Default for DetectionCacheConfig {
    fn default() -> Self {
        Self {
            max_stale_frames: 15,
            confidence_decay_per_frame: 0.85,
            min_cache_confidence: 0.15,
            ego_compensate: true,
        }
    }
}

/// YOLO detection frame cache with ego-motion compensation and trocha awareness.
pub struct DetectionCache {
    config: DetectionCacheConfig,

    /// The cached lane boundary data (if any)
    cached: Option<CachedLaneBoundaries>,

    /// How many frames since the last fresh detection
    stale_frames: u32,

    /// Accumulated ego lateral displacement since last fresh detection (px).
    /// Used to compensate cached positions.
    accumulated_ego_displacement_px: f32,

    /// Current state of the cache
    state: CacheState,

    // ── Metrics ──
    /// Total fresh detections received
    pub total_fresh_detections: u64,
    /// Total frames served from cache
    pub total_cached_frames: u64,
    /// Total frames in expired state (no data)
    pub total_expired_frames: u64,
    /// Number of cache-to-fresh recoveries (occlusion bridges)
    pub cache_recoveries: u64,
    /// Number of cache expirations (transitions to trocha/unmarked)
    pub cache_expirations: u64,
}

/// Result of querying the cache for lane boundaries.
#[derive(Debug, Clone)]
pub struct CachedBoundaryResult {
    /// Left boundary x position (ego-compensated if cached)
    pub left_x: f32,
    /// Right boundary x position (ego-compensated if cached)
    pub right_x: f32,
    /// Effective confidence (decayed if cached)
    pub confidence: f32,
    /// Whether both boundaries were originally detected
    pub both_detected: bool,
    /// Cache state for this result
    pub state: CacheState,
    /// How many frames stale (0 = fresh)
    pub stale_frames: u32,
    /// Cached markings (for crossing detector, road classifier)
    pub markings: Vec<MarkingInfo>,
}

impl DetectionCache {
    pub fn new(config: DetectionCacheConfig) -> Self {
        Self {
            config,
            cached: None,
            stale_frames: 0,
            accumulated_ego_displacement_px: 0.0,
            state: CacheState::Empty,
            total_fresh_detections: 0,
            total_cached_frames: 0,
            total_expired_frames: 0,
            cache_recoveries: 0,
            cache_expirations: 0,
        }
    }

    /// Feed a fresh YOLO-seg detection into the cache.
    ///
    /// Call this every frame when the YOLO model successfully detects lane boundaries.
    /// Pass `None` when no lanes were detected in the current frame.
    pub fn update_fresh(&mut self, detection: Option<CachedLaneBoundaries>) {
        if let Some(det) = detection {
            let was_expired = matches!(self.state, CacheState::Expired | CacheState::Empty);
            if was_expired && self.cached.is_some() {
                self.cache_recoveries += 1;
                info!(
                    "📡 Detection cache: recovered after {} expired frames",
                    self.stale_frames
                );
            }

            self.cached = Some(det);
            self.stale_frames = 0;
            self.accumulated_ego_displacement_px = 0.0;
            self.state = CacheState::Fresh;
            self.total_fresh_detections += 1;
        } else {
            // No fresh detection → age the cache
            self.stale_frames += 1;

            if self.cached.is_some() {
                let decayed_conf = self.effective_confidence();
                let frames_exceeded = self.stale_frames > self.config.max_stale_frames;
                let conf_too_low = decayed_conf < self.config.min_cache_confidence;

                if frames_exceeded || conf_too_low {
                    // Cache expired → trocha / unmarked road mode
                    if self.state != CacheState::Expired {
                        self.cache_expirations += 1;
                        debug!(
                            "📡 Detection cache: EXPIRED after {}f stale (conf={:.2}) → unmarked road mode",
                            self.stale_frames, decayed_conf
                        );
                    }
                    self.state = CacheState::Expired;
                    self.total_expired_frames += 1;
                } else {
                    self.state = CacheState::Cached;
                    self.total_cached_frames += 1;
                }
            } else {
                self.state = CacheState::Empty;
                self.total_expired_frames += 1;
            }
        }
    }

    /// Accumulate ego lateral motion for cache compensation.
    ///
    /// Call this every frame with the ego lateral velocity (px/frame).
    /// Sign convention: positive = rightward, negative = leftward.
    pub fn accumulate_ego_motion(&mut self, lateral_velocity_px: f32) {
        if self.config.ego_compensate && self.state == CacheState::Cached {
            self.accumulated_ego_displacement_px += lateral_velocity_px;
        }
    }

    /// Query the cache for current lane boundaries.
    ///
    /// Returns `None` if no data is available (empty or expired).
    /// Returns ego-compensated, confidence-decayed boundaries if cached.
    /// Returns fresh boundaries if a detection just came in.
    pub fn get_boundaries(&self) -> Option<CachedBoundaryResult> {
        let cached = self.cached.as_ref()?;

        match self.state {
            CacheState::Empty | CacheState::Expired => None,
            CacheState::Fresh => Some(CachedBoundaryResult {
                left_x: cached.left_x,
                right_x: cached.right_x,
                confidence: cached.original_confidence,
                both_detected: cached.both_detected,
                state: CacheState::Fresh,
                stale_frames: 0,
                markings: cached.markings.clone(),
            }),
            CacheState::Cached => {
                // Ego-compensate: as the vehicle moves laterally, the lanes
                // appear to shift in the opposite direction in the camera frame.
                // If the vehicle moves RIGHT by 5px, both lanes shift LEFT by 5px.
                let ego_comp = if self.config.ego_compensate {
                    -self.accumulated_ego_displacement_px
                } else {
                    0.0
                };

                let conf = self.effective_confidence();

                Some(CachedBoundaryResult {
                    left_x: cached.left_x + ego_comp,
                    right_x: cached.right_x + ego_comp,
                    confidence: conf,
                    both_detected: cached.both_detected,
                    state: CacheState::Cached,
                    stale_frames: self.stale_frames,
                    markings: cached.markings.clone(),
                })
            }
        }
    }

    /// Get current cache state.
    pub fn state(&self) -> CacheState {
        self.state
    }

    /// How many frames since last fresh detection.
    pub fn stale_frames(&self) -> u32 {
        self.stale_frames
    }

    /// Is the road currently considered unmarked (trocha)?
    pub fn is_unmarked_road(&self) -> bool {
        matches!(self.state, CacheState::Expired | CacheState::Empty)
    }

    /// Reset the cache (e.g., on scene change).
    pub fn reset(&mut self) {
        self.cached = None;
        self.stale_frames = 0;
        self.accumulated_ego_displacement_px = 0.0;
        self.state = CacheState::Empty;
    }

    /// Effective confidence after decay.
    fn effective_confidence(&self) -> f32 {
        if let Some(ref cached) = self.cached {
            cached.original_confidence
                * self
                    .config
                    .confidence_decay_per_frame
                    .powi(self.stale_frames as i32)
        } else {
            0.0
        }
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Classify a marking's role relative to the ego lane.
fn classify_line_role(
    marking_center_x: f32,
    ego_center_x: f32,
    ego_left_x: Option<f32>,
    ego_right_x: Option<f32>,
    frame_width: f32,
) -> LineRole {
    // If we know the ego lane boundaries, use them for precise classification
    if let (Some(left), Some(right)) = (ego_left_x, ego_right_x) {
        let _lane_center = (left + right) / 2.0;
        let dist_to_left = (marking_center_x - left).abs();
        let dist_to_right = (marking_center_x - right).abs();
        let _dist_to_center = (marking_center_x - _lane_center).abs();
        let lane_width = (right - left).max(1.0);

        // If the marking is close to a known boundary
        if dist_to_left < lane_width * 0.15 {
            return LineRole::LeftBoundary;
        }
        if dist_to_right < lane_width * 0.15 {
            return LineRole::RightBoundary;
        }
    }

    // Fallback: use position relative to frame center
    let center_zone_margin = frame_width * 0.08;
    if (marking_center_x - ego_center_x).abs() < center_zone_margin {
        LineRole::CenterLine
    } else if marking_center_x < ego_center_x {
        LineRole::LeftBoundary
    } else {
        LineRole::RightBoundary
    }
}

/// Determine passing legality for a specific marking (for crossing events).
fn marking_to_passing_legality(marking: &MarkingInfo) -> PassingLegality {
    match marking.class_id {
        // Solid lines → prohibited
        4 | 5 | 6 => PassingLegality::Prohibited,
        // Double solid → definitely prohibited
        7 | 8 => PassingLegality::Prohibited,
        // Dashed → allowed
        9 | 10 => PassingLegality::Allowed,
        // Mixed → need more context (from road classifier)
        99 => PassingLegality::Unknown,
        _ => PassingLegality::Unknown,
    }
}

fn is_lane_line_class(class_id: usize) -> bool {
    matches!(class_id, 4 | 5 | 6 | 7 | 8 | 9 | 10 | 99)
}

/// v6.1c: Dashed markings have short, intermittent segments.
/// Crossing detection needs lower overlap thresholds for these.
fn is_dashed_marking(class_id: usize) -> bool {
    matches!(class_id, 9 | 10) // 9=dashed_single_white, 10=dashed_single_yellow
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color_analysis::MarkingColor;

    fn make_marking(class_id: usize, name: &str, cx: f32, x1: f32, x2: f32) -> MarkingInfo {
        MarkingInfo {
            class_id,
            class_name: name.to_string(),
            center_x: cx,
            bbox: [x1, 400.0, x2, 700.0], // spans reference_y at 0.82 * 720 = 590
            confidence: 0.85,
            detected_color: None,
            mask: Vec::new(),
            mask_width: 0,
            mask_height: 0,
        }
    }

    // ---- Detection Cache tests ----

    #[test]
    fn test_cache_fresh_detection() {
        let mut cache = DetectionCache::new(DetectionCacheConfig::default());
        cache.update_fresh(Some(CachedLaneBoundaries {
            left_x: 300.0,
            right_x: 900.0,
            original_confidence: 0.90,
            both_detected: true,
            markings: vec![],
            captured_frame_id: 1,
            captured_timestamp_ms: 33.0,
        }));

        assert_eq!(cache.state(), CacheState::Fresh);
        let result = cache.get_boundaries().unwrap();
        assert_eq!(result.left_x, 300.0);
        assert_eq!(result.confidence, 0.90);
        assert!(!cache.is_unmarked_road());
    }

    #[test]
    fn test_cache_stale_then_expired() {
        let mut cache = DetectionCache::new(DetectionCacheConfig {
            max_stale_frames: 5,
            confidence_decay_per_frame: 0.80,
            min_cache_confidence: 0.10,
            ego_compensate: false,
        });

        // Fresh detection
        cache.update_fresh(Some(CachedLaneBoundaries {
            left_x: 300.0,
            right_x: 900.0,
            original_confidence: 0.90,
            both_detected: true,
            markings: vec![],
            captured_frame_id: 1,
            captured_timestamp_ms: 33.0,
        }));

        // 3 frames without detection → cached
        for _ in 0..3 {
            cache.update_fresh(None);
        }
        assert_eq!(cache.state(), CacheState::Cached);
        let result = cache.get_boundaries().unwrap();
        assert!(result.confidence < 0.90); // Decayed
        assert_eq!(result.stale_frames, 3);

        // 3 more → exceeds max_stale_frames (5) → expired
        for _ in 0..3 {
            cache.update_fresh(None);
        }
        assert_eq!(cache.state(), CacheState::Expired);
        assert!(cache.is_unmarked_road());
        assert!(cache.get_boundaries().is_none());
    }

    #[test]
    fn test_cache_ego_compensation() {
        let mut cache = DetectionCache::new(DetectionCacheConfig {
            max_stale_frames: 10,
            confidence_decay_per_frame: 0.90,
            min_cache_confidence: 0.10,
            ego_compensate: true,
        });

        cache.update_fresh(Some(CachedLaneBoundaries {
            left_x: 300.0,
            right_x: 900.0,
            original_confidence: 0.90,
            both_detected: true,
            markings: vec![],
            captured_frame_id: 1,
            captured_timestamp_ms: 33.0,
        }));

        // Vehicle moves right by 10px total over 2 frames
        cache.update_fresh(None);
        cache.accumulate_ego_motion(5.0);
        cache.update_fresh(None);
        cache.accumulate_ego_motion(5.0);

        let result = cache.get_boundaries().unwrap();
        // Vehicle moved right → lanes appear to shift left → subtract displacement
        assert!(
            (result.left_x - 290.0).abs() < 0.1,
            "left_x={}",
            result.left_x
        );
        assert!(
            (result.right_x - 890.0).abs() < 0.1,
            "right_x={}",
            result.right_x
        );
    }

    #[test]
    fn test_cache_recovery_from_expired() {
        let mut cache = DetectionCache::new(DetectionCacheConfig {
            max_stale_frames: 3,
            ..Default::default()
        });

        // Fresh → expired
        cache.update_fresh(Some(CachedLaneBoundaries {
            left_x: 300.0,
            right_x: 900.0,
            original_confidence: 0.90,
            both_detected: true,
            markings: vec![],
            captured_frame_id: 1,
            captured_timestamp_ms: 33.0,
        }));
        for _ in 0..5 {
            cache.update_fresh(None);
        }
        assert!(cache.is_unmarked_road());

        // New fresh detection → recovered
        cache.update_fresh(Some(CachedLaneBoundaries {
            left_x: 310.0,
            right_x: 910.0,
            original_confidence: 0.85,
            both_detected: true,
            markings: vec![],
            captured_frame_id: 10,
            captured_timestamp_ms: 333.0,
        }));
        assert_eq!(cache.state(), CacheState::Fresh);
        assert_eq!(cache.cache_recoveries, 1);
    }

    // ---- Line Crossing Detector tests ----

    #[test]
    fn test_crossing_detection_basic() {
        let config = CrossingDetectorConfig {
            ego_half_width_px: 40.0,
            reference_y_ratio: 0.82,
            min_overlap_frames: 2,
            cooldown_frames: 10,
            min_marking_confidence: 0.20,
            min_penetration_ratio: 0.0, // test: no gate
            clear_frames_required: 0,   // test: no debounce
        };
        let mut detector = LineCrossingDetector::new(1280.0, 720.0, config);

        // Place a solid yellow line right at the ego center
        let marking = make_marking(5, "solid_single_yellow", 640.0, 620.0, 660.0);

        // Frame 1: overlap starts
        let r1 = detector.update(&[marking.clone()], Some(300.0), Some(900.0), 1, 33.0);
        assert!(r1.is_none()); // Not confirmed yet (need min_overlap_frames=2)

        // Frame 2: confirmed
        let r2 = detector.update(&[marking.clone()], Some(300.0), Some(900.0), 2, 66.0);
        assert!(r2.is_some());
        let event = r2.unwrap();
        assert_eq!(event.passing_legality, PassingLegality::Prohibited);
    }

    #[test]
    fn test_no_crossing_when_line_is_far() {
        let config = CrossingDetectorConfig::default();
        let mut detector = LineCrossingDetector::new(1280.0, 720.0, config);

        // Line at x=300, far from ego center at 640
        let marking = make_marking(5, "solid_single_yellow", 300.0, 290.0, 310.0);

        for frame in 0..10 {
            let r = detector.update(
                &[marking.clone()],
                Some(300.0),
                Some(900.0),
                frame,
                frame as f64 * 33.0,
            );
            assert!(r.is_none());
        }
    }

    #[test]
    fn test_dashed_line_confirms_on_single_frame() {
        // v6.1c: Dashed markings should confirm with just 1 frame of overlap
        // because dashed segments are short and intermittent
        let config = CrossingDetectorConfig {
            ego_half_width_px: 40.0,
            reference_y_ratio: 0.82,
            min_overlap_frames: 3, // solid lines need 3, but dashed should use 1
            cooldown_frames: 10,
            min_marking_confidence: 0.20,
            min_penetration_ratio: 0.0,
            clear_frames_required: 0,
        };
        let mut detector = LineCrossingDetector::new(1280.0, 720.0, config);

        // Dashed yellow line at ego center (class_id=10 = dashed_single_yellow)
        let marking = make_marking(10, "dashed_single_yellow", 640.0, 620.0, 660.0);

        // Frame 1: Should confirm immediately (dashed → min_overlap=1)
        let r1 = detector.update(&[marking.clone()], Some(300.0), Some(900.0), 1, 33.0);
        assert!(
            r1.is_some(),
            "Dashed marking should confirm on first overlap frame"
        );
        let event = r1.unwrap();
        assert_eq!(event.passing_legality, PassingLegality::Allowed);
        assert_eq!(event.marking_class_id, 10);
    }

    #[test]
    fn test_solid_line_requires_multi_frame() {
        // Solid lines should still require min_overlap_frames=3
        let config = CrossingDetectorConfig {
            ego_half_width_px: 40.0,
            reference_y_ratio: 0.82,
            min_overlap_frames: 3,
            cooldown_frames: 10,
            min_marking_confidence: 0.20,
            min_penetration_ratio: 0.0,
            clear_frames_required: 0,
        };
        let mut detector = LineCrossingDetector::new(1280.0, 720.0, config);

        // Solid yellow line at ego center (class_id=5 = solid_single_yellow)
        let marking = make_marking(5, "solid_single_yellow", 640.0, 620.0, 660.0);

        // Frame 1: Not confirmed yet (solid needs 3 frames)
        let r1 = detector.update(&[marking.clone()], Some(300.0), Some(900.0), 1, 33.0);
        assert!(
            r1.is_none(),
            "Solid marking should NOT confirm on first frame"
        );

        // Frame 2: Still not confirmed
        let r2 = detector.update(&[marking.clone()], Some(300.0), Some(900.0), 2, 66.0);
        assert!(
            r2.is_none(),
            "Solid marking should NOT confirm on second frame"
        );

        // Frame 3: NOW confirmed
        let r3 = detector.update(&[marking.clone()], Some(300.0), Some(900.0), 3, 99.0);
        assert!(r3.is_some(), "Solid marking should confirm on third frame");
    }

    #[test]
    fn test_grazing_contact_suppressed() {
        // v6.1: Low-penetration crossings should be filtered
        let config = CrossingDetectorConfig {
            min_penetration_ratio: 0.12,
            min_overlap_frames: 1,
            cooldown_frames: 5,
            clear_frames_required: 0,
            ..CrossingDetectorConfig::default()
        };
        let mut detector = LineCrossingDetector::new(1280.0, 720.0, config);

        // Place a marking barely touching the ego zone (narrow overlap)
        // ego zone: 600..680, marking: 675..695 → overlap = 5px, width = 20px, pen = 0.25 → passes
        let marking_pass = make_marking(5, "solid_single_yellow", 685.0, 675.0, 695.0);
        let r = detector.update(&[marking_pass], Some(300.0), Some(900.0), 1, 33.0);
        // pen = 5/20 = 0.25, above threshold → should eventually confirm

        // Now test one that grazes: ego zone 600..680, marking 678..682 → overlap=2, width=4, pen=0.5
        // But let's test the inverse: marking just barely overlapping
        // ego zone 600..680, marking: 679..699 → overlap = 1px, width = 20, pen = 0.05 → suppressed
        let mut detector2 = LineCrossingDetector::new(
            1280.0,
            720.0,
            CrossingDetectorConfig {
                min_penetration_ratio: 0.12,
                min_overlap_frames: 1,
                cooldown_frames: 5,
                clear_frames_required: 0,
                ..CrossingDetectorConfig::default()
            },
        );
        let marking_graze = make_marking(5, "solid_single_yellow", 689.0, 679.0, 699.0);
        // This marking is outside the ego zone (ego_zone_right = 640+40 = 680, marking starts at 679)
        // overlap = min(680,699) - max(600,679) = 680 - 679 = 1px, pen = 1/20 = 0.05 → below 0.12
        let r2 = detector2.update(&[marking_graze], Some(300.0), Some(900.0), 1, 33.0);
        assert!(r2.is_none()); // pen < 0.12 → suppressed
    }

    #[test]
    fn test_per_side_debounce() {
        // v6.1: After firing on RIGHT, require clear_frames_required frames before re-fire
        let config = CrossingDetectorConfig {
            ego_half_width_px: 40.0,
            reference_y_ratio: 0.82,
            min_overlap_frames: 1,
            cooldown_frames: 2, // short cooldown so debounce is the gate
            min_marking_confidence: 0.20,
            min_penetration_ratio: 0.0,
            clear_frames_required: 5,
        };
        let mut detector = LineCrossingDetector::new(1280.0, 720.0, config);

        let marking = make_marking(9, "dashed_single_white", 640.0, 620.0, 660.0);

        // Frame 1: first fire (allowed — first time on this side)
        let r1 = detector.update(&[marking.clone()], Some(300.0), Some(900.0), 1, 33.0);
        assert!(r1.is_some(), "First fire should always succeed");

        // Frames 2-3: cooldown
        let r2 = detector.update(&[marking.clone()], Some(300.0), Some(900.0), 2, 66.0);
        assert!(r2.is_none());
        let r3 = detector.update(&[marking.clone()], Some(300.0), Some(900.0), 3, 99.0);
        assert!(r3.is_none());

        // Frames 4-6: overlap again but debounce should block (only 0 clear frames)
        let r4 = detector.update(&[marking.clone()], Some(300.0), Some(900.0), 4, 132.0);
        assert!(r4.is_none(), "Debounce should block re-fire");

        // Frames 7-12: clear (no marking)
        for i in 7..12 {
            detector.update(&[], Some(300.0), Some(900.0), i, i as f64 * 33.0);
        }

        // Frame 12+: overlap again — debounce clear (5+ clear frames)
        let r_final = detector.update(&[marking.clone()], Some(300.0), Some(900.0), 12, 396.0);
        assert!(
            r_final.is_some(),
            "After debounce clear period, should fire again"
        );
    }

    // ---- Line Role Classification ----

    #[test]
    fn test_classify_left_boundary() {
        let role = classify_line_role(300.0, 640.0, Some(300.0), Some(900.0), 1280.0);
        assert_eq!(role, LineRole::LeftBoundary);
    }

    #[test]
    fn test_classify_right_boundary() {
        let role = classify_line_role(900.0, 640.0, Some(300.0), Some(900.0), 1280.0);
        assert_eq!(role, LineRole::RightBoundary);
    }

    #[test]
    fn test_classify_center_line() {
        let role = classify_line_role(640.0, 640.0, None, None, 1280.0);
        assert_eq!(role, LineRole::CenterLine);
    }
}

