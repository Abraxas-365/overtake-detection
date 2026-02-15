// src/lane_crossing_integration.rs
//
// v5.2: Wiring guide — how LineCrossingDetector + DetectionCache integrate
//       with the existing pipeline stages in main.rs
//
// This module provides the glue functions that bridge the new lane_crossing
// module with the existing pipeline architecture.
//
// ════════════════════════════════════════════════════════════════════════════
// ARCHITECTURE OVERVIEW
// ════════════════════════════════════════════════════════════════════════════
//
// Existing pipeline (main.rs):
//   STAGE 1: Vehicle Detection (YoloDetector → vehicle bboxes)
//   STAGE 2: Lane Detection (LaneLegalityDetector → ego boundaries)
//   STAGE 3: Maneuver Detection v2 (LateralShiftDetector + PassDetector + Classifier)
//   STAGE 4: HUD / annotation
//
// New components:
//   DetectionCache sits between STAGE 2 and STAGE 3:
//     - Receives fresh detections from STAGE 2
//     - Provides ego-compensated boundaries when STAGE 2 returns nothing
//     - Reports "unmarked road" (trocha) when cache expires
//
//   LineCrossingDetector runs in STAGE 3 alongside LateralShiftDetector:
//     - Gets markings from the DetectionCache (fresh or cached)
//     - Emits LineCrossingEvent when ego physically drives over a line
//     - These events are fused with LateralShiftEvents for confirmation
//
// Data flow:
//   Frame → YOLO-seg → [boundaries + markings]
//          ↓
//   DetectionCache.update_fresh(boundaries + markings)
//          ↓
//   DetectionCache.get_boundaries() → CachedBoundaryResult
//          ↓                              ↓
//   LateralShiftDetector (offset)    LineCrossingDetector (overlap)
//          ↓                              ↓
//   ManeuverClassifier ←←←← crossing events as corroborating signal

use crate::lane_crossing::{
    CacheState, CachedBoundaryResult, CachedLaneBoundaries, CrossingDetectorConfig, DetectionCache,
    DetectionCacheConfig, LineCrossingDetector, LineCrossingEvent,
};
use crate::road_classification::MarkingInfo;
use tracing::{debug, info};

// ============================================================================
// PIPELINE EXTENSION STATE
// ============================================================================

/// Holds the new lane crossing / cache state alongside the existing pipeline.
/// Add this as a field in PipelineState in main.rs.
pub struct LaneCrossingState {
    /// The YOLO detection frame cache
    pub detection_cache: DetectionCache,
    /// The direct line crossing detector
    pub crossing_detector: LineCrossingDetector,
    /// Last crossing event (for HUD display)
    pub last_crossing_event: Option<LineCrossingEvent>,
    /// Running count of crossing events
    pub crossing_event_count: u64,
    /// Running count of illegal crossings
    pub illegal_crossing_count: u64,
}

impl LaneCrossingState {
    pub fn new(frame_width: f32, frame_height: f32) -> Self {
        Self {
            detection_cache: DetectionCache::new(DetectionCacheConfig::default()),
            crossing_detector: LineCrossingDetector::new(
                frame_width,
                frame_height,
                CrossingDetectorConfig::default(),
            ),
            last_crossing_event: None,
            crossing_event_count: 0,
            illegal_crossing_count: 0,
        }
    }

    pub fn with_config(
        frame_width: f32,
        frame_height: f32,
        cache_config: DetectionCacheConfig,
        crossing_config: CrossingDetectorConfig,
    ) -> Self {
        Self {
            detection_cache: DetectionCache::new(cache_config),
            crossing_detector: LineCrossingDetector::new(
                frame_width,
                frame_height,
                crossing_config,
            ),
            last_crossing_event: None,
            crossing_event_count: 0,
            illegal_crossing_count: 0,
        }
    }
}

// ============================================================================
// PIPELINE INTEGRATION FUNCTIONS
// ============================================================================

/// Call this in STAGE 2 (lane detection), right after the YOLO-seg model runs.
///
/// This replaces the direct use of `estimate_ego_lane_boundaries_stable` output
/// with a cache-aware version. When YOLO finds lanes, it's stored fresh. When
/// YOLO fails, the cache provides ego-compensated boundaries.
///
/// Returns: (left_x, right_x, confidence, cache_state)
///
/// # Example usage in main.rs:
/// ```rust,ignore
/// // In run_lane_detection():
/// let yolo_result = detector.estimate_ego_lane_boundaries_stable(...)?;
///
/// // Feed into cache
/// let (left_x, right_x, conf, cache_state) = update_lane_cache(
///     &mut ps.lane_crossing_state,
///     yolo_result,
///     &all_markings,     // from detector.last_lane_markings()
///     ego_lateral_vel,   // from ego motion estimator
///     frame_count,
///     timestamp_ms,
/// );
///
/// // Use (left_x, right_x, conf) for lane measurement construction
/// // even when yolo_result was None, the cache may provide data
/// ```
pub fn update_lane_cache(
    state: &mut LaneCrossingState,
    yolo_boundaries: Option<(f32, f32, f32)>, // (left_x, right_x, confidence) from YOLO
    markings: &[MarkingInfo],
    ego_lateral_velocity_px: f32,
    frame_id: u64,
    timestamp_ms: f64,
) -> Option<(f32, f32, f32, CacheState)> {
    // 1. Feed ego motion into cache
    state
        .detection_cache
        .accumulate_ego_motion(ego_lateral_velocity_px);

    // 2. Feed fresh detection (or None) into cache
    let fresh = yolo_boundaries.map(|(left_x, right_x, conf)| CachedLaneBoundaries {
        left_x,
        right_x,
        original_confidence: conf,
        both_detected: true, // YOLO always finds both when it returns Some
        markings: markings.to_vec(),
        captured_frame_id: frame_id,
        captured_timestamp_ms: timestamp_ms,
    });
    state.detection_cache.update_fresh(fresh);

    // 3. Query cache for effective boundaries
    let result = state.detection_cache.get_boundaries()?;

    if result.state == CacheState::Cached {
        debug!(
            "📡 Using cached lane boundaries (stale={}f, conf={:.2}): L={:.0} R={:.0}",
            result.stale_frames, result.confidence, result.left_x, result.right_x,
        );
    }

    Some((
        result.left_x,
        result.right_x,
        result.confidence,
        result.state,
    ))
}

/// Call this in STAGE 3 (maneuver detection), every frame.
///
/// Runs the line crossing detector on the current markings (fresh or cached).
/// Returns a crossing event if the ego vehicle is confirmed to be driving over a line.
///
/// # Example usage in main.rs:
/// ```rust,ignore
/// // In run_maneuver_detection():
/// let crossing_event = run_crossing_detection(
///     &mut ps.lane_crossing_state,
///     ego_left_x,   // from cache or fresh detection
///     ego_right_x,
///     frame_count,
///     timestamp_ms,
/// );
///
/// if let Some(ref event) = crossing_event {
///     // Inject as corroborating evidence into ManeuverPipeline
///     // e.g., if event.line_role == LeftBoundary && event.crossing_direction == Leftward
///     //       then this strongly confirms a leftward lane change
/// }
/// ```
pub fn run_crossing_detection(
    state: &mut LaneCrossingState,
    ego_left_x: Option<f32>,
    ego_right_x: Option<f32>,
    frame_id: u64,
    timestamp_ms: f64,
) -> Option<LineCrossingEvent> {
    // Get markings from the cache (these might be cached markings from a recent frame)
    let markings = if let Some(boundaries) = state.detection_cache.get_boundaries() {
        boundaries.markings
    } else {
        Vec::new()
    };

    let event =
        state
            .crossing_detector
            .update(&markings, ego_left_x, ego_right_x, frame_id, timestamp_ms);

    if let Some(ref e) = event {
        state.last_crossing_event = Some(e.clone());
        state.crossing_event_count += 1;
        if matches!(
            e.passing_legality,
            crate::road_classification::PassingLegality::Prohibited
        ) {
            state.illegal_crossing_count += 1;
        }
    }

    event
}

/// Generate diagnostic text for HUD overlay.
pub fn cache_status_text(state: &LaneCrossingState) -> String {
    let cache = &state.detection_cache;
    let cache_state = cache.state();

    match cache_state {
        CacheState::Fresh => "LANE: LIVE".to_string(),
        CacheState::Cached => format!("LANE: CACHED ({}f)", cache.stale_frames()),
        CacheState::Expired => "LANE: TROCHA/UNMARKED".to_string(),
        CacheState::Empty => "LANE: NO DATA".to_string(),
    }
}

/// Generate crossing alert text for HUD overlay.
pub fn crossing_alert_text(state: &LaneCrossingState) -> Option<String> {
    let event = state.last_crossing_event.as_ref()?;

    Some(format!(
        "⚠ CROSSING {} ({}) → {}",
        event.line_role.as_str(),
        event.marking_class,
        event.passing_legality.as_str(),
    ))
}

// ============================================================================
// FUSION: CROSSING + LATERAL SHIFT AGREEMENT
// ============================================================================

/// Evidence from the crossing detector to feed into the maneuver classifier.
///
/// When a line crossing is detected, it provides strong directional evidence
/// about whether a lane change is happening and in which direction.
#[derive(Debug, Clone)]
pub struct CrossingEvidence {
    /// Direction the vehicle is crossing (leftward/rightward)
    pub direction: crate::lane_crossing::CrossingDirection,
    /// Which boundary was crossed
    pub boundary: crate::lane_crossing::LineRole,
    /// Confidence of the crossing detection
    pub confidence: f32,
    /// Legality of crossing this specific line
    pub legality: crate::road_classification::PassingLegality,
    /// How recent the crossing is (frames ago)
    pub age_frames: u32,
}

/// Check if a crossing event corroborates a lateral shift event.
///
/// Returns a confidence boost [0.0, 0.3] if crossing + shift agree on direction.
/// Returns 0.0 if no crossing is active or directions don't match.
///
/// This is meant to be called from the ManeuverClassifier when evaluating
/// a LateralShiftEvent to decide if it's a real lane change.
pub fn crossing_corroboration(
    crossing: Option<&LineCrossingEvent>,
    shift_direction_is_left: bool,
    current_frame_id: u64,
    max_age_frames: u64,
) -> f32 {
    let crossing = match crossing {
        Some(c) => c,
        None => return 0.0,
    };

    // Check recency
    if current_frame_id.saturating_sub(crossing.frame_id) > max_age_frames {
        return 0.0;
    }

    // Check direction agreement
    let crossing_is_left = matches!(
        crossing.crossing_direction,
        crate::lane_crossing::CrossingDirection::Leftward
    );

    if crossing_is_left == shift_direction_is_left {
        // Direction agrees — strong corroboration
        let base_boost = 0.20;
        // Bonus for high penetration (vehicle deeply on the line)
        let penetration_bonus = crossing.penetration_ratio * 0.10;
        base_boost + penetration_bonus
    } else if matches!(
        crossing.crossing_direction,
        crate::lane_crossing::CrossingDirection::Unknown
    ) {
        // Direction unknown but we know a line was crossed — moderate corroboration
        0.10
    } else {
        // Direction disagrees — might be noise or a complex maneuver
        0.0
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lane_crossing::{CrossingDirection, LineRole, PassingLegality};

    #[test]
    fn test_crossing_corroboration_agreement() {
        let crossing = LineCrossingEvent {
            line_role: LineRole::LeftBoundary,
            marking_class: "solid_single_yellow".to_string(),
            marking_class_id: 5,
            passing_legality: crate::road_classification::PassingLegality::Prohibited,
            crossing_direction: CrossingDirection::Leftward,
            confidence: 0.85,
            penetration_ratio: 0.6,
            frame_id: 100,
            timestamp_ms: 3333.0,
        };

        // Crossing left + shift left → agree
        let boost = crossing_corroboration(Some(&crossing), true, 105, 30);
        assert!(boost > 0.15, "boost={}", boost);

        // Crossing left + shift right → disagree
        let boost = crossing_corroboration(Some(&crossing), false, 105, 30);
        assert_eq!(boost, 0.0);
    }

    #[test]
    fn test_crossing_corroboration_expired() {
        let crossing = LineCrossingEvent {
            line_role: LineRole::LeftBoundary,
            marking_class: "solid_single_yellow".to_string(),
            marking_class_id: 5,
            passing_legality: crate::road_classification::PassingLegality::Prohibited,
            crossing_direction: CrossingDirection::Leftward,
            confidence: 0.85,
            penetration_ratio: 0.6,
            frame_id: 100,
            timestamp_ms: 3333.0,
        };

        // Too old → no boost
        let boost = crossing_corroboration(Some(&crossing), true, 200, 30);
        assert_eq!(boost, 0.0);
    }

    #[test]
    fn test_cache_status_text() {
        let state = LaneCrossingState::new(1280.0, 720.0);
        assert_eq!(cache_status_text(&state), "LANE: NO DATA");
    }
}
