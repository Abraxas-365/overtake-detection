// src/analysis/polynomial_tracker_integration.rs
//
// v5.0: Wiring guide — how PolynomialBoundaryTracker integrates with
//       the existing pipeline stages.
//
// ════════════════════════════════════════════════════════════════════════════
// ARCHITECTURE CHANGE
// ════════════════════════════════════════════════════════════════════════════
//
// BEFORE (v4.x):
//   LaneLegalityDetector → (left_x, right_x, confidence)
//       ↓
//   DetectionCache → CachedBoundaryResult (scalar positions, ego-compensated)
//       ↓
//   LateralShiftDetector → LaneMeasurement { lateral_offset_px, lane_width_px }
//
// AFTER (v5.0):
//   LaneLegalityDetector → (left_x, right_x, confidence) + CurvatureEstimate
//       ↓
//   PolynomialBoundaryTracker → smoothed/predicted polynomials + geometric signals
//       ↓
//   DetectionCache (unchanged — still provides markings for crossing detector)
//       ↓
//   LateralShiftDetector → LaneMeasurement { ..., geometric_signals }
//
// The PolynomialBoundaryTracker does NOT replace DetectionCache.
// DetectionCache still manages markings (MarkingInfo) for the crossing detector
// and road classifier. The polynomial tracker provides:
//   1. Better boundary positions (curve-aware prediction vs scalar+ego)
//   2. Geometric lane change signals (new data for lateral_detector)
//   3. Innovation-based lane change detection on recovery from dropout
//
// ════════════════════════════════════════════════════════════════════════════

use super::curvature_estimator::CurvatureEstimate;
use super::lateral_detector::LaneMeasurement;
use super::polynomial_tracker::{
    GeometricLaneChangeSignals, PolynomialBoundaryTracker, PolynomialTrackerConfig,
};
use tracing::debug;

// ============================================================================
// EXTENDED LANE MEASUREMENT
// ============================================================================

/// Extended LaneMeasurement with geometric signals from the polynomial tracker.
///
/// This wraps the existing LaneMeasurement and adds the new fields.
/// Use this during the transition period — eventually the geometric fields
/// should be merged into LaneMeasurement directly.
#[derive(Debug, Clone)]
pub struct EnhancedLaneMeasurement {
    /// Original lane measurement (offset, width, confidence, coherence, curvature)
    pub base: LaneMeasurement,

    /// Geometric lane change signals from polynomial tracker
    pub geometric: GeometricLaneChangeSignals,

    /// Whether the boundary positions came from the polynomial tracker's
    /// prediction (true) vs fresh YOLO detection (false).
    pub is_predicted: bool,

    /// Confidence of the polynomial tracker's estimate.
    /// Based on covariance growth — more principled than DetectionCache's
    /// fixed exponential decay.
    pub tracker_confidence: f32,
}

// ============================================================================
// PIPELINE INTEGRATION FUNCTIONS
// ============================================================================

/// Create an enhanced measurement by combining YOLO output with polynomial tracker.
///
/// Call this in STAGE 2 of the pipeline (after LaneLegalityDetector), before
/// feeding into STAGE 3 (maneuver detection).
///
/// # Arguments
/// * `tracker` — The PolynomialBoundaryTracker (mutable, updates every frame)
/// * `curvature` — Curvature estimate from lane_legality.curvature_estimate()
/// * `yolo_left_x` — Left boundary x from YOLO (None if not detected)
/// * `yolo_right_x` — Right boundary x from YOLO (None if not detected)
/// * `yolo_confidence` — Detection confidence from YOLO
/// * `both_detected` — Whether YOLO detected both boundaries
/// * `ego_lateral_velocity_px` — From ego_motion estimator (px/frame)
/// * `ego_x` — Ego vehicle x position (typically frame_width / 2)
/// * `boundary_coherence` — From lane_legality.boundary_coherence()
///
/// # Returns
/// - Enhanced boundary positions (smoothed or predicted via Kalman filter)
/// - Geometric lane change signals for the lateral detector
pub fn update_tracker_and_build_measurement(
    tracker: &mut PolynomialBoundaryTracker,
    curvature: Option<&CurvatureEstimate>,
    yolo_left_x: Option<f32>,
    yolo_right_x: Option<f32>,
    yolo_confidence: f32,
    both_detected: bool,
    ego_lateral_velocity_px: f32,
    ego_x: f32,
    boundary_coherence: f32,
) -> Option<EnhancedLaneMeasurement> {
    // Extract polynomial fits from the curvature estimate (if available)
    let left_poly = curvature.and_then(|c| c.left_poly.as_ref());
    let right_poly = curvature.and_then(|c| c.right_poly.as_ref());

    // Update the polynomial tracker
    let signals = tracker
        .update(left_poly, right_poly, ego_lateral_velocity_px, ego_x)
        .clone();

    // Determine effective boundary positions:
    // Prefer polynomial tracker (curve-aware) over raw YOLO scalars.
    // Fall back to YOLO when tracker isn't initialized.
    let (eff_left_x, eff_right_x, is_predicted) = if tracker.both_active() {
        let ref_y = 0.82; // near-hood reference y (same as crossing detector)
        (
            tracker.left_x_at(ref_y),
            tracker.right_x_at(ref_y),
            !matches!(
                (tracker.left_state(), tracker.right_state()),
                (
                    super::polynomial_tracker::BoundaryState::Tracking,
                    super::polynomial_tracker::BoundaryState::Tracking,
                )
            ),
        )
    } else if let (Some(lx), Some(rx)) = (yolo_left_x, yolo_right_x) {
        (lx, rx, false)
    } else {
        return None; // No usable boundary data from either source
    };

    let lane_width = (eff_right_x - eff_left_x).max(1.0);
    let lane_center = (eff_left_x + eff_right_x) / 2.0;
    let lateral_offset = ego_x - lane_center;

    // Effective confidence: blend YOLO confidence with tracker confidence
    let tracker_conf = tracker.confidence();
    let eff_confidence = if is_predicted {
        // During prediction: use tracker confidence (decays with covariance)
        tracker_conf * yolo_confidence.max(0.3)
    } else {
        // During tracking: boost confidence (smoothed estimate is better than raw)
        (yolo_confidence * 0.7 + tracker_conf * 0.3).min(1.0)
    };

    let base = LaneMeasurement {
        lateral_offset_px: lateral_offset,
        lane_width_px: lane_width,
        confidence: eff_confidence,
        both_lanes: both_detected || tracker.both_active(),
        boundary_coherence,
        curvature: curvature.cloned(),
    };

    Some(EnhancedLaneMeasurement {
        base,
        geometric: signals,
        is_predicted,
        tracker_confidence: tracker_conf,
    })
}

// ============================================================================
// LATERAL DETECTOR INTEGRATION
// ============================================================================
//
// To integrate geometric signals into LateralShiftDetector's shift-start logic,
// modify the `handle_with_lanes` method. Here's the conceptual change:
//
// BEFORE (v4.13):
//   let shift_triggered = abs_dev > eff_start;
//
// AFTER (v5.0):
//   let offset_signal = abs_dev > eff_start;
//   let geometric_signal = enhanced.geometric.suggests_lane_change
//       && enhanced.geometric.confidence > 0.3;
//
//   let shift_triggered = if self.in_curve_mode {
//       // On curves: require geometric confirmation to prevent false positives
//       offset_signal && geometric_signal
//   } else {
//       // On straights: offset alone is sufficient, or strong geometric signal
//       // with lower offset threshold for faster detection
//       offset_signal || (geometric_signal && abs_dev > eff_start * 0.5)
//   };
//
// Also modify shift confirmation:
//
// BEFORE:
//   let confirmed = abs_dev > self.config.shift_confirm_threshold;
//
// AFTER:
//   let confirmed = abs_dev > self.config.shift_confirm_threshold
//       || (abs_dev > self.config.shift_confirm_threshold * 0.7
//           && enhanced.geometric.suggests_lane_change);
//
// And for the ego-preempt path (v4.10), geometric signals can serve as
// additional corroboration:
//
//   if ego_preempt_conditions_met && enhanced.geometric.suggests_lane_change {
//       // Strong evidence: ego motion + geometric divergence
//       // Can use lower ego velocity threshold
//   }
//
// ════════════════════════════════════════════════════════════════════════════

// ============================================================================
// EXAMPLE: How to add to ManeuverPipeline
// ============================================================================
//
// In src/analysis/maneuver_pipeline.rs:
//
// 1. Add field to ManeuverPipeline:
//
//     pub struct ManeuverPipeline {
//         // ... existing fields ...
//         poly_tracker: PolynomialBoundaryTracker,
//     }
//
// 2. Initialize in constructor:
//
//     impl ManeuverPipeline {
//         pub fn new(frame_w: f32, frame_h: f32) -> Self {
//             Self {
//                 // ... existing ...
//                 poly_tracker: PolynomialBoundaryTracker::new(
//                     PolynomialTrackerConfig::default()
//                 ),
//             }
//         }
//     }
//
// 3. In process_frame(), after ego-motion estimation but before lateral detection:
//
//     // Feed curvature estimate from lane_legality to polynomial tracker
//     let curvature = input.curvature_estimate; // from LaneLegalityDetector
//     let left_poly = curvature.and_then(|c| c.left_poly.as_ref());
//     let right_poly = curvature.and_then(|c| c.right_poly.as_ref());
//
//     let geo_signals = self.poly_tracker.update(
//         left_poly,
//         right_poly,
//         self.last_ego_estimate.lateral_velocity_px,
//         frame_w / 2.0,
//     );
//
// 4. Inject geometric signals into LaneMeasurement before feeding to lateral_detector:
//
//     // If polynomial tracker has better positions, use them
//     if self.poly_tracker.both_active() {
//         if let Some(ref mut meas) = input.lane_measurement {
//             let ref_y = 0.82;
//             let tracked_left = self.poly_tracker.left_x_at(ref_y);
//             let tracked_right = self.poly_tracker.right_x_at(ref_y);
//             let tracked_width = tracked_right - tracked_left;
//             if tracked_width > 50.0 {
//                 let ego_x = frame_w / 2.0;
//                 meas.lateral_offset_px = ego_x - (tracked_left + tracked_right) / 2.0;
//                 meas.lane_width_px = tracked_width;
//                 // Blend confidence
//                 meas.confidence = meas.confidence.max(self.poly_tracker.confidence());
//             }
//         }
//     }
//
// ════════════════════════════════════════════════════════════════════════════

// ============================================================================
// EXAMPLE: Innovation-based lane change detection after dropout
// ============================================================================
//
// When the polynomial tracker transitions from Predicting back to Tracking,
// the innovation tells you what happened during the gap:
//
//     let left_innov = self.poly_tracker.left_innovation();
//     let right_innov = self.poly_tracker.right_innovation();
//
//     // Asymmetric innovation = lane change happened during dropout
//     let innovation_asymmetry = (left_innov - right_innov).abs();
//     let total_innovation = left_innov + right_innov;
//
//     if innovation_asymmetry > 20.0 && total_innovation > 30.0 {
//         // Strong evidence of lane change during dropout
//         // Can emit a synthetic shift event or boost confidence of
//         // any ego-bridged shift that was in progress
//         info!(
//             "📐 Innovation asymmetry after dropout: L={:.1} R={:.1} asym={:.1}",
//             left_innov, right_innov, innovation_asymmetry,
//         );
//     }
//
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::super::curvature_estimator::{CurvatureEstimate, CurveDirection, LanePolynomial};
    use super::*;

    fn make_curvature(left_a: f32, left_c: f32, right_a: f32, right_c: f32) -> CurvatureEstimate {
        CurvatureEstimate {
            left_poly: Some(LanePolynomial {
                a: left_a,
                b: 0.0,
                c: left_c,
                rmse_px: 3.0,
                num_points: 30,
                vertical_span_px: 300.0,
            }),
            right_poly: Some(LanePolynomial {
                a: right_a,
                b: 0.0,
                c: right_c,
                rmse_px: 3.0,
                num_points: 30,
                vertical_span_px: 300.0,
            }),
            mean_curvature: (left_a + right_a) / 2.0,
            curvature_agreement: 0.9,
            is_curve: left_a.abs() > 1.0,
            curve_direction: CurveDirection::Straight,
            confidence: 0.8,
        }
    }

    #[test]
    fn test_full_integration_straight_road() {
        let mut tracker = PolynomialBoundaryTracker::new(PolynomialTrackerConfig::default());
        let ego_x = 640.0;

        // 50 frames of straight road
        for _ in 0..50 {
            let curv = make_curvature(0.0, 400.0, 0.0, 880.0);
            let result = update_tracker_and_build_measurement(
                &mut tracker,
                Some(&curv),
                Some(400.0),
                Some(880.0),
                0.85,
                true,
                0.0,
                ego_x,
                0.0,
            );

            assert!(result.is_some());
            let r = result.unwrap();
            assert!(!r.is_predicted);
            assert!(!r.geometric.suggests_lane_change);
        }
    }

    #[test]
    fn test_dropout_maintains_shape() {
        let mut tracker = PolynomialBoundaryTracker::new(PolynomialTrackerConfig::default());
        let ego_x = 640.0;

        // Stabilize with curved road
        for _ in 0..30 {
            let curv = make_curvature(15.0, 350.0, 15.0, 900.0);
            update_tracker_and_build_measurement(
                &mut tracker,
                Some(&curv),
                Some(350.0),
                Some(900.0),
                0.85,
                true,
                0.0,
                ego_x,
                0.7,
            );
        }

        // Now dropout for 10 frames — tracker should predict
        for _ in 0..10 {
            let result = update_tracker_and_build_measurement(
                &mut tracker,
                None,
                None,
                None,
                0.0,
                false,
                0.0,
                ego_x,
                -1.0,
            );

            assert!(
                result.is_some(),
                "Tracker should provide predicted boundaries"
            );
            let r = result.unwrap();
            assert!(r.is_predicted);
            assert!(r.tracker_confidence > 0.0);
            assert!(
                r.base.lane_width_px > 100.0,
                "Lane width should be reasonable"
            );
        }
    }

    #[test]
    fn test_curve_does_not_trigger_lane_change() {
        let mut tracker = PolynomialBoundaryTracker::new(PolynomialTrackerConfig::default());
        let ego_x = 640.0;

        // Both boundaries curving identically (road curve)
        for _ in 0..50 {
            let curv = make_curvature(20.0, 400.0, 20.0, 880.0);
            update_tracker_and_build_measurement(
                &mut tracker,
                Some(&curv),
                Some(400.0),
                Some(880.0),
                0.85,
                true,
                0.0,
                ego_x,
                0.8,
            );
        }

        let signals = tracker.signals();
        assert!(
            !signals.suggests_lane_change,
            "Road curve should NOT suggest lane change"
        );
    }
}

