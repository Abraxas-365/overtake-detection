// src/analysis/maneuver_pipeline.rs
//
// Orchestrator that wires together the vehicle tracker, pass detector,
// lateral detector, ego-motion estimator, and maneuver classifier.
//
// Single entry point: call process_frame() each frame.
// Replaces LaneChangeAnalyzer + LaneChangeStateMachine.
//
// INTEGRATION NOTE: The pipeline does NOT own the LegalityRingBuffer.
// The existing PipelineState already populates it from YOLOv8-seg.
// We receive it as a reference in ManeuverFrameInput so the classifier
// can look up temporally-correct legality at the actual crossing frame.
//
// v4.4: Reordered steps so ego-motion is computed BEFORE lateral detection.
//       This allows the lateral detector to use ego motion for bridging
//       through lane detection dropout.
//
// v5.0: Added PolynomialBoundaryTracker between ego-motion and lateral
//       detection. Tracks polynomial lane coefficients via Kalman filter
//       for curve-aware prediction during dropout and geometric lane
//       change signals that discriminate curves from lane changes.

use super::ego_motion::{EgoMotionConfig, EgoMotionEstimate, EgoMotionEstimator, GrayFrame};
use super::lateral_detector::{
    EgoMotionInput, LaneMeasurement, LateralDetectorConfig, LateralShiftDetector,
};
use super::maneuver_classifier::{
    ClassifierConfig, ManeuverClassifier, ManeuverEvent, MarkingSnapshot,
    RoadClassificationSnapshot,
};
use super::pass_detector::{PassDetector, PassDetectorConfig};
use super::polynomial_tracker::{
    GeometricLaneChangeSignals, PolynomialBoundaryTracker, PolynomialTrackerConfig,
};
use super::vehicle_tracker::{DetectionInput, Track, TrackerConfig, VehicleTracker};
use crate::analysis::curvature_estimator::LanePolynomial;
use crate::pipeline::legality_buffer::LegalityRingBuffer;
use tracing::{debug, info, warn};

// ============================================================================
// INPUT / OUTPUT
// ============================================================================

pub struct ManeuverFrameInput<'a> {
    pub vehicle_detections: &'a [DetectionInput],
    pub lane_measurement: Option<LaneMeasurement>,
    pub gray_frame: Option<&'a GrayFrame>,
    pub left_marking_name: Option<&'a str>,
    pub right_marking_name: Option<&'a str>,
    /// Reference to the existing legality ring buffer populated by the main pipeline.
    /// Pass None if legality detection is disabled or unavailable.
    pub legality_buffer: Option<&'a LegalityRingBuffer>,
    /// v8.0: Road classification snapshot from the RoadClassifier temporal consensus.
    /// Provides the current center line type and passing legality.
    pub road_classification: Option<RoadClassificationSnapshot>,
    pub timestamp_ms: f64,
    pub frame_id: u64,
}

#[derive(Debug, Clone)]
pub struct ManeuverFrameOutput {
    pub maneuver_events: Vec<ManeuverEvent>,
    pub tracked_vehicle_count: usize,
    pub pass_in_progress: bool,
    pub shift_in_progress: bool,
    pub ego_lateral_velocity: f32,
    pub lateral_state: String,
    /// v5.0: Geometric lane change signals from the polynomial tracker.
    /// Available when both lane boundaries are being tracked.
    pub geometric_signals: Option<GeometricLaneChangeSignals>,
}

// ============================================================================
// CONFIGURATION
// ============================================================================

#[derive(Debug, Clone)]
pub struct ManeuverPipelineConfig {
    pub tracker: TrackerConfig,
    pub pass_detector: PassDetectorConfig,
    pub lateral_detector: LateralDetectorConfig,
    pub ego_motion: EgoMotionConfig,
    pub classifier: ClassifierConfig,
    pub enable_ego_motion: bool,
    /// v5.0: Polynomial boundary tracker configuration.
    pub poly_tracker: PolynomialTrackerConfig,
    /// v5.0: Whether to use polynomial tracker to enhance lane measurements.
    /// When false, the tracker still runs (for signals) but doesn't override
    /// the lane measurement positions. Useful for A/B comparison.
    pub poly_tracker_override_positions: bool,
}

impl Default for ManeuverPipelineConfig {
    fn default() -> Self {
        Self {
            tracker: TrackerConfig::default(),
            pass_detector: PassDetectorConfig::default(),
            lateral_detector: LateralDetectorConfig::default(),
            ego_motion: EgoMotionConfig::default(),
            classifier: ClassifierConfig::default(),
            enable_ego_motion: true,
            poly_tracker: PolynomialTrackerConfig::default(),
            poly_tracker_override_positions: true,
        }
    }
}

impl ManeuverPipelineConfig {
    pub fn mining_route() -> Self {
        Self {
            tracker: TrackerConfig {
                max_coast_frames: 120,
                min_confidence: 0.12,
                min_iou: 0.05,
                ..TrackerConfig::default()
            },
            pass_detector: PassDetectorConfig {
                min_beside_duration_ms: 300.0,
                max_pass_duration_ms: 90000.0,
                min_beside_frames: 5,
                disappearance_grace_frames: 90,
                ..PassDetectorConfig::default()
            },
            lateral_detector: LateralDetectorConfig {
                min_lane_confidence: 0.20,
                shift_start_threshold: 0.35,
                shift_confirm_threshold: 0.50,
                shift_end_threshold: 0.20,
                min_shift_frames: 15,
                baseline_alpha_stable: 0.002,
                baseline_warmup_frames: 25,
                occlusion_reset_frames: 60,
                post_reset_freeze_frames: 60,
                // v4.4: ego-motion fusion for mining
                ego_motion_min_velocity: 1.5,
                ego_shift_start_frames: 8,
                ego_bridge_max_frames: 120,
                ego_px_per_norm_unit: 800.0,
                ego_only_confidence_penalty: 0.20,
                ego_shift_max_frames: 150,
                ..LateralDetectorConfig::default()
            },
            ego_motion: EgoMotionConfig {
                min_displacement: 1.0,
                min_consensus: 0.35,
                ..EgoMotionConfig::default()
            },
            classifier: ClassifierConfig {
                max_correlation_gap_ms: 30000.0,
                min_single_source_confidence: 0.35,
                correlation_window_ms: 40000.0,
                min_combined_confidence: 0.30,
                ..ClassifierConfig::default()
            },
            enable_ego_motion: true,
            poly_tracker: PolynomialTrackerConfig::default(),
            poly_tracker_override_positions: true,
        }
    }
}

// ============================================================================
// PIPELINE
// ============================================================================

pub struct ManeuverPipeline {
    pub tracker: VehicleTracker,
    pass_detector: PassDetector,
    lateral_detector: LateralShiftDetector,
    ego_motion: EgoMotionEstimator,
    classifier: ManeuverClassifier,
    enable_ego_motion: bool,
    frame_count: u64,
    last_ego_estimate: EgoMotionEstimate,
    last_tracked_count: usize,
    /// v5.0: Polynomial boundary tracker for curve-aware lane geometry.
    poly_tracker: PolynomialBoundaryTracker,
    /// v5.0: Whether to override lane measurement positions with tracker output.
    poly_tracker_override_positions: bool,
    /// Stored frame width for ego_x computation.
    frame_w: f32,
}

impl ManeuverPipeline {
    pub fn new(frame_w: f32, frame_h: f32) -> Self {
        Self::with_config(ManeuverPipelineConfig::default(), frame_w, frame_h)
    }

    pub fn last_ego_velocity(&self) -> Option<f32> {
        if self.last_ego_estimate.confidence > 0.0 {
            Some(self.last_ego_estimate.lateral_velocity_px)
        } else {
            None
        }
    }

    pub fn with_config(config: ManeuverPipelineConfig, frame_w: f32, frame_h: f32) -> Self {
        Self {
            tracker: VehicleTracker::new(config.tracker, frame_w, frame_h),
            pass_detector: PassDetector::new(config.pass_detector, frame_h),
            lateral_detector: LateralShiftDetector::new(config.lateral_detector),
            ego_motion: EgoMotionEstimator::new(config.ego_motion),
            classifier: ManeuverClassifier::new(config.classifier),
            enable_ego_motion: config.enable_ego_motion,
            frame_count: 0,
            last_ego_estimate: EgoMotionEstimate::none(),
            last_tracked_count: 0,
            poly_tracker: PolynomialBoundaryTracker::new(config.poly_tracker),
            poly_tracker_override_positions: config.poly_tracker_override_positions,
            frame_w,
        }
    }

    /// Process one frame through the entire pipeline.
    pub fn process_frame(&mut self, input: ManeuverFrameInput) -> ManeuverFrameOutput {
        self.frame_count += 1;

        // ══════════════════════════════════════════════════════════════════
        // DIAGNOSTIC: Count raw detections before filtering
        // ══════════════════════════════════════════════════════════════════
        let raw_det_count = input.vehicle_detections.len();

        let valid_det_count = input
            .vehicle_detections
            .iter()
            .filter(|d| {
                d.confidence >= self.tracker.config.min_confidence
                    && self.tracker.config.vehicle_class_ids.contains(&d.class_id)
            })
            .count();

        // ══════════════════════════════════════════════════════════════════
        // 1. VEHICLE TRACKING
        // ══════════════════════════════════════════════════════════════════
        self.tracker
            .update(input.vehicle_detections, input.timestamp_ms, input.frame_id);
        let tracked_count = self.tracker.confirmed_count();
        let tracks = self.tracker.confirmed_tracks();

        let track_info: Vec<String> = tracks
            .iter()
            .map(|t| format!("T{}:{}", t.id, t.zone.as_str()))
            .collect();

        let mut class_counts: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for det in input.vehicle_detections {
            *class_counts.entry(det.class_id).or_default() += 1;
        }

        // ══════════════════════════════════════════════════════════════════
        // 2. PASS DETECTION + FEED TO CLASSIFIER
        // ══════════════════════════════════════════════════════════════════
        let pass_events = self
            .pass_detector
            .update(&tracks, input.timestamp_ms, input.frame_id);

        for pass_event in pass_events {
            self.classifier.feed_pass(pass_event);
        }

        // ══════════════════════════════════════════════════════════════════
        // 3. EGO-MOTION ESTIMATION (moved BEFORE lateral detection)
        //    v4.4: Must run first so lateral detector can use ego motion
        //    to bridge through lane detection dropout.
        // ══════════════════════════════════════════════════════════════════
        let ego_velocity = if self.enable_ego_motion {
            if let Some(gray) = input.gray_frame {
                let estimate = self.ego_motion.update(gray);
                self.last_ego_estimate = estimate;

                self.classifier
                    .feed_ego_motion(estimate, input.timestamp_ms);
                estimate.lateral_velocity_px
            } else {
                0.0
            }
        } else {
            0.0
        };

        // ══════════════════════════════════════════════════════════════════
        // 3.5 POLYNOMIAL BOUNDARY TRACKER (v5.0)
        //
        //     Feeds polynomial fits from curvature_estimator (via LaneMeasurement.curvature)
        //     into Kalman filters that track lane boundary geometry.
        //
        //     Outputs:
        //       - Smoothed/predicted boundary positions (curve-aware)
        //       - Geometric lane change signals (boundary divergence, width rate, etc.)
        //       - Innovation magnitudes for dropout recovery detection
        //
        //     When poly_tracker_override_positions is true, replaces the
        //     offset/width in the LaneMeasurement with tracker output before
        //     feeding to the lateral detector. This gives:
        //       - Curve-aware prediction during YOLO dropout (vs scalar+ego)
        //       - Smoother boundaries (Kalman filtering vs raw YOLO noise)
        //       - Principled confidence decay (covariance-based vs 0.85^n)
        // ══════════════════════════════════════════════════════════════════
        let ego_x = self.frame_w / 2.0;

        // Extract polynomial fits from the curvature estimate (if present in measurement).
        // When the curvature estimator can't produce a polynomial (e.g., dashed line
        // with too few spine points), synthesize a straight-line polynomial from the
        // known boundary positions. This gives the Kalman tracker a position anchor
        // while the adaptive R scaling (12× noise for synthetic) ensures the filter
        // trusts its shape prediction over the synthetic measurement.
        let left_poly = input
            .lane_measurement
            .as_ref()
            .and_then(|m| m.curvature.as_ref())
            .and_then(|c| c.left_poly);
        let right_poly = input
            .lane_measurement
            .as_ref()
            .and_then(|m| m.curvature.as_ref())
            .and_then(|c| c.right_poly);

        // Derive boundary positions for synthetic fallback
        let (left_boundary_x, right_boundary_x) = if let Some(ref meas) = input.lane_measurement {
            if meas.both_lanes && meas.lane_width_px > 50.0 && meas.confidence > 0.3 {
                let center_x = ego_x - meas.lateral_offset_px;
                let half_w = meas.lane_width_px / 2.0;
                (Some(center_x - half_w), Some(center_x + half_w))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        // Synthesize straight-line polys for boundaries missing polynomial fits.
        // a=0, b=0, c=boundary_x | high RMSE + low points → KF adaptive R will
        // scale measurement noise 12× so the filter trusts prediction for shape.
        let effective_left = left_poly.or_else(|| {
            left_boundary_x.map(|x| LanePolynomial {
                a: 0.0,
                b: 0.0,
                c: x,
                rmse_px: 15.0, // synthetic — triggers high measurement noise
                num_points: 5, // synthetic — above min_fit_points threshold
                vertical_span_px: 200.0,
            })
        });
        let effective_right = right_poly.or_else(|| {
            right_boundary_x.map(|x| LanePolynomial {
                a: 0.0,
                b: 0.0,
                c: x,
                rmse_px: 15.0,
                num_points: 5,
                vertical_span_px: 200.0,
            })
        });

        // Update the polynomial tracker
        let geo_signals = self
            .poly_tracker
            .update(
                effective_left.as_ref(),
                effective_right.as_ref(),
                self.last_ego_estimate.lateral_velocity_px,
                ego_x,
            )
            .clone();

        // Optionally override lane measurement positions with tracker output
        let enhanced_measurement =
            if self.poly_tracker_override_positions && self.poly_tracker.both_active() {
                if let Some(meas) = input.lane_measurement {
                    let ref_y = 0.82;
                    let tracked_left = self.poly_tracker.left_x_at(ref_y);
                    let tracked_right = self.poly_tracker.right_x_at(ref_y);
                    let tracked_width = tracked_right - tracked_left;

                    if tracked_width > 50.0 {
                        // Use tracker's smoothed/predicted positions
                        let tracked_offset = ego_x - (tracked_left + tracked_right) / 2.0;
                        let tracker_conf = self.poly_tracker.confidence();

                        // Blend confidence: tracker can boost low-confidence YOLO
                        // or provide principled decay during prediction
                        let is_predicting = !matches!(
                            (
                                self.poly_tracker.left_state(),
                                self.poly_tracker.right_state()
                            ),
                            (
                                super::polynomial_tracker::BoundaryState::Tracking,
                                super::polynomial_tracker::BoundaryState::Tracking,
                            )
                        );

                        let blended_conf = if is_predicting {
                            // During prediction: use tracker's covariance-based confidence
                            tracker_conf * meas.confidence.max(0.3)
                        } else {
                            // During tracking: blend (smoothed estimate is more stable)
                            (meas.confidence * 0.7 + tracker_conf * 0.3).min(1.0)
                        };

                        if is_predicting {
                            debug!(
                                "📐 PolyTracker providing predicted boundaries: L={:.0} R={:.0} \
                             W={:.0} conf={:.2} (stale L={}f R={}f)",
                                tracked_left,
                                tracked_right,
                                tracked_width,
                                blended_conf,
                                self.poly_tracker.left_kf_stale(),
                                self.poly_tracker.right_kf_stale(),
                            );
                        }

                        Some(LaneMeasurement {
                            lateral_offset_px: tracked_offset,
                            lane_width_px: tracked_width,
                            confidence: blended_conf,
                            both_lanes: true, // tracker has both if both_active()
                            boundary_coherence: meas.boundary_coherence,
                            curvature: meas.curvature,
                        })
                    } else {
                        // Tracked width too narrow — bad state, use original
                        Some(meas)
                    }
                } else if self.poly_tracker.both_active() {
                    // No YOLO measurement at all, but tracker is still predicting
                    let ref_y = 0.82;
                    let tracked_left = self.poly_tracker.left_x_at(ref_y);
                    let tracked_right = self.poly_tracker.right_x_at(ref_y);
                    let tracked_width = tracked_right - tracked_left;

                    if tracked_width > 50.0 {
                        let tracked_offset = ego_x - (tracked_left + tracked_right) / 2.0;
                        let tracker_conf = self.poly_tracker.confidence();

                        debug!(
                            "📐 PolyTracker bridging dropout: L={:.0} R={:.0} \
                         W={:.0} conf={:.2}",
                            tracked_left, tracked_right, tracked_width, tracker_conf,
                        );

                        Some(LaneMeasurement {
                            lateral_offset_px: tracked_offset,
                            lane_width_px: tracked_width,
                            confidence: tracker_conf,
                            both_lanes: true,
                            boundary_coherence: -1.0, // no fresh coherence data
                            curvature: None,
                        })
                    } else {
                        None
                    }
                } else {
                    input.lane_measurement
                }
            } else {
                input.lane_measurement
            };

        // Capture geometric signals for output
        let output_geo = if self.poly_tracker.both_active() {
            Some(geo_signals)
        } else {
            None
        };

        // ══════════════════════════════════════════════════════════════════
        // 4. LATERAL SHIFT DETECTION + FEED TO CLASSIFIER
        //    v4.4: Now receives ego-motion input for fusion/bridging.
        //    v5.0: Receives enhanced measurement from polynomial tracker.
        // ══════════════════════════════════════════════════════════════════
        let ego_input = if self.enable_ego_motion {
            Some(EgoMotionInput {
                lateral_velocity: self.last_ego_estimate.lateral_velocity_px,
                confidence: self.last_ego_estimate.confidence,
            })
        } else {
            None
        };

        let lateral_result = self.lateral_detector.update(
            enhanced_measurement,
            ego_input,
            input.timestamp_ms,
            input.frame_id,
        );
        // v7.0: Attach geometric signals from polynomial tracker to the
        // shift event. These signals help the classifier distinguish real
        // lane changes from curve-induced perspective artifacts.
        if let Some(mut shift) = lateral_result.completed_shift {
            if self.poly_tracker.both_active() {
                shift.geometric_signals = Some(*self.poly_tracker.signals());
            }
            self.classifier.feed_shift(shift);
        }
        // v7.0: Feed early "shift confirmed" notification to classifier for
        // immediate LANE_CHANGE emission (entry LC fires before shift completes).
        if let Some(mut confirmed) = lateral_result.confirmed_in_progress {
            if self.poly_tracker.both_active() {
                confirmed.geometric_signals = Some(*self.poly_tracker.signals());
            }
            self.classifier.feed_shift_confirmed(confirmed);
        }

        // ══════════════════════════════════════════════════════════════════
        // 5. ROAD MARKING UPDATE (for legality context)
        // ══════════════════════════════════════════════════════════════════
        self.classifier.update_markings(MarkingSnapshot {
            left_name: input.left_marking_name.map(|s| s.to_string()),
            right_name: input.right_marking_name.map(|s| s.to_string()),
            frame_id: input.frame_id,
        });

        // v8.0: Feed road classification to classifier for curve + legality awareness
        if let Some(rc) = input.road_classification {
            self.classifier.update_road_classification(rc);
        }

        // ══════════════════════════════════════════════════════════════════
        // 6. CLASSIFICATION / FUSION
        // ══════════════════════════════════════════════════════════════════
        // v4.13b: Pass curve mode to classifier for zone-oscillation suppression.
        self.classifier
            .set_curve_mode(self.lateral_detector.in_curve_mode());

        let maneuver_events =
            self.classifier
                .classify(input.timestamp_ms, input.frame_id, input.legality_buffer);

        // ══════════════════════════════════════════════════════════════════
        // 7. PERIODIC DIAGNOSTICS (ENHANCED)
        // ══════════════════════════════════════════════════════════════════

        if raw_det_count > 0 && tracked_count == 0 {
            warn!(
            "⚠️  F{}: TRACKING FAILURE | raw_dets={} | valid_dets={} | tracks={} | classes={:?}",
            self.frame_count,
            raw_det_count,
            valid_det_count,
            tracked_count,
            class_counts,
        );
        }

        if self.frame_count % 150 == 0 {
            // v5.0: Include polynomial tracker state in diagnostics
            let poly_status = if self.poly_tracker.both_active() {
                format!(
                    "poly=L:{}/R:{} conf={:.2} div={:.2}",
                    self.poly_tracker.left_state().as_str(),
                    self.poly_tracker.right_state().as_str(),
                    self.poly_tracker.confidence(),
                    self.poly_tracker.signals().boundary_velocity_divergence,
                )
            } else {
                format!(
                    "poly=L:{}/R:{}",
                    self.poly_tracker.left_state().as_str(),
                    self.poly_tracker.right_state().as_str(),
                )
            };

            info!(
            "📊 Pipeline v2 (F{}): raw_dets={} | valid_dets={} | tracks={} [{}] | passes={} | lateral={} | ego={:.2}px/f | maneuvers={} | {}",
            self.frame_count,
            raw_det_count,
            valid_det_count,
            tracked_count,
            track_info.join(", "),
            self.pass_detector.total_passes(),
            self.lateral_detector.state_str(),
            ego_velocity,
            self.classifier.total_maneuvers(),
            poly_status,
        );

            if !class_counts.is_empty() {
                info!("    └─ Class distribution: {:?}", class_counts);
            }
        }

        if tracked_count != self.last_tracked_count {
            if tracked_count > self.last_tracked_count {
                info!(
                    "✅ F{}: Tracks increased {} → {} | new tracks: {:?}",
                    self.frame_count, self.last_tracked_count, tracked_count, track_info,
                );
            } else {
                warn!(
                    "❌ F{}: Tracks decreased {} → {} | remaining: {:?}",
                    self.frame_count, self.last_tracked_count, tracked_count, track_info,
                );
            }
            self.last_tracked_count = tracked_count;
        }

        ManeuverFrameOutput {
            maneuver_events,
            tracked_vehicle_count: tracked_count,
            pass_in_progress: self.pass_detector.has_active_pass(),
            shift_in_progress: self.lateral_detector.is_shifting(),
            ego_lateral_velocity: ego_velocity,
            lateral_state: self.lateral_detector.state_str().to_string(),
            geometric_signals: output_geo,
        }
    }

    pub fn tracked_vehicles(&self) -> Vec<&Track> {
        self.tracker.confirmed_tracks()
    }

    pub fn total_passes(&self) -> usize {
        self.pass_detector.total_passes()
    }

    pub fn total_maneuvers(&self) -> u64 {
        self.classifier.total_maneuvers()
    }

    /// v5.0: Access the polynomial boundary tracker (read-only).
    pub fn poly_tracker(&self) -> &PolynomialBoundaryTracker {
        &self.poly_tracker
    }

    /// v5.0: Get the latest geometric lane change signals.
    pub fn geometric_signals(&self) -> &GeometricLaneChangeSignals {
        self.poly_tracker.signals()
    }

    pub fn reset(&mut self) {
        self.tracker.reset();
        self.pass_detector.reset();
        self.lateral_detector.reset();
        self.ego_motion.reset();
        self.classifier.reset();
        self.poly_tracker.reset();
        self.frame_count = 0;
        self.last_ego_estimate = EgoMotionEstimate::none();
    }
}
