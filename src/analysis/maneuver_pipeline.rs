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

use super::ego_motion::{EgoMotionConfig, EgoMotionEstimate, EgoMotionEstimator, GrayFrame};
use super::lateral_detector::{
    EgoMotionInput, LaneMeasurement, LateralDetectorConfig, LateralShiftDetector,
};
use super::maneuver_classifier::{
    ClassifierConfig, ManeuverClassifier, ManeuverEvent, MarkingSnapshot,
};
use super::pass_detector::{PassDetector, PassDetectorConfig};
use super::vehicle_tracker::{DetectionInput, Track, TrackerConfig, VehicleTracker};
use crate::pipeline::legality_buffer::LegalityRingBuffer;
use tracing::info;

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
    pub timestamp_ms: f64,
    pub frame_id: u64,
}

#[derive(Debug)]
pub struct ManeuverFrameOutput {
    pub maneuver_events: Vec<ManeuverEvent>,
    pub tracked_vehicle_count: usize,
    pub pass_in_progress: bool,
    pub shift_in_progress: bool,
    pub ego_lateral_velocity: f32,
    pub lateral_state: String,
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
    last_tracked_count: usize, // 🆕 Add this
}

impl ManeuverPipeline {
    pub fn new(frame_w: f32, frame_h: f32) -> Self {
        Self::with_config(ManeuverPipelineConfig::default(), frame_w, frame_h)
    }

    pub fn with_config(config: ManeuverPipelineConfig, frame_w: f32, frame_h: f32) -> Self {
        Self {
            tracker: VehicleTracker::new(config.tracker, frame_w, frame_h),
            pass_detector: PassDetector::new(config.pass_detector),
            lateral_detector: LateralShiftDetector::new(config.lateral_detector),
            ego_motion: EgoMotionEstimator::new(config.ego_motion),
            classifier: ManeuverClassifier::new(config.classifier),
            enable_ego_motion: config.enable_ego_motion,
            frame_count: 0,
            last_ego_estimate: EgoMotionEstimate::none(),
            last_tracked_count: 0, // 🆕 Initialize to 0
        }
    }

    /// Process one frame through the entire pipeline.
    pub fn process_frame(&mut self, input: ManeuverFrameInput) -> ManeuverFrameOutput {
        self.frame_count += 1;

        // ══════════════════════════════════════════════════════════════════
        // 🆕 DIAGNOSTIC: Count raw detections before filtering
        // ══════════════════════════════════════════════════════════════════
        let raw_det_count = input.vehicle_detections.len();

        // Count valid detections (would pass tracker's filters)
        // This replicates the filtering logic from VehicleTracker::update()
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

        // 🆕 DIAGNOSTIC: Track IDs and zones
        let track_info: Vec<String> = tracks
            .iter()
            .map(|t| format!("T{}:{}", t.id, t.zone.as_str()))
            .collect();

        // 🆕 DIAGNOSTIC: Class ID distribution in raw detections
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
                self.classifier.feed_ego_motion(estimate);
                estimate.lateral_velocity_px
            } else {
                0.0
            }
        } else {
            0.0
        };

        // ══════════════════════════════════════════════════════════════════
        // 4. LATERAL SHIFT DETECTION + FEED TO CLASSIFIER
        //    v4.4: Now receives ego-motion input for fusion/bridging.
        // ══════════════════════════════════════════════════════════════════
        let ego_input = if self.enable_ego_motion {
            Some(EgoMotionInput {
                lateral_velocity: self.last_ego_estimate.lateral_velocity_px,
                confidence: self.last_ego_estimate.confidence,
            })
        } else {
            None
        };

        let shift_event = self.lateral_detector.update(
            input.lane_measurement,
            ego_input,
            input.timestamp_ms,
            input.frame_id,
        );
        if let Some(ref shift) = shift_event {
            self.classifier.feed_shift(shift.clone());
        }

        // ══════════════════════════════════════════════════════════════════
        // 5. ROAD MARKING UPDATE (for legality context)
        // ══════════════════════════════════════════════════════════════════
        self.classifier.update_markings(MarkingSnapshot {
            left_name: input.left_marking_name.map(|s| s.to_string()),
            right_name: input.right_marking_name.map(|s| s.to_string()),
            frame_id: input.frame_id,
        });

        // ══════════════════════════════════════════════════════════════════
        // 6. CLASSIFICATION / FUSION
        // ══════════════════════════════════════════════════════════════════
        let maneuver_events =
            self.classifier
                .classify(input.timestamp_ms, input.frame_id, input.legality_buffer);

        // ══════════════════════════════════════════════════════════════════
        // 7. PERIODIC DIAGNOSTICS (ENHANCED)
        // ══════════════════════════════════════════════════════════════════

        // 🆕 Always log critical tracking failures
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

        // 🆕 Enhanced periodic diagnostics
        if self.frame_count % 150 == 0 {
            info!(
            "📊 Pipeline v2 (F{}): raw_dets={} | valid_dets={} | tracks={} [{}] | passes={} | lateral={} | ego={:.2}px/f | maneuvers={}",
            self.frame_count,
            raw_det_count,
            valid_det_count,
            tracked_count,
            track_info.join(", "),
            self.pass_detector.total_passes(),
            self.lateral_detector.state_str(),
            ego_velocity,
            self.classifier.total_maneuvers(),
        );

            if !class_counts.is_empty() {
                info!("    └─ Class distribution: {:?}", class_counts);
            }
        }

        // 🆕 Log when tracks appear/disappear
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

    pub fn reset(&mut self) {
        self.tracker.reset();
        self.pass_detector.reset();
        self.lateral_detector.reset();
        self.ego_motion.reset();
        self.classifier.reset();
        self.frame_count = 0;
        self.last_ego_estimate = EgoMotionEstimate::none();
    }
}
