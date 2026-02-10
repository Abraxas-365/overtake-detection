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

use super::ego_motion::{EgoMotionConfig, EgoMotionEstimator, GrayFrame};
use super::lateral_detector::{LaneMeasurement, LateralDetectorConfig, LateralShiftDetector};
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
                min_beside_duration_ms: 400.0, // Lower threshold
                max_pass_duration_ms: 90000.0,
                min_beside_frames: 6,
                disappearance_grace_frames: 90,
                ..PassDetectorConfig::default()
            },
            lateral_detector: LateralDetectorConfig {
                min_lane_confidence: 0.20,     // ✅ Reverted to allow detection
                shift_start_threshold: 0.18,   // ✅ More sensitive (was 0.28)
                shift_confirm_threshold: 0.25, // ✅ More sensitive (was 0.38)
                shift_end_threshold: 0.12,
                min_shift_frames: 10, // ✅ Less strict (was 20)
                baseline_alpha_stable: 0.003,
                baseline_warmup_frames: 20,
                occlusion_reset_frames: 60,
                post_reset_freeze_frames: 75,
                ..LateralDetectorConfig::default()
            },
            ego_motion: EgoMotionConfig {
                min_displacement: 1.0,
                min_consensus: 0.35,
                ..EgoMotionConfig::default()
            },
            classifier: ClassifierConfig {
                max_correlation_gap_ms: 25000.0,
                min_single_source_confidence: 0.40, // ✅ Lowered to catch passes (was 0.50)
                correlation_window_ms: 35000.0,
                min_combined_confidence: 0.35, // ✅ Lowered (was 0.45)
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
    tracker: VehicleTracker,
    pass_detector: PassDetector,
    lateral_detector: LateralShiftDetector,
    ego_motion: EgoMotionEstimator,
    classifier: ManeuverClassifier,
    enable_ego_motion: bool,
    frame_count: u64,
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
        }
    }

    /// Process one frame through the entire pipeline.
    pub fn process_frame(&mut self, input: ManeuverFrameInput) -> ManeuverFrameOutput {
        self.frame_count += 1;

        self.tracker
            .update(input.vehicle_detections, input.timestamp_ms, input.frame_id);
        let tracked_count = self.tracker.confirmed_count();
        let tracks = self.tracker.confirmed_tracks();

        // 2. PASS DETECTION
        let pass_events = self
            .pass_detector
            .update(&tracks, input.timestamp_ms, input.frame_id);

        // 3. LATERAL SHIFT DETECTION
        let shift_event = self.lateral_detector.update(
            input.lane_measurement,
            input.timestamp_ms,
            input.frame_id,
        );
        if let Some(ref shift) = shift_event {
            self.classifier.feed_shift(shift.clone());
        }

        // 4. EGO-MOTION ESTIMATION
        let ego_velocity = if self.enable_ego_motion {
            if let Some(gray) = input.gray_frame {
                let estimate = self.ego_motion.update(gray);
                self.classifier.feed_ego_motion(estimate);
                estimate.lateral_velocity_px
            } else {
                0.0
            }
        } else {
            0.0
        };

        // 5. ROAD MARKING UPDATE (for legality context)
        self.classifier.update_markings(MarkingSnapshot {
            left_name: input.left_marking_name.map(|s| s.to_string()),
            right_name: input.right_marking_name.map(|s| s.to_string()),
            frame_id: input.frame_id,
        });

        // 6. CLASSIFICATION / FUSION
        // Pass the existing legality buffer through for temporally-correct lookups
        let maneuver_events =
            self.classifier
                .classify(input.timestamp_ms, input.frame_id, input.legality_buffer);

        // 7. PERIODIC DIAGNOSTICS
        if self.frame_count % 150 == 0 {
            info!(
                "📊 Pipeline v2: tracks={} | passes={} | lateral={} | maneuvers={}",
                tracked_count,
                self.pass_detector.total_passes(),
                self.lateral_detector.state_str(),
                self.classifier.total_maneuvers(),
            );
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
    }
}
