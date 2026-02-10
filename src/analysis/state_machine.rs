// src/analysis/state_machine.rs
//
// LANE CHANGE DETECTION v5.0 - ADAPTIVE MINING ROUTE EDITION
//
// v5.0: Adaptive system for Peru mining routes:
//   1. Adaptive thresholds based on noise floor
//   2. Context detection (dust/unpaved/paved/highway)
//   3. Mining-specific profiles with tailored parameters
//   4. Profile-based validation requirements
//   5. Real-time threshold adjustment
//
// v4.1 features retained:
//   - Path-specific durations
//   - Tiered validation
//   - Curve rejection
//   - Baseline occlusion handling

use super::boundary_detector::CrossingType;
use super::curve_detector::CurveDetector;
use super::velocity_tracker::LateralVelocityTracker;
use crate::types::{Direction, LaneChangeConfig, LaneChangeEvent, LaneChangeState, VehicleState};
use std::collections::VecDeque;
use tracing::{debug, info, warn};

// 🆕 ADAPTIVE SYSTEM IMPORTS
use crate::analysis::adaptive::{
    AdaptiveThresholdSet, ContextDetector, MiningRouteProfile, RoadContext,
};

// ============================================================================
// VALIDATION THRESHOLDS (BALANCED) - Now used as DEFAULTS
// ============================================================================
const VERY_HIGH_OFFSET_THRESHOLD: f32 = 0.55;
const HIGH_OFFSET_THRESHOLD: f32 = 0.45;
const MEDIUM_OFFSET_THRESHOLD: f32 = 0.35;
const LOW_OFFSET_THRESHOLD: f32 = 0.28;

// Path-specific minimum offsets
const MIN_OFFSET_BOUNDARY: f32 = 0.25;
const MIN_OFFSET_HIGH_VEL: f32 = 0.27;
const MIN_OFFSET_GRADUAL: f32 = 0.32;
const MIN_OFFSET_DEFAULT: f32 = 0.28;

const MIN_DURATION_VERY_HIGH: f64 = 500.0;
const MIN_DURATION_HIGH: f64 = 800.0;
const MIN_DURATION_MEDIUM: f64 = 1200.0;
const MIN_DURATION_LOW: f64 = 2000.0;

const MIN_NET_DISPLACEMENT: f32 = 0.15;

// ============================================================================
// VELOCITY THRESHOLDS
// ============================================================================
const MIN_VELOCITY_FAST: f32 = 120.0;
const MIN_VELOCITY_MEDIUM: f32 = 60.0;
const MIN_VELOCITY_SLOW: f32 = 20.0;
const VELOCITY_SPIKE_THRESHOLD: f32 = 200.0;

// ============================================================================
// ANALYSIS WINDOWS
// ============================================================================
const ANALYSIS_WINDOW_MS: f64 = 4000.0;
const TRAJECTORY_WINDOW_SIZE: usize = 90;

// ============================================================================
// DEVIATION THRESHOLDS - Now used as NOMINAL values for adaptive system
// ============================================================================
const DEVIATION_DRIFT_START: f32 = 0.25;
const DEVIATION_CROSSING: f32 = 0.30;
const DEVIATION_LANE_CENTER: f32 = 0.55;
const DEVIATION_SIGNIFICANT: f32 = 0.50;

// ============================================================================
// STATE MACHINE BEHAVIOR
// ============================================================================
const HYSTERESIS_EXIT: f32 = 0.5;
const DIRECTION_CONSISTENCY_THRESHOLD: f32 = 0.65; // Nominal, adaptive system will adjust
const POST_CHANGE_GRACE_FRAMES: u32 = 90;
const MAX_DRIFTING_MS: f64 = 10000.0;

// ============================================================================
// KALMAN FILTER
// ============================================================================
const KALMAN_PROCESS_NOISE: f32 = 0.001;
const KALMAN_MEASUREMENT_NOISE: f32 = 0.01;

// ============================================================================
// ADAPTIVE BASELINE
// ============================================================================
const EWMA_ALPHA_STABLE: f32 = 0.003;
const EWMA_ALPHA_ADAPTING: f32 = 0.015;
const EWMA_ALPHA_FAST_RECOVER: f32 = 0.030;
const EWMA_MIN_SAMPLES: u32 = 20;
const STABILITY_VARIANCE_THRESHOLD: f32 = 0.005;
const INSTABILITY_VARIANCE_THRESHOLD: f32 = 0.05;
const BASELINE_MAX_DRIFT: f32 = 0.25;
const BASELINE_SANITY_CHECK: f32 = 0.35;

const OCCLUSION_SHORT_THRESHOLD: u32 = 30;
const OCCLUSION_LONG_THRESHOLD: u32 = 60;

// ============================================================================
// CURVE DETECTION
// ============================================================================
const CURVE_COMPENSATION_FACTOR: f32 = 1.0;
const MIN_DRIFT_RATE_PER_SEC: f32 = 5.0;

// ============================================================================
// TRAJECTORY ANALYSIS
// ============================================================================
const RETURN_TO_CENTER_THRESHOLD: f32 = 0.35;
const MIN_EXCURSION_FOR_OVERTAKE: f32 = 0.30;
const TRAJECTORY_SMOOTHNESS_THRESHOLD: f32 = 0.25;
const MIN_TRAJECTORY_POINTS: usize = 8;
const POSITION_CHANGE_THRESHOLD: f32 = 0.15;

// ============================================================================
// KALMAN FILTER
// ============================================================================

#[derive(Clone)]
struct SimpleKalmanFilter {
    x: f32,
    p: f32,
    q: f32,
    r: f32,
    initialized: bool,
}

impl SimpleKalmanFilter {
    fn new() -> Self {
        Self {
            x: 0.0,
            p: 1.0,
            q: KALMAN_PROCESS_NOISE,
            r: KALMAN_MEASUREMENT_NOISE,
            initialized: false,
        }
    }

    fn update(&mut self, measurement: f32) -> f32 {
        if !self.initialized {
            self.x = measurement;
            self.p = self.r;
            self.initialized = true;
            return measurement;
        }
        let p_pred = self.p + self.q;
        let k = p_pred / (p_pred + self.r);
        self.x = self.x + k * (measurement - self.x);
        self.p = (1.0 - k) * p_pred;
        self.x
    }

    fn reset(&mut self) {
        self.x = 0.0;
        self.p = 1.0;
        self.initialized = false;
    }

    fn get_last_value(&self) -> Option<f32> {
        if self.initialized {
            Some(self.x)
        } else {
            None
        }
    }
}

// ============================================================================
// ADAPTIVE BASELINE WITH OCCLUSION RECOVERY
// ============================================================================

#[derive(Clone)]
struct AdaptiveBaseline {
    value: f32,
    initial_value: f32,
    variance: f32,
    sample_count: u32,
    is_valid: bool,
    recent_samples: VecDeque<f32>,
    is_adapting: bool,
    stable_frames: u32,
    is_frozen: bool,
    frozen_value: f32,
    has_initial: bool,

    frames_without_lanes: u32,
    just_recovered_from_occlusion: bool,
    fast_recovery_frames: u32,
    seed_lock_frames: u32,
}

impl AdaptiveBaseline {
    fn new() -> Self {
        Self {
            value: 0.0,
            initial_value: 0.0,
            variance: 1.0,
            sample_count: 0,
            is_valid: false,
            recent_samples: VecDeque::with_capacity(30),
            is_adapting: true,
            stable_frames: 0,
            is_frozen: false,
            frozen_value: 0.0,
            has_initial: false,
            frames_without_lanes: 0,
            just_recovered_from_occlusion: false,
            fast_recovery_frames: 0,
            seed_lock_frames: 0,
        }
    }

    fn update(&mut self, measurement: f32) -> f32 {
        self.sample_count += 1;

        if self.frames_without_lanes > OCCLUSION_SHORT_THRESHOLD {
            info!(
                "🚨 Fast recovery mode after {}s occlusion",
                self.frames_without_lanes as f32 / 30.0
            );
            self.is_adapting = true;
            self.stable_frames = 0;
            self.just_recovered_from_occlusion = true;
            self.fast_recovery_frames = 30;
        }

        self.frames_without_lanes = 0;

        self.recent_samples.push_back(measurement);
        if self.recent_samples.len() > 30 {
            self.recent_samples.pop_front();
        }

        if self.recent_samples.len() >= 10 {
            let samples: Vec<f32> = self.recent_samples.iter().copied().collect();
            let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
            self.variance =
                samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / samples.len() as f32;
        }

        if self.is_frozen {
            return self.frozen_value;
        }

        let deviation_from_baseline = (measurement - self.value).abs();

        if self.variance < STABILITY_VARIANCE_THRESHOLD {
            self.stable_frames += 1;
            if self.stable_frames > 30 {
                self.is_adapting = false;
            }
        } else {
            self.stable_frames = 0;
            if deviation_from_baseline > 0.15 {
                self.is_adapting = true;
            }
        }

        let alpha = if self.seed_lock_frames > 0 {
            self.seed_lock_frames -= 1;
            if self.seed_lock_frames == 0 {
                info!("🔓 Baseline seed lock expired");
            }
            0.0
        } else if self.fast_recovery_frames > 0 {
            self.fast_recovery_frames -= 1;
            if self.fast_recovery_frames == 0 {
                self.just_recovered_from_occlusion = false;
                info!("✅ Fast recovery complete");
            }
            EWMA_ALPHA_FAST_RECOVER
        } else if self.is_adapting || !self.is_valid {
            EWMA_ALPHA_ADAPTING
        } else {
            EWMA_ALPHA_STABLE
        };

        if self.sample_count == 1 {
            self.value = measurement;
        } else {
            let new_value = alpha * measurement + (1.0 - alpha) * self.value;

            if self.has_initial {
                let drift = (new_value - self.initial_value).abs();
                if drift <= BASELINE_MAX_DRIFT {
                    self.value = new_value;
                }
            } else {
                self.value = new_value;
            }
        }

        if !self.has_initial
            && self.sample_count >= EWMA_MIN_SAMPLES
            && self.variance < INSTABILITY_VARIANCE_THRESHOLD
        {
            self.initial_value = self.value;
            self.has_initial = true;
            self.is_valid = true;
        }

        if self.sample_count >= EWMA_MIN_SAMPLES && self.variance < INSTABILITY_VARIANCE_THRESHOLD {
            self.is_valid = true;
        }

        self.value
    }

    fn reset_to(&mut self, initial_value: f32) {
        self.value = initial_value;
        self.initial_value = initial_value;
        self.variance = 1.0;
        self.sample_count = 0;
        self.is_valid = false;
        self.recent_samples.clear();
        self.is_adapting = true;
        self.stable_frames = 0;
        self.is_frozen = false;
        self.frozen_value = 0.0;
        self.has_initial = false;
        self.frames_without_lanes = 0;
        self.just_recovered_from_occlusion = false;
        self.fast_recovery_frames = 0;
        self.seed_lock_frames = 0;

        info!("🎯 Baseline seeded at {:.1}%", initial_value * 100.0);
    }

    fn freeze(&mut self) {
        if !self.is_frozen {
            self.is_frozen = true;
            self.frozen_value = self.value;
            info!("🧊 Baseline frozen at {:.1}%", self.frozen_value * 100.0);
        }
    }

    fn unfreeze(&mut self) {
        if self.is_frozen {
            self.is_frozen = false;
            debug!("🔥 Baseline unfrozen");
        }
    }

    fn effective_value(&self) -> f32 {
        if self.is_frozen {
            self.frozen_value
        } else {
            self.value
        }
    }

    fn is_sane(&self) -> bool {
        self.value.abs() < BASELINE_SANITY_CHECK
    }

    fn mark_no_lanes(&mut self) {
        self.frames_without_lanes += 1;

        if self.frames_without_lanes > OCCLUSION_LONG_THRESHOLD {
            if self.is_valid {
                warn!(
                    "🗑️  Baseline invalidated after {}s occlusion",
                    self.frames_without_lanes as f32 / 30.0
                );
            }
            self.is_valid = false;
            self.has_initial = false;
        }
    }

    fn frames_without_lanes_recent(&self) -> u32 {
        if self.just_recovered_from_occlusion {
            1
        } else {
            0
        }
    }

    fn reset(&mut self) {
        self.value = 0.0;
        self.initial_value = 0.0;
        self.variance = 1.0;
        self.sample_count = 0;
        self.is_valid = false;
        self.recent_samples.clear();
        self.is_adapting = true;
        self.stable_frames = 0;
        self.is_frozen = false;
        self.frozen_value = 0.0;
        self.has_initial = false;
        self.frames_without_lanes = 0;
        self.just_recovered_from_occlusion = false;
        self.fast_recovery_frames = 0;
    }
}

// ============================================================================
// TRAJECTORY ANALYZER
// ============================================================================

#[derive(Clone)]
struct TrajectoryAnalyzer {
    positions: VecDeque<f32>,
    timestamps: VecDeque<f64>,
    velocities: VecDeque<f32>,
}

#[derive(Default, Debug, Clone)]
struct OvertakeAnalysis {
    excursion_sufficient: bool,
    returned_to_start: bool,
    is_smooth: bool,
    has_reversal: bool,
    shape_score: f32,
    smoothness: f32,
    is_valid_overtake: bool,
}

impl TrajectoryAnalyzer {
    fn new() -> Self {
        Self {
            positions: VecDeque::with_capacity(TRAJECTORY_WINDOW_SIZE),
            timestamps: VecDeque::with_capacity(TRAJECTORY_WINDOW_SIZE),
            velocities: VecDeque::with_capacity(TRAJECTORY_WINDOW_SIZE),
        }
    }

    fn add_sample(&mut self, position: f32, timestamp: f64, velocity: f32) {
        self.positions.push_back(position);
        self.timestamps.push_back(timestamp);
        self.velocities.push_back(velocity);

        if self.positions.len() > TRAJECTORY_WINDOW_SIZE {
            self.positions.pop_front();
            self.timestamps.pop_front();
            self.velocities.pop_front();
        }
    }

    fn analyze_overtake_pattern(
        &self,
        start_pos: f32,
        current_pos: f32,
        max_excursion: f32,
    ) -> OvertakeAnalysis {
        let mut analysis = OvertakeAnalysis::default();

        if self.positions.len() < MIN_TRAJECTORY_POINTS {
            analysis.excursion_sufficient = max_excursion >= MIN_EXCURSION_FOR_OVERTAKE;
            analysis.shape_score = 0.6;
            analysis.is_valid_overtake = analysis.excursion_sufficient;
            return analysis;
        }

        analysis.excursion_sufficient = max_excursion >= MIN_EXCURSION_FOR_OVERTAKE;

        let return_distance = (current_pos - start_pos).abs();
        analysis.returned_to_start = return_distance < RETURN_TO_CENTER_THRESHOLD;

        analysis.smoothness = self.calculate_smoothness();
        analysis.is_smooth = analysis.smoothness < TRAJECTORY_SMOOTHNESS_THRESHOLD;

        analysis.has_reversal = self.detect_direction_reversal();
        analysis.shape_score = self.calculate_shape_score(start_pos, max_excursion);

        analysis.is_valid_overtake = analysis.excursion_sufficient
            || analysis.shape_score >= 0.5
            || (analysis.has_reversal && max_excursion >= 0.25);

        analysis
    }

    fn calculate_smoothness(&self) -> f32 {
        if self.positions.len() < 3 {
            return 0.0;
        }

        let positions: Vec<f32> = self.positions.iter().copied().collect();
        let mut jitter_sum = 0.0;
        let mut count = 0;

        for i in 2..positions.len() {
            let accel = positions[i] - 2.0 * positions[i - 1] + positions[i - 2];
            jitter_sum += accel.abs();
            count += 1;
        }

        if count > 0 {
            jitter_sum / count as f32
        } else {
            0.0
        }
    }

    fn detect_direction_reversal(&self) -> bool {
        if self.velocities.len() < 8 {
            return true;
        }

        let velocities: Vec<f32> = self.velocities.iter().copied().collect();
        let mut sign_changes = 0;
        let mut last_sign: Option<bool> = None;

        for v in &velocities {
            if v.abs() > 15.0 {
                let current_sign = *v > 0.0;
                if let Some(last) = last_sign {
                    if current_sign != last {
                        sign_changes += 1;
                    }
                }
                last_sign = Some(current_sign);
            }
        }

        sign_changes >= 1
    }

    fn calculate_shape_score(&self, start_pos: f32, max_excursion: f32) -> f32 {
        if self.positions.len() < MIN_TRAJECTORY_POINTS {
            return 0.6;
        }

        let positions: Vec<f32> = self.positions.iter().copied().collect();
        let n = positions.len();

        let mut peak_idx = 0;
        let mut peak_deviation = 0.0;
        for (i, &pos) in positions.iter().enumerate() {
            let deviation = (pos - start_pos).abs();
            if deviation > peak_deviation {
                peak_deviation = deviation;
                peak_idx = i;
            }
        }

        let peak_position_score = {
            let relative_pos = peak_idx as f32 / n as f32;
            if relative_pos >= 0.2 && relative_pos <= 0.8 {
                1.0
            } else if relative_pos >= 0.1 && relative_pos <= 0.9 {
                0.7
            } else {
                0.4
            }
        };

        let return_score = {
            let end_pos = positions[n - 1];
            let return_distance = (end_pos - start_pos).abs();
            if return_distance < 0.20 {
                1.0
            } else if return_distance < 0.35 {
                0.7
            } else if return_distance < 0.50 {
                0.5
            } else {
                0.3
            }
        };

        let magnitude_score = if max_excursion >= 0.55 {
            1.0
        } else if max_excursion >= 0.45 {
            0.9
        } else if max_excursion >= 0.35 {
            0.7
        } else if max_excursion >= 0.25 {
            0.5
        } else {
            0.3
        };

        (peak_position_score * 0.25 + return_score * 0.35 + magnitude_score * 0.40)
    }

    fn clear(&mut self) {
        self.positions.clear();
        self.timestamps.clear();
        self.velocities.clear();
    }
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Clone, Copy, Debug)]
struct OffsetSample {
    normalized_offset: f32,
    deviation: f32,
    timestamp_ms: f64,
    lateral_velocity: f32,
    direction: Direction,
    confidence: f32, // 🆕 Added for adaptive system
}

#[derive(Debug, Default)]
struct WindowMetrics {
    total_displacement: f32,
    max_deviation: f32,
    avg_velocity: f32,
    peak_velocity: f32,
    direction_consistency: f32,
    time_span_ms: f64,
    is_intentional_change: bool,
    is_sustained_movement: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum DetectionPath {
    BoundaryCrossing,
    HighVelocity,
    MediumDeviation,
    GradualChange,
    LargeDeviation,
    VelocitySpike,
    PostOcclusion,
    BlackoutRecovery,
    StaticHighDeviation,
}

// ============================================================================
// CONFIDENCE CALCULATOR
// ============================================================================

struct ConfidenceCalculator;

impl ConfidenceCalculator {
    fn calculate(
        max_offset: f32,
        duration_ms: f64,
        trajectory_analysis: &OvertakeAnalysis,
        detection_path: Option<DetectionPath>,
    ) -> f32 {
        let offset_confidence = Self::offset_to_confidence(max_offset);
        let duration_confidence = Self::duration_to_confidence(duration_ms);
        let trajectory_confidence = trajectory_analysis.shape_score;

        let path_bonus = match detection_path {
            Some(DetectionPath::BoundaryCrossing) => 0.05,
            Some(DetectionPath::HighVelocity) => 0.03,
            Some(DetectionPath::PostOcclusion) => 0.02,
            Some(DetectionPath::BlackoutRecovery) => 0.00,
            Some(DetectionPath::StaticHighDeviation) => 0.01,
            _ => 0.0,
        };
        let base_confidence = offset_confidence * 0.45
            + duration_confidence * 0.25
            + trajectory_confidence * 0.20
            + path_bonus
            + 0.05;

        base_confidence.max(0.4).min(0.98)
    }

    fn offset_to_confidence(max_offset: f32) -> f32 {
        if max_offset >= 0.60 {
            0.95
        } else if max_offset >= 0.50 {
            0.85
        } else if max_offset >= 0.40 {
            0.75
        } else if max_offset >= 0.30 {
            0.60
        } else {
            0.45
        }
    }

    fn duration_to_confidence(duration_ms: f64) -> f32 {
        if duration_ms >= 2000.0 && duration_ms <= 6000.0 {
            0.90
        } else if duration_ms >= 1000.0 && duration_ms <= 8000.0 {
            0.75
        } else if duration_ms >= 500.0 && duration_ms <= 10000.0 {
            0.55
        } else {
            0.35
        }
    }
}

// ============================================================================
// MAIN STATE MACHINE - WITH ADAPTIVE SYSTEM
// ============================================================================

pub struct LaneChangeStateMachine {
    config: LaneChangeConfig,
    source_id: String,

    state: LaneChangeState,
    frames_in_state: u32,
    pending_state: Option<LaneChangeState>,
    pending_frames: u32,

    change_direction: Direction,
    change_start_frame: Option<u64>,
    change_start_time: Option<f64>,
    change_detection_path: Option<DetectionPath>,
    max_offset_in_change: f32,
    initial_position_frozen: Option<f32>,

    cooldown_remaining: u32,
    total_frames_processed: u64,
    post_lane_change_grace: u32,

    position_filter: SimpleKalmanFilter,
    adaptive_baseline: AdaptiveBaseline,

    offset_history: Vec<f32>,
    velocity_history: VecDeque<f32>,
    offset_samples: VecDeque<OffsetSample>,
    direction_samples: VecDeque<Direction>,
    recent_deviations: Vec<f32>,

    stable_deviation_frames: u32,
    last_deviation: f32,

    peak_deviation_in_window: f32,
    peak_velocity_in_window: f32,
    peak_direction: Direction,

    curve_detector: CurveDetector,
    velocity_tracker: LateralVelocityTracker,
    trajectory_analyzer: TrajectoryAnalyzer,

    is_in_curve: bool,
    curve_compensation_factor: f32,

    pending_change_direction: Direction,
    pending_max_offset: f32,

    pending_span_duration: Option<f64>,
    detection_start_timestamp: Option<f64>,

    in_blackout: bool,
    blackout_consecutive_frames: u32,
    position_before_blackout: Option<f32>,
    blackout_started_frame: Option<u64>,
    blackout_started_time: Option<f64>,

    // 🆕 ADAPTIVE SYSTEM COMPONENTS
    adaptive_thresholds: AdaptiveThresholdSet,
    context_detector: ContextDetector,
    current_profile: MiningRouteProfile,
    last_profile_context: RoadContext,
}

impl LaneChangeStateMachine {
    pub fn new(config: LaneChangeConfig) -> Self {
        Self {
            config,
            source_id: String::new(),
            state: LaneChangeState::Centered,
            frames_in_state: 0,
            pending_state: None,
            pending_frames: 0,
            change_direction: Direction::Unknown,
            change_start_frame: None,
            change_start_time: None,
            change_detection_path: None,
            max_offset_in_change: 0.0,
            initial_position_frozen: None,
            cooldown_remaining: 0,
            total_frames_processed: 0,
            post_lane_change_grace: 0,
            position_filter: SimpleKalmanFilter::new(),
            adaptive_baseline: AdaptiveBaseline::new(),
            offset_history: Vec::with_capacity(90),
            velocity_history: VecDeque::with_capacity(30),
            offset_samples: VecDeque::with_capacity(150),
            direction_samples: VecDeque::with_capacity(30),
            recent_deviations: Vec::with_capacity(30),
            stable_deviation_frames: 0,
            last_deviation: 0.0,
            peak_deviation_in_window: 0.0,
            peak_velocity_in_window: 0.0,
            peak_direction: Direction::Unknown,
            curve_detector: CurveDetector::new(),
            velocity_tracker: LateralVelocityTracker::new(),
            trajectory_analyzer: TrajectoryAnalyzer::new(),
            is_in_curve: false,
            curve_compensation_factor: 1.0,
            pending_change_direction: Direction::Unknown,
            pending_max_offset: 0.0,
            pending_span_duration: None,
            detection_start_timestamp: None,
            in_blackout: false,
            blackout_consecutive_frames: 0,
            position_before_blackout: None,
            blackout_started_frame: None,
            blackout_started_time: None,

            // 🆕 Initialize adaptive components
            adaptive_thresholds: AdaptiveThresholdSet::new(),
            context_detector: ContextDetector::new(),
            current_profile: MiningRouteProfile::paved(), // Start with default
            last_profile_context: RoadContext::Unknown,
        }
    }

    // 🆕 PRIMARY UPDATE METHOD - Now uses adaptive system
    pub fn update(
        &mut self,
        vehicle_state: &VehicleState,
        frame_id: u64,
        timestamp_ms: f64,
        crossing_type: CrossingType,
    ) -> Option<LaneChangeEvent> {
        self.total_frames_processed += 1;

        if self.total_frames_processed < self.config.skip_initial_frames {
            return None;
        }

        // ══════════════════════════════════════════════════════════════
        // STEP 1: CONTEXT DETECTION & PROFILE SELECTION
        // ══════════════════════════════════════════════════════════════
        let detected_context = self.context_detector.update(vehicle_state);

        // Load new profile if context changed
        if detected_context != self.last_profile_context && detected_context != RoadContext::Unknown
        {
            self.current_profile = MiningRouteProfile::for_context(detected_context);
            self.last_profile_context = detected_context;

            info!(
                "📋 Profile loaded: {} ({}) | context={:?}",
                self.current_profile.name, self.current_profile.description, detected_context
            );
        }

        // ══════════════════════════════════════════════════════════════
        // STEP 2: ADAPTIVE THRESHOLD ADJUSTMENT (every 1 second)
        // ══════════════════════════════════════════════════════════════
        if frame_id % 30 == 0 && self.state == LaneChangeState::Centered {
            self.adaptive_thresholds.adapt(
                vehicle_state.detection_confidence,
                &self.recent_deviations,
                frame_id,
            );

            // Apply context-specific overrides
            self.adaptive_thresholds
                .apply_context_overrides(&detected_context);
        }

        // ══════════════════════════════════════════════════════════════
        // STEP 3: RUN DETECTION WITH ADAPTIVE THRESHOLDS
        // ══════════════════════════════════════════════════════════════
        self.update_with_adaptive_thresholds(vehicle_state, frame_id, timestamp_ms, crossing_type)
    }

    fn update_with_adaptive_thresholds(
        &mut self,
        vehicle_state: &VehicleState,
        frame_id: u64,
        timestamp_ms: f64,
        crossing_type: CrossingType,
    ) -> Option<LaneChangeEvent> {
        // Cooldown handling
        if self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
            if self.cooldown_remaining == 0 {
                self.state = LaneChangeState::Centered;
                self.frames_in_state = 0;
            }
            return None;
        }

        if self.post_lane_change_grace > 0 {
            self.post_lane_change_grace -= 1;
        }

        // Timeout handling
        if let Some(event) = self.handle_timeouts(frame_id, timestamp_ms) {
            return Some(event);
        }

        // Invalid state handling
        if !vehicle_state.is_valid() {
            self.adaptive_baseline.mark_no_lanes();
            return None;
        }

        // Calculate normalized position
        let lane_width = vehicle_state.lane_width.unwrap();
        let raw_offset = vehicle_state.lateral_offset / lane_width;
        let normalized_offset = self.position_filter.update(raw_offset);

        let lateral_velocity = self
            .velocity_tracker
            .get_velocity(vehicle_state.lateral_offset, timestamp_ms);

        // Update histories
        self.velocity_history.push_back(lateral_velocity);
        if self.velocity_history.len() > 30 {
            self.velocity_history.pop_front();
        }

        self.offset_history.push(normalized_offset);
        if self.offset_history.len() > 60 {
            self.offset_history.remove(0);
        }

        // Grace period - just update baseline
        if self.post_lane_change_grace > 0 {
            self.adaptive_baseline.update(normalized_offset);
            return None;
        }

        // Update baseline
        self.adaptive_baseline.update(normalized_offset);

        if !self.adaptive_baseline.is_valid {
            return None;
        }

        // Calculate deviation
        let baseline = self.adaptive_baseline.effective_value();
        let signed_deviation = normalized_offset - baseline;
        let deviation = signed_deviation.abs();

        // Wrap-around detection
        let is_wrap_around = self.adaptive_baseline.seed_lock_frames > 0
            && deviation > 0.20
            && baseline.signum() != normalized_offset.signum();

        let current_direction = if is_wrap_around {
            if baseline > 0.05 && signed_deviation < -0.20 {
                Direction::Right
            } else if baseline < -0.05 && signed_deviation > 0.20 {
                Direction::Left
            } else {
                Direction::from_offset(signed_deviation)
            }
        } else {
            Direction::from_offset(signed_deviation)
        };

        // Direction correction
        if self.pending_state == Some(LaneChangeState::Drifting) && is_wrap_around {
            if self.pending_change_direction != current_direction
                && current_direction != Direction::Unknown
            {
                info!(
                    "🔄 Direction correction: {} -> {} (Wrap-Around detected)",
                    self.pending_change_direction.as_str(),
                    current_direction.as_str()
                );
                self.pending_change_direction = current_direction;
            }
        }

        // Track max offset
        if self.pending_state == Some(LaneChangeState::Drifting)
            || self.state == LaneChangeState::Drifting
            || self.state == LaneChangeState::Crossing
        {
            if deviation > self.max_offset_in_change {
                self.max_offset_in_change = deviation;
            }
            if deviation > self.pending_max_offset {
                self.pending_max_offset = deviation;
            }
        }

        // Update recent deviations
        self.recent_deviations.push(deviation);
        if self.recent_deviations.len() > 30 {
            self.recent_deviations.remove(0);
        }

        // 🆕 Feed measurement to adaptive threshold (for continuous learning)
        self.adaptive_thresholds
            .drift_threshold
            .feed_measurement(deviation);

        // Store sample with confidence
        let sample = OffsetSample {
            normalized_offset,
            deviation,
            timestamp_ms,
            lateral_velocity,
            direction: current_direction,
            confidence: vehicle_state.detection_confidence, // 🆕
        };
        self.offset_samples.push_back(sample);

        // Clean old samples
        while let Some(oldest) = self.offset_samples.front() {
            if timestamp_ms - oldest.timestamp_ms > ANALYSIS_WINDOW_MS {
                self.offset_samples.pop_front();
            } else {
                break;
            }
        }

        // Direction tracking
        self.direction_samples.push_back(current_direction);
        if self.direction_samples.len() > 30 {
            self.direction_samples.pop_front();
        }

        // Peak tracking
        if deviation > self.peak_deviation_in_window {
            self.peak_deviation_in_window = deviation;
            self.peak_direction = current_direction;
        }
        if lateral_velocity.abs() > self.peak_velocity_in_window {
            self.peak_velocity_in_window = lateral_velocity.abs();
        }

        // Calculate window metrics
        let window_metrics = self.calculate_window_metrics(timestamp_ms, lane_width);

        // 🆕 DETERMINE TARGET STATE WITH ADAPTIVE THRESHOLDS
        let target_state = self.determine_target_state_adaptive(
            deviation,
            crossing_type,
            lateral_velocity,
            current_direction,
            &window_metrics,
        );

        // Logging
        if frame_id % 30 == 0 {
            info!(
                "F{}: dev={:.1}%, state={:?}->{:?}, dir={}, ctx={:?}",
                frame_id,
                deviation * 100.0,
                self.state,
                target_state,
                current_direction.as_str(),
                self.last_profile_context
            );
        }

        // State transition
        self.check_transition(
            target_state,
            current_direction,
            frame_id,
            timestamp_ms,
            deviation,
            normalized_offset,
        )
    }

    // 🆕 DETERMINE TARGET STATE - Using adaptive thresholds
    fn determine_target_state_adaptive(
        &mut self,
        deviation: f32,
        crossing_type: CrossingType,
        lateral_velocity: f32,
        current_direction: Direction,
        metrics: &WindowMetrics,
    ) -> LaneChangeState {
        // Get adaptive threshold values
        let drift_threshold = self.adaptive_thresholds.drift_threshold.get();
        let crossing_threshold = self.adaptive_thresholds.crossing_threshold.get();
        let consistency_threshold = self.adaptive_thresholds.consistency_threshold.get();

        let vel_fast = MIN_VELOCITY_FAST;
        let vel_medium = MIN_VELOCITY_MEDIUM;

        match self.state {
            LaneChangeState::Centered => {
                if !self.adaptive_baseline.is_sane() {
                    return LaneChangeState::Centered;
                }

                // 🚨 CRITICAL: Reject new detections during post-occlusion stabilization
                let frames_since_occlusion = self.adaptive_baseline.frames_without_lanes_recent();
                if frames_since_occlusion > 0 {
                    let freeze_frames = self.current_profile.post_occlusion_freeze_frames;
                    if self.adaptive_baseline.frames_without_lanes < freeze_frames {
                        debug!(
                            "⏸️  Post-occlusion freeze: {}/{} frames",
                            self.adaptive_baseline.frames_without_lanes, freeze_frames
                        );
                        return LaneChangeState::Centered;
                    }
                }

                // PATH 1: BOUNDARY CROSSING
                if crossing_type != CrossingType::None && lateral_velocity.abs() > vel_fast {
                    if self.is_deviation_sustained(drift_threshold) {
                        self.change_detection_path = Some(DetectionPath::BoundaryCrossing);
                        info!(
                            "🚨 [BOUNDARY] {:?}, vel={:.1}px/s",
                            crossing_type, lateral_velocity
                        );
                        return LaneChangeState::Drifting;
                    }
                }

                // PATH 2: HIGH VELOCITY + DEVIATION
                if lateral_velocity.abs() > vel_fast && deviation >= drift_threshold {
                    if self.is_velocity_sustained(vel_medium) {
                        self.change_detection_path = Some(DetectionPath::HighVelocity);
                        info!(
                            "🚨 [HIGH-VEL] vel={:.1}px/s, dev={:.1}%",
                            lateral_velocity,
                            deviation * 100.0
                        );
                        return LaneChangeState::Drifting;
                    }
                }

                // PATH 3: VELOCITY SPIKE
                if lateral_velocity.abs() > VELOCITY_SPIKE_THRESHOLD && deviation >= drift_threshold
                {
                    if self.is_velocity_sustained(vel_fast) {
                        let position_change = self.get_recent_position_change();
                        if position_change >= POSITION_CHANGE_THRESHOLD {
                            self.change_detection_path = Some(DetectionPath::VelocitySpike);
                            info!(
                                "🚨 [VELOCITY-SPIKE] vel={:.1}px/s, pos_change={:.1}%",
                                lateral_velocity,
                                position_change * 100.0
                            );
                            return LaneChangeState::Drifting;
                        }
                    }
                }

                // PATH 4: MEDIUM SPEED + HIGH DEVIATION
                if deviation >= drift_threshold + 0.10 && lateral_velocity.abs() > vel_medium {
                    if self.is_deviation_sustained(drift_threshold) {
                        self.change_detection_path = Some(DetectionPath::MediumDeviation);
                        info!(
                            "🚨 [MEDIUM] dev={:.1}%, vel={:.1}px/s",
                            deviation * 100.0,
                            lateral_velocity
                        );
                        return LaneChangeState::Drifting;
                    }
                }

                // PATH 4.5: SUSTAINED DEVIATION (profile-gated)
                if self.current_profile.allow_sustained_path {
                    if deviation >= DEVIATION_DRIFT_START && deviation < MEDIUM_OFFSET_THRESHOLD {
                        // Curve rejection
                        if self.is_in_curve {
                            if metrics.is_sustained_movement && metrics.time_span_ms >= 3000.0 {
                                info!(
                                    "🌀 Curve rejection [SUSTAINED]: dev={:.1}%, span={:.1}s",
                                    deviation * 100.0,
                                    metrics.time_span_ms / 1000.0
                                );
                            }
                            return LaneChangeState::Centered;
                        }

                        if self.is_deviation_sustained_long(DEVIATION_DRIFT_START)
                            && metrics.time_span_ms >= 3000.0
                            && metrics.is_sustained_movement
                            && metrics.direction_consistency >= consistency_threshold
                            && metrics.direction_consistency <= 0.95
                        {
                            self.change_detection_path = Some(DetectionPath::MediumDeviation);
                            info!(
                                "🚨 [SUSTAINED] dev={:.1}%, span={:.1}s, consistency={:.1}%",
                                deviation * 100.0,
                                metrics.time_span_ms / 1000.0,
                                metrics.direction_consistency * 100.0
                            );
                            return LaneChangeState::Drifting;
                        }
                    }
                }

                // PATH 5: GRADUAL CHANGE (profile-gated)
                if self.current_profile.allow_sustained_path {
                    if metrics.is_intentional_change
                        && metrics.max_deviation >= DEVIATION_SIGNIFICANT
                    {
                        if self.is_in_curve {
                            info!(
                                "🌀 Curve rejection [GRADUAL]: max={:.1}%, span={:.1}s",
                                metrics.max_deviation * 100.0,
                                metrics.time_span_ms / 1000.0
                            );
                            return LaneChangeState::Centered;
                        }

                        if self.is_deviation_sustained_long(DEVIATION_DRIFT_START)
                            && metrics.time_span_ms >= 5000.0
                        {
                            self.change_detection_path = Some(DetectionPath::GradualChange);
                            self.pending_span_duration = Some(metrics.time_span_ms);

                            info!(
                                "🚨 [GRADUAL] max={:.1}%, span={:.1}s",
                                metrics.max_deviation * 100.0,
                                metrics.time_span_ms / 1000.0
                            );
                            return LaneChangeState::Drifting;
                        }
                    }
                }

                // PATH 6: LARGE DEVIATION
                if deviation >= DEVIATION_LANE_CENTER {
                    if self.is_deviation_sustained(drift_threshold) {
                        self.change_detection_path = Some(DetectionPath::LargeDeviation);
                        info!("🚨 [LARGE] dev={:.1}%", deviation * 100.0);
                        return LaneChangeState::Drifting;
                    }
                }

                LaneChangeState::Centered
            }

            LaneChangeState::Drifting => {
                if deviation >= crossing_threshold {
                    return LaneChangeState::Crossing;
                }

                if self.max_offset_in_change >= crossing_threshold {
                    return LaneChangeState::Crossing;
                }

                if self.frames_in_state > 45 && self.max_offset_in_change >= MEDIUM_OFFSET_THRESHOLD
                {
                    if self.is_deviation_stable() {
                        info!(
                            "✅ DRIFTING stabilized with max={:.1}%",
                            self.max_offset_in_change * 100.0
                        );
                        return LaneChangeState::Completed;
                    }
                }

                let cancel_threshold = drift_threshold * HYSTERESIS_EXIT;
                if deviation < cancel_threshold && self.max_offset_in_change < LOW_OFFSET_THRESHOLD
                {
                    warn!(
                        "❌ Cancelled: max={:.1}% < {:.1}%",
                        self.max_offset_in_change * 100.0,
                        LOW_OFFSET_THRESHOLD * 100.0
                    );
                    return LaneChangeState::Centered;
                }

                LaneChangeState::Drifting
            }

            LaneChangeState::Crossing => {
                let deviation_change = (deviation - self.last_deviation).abs();
                if deviation_change < 0.03 {
                    self.stable_deviation_frames += 1;
                } else {
                    self.stable_deviation_frames = 0;
                }
                self.last_deviation = deviation;

                if self.is_deviation_stable() && deviation < 0.40 {
                    return LaneChangeState::Completed;
                }

                let return_threshold = drift_threshold * HYSTERESIS_EXIT;
                if deviation < return_threshold {
                    return LaneChangeState::Completed;
                }

                if self.stable_deviation_frames >= 25 && deviation < 0.50 {
                    return LaneChangeState::Completed;
                }

                if self.max_offset_in_change >= crossing_threshold
                    && current_direction != self.change_direction
                    && current_direction != Direction::Unknown
                {
                    let reversal_count = self
                        .direction_samples
                        .iter()
                        .rev()
                        .take(10)
                        .filter(|&&d| d != self.change_direction && d != Direction::Unknown)
                        .count();
                    if reversal_count >= 6 {
                        return LaneChangeState::Completed;
                    }
                }

                LaneChangeState::Crossing
            }

            LaneChangeState::Completed => LaneChangeState::Centered,
        }
    }

    pub fn current_state(&self) -> &str {
        self.state.as_str()
    }

    pub fn update_curve_detector(&mut self, lanes: &[crate::types::Lane]) -> bool {
        self.is_in_curve = self.curve_detector.is_in_curve(lanes);
        self.curve_compensation_factor = CURVE_COMPENSATION_FACTOR;
        self.is_in_curve
    }

    fn reset_blackout_state(&mut self) {
        self.in_blackout = false;
        self.blackout_consecutive_frames = 0;
        self.position_before_blackout = None;
        self.blackout_started_frame = None;
        self.blackout_started_time = None;
    }

    fn handle_timeouts(&mut self, frame_id: u64, timestamp_ms: f64) -> Option<LaneChangeEvent> {
        if self.state == LaneChangeState::Drifting {
            if let Some(start_time) = self.change_start_time {
                let elapsed = timestamp_ms - start_time;

                if elapsed > MAX_DRIFTING_MS {
                    if self.max_offset_in_change >= LOW_OFFSET_THRESHOLD {
                        let drift_duration_s = (elapsed / 1000.0) as f32;
                        let drift_rate = if drift_duration_s > 0.0 {
                            (self.max_offset_in_change * 100.0) / drift_duration_s
                        } else {
                            999.0
                        };
                        if drift_rate < MIN_DRIFT_RATE_PER_SEC {
                            warn!(
                                "❌ Force-complete rejected: drift rate {:.1}%/s < {:.1}%/s min",
                                drift_rate, MIN_DRIFT_RATE_PER_SEC
                            );
                            self.adaptive_baseline.unfreeze();
                            self.reset_lane_change();
                            self.cooldown_remaining = 60;
                            return None;
                        }
                        info!(
                            "⏰ Long DRIFTING ({:.0}ms) with offset ({:.1}%), rate={:.1}%/s — auto-completing",
                            elapsed,
                            self.max_offset_in_change * 100.0,
                            drift_rate
                        );
                        return self.force_complete(frame_id, timestamp_ms);
                    } else {
                        warn!("⏰ Timeout with low offset — cancelling");
                        self.adaptive_baseline.unfreeze();
                        self.reset_lane_change();
                        self.cooldown_remaining = 30;
                        return None;
                    }
                }

                if elapsed > self.config.max_duration_ms {
                    if self.max_offset_in_change >= LOW_OFFSET_THRESHOLD {
                        let drift_duration_s = (elapsed / 1000.0) as f32;
                        let drift_rate = if drift_duration_s > 0.0 {
                            (self.max_offset_in_change * 100.0) / drift_duration_s
                        } else {
                            999.0
                        };
                        if drift_rate < MIN_DRIFT_RATE_PER_SEC {
                            warn!(
                                "❌ Timeout force-complete rejected: drift rate {:.1}%/s < {:.1}%/s min",
                                drift_rate, MIN_DRIFT_RATE_PER_SEC
                            );
                            self.adaptive_baseline.unfreeze();
                            self.reset_lane_change();
                            self.cooldown_remaining = 60;
                            return None;
                        }
                        return self.force_complete(frame_id, timestamp_ms);
                    }
                    warn!(
                        "⏰ Timeout after {:.0}ms with max={:.1}%",
                        elapsed,
                        self.max_offset_in_change * 100.0
                    );
                    self.adaptive_baseline.unfreeze();
                    self.reset_lane_change();
                    self.cooldown_remaining = 30;
                }
            }
        }

        if self.state == LaneChangeState::Crossing {
            if let Some(start_time) = self.change_start_time {
                let elapsed = timestamp_ms - start_time;
                if elapsed > self.config.max_duration_ms {
                    return self.force_complete(frame_id, timestamp_ms);
                }
            }
        }

        None
    }

    fn force_complete(&mut self, frame_id: u64, timestamp_ms: f64) -> Option<LaneChangeEvent> {
        let start_frame = self.change_start_frame.unwrap_or(frame_id);
        let start_time = self.change_start_time.unwrap_or(timestamp_ms);
        let duration_ms = Some(timestamp_ms - start_time);
        let duration = duration_ms.unwrap_or(0.0);

        let final_position = self.offset_history.last().copied().unwrap_or(0.0);
        let net_displacement = self
            .initial_position_frozen
            .map(|initial| (final_position - initial).abs())
            .unwrap_or(0.0);

        let trajectory_analysis = self.trajectory_analyzer.analyze_overtake_pattern(
            self.initial_position_frozen.unwrap_or(0.0),
            final_position,
            self.max_offset_in_change,
        );

        if !self.validate_lane_change_adaptive(duration, net_displacement, &trajectory_analysis) {
            warn!(
                "⏰ Force complete rejected: max={:.1}%, dur={:.0}ms",
                self.max_offset_in_change * 100.0,
                duration
            );
            self.adaptive_baseline.unfreeze();
            self.reset_lane_change();
            self.cooldown_remaining = 60;
            return None;
        }

        let confidence = ConfidenceCalculator::calculate(
            self.max_offset_in_change,
            duration,
            &trajectory_analysis,
            self.change_detection_path,
        );

        let mut event = LaneChangeEvent::new(
            start_time,
            start_frame,
            frame_id,
            self.change_direction,
            confidence,
        );
        event.duration_ms = duration_ms;
        event.source_id = self.source_id.clone();

        info!(
            "✅ FORCE CONFIRMED: {} at {:.2}s, dur={:.0}ms, max={:.1}%, conf={:.2}",
            event.direction_name(),
            start_time / 1000.0,
            duration,
            self.max_offset_in_change * 100.0,
            confidence
        );

        let current_pos = self.offset_history.last().copied().unwrap_or(0.0);
        self.adaptive_baseline.reset_to(current_pos);
        self.adaptive_baseline.seed_lock_frames = 900;

        self.position_filter.reset();
        self.post_lane_change_grace = POST_CHANGE_GRACE_FRAMES;
        self.offset_samples.clear();
        self.cooldown_remaining = self.config.cooldown_frames;

        info!(
            "🔄 Baseline seeded at {:.1}% with 900-frame lock — ready for return detection",
            current_pos * 100.0
        );
        self.reset_lane_change();

        Some(event)
    }

    // 🆕 PROFILE-BASED VALIDATION
    fn validate_lane_change_adaptive(
        &self,
        duration: f64,
        net_displacement: f32,
        trajectory_analysis: &OvertakeAnalysis,
    ) -> bool {
        let profile = &self.current_profile;

        // ══════════════════════════════════════════════════════════════
        // PROFILE GATE 1: Minimum average confidence
        // ══════════════════════════════════════════════════════════════
        let avg_confidence = self.get_average_detection_confidence();
        if avg_confidence < profile.min_lane_confidence {
            warn!(
                "❌ [{}] Rejected: avg_confidence {:.2} < {:.2}",
                profile.name, avg_confidence, profile.min_lane_confidence
            );
            return false;
        }

        // ══════════════════════════════════════════════════════════════
        // PROFILE GATE 2: Minimum duration
        // ══════════════════════════════════════════════════════════════
        let effective_duration = if matches!(
            self.change_detection_path,
            Some(DetectionPath::GradualChange)
        ) {
            self.pending_span_duration.unwrap_or(duration).max(duration)
        } else {
            duration
        };

        if effective_duration < profile.min_duration_ms {
            warn!(
                "❌ [{}] Rejected: duration {:.0}ms < {:.0}ms",
                profile.name, effective_duration, profile.min_duration_ms
            );
            return false;
        }

        // ══════════════════════════════════════════════════════════════
        // PROFILE GATE 3: YOLOv8 validation requirement
        // ══════════════════════════════════════════════════════════════
        if profile.require_yolo_validation {
            if !self.has_yolo_confirmation() {
                warn!(
                    "❌ [{}] Rejected: YOLOv8 validation required but not present",
                    profile.name
                );
                return false;
            }
        }

        // ══════════════════════════════════════════════════════════════
        // PROFILE GATE 4: Boundary crossing requirement
        // ══════════════════════════════════════════════════════════════
        if profile.require_boundary_crossing {
            if !self.had_boundary_crossing() {
                warn!("❌ [{}] Rejected: Boundary crossing required", profile.name);
                return false;
            }
        }

        // ══════════════════════════════════════════════════════════════
        // PROFILE GATE 5: SUSTAINED path allowed?
        // ══════════════════════════════════════════════════════════════
        if !profile.allow_sustained_path {
            if matches!(
                self.change_detection_path,
                Some(DetectionPath::GradualChange)
                    | Some(DetectionPath::MediumDeviation)
                    | Some(DetectionPath::PostOcclusion)
            ) {
                warn!(
                    "❌ [{}] Rejected: SUSTAINED path disabled in this profile",
                    profile.name
                );
                return false;
            }
        }

        // ══════════════════════════════════════════════════════════════
        // STANDARD TIER-BASED VALIDATION (original logic)
        // ══════════════════════════════════════════════════════════════
        self.validate_lane_change(duration, net_displacement, trajectory_analysis)
    }

    fn validate_lane_change(
        &self,
        duration: f64,
        net_displacement: f32,
        trajectory_analysis: &OvertakeAnalysis,
    ) -> bool {
        // (Keep your existing validation logic exactly as is)
        // This provides the tier-based checks (TIER 1-4)

        let (min_duration_ms, effective_duration, min_offset, path_name) =
            match self.change_detection_path {
                Some(DetectionPath::GradualChange) => {
                    let span = self.pending_span_duration.unwrap_or(duration);
                    let effective = span.max(duration);
                    (3000.0, effective, MIN_OFFSET_GRADUAL, "GRADUAL")
                }

                Some(DetectionPath::BoundaryCrossing) => {
                    (600.0, duration, MIN_OFFSET_BOUNDARY, "BOUNDARY")
                }

                Some(DetectionPath::HighVelocity) | Some(DetectionPath::VelocitySpike) => {
                    (800.0, duration, MIN_OFFSET_HIGH_VEL, "HIGH-VEL")
                }

                Some(DetectionPath::PostOcclusion)
                | Some(DetectionPath::BlackoutRecovery)
                | Some(DetectionPath::StaticHighDeviation) => {
                    (1500.0, duration, MIN_OFFSET_GRADUAL, "POST-OCCLUSION")
                }

                Some(DetectionPath::MediumDeviation) | Some(DetectionPath::LargeDeviation) => {
                    (1000.0, duration, MIN_OFFSET_DEFAULT, "DEVIATION")
                }

                _ => (1200.0, duration, MIN_OFFSET_DEFAULT, "DEFAULT"),
            };

        if effective_duration < min_duration_ms {
            warn!(
                "❌ Rejected [{}]: effective_dur={:.0}ms < {:.0}ms minimum (state_dur={:.0}ms)",
                path_name, effective_duration, min_duration_ms, duration
            );
            return false;
        }

        if self.max_offset_in_change < min_offset {
            warn!(
                "❌ Rejected [{}]: max offset {:.1}% < {:.1}% path minimum",
                path_name,
                self.max_offset_in_change * 100.0,
                min_offset * 100.0
            );
            return false;
        }

        if !self.adaptive_baseline.is_sane()
            && self.max_offset_in_change < VERY_HIGH_OFFSET_THRESHOLD
        {
            warn!(
                "❌ Rejected [{}]: baseline at {:.1}% and offset only {:.1}%",
                path_name,
                self.adaptive_baseline.frozen_value * 100.0,
                self.max_offset_in_change * 100.0
            );
            return false;
        }

        // TIER 1: Very high offset
        if self.max_offset_in_change >= VERY_HIGH_OFFSET_THRESHOLD {
            if effective_duration >= MIN_DURATION_VERY_HIGH {
                info!(
                    "✅ Valid [{}]: very high offset {:.1}% with dur={:.0}ms",
                    path_name,
                    self.max_offset_in_change * 100.0,
                    effective_duration
                );
                return true;
            } else {
                warn!(
                "❌ Rejected [{}]: very high offset {:.1}% but dur={:.0}ms < {:.0}ms (likely noise spike)",
                path_name,
                self.max_offset_in_change * 100.0,
                effective_duration,
                MIN_DURATION_VERY_HIGH
            );
                return false;
            }
        }

        // TIER 2: High offset
        if self.max_offset_in_change >= HIGH_OFFSET_THRESHOLD {
            let has_duration = effective_duration >= MIN_DURATION_HIGH;
            let has_trajectory =
                trajectory_analysis.is_valid_overtake && trajectory_analysis.shape_score >= 0.65;

            let is_strong_path = matches!(
                self.change_detection_path,
                Some(DetectionPath::BoundaryCrossing) | Some(DetectionPath::HighVelocity)
            );

            if has_duration && (has_trajectory || is_strong_path) {
                info!(
                    "✅ Valid [{}]: high offset {:.1}%, dur={:.0}ms, traj={:.2}",
                    path_name,
                    self.max_offset_in_change * 100.0,
                    effective_duration,
                    trajectory_analysis.shape_score
                );
                return true;
            }

            warn!(
                "❌ Rejected [{}]: high offset {:.1}% but dur={} traj={}",
                path_name,
                self.max_offset_in_change * 100.0,
                if has_duration { "✓" } else { "✗" },
                if has_trajectory { "✓" } else { "✗" }
            );
            return false;
        }

        // TIER 3: Medium offset
        if self.max_offset_in_change >= MEDIUM_OFFSET_THRESHOLD {
            let has_duration = effective_duration >= MIN_DURATION_MEDIUM;
            let has_trajectory =
                trajectory_analysis.is_valid_overtake && trajectory_analysis.shape_score >= 0.55;
            let has_displacement = net_displacement >= MIN_NET_DISPLACEMENT;

            let conditions_met = [has_duration, has_trajectory, has_displacement]
                .iter()
                .filter(|&&x| x)
                .count();

            if conditions_met >= 2 {
                info!(
                "✅ Valid [{}]: medium offset {:.1}%, {}/3 criteria met (dur={} traj={} disp={})",
                path_name,
                self.max_offset_in_change * 100.0,
                conditions_met,
                if has_duration { "✓" } else { "✗" },
                if has_trajectory { "✓" } else { "✗" },
                if has_displacement { "✓" } else { "✗" }
            );
                return true;
            }

            warn!(
                "❌ Rejected [{}]: medium offset {:.1}%, dur={} traj={} disp={} (need 2 of 3)",
                path_name,
                self.max_offset_in_change * 100.0,
                if has_duration { "✓" } else { "✗" },
                if has_trajectory { "✓" } else { "✗" },
                if has_displacement { "✓" } else { "✗" }
            );
            return false;
        }

        // TIER 4: Low offset
        let drift_duration_s = (effective_duration / 1000.0) as f32;
        let drift_rate = if drift_duration_s > 0.0 {
            (self.max_offset_in_change * 100.0) / drift_duration_s
        } else {
            999.0
        };

        if drift_rate < MIN_DRIFT_RATE_PER_SEC {
            warn!(
            "❌ Rejected [{}]: LOW offset {:.1}% with slow drift rate {:.1}%/s (< {:.1}%/s min)",
            path_name,
            self.max_offset_in_change * 100.0,
            drift_rate,
            MIN_DRIFT_RATE_PER_SEC
        );
            return false;
        }

        if self.change_detection_path == Some(DetectionPath::BoundaryCrossing) {
            if effective_duration >= 1500.0 && drift_rate >= MIN_DRIFT_RATE_PER_SEC {
                info!(
                    "✅ Valid [BOUNDARY]: low offset {:.1}%, rate={:.1}%/s, dur={:.0}ms",
                    self.max_offset_in_change * 100.0,
                    drift_rate,
                    effective_duration
                );
                return true;
            }
        }

        let has_duration = effective_duration >= MIN_DURATION_LOW;
        let has_trajectory = trajectory_analysis.is_valid_overtake;
        let has_displacement = net_displacement >= MIN_NET_DISPLACEMENT;

        let conditions_met = [has_duration, has_trajectory, has_displacement]
            .iter()
            .filter(|&&x| x)
            .count();

        if conditions_met >= 2 && drift_rate >= MIN_DRIFT_RATE_PER_SEC {
            info!(
                "✅ Valid [{}]: lower offset {:.1}%, {}/3 conditions met, rate={:.1}%/s",
                path_name,
                self.max_offset_in_change * 100.0,
                conditions_met,
                drift_rate
            );
            return true;
        }

        warn!(
        "❌ Rejected [{}]: offset {:.1}%, dur={} traj={} disp={}, rate={:.1}%/s (need 2 of 3 + good rate)",
        path_name,
        self.max_offset_in_change * 100.0,
        if has_duration { "✓" } else { "✗" },
        if has_trajectory { "✓" } else { "✗" },
        if has_displacement { "✓" } else { "✗" },
        drift_rate
    );
        false
    }

    // 🆕 HELPER: Get average detection confidence during maneuver
    fn get_average_detection_confidence(&self) -> f32 {
        if self.offset_samples.is_empty() {
            return 0.0;
        }

        let start_time = self.change_start_time.unwrap_or(0.0);
        let relevant_samples: Vec<f32> = self
            .offset_samples
            .iter()
            .filter(|s| s.timestamp_ms >= start_time)
            .map(|s| s.confidence)
            .collect();

        if relevant_samples.is_empty() {
            return 0.0;
        }

        relevant_samples.iter().sum::<f32>() / relevant_samples.len() as f32
    }

    // 🆕 HELPER: Check if YOLOv8 confirmed the detection
    fn has_yolo_confirmation(&self) -> bool {
        // For now, proxy with boundary crossing
        // In full implementation, track YOLOv8 detections separately
        matches!(
            self.change_detection_path,
            Some(DetectionPath::BoundaryCrossing)
        )
    }

    // 🆕 HELPER: Check if boundary crossing occurred
    fn had_boundary_crossing(&self) -> bool {
        matches!(
            self.change_detection_path,
            Some(DetectionPath::BoundaryCrossing)
        )
    }

    fn get_recent_position_change(&self) -> f32 {
        if self.offset_history.len() < 10 {
            return 0.0;
        }

        let recent = &self.offset_history[self.offset_history.len() - 10..];
        let first = recent[0];
        let last = recent[recent.len() - 1];

        let max = recent.iter().fold(f32::MIN, |a, &b| a.max(b));
        let min = recent.iter().fold(f32::MAX, |a, &b| a.min(b));
        let swing = max - min;

        (last - first).abs().max(swing)
    }

    fn calculate_window_metrics(&self, _current_time_ms: f64, _lane_width: f32) -> WindowMetrics {
        let mut metrics = WindowMetrics::default();

        if self.offset_samples.len() < 10 {
            return metrics;
        }

        let first = self.offset_samples.front().unwrap();
        let last = self.offset_samples.back().unwrap();

        metrics.time_span_ms = last.timestamp_ms - first.timestamp_ms;
        if metrics.time_span_ms < 300.0 {
            return metrics;
        }

        metrics.total_displacement = (last.deviation - first.deviation).abs();
        metrics.max_deviation = self
            .offset_samples
            .iter()
            .map(|s| s.deviation)
            .fold(0.0f32, |a, b| a.max(b));

        let velocities: Vec<f32> = self
            .offset_samples
            .iter()
            .map(|s| s.lateral_velocity)
            .collect();
        metrics.avg_velocity = velocities.iter().sum::<f32>() / velocities.len() as f32;
        metrics.peak_velocity = velocities
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, |a, b| a.max(b));

        if !self.direction_samples.is_empty() {
            let target_dir = self.peak_direction;
            let consistent = self
                .direction_samples
                .iter()
                .filter(|&&d| d == target_dir)
                .count();
            metrics.direction_consistency = consistent as f32 / self.direction_samples.len() as f32;
        }

        // Use adaptive threshold for consistency
        let consistency_threshold = self.adaptive_thresholds.consistency_threshold.get();

        metrics.is_sustained_movement = metrics.direction_consistency >= consistency_threshold
            && metrics.time_span_ms >= 1000.0;
        metrics.is_intentional_change = metrics.max_deviation >= DEVIATION_DRIFT_START
            && metrics.is_sustained_movement
            && (metrics.avg_velocity.abs() > MIN_VELOCITY_SLOW || metrics.time_span_ms >= 2000.0);

        metrics
    }

    fn is_deviation_sustained(&self, threshold: f32) -> bool {
        if self.offset_history.len() < 8 {
            return false;
        }
        let baseline = self.adaptive_baseline.effective_value();
        self.offset_history
            .iter()
            .rev()
            .take(6)
            .filter(|o| (*o - baseline).abs() >= threshold)
            .count()
            >= 4
    }

    fn is_deviation_sustained_long(&self, threshold: f32) -> bool {
        if self.offset_history.len() < 20 {
            return false;
        }
        let baseline = self.adaptive_baseline.effective_value();
        self.offset_history
            .iter()
            .rev()
            .take(15)
            .filter(|o| (*o - baseline).abs() >= threshold)
            .count()
            >= 10
    }

    fn is_velocity_sustained(&self, threshold: f32) -> bool {
        if self.velocity_history.len() < 5 {
            return false;
        }
        self.velocity_history
            .iter()
            .rev()
            .take(5)
            .filter(|v| v.abs() >= threshold)
            .count()
            >= 3
    }

    fn is_deviation_stable(&self) -> bool {
        if self.recent_deviations.len() < 12 {
            return false;
        }
        let recent = &self.recent_deviations[self.recent_deviations.len() - 12..];
        let max = recent.iter().fold(f32::MIN, |a, &b| a.max(b));
        let min = recent.iter().fold(f32::MAX, |a, &b| a.min(b));
        if max - min > 0.12 {
            return false;
        }
        recent
            .windows(2)
            .filter(|w| (w[1] - w[0]).abs() > 0.04)
            .count()
            <= 4
    }

    fn reset_lane_change(&mut self) {
        self.state = LaneChangeState::Centered;
        self.frames_in_state = 0;
        self.pending_state = None;
        self.pending_frames = 0;
        self.change_direction = Direction::Unknown;
        self.change_start_frame = None;
        self.change_start_time = None;
        self.change_detection_path = None;
        self.max_offset_in_change = 0.0;
        self.initial_position_frozen = None;
        self.stable_deviation_frames = 0;
        self.last_deviation = 0.0;
        self.recent_deviations.clear();
        self.peak_deviation_in_window = 0.0;
        self.peak_velocity_in_window = 0.0;
        self.peak_direction = Direction::Unknown;
        self.pending_change_direction = Direction::Unknown;
        self.pending_max_offset = 0.0;
        self.trajectory_analyzer.clear();

        self.pending_span_duration = None;
        self.detection_start_timestamp = None;
    }

    fn finalize_lane_change(&mut self) {
        let current_pos = self.offset_history.last().copied().unwrap_or(0.0);
        self.adaptive_baseline.reset_to(current_pos);
        self.adaptive_baseline.seed_lock_frames = 900;

        self.position_filter.reset();
        self.post_lane_change_grace = POST_CHANGE_GRACE_FRAMES;
        self.offset_samples.clear();
        self.cooldown_remaining = self.config.cooldown_frames;

        info!(
            "🔄 Baseline seeded at {:.1}% with 900-frame lock — ready for return detection",
            current_pos * 100.0
        );
        self.reset_lane_change();
    }

    fn check_transition(
        &mut self,
        target_state: LaneChangeState,
        direction: Direction,
        frame_id: u64,
        timestamp_ms: f64,
        current_deviation: f32,
        normalized_offset: f32,
    ) -> Option<LaneChangeEvent> {
        if target_state == self.state {
            self.pending_state = None;
            self.pending_frames = 0;
            self.frames_in_state += 1;
            return None;
        }

        if target_state == LaneChangeState::Drifting && self.state == LaneChangeState::Centered {
            if self.pending_state != Some(LaneChangeState::Drifting) {
                self.adaptive_baseline.freeze();
                self.initial_position_frozen = Some(normalized_offset);
                self.pending_change_direction = direction;
                self.pending_max_offset = current_deviation;
                info!(
                    "🧊 Baseline frozen at {:.1}%, initial position: {:.1}%",
                    self.adaptive_baseline.effective_value() * 100.0,
                    normalized_offset * 100.0
                );
            }
        }

        if self.pending_state == Some(target_state) {
            self.pending_frames += 1;
        } else {
            self.pending_state = Some(target_state);
            self.pending_frames = 1;
        }

        if target_state != LaneChangeState::Drifting
            && self.adaptive_baseline.is_frozen
            && self.state == LaneChangeState::Centered
            && self.pending_frames < self.config.min_frames_confirm
        {
            self.adaptive_baseline.unfreeze();
            self.initial_position_frozen = None;
            self.pending_max_offset = 0.0;
        }

        if self.pending_frames < self.config.min_frames_confirm {
            return None;
        }

        self.execute_transition(
            target_state,
            direction,
            frame_id,
            timestamp_ms,
            normalized_offset,
        )
    }

    fn execute_transition(
        &mut self,
        target_state: LaneChangeState,
        direction: Direction,
        frame_id: u64,
        timestamp_ms: f64,
        final_position: f32,
    ) -> Option<LaneChangeEvent> {
        let from_state = self.state;

        info!(
            "State: {:?} → {:?} at frame {} ({:.2}s)",
            from_state,
            target_state,
            frame_id,
            timestamp_ms / 1000.0
        );

        if target_state == LaneChangeState::Drifting && from_state == LaneChangeState::Centered {
            self.change_direction = if self.pending_change_direction != Direction::Unknown {
                self.pending_change_direction
            } else {
                direction
            };
            self.change_start_frame = Some(frame_id);
            self.change_start_time = Some(timestamp_ms);
            self.max_offset_in_change = self.pending_max_offset;
            self.stable_deviation_frames = 0;
            self.last_deviation = 0.0;

            if !self.adaptive_baseline.is_frozen {
                self.adaptive_baseline.freeze();
                self.initial_position_frozen = Some(final_position);
            }

            info!(
                "🚗 Lane change started: {} at {:.2}s via {:?} (max: {:.1}%)",
                self.change_direction.as_str(),
                timestamp_ms / 1000.0,
                self.change_detection_path,
                self.max_offset_in_change * 100.0
            );
        }

        if target_state == LaneChangeState::Centered && from_state == LaneChangeState::Drifting {
            info!("↩️ Cancelled");
            self.adaptive_baseline.unfreeze();
            self.reset_lane_change();
            self.cooldown_remaining = 30;
            return None;
        }

        let duration_ms = if target_state == LaneChangeState::Completed {
            self.change_start_time.map(|start| timestamp_ms - start)
        } else {
            None
        };

        self.state = target_state;
        self.frames_in_state = 0;
        self.pending_state = None;
        self.pending_frames = 0;

        if target_state == LaneChangeState::Completed {
            let duration = duration_ms.unwrap_or(0.0);

            let net_displacement = if let Some(initial_pos) = self.initial_position_frozen {
                (final_position - initial_pos).abs()
            } else {
                0.0
            };

            let trajectory_analysis = self.trajectory_analyzer.analyze_overtake_pattern(
                self.initial_position_frozen.unwrap_or(0.0),
                final_position,
                self.max_offset_in_change,
            );

            info!(
                "📊 Trajectory: excursion={}, returned={}, shape={:.2}",
                trajectory_analysis.excursion_sufficient,
                trajectory_analysis.returned_to_start,
                trajectory_analysis.shape_score
            );

            // 🆕 Use adaptive validation
            if !self.validate_lane_change_adaptive(duration, net_displacement, &trajectory_analysis)
            {
                self.adaptive_baseline.unfreeze();
                self.reset_lane_change();
                self.cooldown_remaining = 60;
                return None;
            }

            let confidence = ConfidenceCalculator::calculate(
                self.max_offset_in_change,
                duration,
                &trajectory_analysis,
                self.change_detection_path,
            );

            let start_frame = self.change_start_frame.unwrap_or(frame_id);
            let start_time = self.change_start_time.unwrap_or(timestamp_ms);

            let mut event = LaneChangeEvent::new(
                start_time,
                start_frame,
                frame_id,
                self.change_direction,
                confidence,
            );
            event.duration_ms = duration_ms;
            event.source_id = self.source_id.clone();

            event.metadata.insert(
                "max_offset_normalized".to_string(),
                serde_json::json!(self.max_offset_in_change),
            );
            event.metadata.insert(
                "trajectory_shape_score".to_string(),
                serde_json::json!(trajectory_analysis.shape_score),
            );
            event.metadata.insert(
                "detection_profile".to_string(),
                serde_json::json!(self.current_profile.name),
            );
            event.metadata.insert(
                "road_context".to_string(),
                serde_json::json!(format!("{:?}", self.last_profile_context)),
            );

            if let Some(path) = self.change_detection_path {
                event.metadata.insert(
                    "detection_path".to_string(),
                    serde_json::json!(format!("{:?}", path)),
                );
            }

            info!(
                "✅ CONFIRMED [{}]: {} at {:.2}s, dur={:.0}ms, max={:.1}%, conf={:.2}",
                self.current_profile.name,
                event.direction_name(),
                start_time / 1000.0,
                duration,
                self.max_offset_in_change * 100.0,
                confidence
            );

            self.finalize_lane_change();
            return Some(event);
        }

        None
    }

    pub fn reset(&mut self) {
        self.reset_lane_change();
        self.adaptive_baseline.unfreeze();
        self.cooldown_remaining = 0;
        self.total_frames_processed = 0;
        self.post_lane_change_grace = 0;
        self.position_filter.reset();
        self.adaptive_baseline.reset();
        self.offset_history.clear();
        self.velocity_history.clear();
        self.curve_detector.reset();
        self.velocity_tracker.reset();
        self.offset_samples.clear();
        self.direction_samples.clear();
        self.trajectory_analyzer.clear();
        self.is_in_curve = false;
        self.curve_compensation_factor = 1.0;
        self.reset_blackout_state();

        // 🆕 Reset adaptive components
        self.adaptive_thresholds.reset();
        self.context_detector.reset();
        self.current_profile = MiningRouteProfile::paved();
        self.last_profile_context = RoadContext::Unknown;
    }

    pub fn set_source_id(&mut self, source_id: String) {
        self.source_id = source_id;
    }

    // Note: update_perfect() method kept for blackout recovery
    // (Same as your existing implementation)
    pub fn update_perfect(
        &mut self,
        vehicle_state: &VehicleState,
        frame_id: u64,
        timestamp_ms: f64,
        crossing_type: CrossingType,
    ) -> Option<LaneChangeEvent> {
        // Just call the main update - adaptive system handles everything
        self.update(vehicle_state, frame_id, timestamp_ms, crossing_type)
    }
}

