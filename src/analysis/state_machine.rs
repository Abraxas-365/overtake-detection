// src/analysis/state_machine.rs
//
// LANE CHANGE DETECTION v3.8.1 - BLACKOUT RECOVERY (FIXED)
//
// v3.8.1: Fixed 3 critical bugs from v3.8:
//   1. reset_blackout_state() no longer corrupts adaptive_baseline.frames_without_lanes
//   2. Blackout recovery uses SEPARATE counter (not baseline's shared counter)
//   3. Kalman filter no longer double-updated on recovery frames
//

use super::boundary_detector::CrossingType;
use super::curve_detector::CurveDetector;
use super::velocity_tracker::LateralVelocityTracker;
use crate::types::{Direction, LaneChangeConfig, LaneChangeEvent, LaneChangeState, VehicleState};
use std::collections::VecDeque;
use tracing::{debug, info, warn};

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
// DEVIATION THRESHOLDS
// ============================================================================
const DEVIATION_DRIFT_START: f32 = 0.22;
const DEVIATION_CROSSING: f32 = 0.25;
const DEVIATION_LANE_CENTER: f32 = 0.55;
const DEVIATION_SIGNIFICANT: f32 = 0.45;

// ============================================================================
// STATE MACHINE BEHAVIOR
// ============================================================================
const HYSTERESIS_EXIT: f32 = 0.5;
const DIRECTION_CONSISTENCY_THRESHOLD: f32 = 0.65;
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
const CURVE_CURVATURE_THRESHOLD: f32 = 0.015;
const MIN_DRIFT_RATE_PER_SEC: f32 = 5.0;

// ============================================================================
// VALIDATION THRESHOLDS
// ============================================================================
const VERY_HIGH_OFFSET_THRESHOLD: f32 = 0.55;
const HIGH_OFFSET_THRESHOLD: f32 = 0.45;
const MEDIUM_OFFSET_THRESHOLD: f32 = 0.35;
const LOW_OFFSET_THRESHOLD: f32 = 0.25;

const MIN_DURATION_VERY_HIGH: f64 = 400.0;
const MIN_DURATION_HIGH: f64 = 600.0;
const MIN_DURATION_MEDIUM: f64 = 1000.0;
const MIN_DURATION_LOW: f64 = 1800.0;

const MIN_NET_DISPLACEMENT: f32 = 0.15;

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

        // 🔧 NEW: Seed lock takes priority — keeps baseline anchored after lane change
        let alpha = if self.seed_lock_frames > 0 {
            self.seed_lock_frames -= 1;
            if self.seed_lock_frames == 0 {
                info!("🔓 Baseline seed lock expired");
            }
            0.0005 // Ultra-slow: baseline barely moves during return detection window
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

    fn update_guarded(&mut self, measurement: f32, confidence: f32) -> f32 {
        if confidence < 0.4 {
            self.frames_without_lanes += 1;
            return self.value;
        }

        self.sample_count += 1;
        self.frames_without_lanes = 0;

        if self.is_frozen {
            return self.frozen_value;
        }

        if self.sample_count == 1 {
            self.value = measurement;
            return self.value;
        }

        // 🔧 Seed lock takes priority — keeps baseline anchored after lane change
        let alpha = if self.seed_lock_frames > 0 {
            self.seed_lock_frames -= 1;
            if self.seed_lock_frames == 0 {
                info!("🔓 Baseline seed lock expired");
            }
            0.0005
        } else if !self.is_valid {
            EWMA_ALPHA_ADAPTING
        } else if self.is_adapting {
            EWMA_ALPHA_ADAPTING
        } else {
            EWMA_ALPHA_STABLE
        };

        self.value = alpha * measurement + (1.0 - alpha) * self.value;
        self.is_valid = self.sample_count >= EWMA_MIN_SAMPLES;

        self.recent_samples.push_back(measurement);
        if self.recent_samples.len() > 30 {
            self.recent_samples.pop_front();
        }
        if self.recent_samples.len() >= 10 {
            let samples: Vec<f32> = self.recent_samples.iter().copied().collect();
            let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
            self.variance =
                samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / samples.len() as f32;

            if self.variance < STABILITY_VARIANCE_THRESHOLD {
                self.stable_frames += 1;
                if self.stable_frames > 30 {
                    self.is_adapting = false;
                }
            } else {
                self.stable_frames = 0;
            }
        }

        self.value
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
// MAIN STATE MACHINE
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

    // 🔧 FIX 1: Blackout uses its OWN counter, completely separate from baseline
    in_blackout: bool,
    blackout_consecutive_frames: u32,
    position_before_blackout: Option<f32>,
    blackout_started_frame: Option<u64>,
    blackout_started_time: Option<f64>,
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

            // 🔧 FIX 1: Separate blackout state
            in_blackout: false,
            blackout_consecutive_frames: 0,
            position_before_blackout: None,
            blackout_started_frame: None,
            blackout_started_time: None,
        }
    }

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

        if let Some(event) = self.handle_timeouts(frame_id, timestamp_ms) {
            return Some(event);
        }

        if !vehicle_state.is_valid() {
            self.adaptive_baseline.mark_no_lanes();
            return None;
        }

        let lane_width = vehicle_state.lane_width.unwrap();
        let raw_offset = vehicle_state.lateral_offset / lane_width;
        let normalized_offset = self.position_filter.update(raw_offset);

        let lateral_velocity = self
            .velocity_tracker
            .get_velocity(vehicle_state.lateral_offset, timestamp_ms);

        self.update_histories(normalized_offset, lateral_velocity);

        if self.post_lane_change_grace > 0 {
            self.adaptive_baseline.update(normalized_offset);
            return None;
        }

        self.adaptive_baseline.update(normalized_offset);

        if !self.adaptive_baseline.is_valid {
            return None;
        }

        if self.adaptive_baseline.sample_count == EWMA_MIN_SAMPLES {
            info!(
                "✅ Adaptive baseline ready: {:.1}% at frame {} ({:.1}s)",
                self.adaptive_baseline.value * 100.0,
                frame_id,
                timestamp_ms / 1000.0
            );
        }

        let baseline = self.adaptive_baseline.effective_value();
        let signed_deviation = normalized_offset - baseline;
        let deviation = signed_deviation.abs();

        // 🔧 CRITICAL FIX: Handle Lane Coordinate Wrapping (Polarity Flip)
        // When swapping lanes, coordinates jump from Positive Edge (+0.4) to Negative Edge (-0.4) or vice versa.
        // A linear check sees this as a massive Negative change (LEFT), but physically it's a RIGHT move.
        // Logic:
        //   +Baseline to -Current => Moving RIGHT (crossed right boundary)
        //   -Baseline to +Current => Moving LEFT (crossed left boundary)
        let current_direction = if self.adaptive_baseline.is_frozen
            && baseline.abs() > 0.10
            && normalized_offset.abs() > 0.10
            && baseline.signum() != normalized_offset.signum()
            && deviation > 0.40
        {
            if baseline > 0.0 {
                // + to - (e.g. 0.3 to -0.3): Wrapped around Right Edge
                Direction::Right
            } else {
                // - to + (e.g. -0.3 to 0.3): Wrapped around Left Edge
                Direction::Left
            }
        } else {
            // Standard linear drift within lane
            Direction::from_offset(signed_deviation)
        };

        // Track max offset during pending and active phases
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

        self.trajectory_analyzer
            .add_sample(normalized_offset, timestamp_ms, lateral_velocity);
        self.track_max_offset(deviation);
        self.update_samples(
            normalized_offset,
            deviation,
            timestamp_ms,
            lateral_velocity,
            current_direction,
        );

        let window_metrics = self.calculate_window_metrics(timestamp_ms, lane_width);

        let target_state = self.determine_target_state(
            deviation,
            crossing_type,
            lateral_velocity,
            current_direction,
            &window_metrics,
        );

        debug!(
            "F{}: off={:.1}%, base={:.1}%{}, dev={:.1}%, max={:.1}%, state={:?}→{:?}",
            frame_id,
            normalized_offset * 100.0,
            baseline * 100.0,
            if self.adaptive_baseline.is_frozen {
                "🧊"
            } else {
                ""
            },
            deviation * 100.0,
            self.max_offset_in_change * 100.0,
            self.state,
            target_state
        );

        self.check_transition(
            target_state,
            current_direction,
            frame_id,
            timestamp_ms,
            deviation,
            normalized_offset,
        )
    }

    pub fn current_state(&self) -> &str {
        self.state.as_str()
    }

    pub fn update_curve_detector(&mut self, lanes: &[crate::types::Lane]) -> bool {
        self.is_in_curve = self.curve_detector.is_in_curve(lanes);
        self.curve_compensation_factor = CURVE_COMPENSATION_FACTOR;
        self.is_in_curve
    }

    // 🔧 FIX 1: Does NOT touch adaptive_baseline.frames_without_lanes
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

    fn update_histories(&mut self, normalized_offset: f32, lateral_velocity: f32) {
        self.velocity_history.push_back(lateral_velocity);
        if self.velocity_history.len() > 30 {
            self.velocity_history.pop_front();
        }

        self.offset_history.push(normalized_offset);
        if self.offset_history.len() > 90 {
            self.offset_history.remove(0);
        }
    }

    fn track_max_offset(&mut self, deviation: f32) {
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
    }

    fn update_samples(
        &mut self,
        normalized_offset: f32,
        deviation: f32,
        timestamp_ms: f64,
        lateral_velocity: f32,
        current_direction: Direction,
    ) {
        self.recent_deviations.push(deviation);
        if self.recent_deviations.len() > 30 {
            self.recent_deviations.remove(0);
        }

        let sample = OffsetSample {
            normalized_offset,
            deviation,
            timestamp_ms,
            lateral_velocity,
            direction: current_direction,
        };
        self.offset_samples.push_back(sample);

        while let Some(oldest) = self.offset_samples.front() {
            if timestamp_ms - oldest.timestamp_ms > ANALYSIS_WINDOW_MS {
                self.offset_samples.pop_front();
            } else {
                break;
            }
        }

        self.direction_samples.push_back(current_direction);
        if self.direction_samples.len() > 30 {
            self.direction_samples.pop_front();
        }

        if deviation > self.peak_deviation_in_window {
            self.peak_deviation_in_window = deviation;
            self.peak_direction = current_direction;
        }
        if lateral_velocity.abs() > self.peak_velocity_in_window {
            self.peak_velocity_in_window = lateral_velocity.abs();
        }
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

        if !self.validate_lane_change(duration, net_displacement, &trajectory_analysis) {
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

        // 🔧 FIX: Lock baseline so it doesn't drift during cooldown/grace.
        // Without this, the EWMA baseline tracks the vehicle back toward
        // the original lane during the 90-frame cooldown + 90-frame grace,
        // eating the deviation signal and preventing return detection.
        // 300 frames ≈ 10s at 30fps — covers the typical return window.
        self.adaptive_baseline.seed_lock_frames = 300;

        self.position_filter.reset();
        self.post_lane_change_grace = POST_CHANGE_GRACE_FRAMES;
        self.offset_samples.clear();
        self.cooldown_remaining = self.config.cooldown_frames;

        info!(
            "🔄 Baseline seeded at {:.1}% with 300-frame lock — ready for return detection",
            current_pos * 100.0
        );
        self.reset_lane_change();

        Some(event)
    }

    fn validate_lane_change(
        &self,
        duration: f64,
        net_displacement: f32,
        trajectory_analysis: &OvertakeAnalysis,
    ) -> bool {
        const ABSOLUTE_MIN_DURATION: f64 = 200.0;

        if duration < ABSOLUTE_MIN_DURATION {
            warn!(
                "❌ Rejected: duration {:.0}ms < {:.0}ms absolute minimum",
                duration, ABSOLUTE_MIN_DURATION
            );
            return false;
        }

        if !self.adaptive_baseline.is_sane()
            && self.max_offset_in_change < VERY_HIGH_OFFSET_THRESHOLD
        {
            warn!(
                "❌ Rejected: baseline at {:.1}% and offset only {:.1}%",
                self.adaptive_baseline.frozen_value * 100.0,
                self.max_offset_in_change * 100.0
            );
            return false;
        }

        if self.max_offset_in_change < LOW_OFFSET_THRESHOLD {
            warn!(
                "❌ Rejected: max offset {:.1}% < {:.1}% minimum",
                self.max_offset_in_change * 100.0,
                LOW_OFFSET_THRESHOLD * 100.0
            );
            return false;
        }

        if self.max_offset_in_change >= VERY_HIGH_OFFSET_THRESHOLD {
            info!(
                "✅ Valid: very high offset {:.1}% with dur={:.0}ms",
                self.max_offset_in_change * 100.0,
                duration
            );
            return true;
        }

        if self.max_offset_in_change >= HIGH_OFFSET_THRESHOLD {
            if duration >= MIN_DURATION_HIGH {
                info!(
                    "✅ Valid: high offset {:.1}% with dur={:.0}ms",
                    self.max_offset_in_change * 100.0,
                    duration
                );
                return true;
            }
            if trajectory_analysis.is_valid_overtake && trajectory_analysis.shape_score >= 0.6 {
                info!(
                    "✅ Valid: high offset {:.1}%, short dur but excellent trajectory (score={:.2})",
                    self.max_offset_in_change * 100.0,
                    trajectory_analysis.shape_score
                );
                return true;
            }
            warn!(
                "❌ Rejected: high offset {:.1}% but dur={:.0}ms < {:.0}ms and traj_score={:.2}",
                self.max_offset_in_change * 100.0,
                duration,
                MIN_DURATION_HIGH,
                trajectory_analysis.shape_score
            );
            return false;
        }

        if self.max_offset_in_change >= MEDIUM_OFFSET_THRESHOLD {
            let has_duration = duration >= MIN_DURATION_MEDIUM;
            let has_trajectory =
                trajectory_analysis.is_valid_overtake && trajectory_analysis.shape_score >= 0.5;
            let has_displacement = net_displacement >= MIN_NET_DISPLACEMENT;

            if has_duration && (has_trajectory || has_displacement) {
                info!(
                    "✅ Valid: medium offset {:.1}%, dur=✓ + traj={} disp={}",
                    self.max_offset_in_change * 100.0,
                    if has_trajectory { "✓" } else { "✗" },
                    if has_displacement { "✓" } else { "✗" }
                );
                return true;
            }

            let conditions_met = [has_duration, has_trajectory, has_displacement]
                .iter()
                .filter(|&&x| x)
                .count();

            if conditions_met >= 2 {
                info!(
                    "✅ Valid: medium offset {:.1}%, {}/3 conditions met (dur={} traj={} disp={})",
                    self.max_offset_in_change * 100.0,
                    conditions_met,
                    if has_duration { "✓" } else { "✗" },
                    if has_trajectory { "✓" } else { "✗" },
                    if has_displacement { "✓" } else { "✗" }
                );
                return true;
            }

            warn!(
                "❌ Rejected: medium offset {:.1}%, dur={} traj={} disp={} (need dur+1 or 2of3)",
                self.max_offset_in_change * 100.0,
                if has_duration { "✓" } else { "✗" },
                if has_trajectory { "✓" } else { "✗" },
                if has_displacement { "✓" } else { "✗" }
            );
            return false;
        }

        let drift_duration_s = (duration / 1000.0) as f32;
        let drift_rate = if drift_duration_s > 0.0 {
            (self.max_offset_in_change * 100.0) / drift_duration_s
        } else {
            999.0
        };
        if drift_rate < MIN_DRIFT_RATE_PER_SEC {
            warn!(
                "❌ Rejected: LOW offset {:.1}% over {:.1}s = {:.1}%/s drift rate (< {:.1}%/s min)",
                self.max_offset_in_change * 100.0,
                drift_duration_s,
                drift_rate,
                MIN_DRIFT_RATE_PER_SEC
            );
            return false;
        }

        let has_duration = duration >= MIN_DURATION_LOW;
        let has_trajectory = trajectory_analysis.is_valid_overtake;
        let has_displacement = net_displacement >= MIN_NET_DISPLACEMENT;

        let conditions_met = [has_duration, has_trajectory, has_displacement]
            .iter()
            .filter(|&&x| x)
            .count();

        if conditions_met >= 2 {
            info!(
                "✅ Valid: lower offset {:.1}%, {}/3 conditions met (dur={} traj={} disp={}), rate={:.1}%/s",
                self.max_offset_in_change * 100.0,
                conditions_met,
                if has_duration { "✓" } else { "✗" },
                if has_trajectory { "✓" } else { "✗" },
                if has_displacement { "✓" } else { "✗" },
                drift_rate
            );
            return true;
        }

        warn!(
            "❌ Rejected: offset {:.1}%, dur={} traj={} disp={} (need 2 of 3)",
            self.max_offset_in_change * 100.0,
            if has_duration { "✓" } else { "✗" },
            if has_trajectory { "✓" } else { "✗" },
            if has_displacement { "✓" } else { "✗" }
        );
        false
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

        metrics.is_sustained_movement = metrics.direction_consistency
            >= DIRECTION_CONSISTENCY_THRESHOLD
            && metrics.time_span_ms >= 1000.0;
        metrics.is_intentional_change = metrics.max_deviation >= DEVIATION_DRIFT_START
            && metrics.is_sustained_movement
            && (metrics.avg_velocity.abs() > MIN_VELOCITY_SLOW || metrics.time_span_ms >= 2000.0);

        metrics
    }

    fn determine_target_state(
        &mut self,
        deviation: f32,
        crossing_type: CrossingType,
        lateral_velocity: f32,
        current_direction: Direction,
        metrics: &WindowMetrics,
    ) -> LaneChangeState {
        let drift_threshold = self.config.drift_threshold;
        let crossing_threshold = self.config.crossing_threshold;

        let vel_fast = MIN_VELOCITY_FAST;
        let vel_medium = MIN_VELOCITY_MEDIUM;

        match self.state {
            LaneChangeState::Centered => {
                if !self.adaptive_baseline.is_sane() {
                    return LaneChangeState::Centered;
                }

                // PATH 0: POST-OCCLUSION FAST DETECTION
                if self.adaptive_baseline.frames_without_lanes_recent() > 0
                    && lateral_velocity.abs() > vel_medium
                    && deviation >= drift_threshold * 0.8
                {
                    self.change_detection_path = Some(DetectionPath::PostOcclusion);
                    info!(
                        "🚨 [POST-OCCLUSION] Maneuver likely in progress: vel={:.1}, dev={:.1}%",
                        lateral_velocity,
                        deviation * 100.0
                    );
                    return LaneChangeState::Drifting;
                }

                // PATH 1: BOUNDARY CROSSING
                if crossing_type != CrossingType::None && lateral_velocity.abs() > vel_fast {
                    if self.is_deviation_sustained(drift_threshold * 0.9) {
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

                // PATH 5: GRADUAL CHANGE
                if metrics.is_intentional_change && metrics.max_deviation >= DEVIATION_SIGNIFICANT {
                    if self.is_deviation_sustained_long(DEVIATION_DRIFT_START) {
                        self.change_detection_path = Some(DetectionPath::GradualChange);
                        info!(
                            "🚨 [GRADUAL] max={:.1}%, span={:.1}s",
                            metrics.max_deviation * 100.0,
                            metrics.time_span_ms / 1000.0
                        );
                        return LaneChangeState::Drifting;
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

                // PATH 7: STATIC HIGH DEVIATION (missed lane change during blackout)
                // Vehicle is far from baseline but NOT moving — the return already
                // happened during a detection gap. Only fires when seed_lock is
                // active (i.e., we're actively watching for a return after a
                // confirmed lane change).
                if self.adaptive_baseline.seed_lock_frames > 0
                    && deviation >= MEDIUM_OFFSET_THRESHOLD
                    && self.is_deviation_sustained_long(DEVIATION_DRIFT_START)
                {
                    self.change_detection_path = Some(DetectionPath::StaticHighDeviation);
                    info!(
                        "🚨 [STATIC-HIGH-DEV] dev={:.1}%, vel={:.1}px/s — missed return detected",
                        deviation * 100.0,
                        lateral_velocity
                    );
                    return LaneChangeState::Drifting;
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
    }

    fn finalize_lane_change(&mut self) {
        let current_pos = self.offset_history.last().copied().unwrap_or(0.0);
        self.adaptive_baseline.reset_to(current_pos);

        // 🔧 FIX: Lock baseline so it doesn't drift during cooldown/grace.
        // After a confirmed lane change (e.g. the initial LEFT of an overtake),
        // the baseline must stay anchored at the overtaking-lane position.
        // Otherwise the EWMA adaptation (α=0.015) causes the baseline to
        // follow the vehicle back toward center during the grace period,
        // suppressing the deviation and making the return RIGHT undetectable.
        // 300 frames ≈ 10s at 30fps — long enough to catch the return.
        self.adaptive_baseline.seed_lock_frames = 300;

        self.position_filter.reset();
        self.post_lane_change_grace = POST_CHANGE_GRACE_FRAMES;
        self.offset_samples.clear();
        self.cooldown_remaining = self.config.cooldown_frames;

        info!(
            "🔄 Baseline seeded at {:.1}% with 300-frame lock — ready for return detection",
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

            debug!(
                "📊 Trajectory: excursion={}, returned={}, shape={:.2}",
                trajectory_analysis.excursion_sufficient,
                trajectory_analysis.returned_to_start,
                trajectory_analysis.shape_score
            );

            if !self.validate_lane_change(duration, net_displacement, &trajectory_analysis) {
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
                "trajectory_smoothness".to_string(),
                serde_json::json!(trajectory_analysis.smoothness),
            );
            event.metadata.insert(
                "has_direction_reversal".to_string(),
                serde_json::json!(trajectory_analysis.has_reversal),
            );
            event.metadata.insert(
                "peak_lateral_velocity".to_string(),
                serde_json::json!(self.peak_velocity_in_window),
            );
            event.metadata.insert(
                "baseline_offset".to_string(),
                serde_json::json!(self.adaptive_baseline.effective_value()),
            );
            event.metadata.insert(
                "baseline_frozen".to_string(),
                serde_json::json!(self.adaptive_baseline.is_frozen),
            );

            if let Some(initial) = self.initial_position_frozen {
                event
                    .metadata
                    .insert("initial_position".to_string(), serde_json::json!(initial));
                event.metadata.insert(
                    "final_position".to_string(),
                    serde_json::json!(final_position),
                );
                event.metadata.insert(
                    "net_displacement".to_string(),
                    serde_json::json!(net_displacement),
                );
            }
            if let Some(path) = self.change_detection_path {
                event.metadata.insert(
                    "detection_path".to_string(),
                    serde_json::json!(format!("{:?}", path)),
                );
            }

            info!(
                "✅ CONFIRMED: {} at {:.2}s, dur={:.0}ms, max={:.1}%, conf={:.2}",
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
    }

    pub fn set_source_id(&mut self, source_id: String) {
        self.source_id = source_id;
    }

    // ============================================================================
    // update_perfect WITH BLACKOUT RECOVERY (FIXED)
    // ============================================================================

    pub fn update_perfect(
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

        if let Some(event) = self.handle_timeouts(frame_id, timestamp_ms) {
            return Some(event);
        }

        // ============================================================
        // BLACKOUT TRACKING
        // 🔧 FIX 2: Uses blackout_consecutive_frames (OWN counter)
        //            NOT adaptive_baseline.frames_without_lanes
        // ============================================================

        if !vehicle_state.is_valid() {
            self.adaptive_baseline.mark_no_lanes(); // Baseline manages itself
            self.blackout_consecutive_frames += 1; // Our own independent counter

            // Enter blackout state after sustained gap
            if !self.in_blackout
                && self.blackout_consecutive_frames >= self.config.blackout_detection_frames
                && self.config.enable_blackout_recovery
            {
                self.in_blackout = true;
                self.blackout_started_frame = Some(frame_id);
                self.blackout_started_time = Some(timestamp_ms);

                // Store last known FILTERED position
                if let Some(last_pos) = self.position_filter.get_last_value() {
                    self.position_before_blackout = Some(last_pos);
                    warn!(
                        "⚫ Blackout started at frame {} ({:.1}s), position before: {:.1}%",
                        frame_id,
                        timestamp_ms / 1000.0,
                        last_pos * 100.0
                    );
                }
            }

            return None;
        }

        // ============================================================
        // BLACKOUT RECOVERY CHECK
        // 🔧 FIX 3: Does NOT call position_filter.update() here.
        //            Only peeks at the raw value. Normal processing
        //            below will do the actual Kalman update.
        // ============================================================

        if self.in_blackout && self.config.enable_blackout_recovery {
            let blackout_frames = self.blackout_consecutive_frames;

            // 🔧 FIX 3: Peek at raw position WITHOUT updating Kalman filter
            let lane_width = vehicle_state.lane_width.unwrap_or(450.0);
            let raw_new_position = vehicle_state.lateral_offset / lane_width;

            if let Some(old_position) = self.position_before_blackout {
                let position_delta = (raw_new_position - old_position).abs();

                info!(
                    "🔄 Recovered from {}-frame ({:.1}s) blackout: {:.1}% → {:.1}% (Δ={:.1}%)",
                    blackout_frames,
                    blackout_frames as f32 / 30.0,
                    old_position * 100.0,
                    raw_new_position * 100.0,
                    position_delta * 100.0
                );

                // Only infer lane change if:
                //  1. Jump is large enough
                //  2. Blackout was genuinely long (not a brief flicker)
                if position_delta >= self.config.blackout_jump_threshold
                    && blackout_frames >= self.config.blackout_detection_frames * 2
                {
                    let direction = if raw_new_position > old_position {
                        Direction::Right
                    } else {
                        Direction::Left
                    };

                    warn!(
                        "🎯 INFERRED LANE CHANGE during {}-frame blackout: {} (Δ={:.1}%)",
                        blackout_frames,
                        direction.as_str(),
                        position_delta * 100.0
                    );

                    let start_time = self.blackout_started_time.unwrap_or(timestamp_ms);
                    let start_frame = self.blackout_started_frame.unwrap_or(frame_id);
                    let duration_ms = timestamp_ms - start_time;

                    let trajectory_analysis = OvertakeAnalysis {
                        excursion_sufficient: true,
                        returned_to_start: false,
                        is_smooth: false,
                        has_reversal: false,
                        shape_score: 0.5,
                        smoothness: 0.0,
                        is_valid_overtake: true,
                    };

                    let confidence = ConfidenceCalculator::calculate(
                        position_delta,
                        duration_ms,
                        &trajectory_analysis,
                        Some(DetectionPath::BlackoutRecovery),
                    ) * 0.75; // Reduce confidence for inferred events

                    let mut event = LaneChangeEvent::new(
                        start_time,
                        start_frame,
                        frame_id,
                        direction,
                        confidence,
                    );
                    event.duration_ms = Some(duration_ms);
                    event.source_id = self.source_id.clone();
                    event.metadata.insert(
                        "detection_path".to_string(),
                        serde_json::json!("BlackoutRecovery"),
                    );
                    event.metadata.insert(
                        "blackout_frames".to_string(),
                        serde_json::json!(blackout_frames),
                    );
                    event.metadata.insert(
                        "position_jump".to_string(),
                        serde_json::json!(position_delta),
                    );

                    info!(
                        "✅ BLACKOUT CONFIRMED: {} at {:.2}s, dur={:.0}ms, jump={:.1}%, conf={:.2}",
                        direction.as_str(),
                        start_time / 1000.0,
                        duration_ms,
                        position_delta * 100.0,
                        confidence
                    );

                    // Seed baseline at recovered position
                    self.adaptive_baseline.reset_to(raw_new_position);
                    self.position_filter.reset();
                    self.post_lane_change_grace = POST_CHANGE_GRACE_FRAMES;
                    self.offset_samples.clear();
                    self.cooldown_remaining = self.config.cooldown_frames;
                    self.reset_lane_change();
                    self.reset_blackout_state();

                    return Some(event);
                }
            }

            // No jump detected — clear blackout, continue normally
            self.reset_blackout_state();
        }

        // Reset consecutive counter on valid frame
        self.blackout_consecutive_frames = 0;

        // ============================================================
        // NORMAL PROCESSING (identical to v3.7)
        // ============================================================

        let lane_width = vehicle_state.lane_width.unwrap_or(450.0);
        let raw_offset = vehicle_state.lateral_offset / lane_width;
        let normalized_offset = self.position_filter.update(raw_offset);

        let lateral_velocity = if vehicle_state.detection_confidence > 0.5 {
            self.velocity_tracker
                .get_velocity(vehicle_state.lateral_offset, timestamp_ms)
        } else {
            0.0
        };

        self.update_histories(normalized_offset, lateral_velocity);

        self.adaptive_baseline
            .update_guarded(normalized_offset, vehicle_state.detection_confidence);

        if !self.adaptive_baseline.is_valid {
            return None;
        }

        let baseline = self.adaptive_baseline.effective_value();
        let signed_deviation = normalized_offset - baseline;
        let deviation = signed_deviation.abs();

        // 🔧 CRITICAL FIX: Handle Lane Coordinate Wrapping (Polarity Flip)
        // When swapping lanes, coordinates jump from Positive Edge (+0.4) to Negative Edge (-0.4) or vice versa.
        // A linear check sees this as a massive Negative change (LEFT), but physically it's a RIGHT move.
        // Logic:
        //   +Baseline to -Current => Moving RIGHT (crossed right boundary)
        //   -Baseline to +Current => Moving LEFT (crossed left boundary)
        let current_direction = if self.adaptive_baseline.is_frozen
            && baseline.abs() > 0.10
            && normalized_offset.abs() > 0.10
            && baseline.signum() != normalized_offset.signum()
            && deviation > 0.40
        {
            if baseline > 0.0 {
                // + to - (e.g. 0.3 to -0.3): Wrapped around Right Edge
                Direction::Right
            } else {
                // - to + (e.g. -0.3 to 0.3): Wrapped around Left Edge
                Direction::Left
            }
        } else {
            // Standard linear drift within lane
            Direction::from_offset(signed_deviation)
        };

        self.trajectory_analyzer
            .add_sample(normalized_offset, timestamp_ms, lateral_velocity);
        self.track_max_offset(deviation);
        self.update_samples(
            normalized_offset,
            deviation,
            timestamp_ms,
            lateral_velocity,
            current_direction,
        );

        let window_metrics = self.calculate_window_metrics(timestamp_ms, lane_width);

        let target_state = self.determine_target_state(
            deviation,
            crossing_type,
            lateral_velocity,
            current_direction,
            &window_metrics,
        );

        if frame_id % 30 == 0 {
            debug!(
                "F{}: dev={:.1}%, state={:?}->{:?}, conf={:.2}",
                frame_id,
                deviation * 100.0,
                self.state,
                target_state,
                vehicle_state.detection_confidence
            );
        }

        self.check_transition(
            target_state,
            current_direction,
            frame_id,
            timestamp_ms,
            deviation,
            normalized_offset,
        )
    }
}
