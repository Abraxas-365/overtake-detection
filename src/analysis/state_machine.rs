// src/analysis/state_machine.rs
//
// LANE CHANGE DETECTION v3.0 - STATE OF THE ART
//
// Improvements:
// 1. Advanced curve detection with curvature analysis
// 2. Bidirectional movement validation (out-and-back pattern)
// 3. Dynamic thresholds based on movement characteristics
// 4. Trajectory shape analysis
// 5. Probabilistic confidence scoring (HMM-inspired)
// 6. Multi-signal fusion for robust detection
//

use super::boundary_detector::CrossingType;
use super::curve_detector::CurveDetector;
use super::velocity_tracker::LateralVelocityTracker;
use crate::types::{Direction, LaneChangeConfig, LaneChangeEvent, LaneChangeState, VehicleState};
use std::collections::VecDeque;
use tracing::{debug, info, warn};

// ============================================================================
// CONSTANTS - TUNED FOR STATE-OF-THE-ART PERFORMANCE
// ============================================================================

// Velocity thresholds (pixels/second)
const MIN_VELOCITY_FAST: f32 = 120.0;
const MIN_VELOCITY_MEDIUM: f32 = 60.0;
const MIN_VELOCITY_SLOW: f32 = 20.0;
const VELOCITY_SPIKE_THRESHOLD: f32 = 180.0;

// Time-to-Lane-Crossing
const TLC_WARNING_THRESHOLD: f64 = 1.5;

// Analysis windows
const ANALYSIS_WINDOW_MS: f64 = 4000.0;
const TRAJECTORY_WINDOW_SIZE: usize = 60;

// Deviation thresholds (fraction of lane width)
const DEVIATION_DRIFT_START: f32 = 0.20;
const DEVIATION_CROSSING: f32 = 0.30;
const DEVIATION_LANE_CENTER: f32 = 0.50;
const DEVIATION_SIGNIFICANT: f32 = 0.40;

// State machine behavior
const HYSTERESIS_EXIT: f32 = 0.6;
const DIRECTION_CONSISTENCY_THRESHOLD: f32 = 0.65;
const POST_CHANGE_GRACE_FRAMES: u32 = 90;
const MAX_DRIFTING_MS: f64 = 8000.0;

// Kalman filter parameters
const KALMAN_PROCESS_NOISE: f32 = 0.001;
const KALMAN_MEASUREMENT_NOISE: f32 = 0.01;

// Adaptive baseline parameters
const EWMA_ALPHA_STABLE: f32 = 0.003;
const EWMA_ALPHA_ADAPTING: f32 = 0.015;
const EWMA_MIN_SAMPLES: u32 = 30;
const STABILITY_VARIANCE_THRESHOLD: f32 = 0.005;
const INSTABILITY_VARIANCE_THRESHOLD: f32 = 0.05;
const BASELINE_MAX_DRIFT: f32 = 0.25; // Max baseline can drift from center

// Curve detection
const CURVE_COMPENSATION_FACTOR: f32 = 1.15;
const CURVE_CURVATURE_THRESHOLD: f32 = 0.008; // Radians per pixel
const CURVE_REJECTION_OFFSET: f32 = 0.70; // Must exceed this in curve to be valid

// Position change for velocity spike
const POSITION_CHANGE_THRESHOLD: f32 = 0.15;

// ============================================================================
// VALIDATION THRESHOLDS - DYNAMIC
// ============================================================================
const HIGH_OFFSET_THRESHOLD: f32 = 0.65; // Auto-valid if exceeded
const MEDIUM_OFFSET_THRESHOLD: f32 = 0.55; // Valid with duration
const LOW_OFFSET_THRESHOLD: f32 = 0.35; // Valid with duration + displacement
const MIN_NET_DISPLACEMENT: f32 = 0.25;
const MIN_DURATION_FOR_VALIDATION: f64 = 3000.0;

// Bidirectional movement validation
const RETURN_TO_CENTER_THRESHOLD: f32 = 0.20; // Must return within this of start
const MIN_EXCURSION_FOR_OVERTAKE: f32 = 0.45; // Must go at least this far out

// Trajectory shape analysis
const TRAJECTORY_SMOOTHNESS_THRESHOLD: f32 = 0.15; // Max acceptable jitter
const MIN_TRAJECTORY_POINTS: usize = 15;

// ============================================================================
// KALMAN FILTER
// ============================================================================

#[derive(Clone)]
struct SimpleKalmanFilter {
    x: f32, // State estimate
    p: f32, // Error covariance
    q: f32, // Process noise
    r: f32, // Measurement noise
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

        // Prediction step
        let p_pred = self.p + self.q;

        // Update step
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
}

// ============================================================================
// ADAPTIVE BASELINE WITH DRIFT LIMITING
// ============================================================================

#[derive(Clone)]
struct AdaptiveBaseline {
    value: f32,
    initial_value: f32, // First stable value (reference point)
    variance: f32,
    sample_count: u32,
    is_valid: bool,
    recent_samples: VecDeque<f32>,
    is_adapting: bool,
    stable_frames: u32,
    is_frozen: bool,
    frozen_value: f32,
    has_initial: bool,
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
        }
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

    fn update(&mut self, measurement: f32) -> f32 {
        self.sample_count += 1;

        self.recent_samples.push_back(measurement);
        if self.recent_samples.len() > 30 {
            self.recent_samples.pop_front();
        }

        // Calculate variance
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

        // Track stability
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

        let alpha = if self.is_adapting || !self.is_valid {
            EWMA_ALPHA_ADAPTING
        } else {
            EWMA_ALPHA_STABLE
        };

        if self.sample_count == 1 {
            self.value = measurement;
        } else {
            let new_value = alpha * measurement + (1.0 - alpha) * self.value;

            // IMPROVEMENT: Limit baseline drift from initial position
            if self.has_initial {
                let drift = (new_value - self.initial_value).abs();
                if drift > BASELINE_MAX_DRIFT {
                    // Don't adapt - vehicle has likely changed lanes
                    debug!(
                        "🚫 Baseline drift limited: {:.1}% > {:.1}%",
                        drift * 100.0,
                        BASELINE_MAX_DRIFT * 100.0
                    );
                } else {
                    self.value = new_value;
                }
            } else {
                self.value = new_value;
            }
        }

        // Set initial value once stable
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
    }
}

// ============================================================================
// TRAJECTORY ANALYZER - For shape analysis
// ============================================================================

#[derive(Clone)]
struct TrajectoryAnalyzer {
    positions: VecDeque<f32>,
    timestamps: VecDeque<f64>,
    velocities: VecDeque<f32>,
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

    /// Analyzes if the trajectory shows a clear out-and-back pattern (overtake)
    fn analyze_overtake_pattern(
        &self,
        start_pos: f32,
        current_pos: f32,
        max_excursion: f32,
    ) -> OvertakeAnalysis {
        let mut analysis = OvertakeAnalysis::default();

        if self.positions.len() < MIN_TRAJECTORY_POINTS {
            return analysis;
        }

        // Check 1: Did we go far enough out?
        analysis.excursion_sufficient = max_excursion >= MIN_EXCURSION_FOR_OVERTAKE;

        // Check 2: Did we return close to starting position?
        let return_distance = (current_pos - start_pos).abs();
        analysis.returned_to_start = return_distance < RETURN_TO_CENTER_THRESHOLD;

        // Check 3: Was the movement smooth (not oscillating)?
        analysis.smoothness = self.calculate_smoothness();
        analysis.is_smooth = analysis.smoothness < TRAJECTORY_SMOOTHNESS_THRESHOLD;

        // Check 4: Was there a clear direction reversal?
        analysis.has_reversal = self.detect_direction_reversal();

        // Check 5: Calculate trajectory shape score
        analysis.shape_score = self.calculate_shape_score(start_pos, max_excursion);

        // Overall validity
        analysis.is_valid_overtake = analysis.excursion_sufficient
            && analysis.returned_to_start
            && analysis.is_smooth
            && analysis.has_reversal
            && analysis.shape_score > 0.6;

        analysis
    }

    /// Calculate trajectory smoothness (lower = smoother)
    fn calculate_smoothness(&self) -> f32 {
        if self.positions.len() < 3 {
            return 1.0;
        }

        let positions: Vec<f32> = self.positions.iter().copied().collect();
        let mut jitter_sum = 0.0;
        let mut count = 0;

        // Calculate second derivative (acceleration) of position
        for i in 2..positions.len() {
            let accel = positions[i] - 2.0 * positions[i - 1] + positions[i - 2];
            jitter_sum += accel.abs();
            count += 1;
        }

        if count > 0 {
            jitter_sum / count as f32
        } else {
            1.0
        }
    }

    /// Detect if there was a clear direction reversal
    fn detect_direction_reversal(&self) -> bool {
        if self.velocities.len() < 10 {
            return false;
        }

        let velocities: Vec<f32> = self.velocities.iter().copied().collect();

        // Find sign changes in velocity
        let mut sign_changes = 0;
        let mut last_sign: Option<bool> = None;

        for v in &velocities {
            if v.abs() > 10.0 {
                // Ignore near-zero velocities
                let current_sign = *v > 0.0;
                if let Some(last) = last_sign {
                    if current_sign != last {
                        sign_changes += 1;
                    }
                }
                last_sign = Some(current_sign);
            }
        }

        // A valid overtake should have exactly 1 major direction reversal
        sign_changes >= 1 && sign_changes <= 3
    }

    /// Calculate how well the trajectory matches an ideal overtake shape
    fn calculate_shape_score(&self, start_pos: f32, max_excursion: f32) -> f32 {
        if self.positions.len() < MIN_TRAJECTORY_POINTS {
            return 0.0;
        }

        let positions: Vec<f32> = self.positions.iter().copied().collect();
        let n = positions.len();

        // Ideal overtake shape: start → peak → return
        // Find the peak (max deviation from start)
        let mut peak_idx = 0;
        let mut peak_deviation = 0.0;
        for (i, &pos) in positions.iter().enumerate() {
            let deviation = (pos - start_pos).abs();
            if deviation > peak_deviation {
                peak_deviation = deviation;
                peak_idx = i;
            }
        }

        // Score based on:
        // 1. Peak should be roughly in the middle (not at edges)
        let peak_position_score = {
            let relative_pos = peak_idx as f32 / n as f32;
            // Best if peak is between 30% and 70% of trajectory
            if relative_pos >= 0.3 && relative_pos <= 0.7 {
                1.0
            } else if relative_pos >= 0.2 && relative_pos <= 0.8 {
                0.7
            } else {
                0.3
            }
        };

        // 2. Trajectory should be monotonic before and after peak
        let monotonic_score = {
            let mut before_monotonic = true;
            let mut after_monotonic = true;
            let direction = if positions[peak_idx] > start_pos {
                1.0
            } else {
                -1.0
            };

            // Check before peak
            for i in 1..peak_idx {
                let expected_direction = (positions[i] - positions[i - 1]) * direction;
                if expected_direction < -0.02 {
                    // Allow small violations
                    before_monotonic = false;
                    break;
                }
            }

            // Check after peak
            for i in (peak_idx + 1)..n {
                let expected_direction = (positions[i - 1] - positions[i]) * direction;
                if expected_direction < -0.02 {
                    after_monotonic = false;
                    break;
                }
            }

            match (before_monotonic, after_monotonic) {
                (true, true) => 1.0,
                (true, false) | (false, true) => 0.6,
                (false, false) => 0.3,
            }
        };

        // 3. End position should be close to start
        let return_score = {
            let end_pos = positions[n - 1];
            let return_distance = (end_pos - start_pos).abs();
            if return_distance < 0.10 {
                1.0
            } else if return_distance < 0.20 {
                0.7
            } else if return_distance < 0.30 {
                0.4
            } else {
                0.1
            }
        };

        // Weighted average
        (peak_position_score * 0.3 + monotonic_score * 0.4 + return_score * 0.3)
    }

    fn clear(&mut self) {
        self.positions.clear();
        self.timestamps.clear();
        self.velocities.clear();
    }
}

#[derive(Default, Debug)]
struct OvertakeAnalysis {
    excursion_sufficient: bool,
    returned_to_start: bool,
    is_smooth: bool,
    has_reversal: bool,
    shape_score: f32,
    smoothness: f32,
    is_valid_overtake: bool,
}

// ============================================================================
// ENHANCED CURVE ANALYZER
// ============================================================================

#[derive(Clone)]
struct EnhancedCurveAnalyzer {
    curvature_history: VecDeque<f32>,
    is_in_curve: bool,
    curve_direction: Direction,
    curve_intensity: f32,
}

impl EnhancedCurveAnalyzer {
    fn new() -> Self {
        Self {
            curvature_history: VecDeque::with_capacity(30),
            is_in_curve: false,
            curve_direction: Direction::Unknown,
            curve_intensity: 0.0,
        }
    }

    fn update(&mut self, lanes: &[crate::types::Lane]) -> bool {
        let curvature = self.calculate_lane_curvature(lanes);

        self.curvature_history.push_back(curvature);
        if self.curvature_history.len() > 30 {
            self.curvature_history.pop_front();
        }

        // Average curvature over recent frames
        let avg_curvature: f32 = if !self.curvature_history.is_empty() {
            self.curvature_history.iter().sum::<f32>() / self.curvature_history.len() as f32
        } else {
            0.0
        };

        self.curve_intensity = avg_curvature.abs();
        self.is_in_curve = self.curve_intensity > CURVE_CURVATURE_THRESHOLD;

        if self.is_in_curve {
            self.curve_direction = if avg_curvature > 0.0 {
                Direction::Left
            } else {
                Direction::Right
            };
        } else {
            self.curve_direction = Direction::Unknown;
        }

        self.is_in_curve
    }

    fn calculate_lane_curvature(&self, lanes: &[crate::types::Lane]) -> f32 {
        if lanes.is_empty() {
            return 0.0;
        }

        // Use the lane with most points
        let best_lane = lanes.iter().max_by_key(|l| l.points.len());

        if let Some(lane) = best_lane {
            if lane.points.len() < 5 {
                return 0.0;
            }

            // Calculate curvature using three-point method
            let points = &lane.points;
            let n = points.len();

            // Sample points at start, middle, and end
            let p1 = &points[0];
            let p2 = &points[n / 2];
            let p3 = &points[n - 1];

            // Calculate curvature using Menger curvature formula
            // κ = 4 * Area(triangle) / (|p1-p2| * |p2-p3| * |p3-p1|)
            let area = 0.5 * ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)).abs();

            let d12 = ((p2.x - p1.x).powi(2) + (p2.y - p1.y).powi(2)).sqrt();
            let d23 = ((p3.x - p2.x).powi(2) + (p3.y - p2.y).powi(2)).sqrt();
            let d31 = ((p1.x - p3.x).powi(2) + (p1.y - p3.y).powi(2)).sqrt();

            let denominator = d12 * d23 * d31;
            if denominator > 0.0 {
                let curvature = 4.0 * area / denominator;
                // Signed curvature based on cross product
                let cross = (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y);
                return if cross > 0.0 { curvature } else { -curvature };
            }
        }

        0.0
    }

    fn reset(&mut self) {
        self.curvature_history.clear();
        self.is_in_curve = false;
        self.curve_direction = Direction::Unknown;
        self.curve_intensity = 0.0;
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
    tlc_estimate: Option<f64>,
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
    TLCBased,
    CumulativeDisplacement,
    VelocitySpike,
}

// ============================================================================
// PROBABILISTIC CONFIDENCE CALCULATOR (HMM-inspired)
// ============================================================================

struct ConfidenceCalculator;

impl ConfidenceCalculator {
    /// Calculate overall confidence using multiple factors
    fn calculate(
        max_offset: f32,
        duration_ms: f64,
        trajectory_analysis: &OvertakeAnalysis,
        curve_intensity: f32,
        detection_path: Option<DetectionPath>,
    ) -> f32 {
        // Base confidence from offset magnitude
        let offset_confidence = Self::offset_to_confidence(max_offset);

        // Duration confidence
        let duration_confidence = Self::duration_to_confidence(duration_ms);

        // Trajectory shape confidence
        let trajectory_confidence = trajectory_analysis.shape_score;

        // Curve penalty (reduce confidence if in curve)
        let curve_penalty = if curve_intensity > CURVE_CURVATURE_THRESHOLD {
            0.8 - (curve_intensity - CURVE_CURVATURE_THRESHOLD) * 10.0
        } else {
            1.0
        }
        .max(0.5);

        // Detection path bonus
        let path_bonus = match detection_path {
            Some(DetectionPath::BoundaryCrossing) => 0.10,
            Some(DetectionPath::HighVelocity) => 0.05,
            Some(DetectionPath::TLCBased) => 0.08,
            Some(DetectionPath::VelocitySpike) => 0.03,
            _ => 0.0,
        };

        // Combine using weighted product (multiplicative fusion)
        let base_confidence = (offset_confidence * 0.35
            + duration_confidence * 0.25
            + trajectory_confidence * 0.30
            + path_bonus)
            * curve_penalty;

        // Clamp to valid range
        base_confidence.max(0.1).min(0.98)
    }

    fn offset_to_confidence(max_offset: f32) -> f32 {
        if max_offset >= 0.80 {
            0.95
        } else if max_offset >= 0.70 {
            0.85
        } else if max_offset >= 0.60 {
            0.75
        } else if max_offset >= 0.50 {
            0.65
        } else if max_offset >= 0.40 {
            0.55
        } else {
            0.40
        }
    }

    fn duration_to_confidence(duration_ms: f64) -> f32 {
        if duration_ms >= 3500.0 && duration_ms <= 7000.0 {
            0.90 // Ideal range
        } else if duration_ms >= 2500.0 && duration_ms <= 9000.0 {
            0.75
        } else if duration_ms >= 1500.0 && duration_ms <= 12000.0 {
            0.55
        } else {
            0.35
        }
    }
}

// ============================================================================
// DYNAMIC THRESHOLD CALCULATOR
// ============================================================================

struct DynamicThresholds;

impl DynamicThresholds {
    /// Get offset threshold based on context
    fn get_offset_threshold(curve_intensity: f32, avg_velocity: f32) -> f32 {
        let mut threshold = MEDIUM_OFFSET_THRESHOLD;

        // In curves, require higher offset
        if curve_intensity > CURVE_CURVATURE_THRESHOLD {
            threshold += 0.10;
        }

        // With high velocity, can use lower threshold (more confident detection)
        if avg_velocity.abs() > MIN_VELOCITY_FAST {
            threshold -= 0.05;
        }

        threshold.max(0.45).min(0.70)
    }

    /// Get duration threshold based on offset
    fn get_duration_threshold(max_offset: f32) -> f64 {
        // Higher offset = can accept shorter duration
        if max_offset >= 0.75 {
            2000.0
        } else if max_offset >= 0.65 {
            2500.0
        } else if max_offset >= 0.55 {
            3000.0
        } else {
            3500.0
        }
    }
}

// ============================================================================
// MAIN STATE MACHINE
// ============================================================================

pub struct LaneChangeStateMachine {
    config: LaneChangeConfig,
    source_id: String,

    // State
    state: LaneChangeState,
    frames_in_state: u32,
    pending_state: Option<LaneChangeState>,
    pending_frames: u32,

    // Current change tracking
    change_direction: Direction,
    change_start_frame: Option<u64>,
    change_start_time: Option<f64>,
    change_detection_path: Option<DetectionPath>,
    max_offset_in_change: f32,
    initial_position_frozen: Option<f32>,

    // Cooldowns and grace periods
    cooldown_remaining: u32,
    total_frames_processed: u64,
    post_lane_change_grace: u32,

    // Signal processing
    position_filter: SimpleKalmanFilter,
    adaptive_baseline: AdaptiveBaseline,

    // History buffers
    offset_history: Vec<f32>,
    velocity_history: VecDeque<f32>,
    offset_samples: VecDeque<OffsetSample>,
    direction_samples: VecDeque<Direction>,
    recent_deviations: Vec<f32>,

    // Stability tracking
    stable_deviation_frames: u32,
    last_deviation: f32,

    // Peak tracking
    peak_deviation_in_window: f32,
    peak_velocity_in_window: f32,
    peak_direction: Direction,

    // Advanced analyzers
    curve_detector: CurveDetector,
    enhanced_curve_analyzer: EnhancedCurveAnalyzer,
    velocity_tracker: LateralVelocityTracker,
    trajectory_analyzer: TrajectoryAnalyzer,

    // Curve state
    is_in_curve: bool,
    curve_compensation_factor: f32,
    curve_intensity: f32,

    // Pending state tracking
    pending_change_direction: Direction,
    pending_max_offset: f32,
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
            offset_history: Vec::with_capacity(60),
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
            enhanced_curve_analyzer: EnhancedCurveAnalyzer::new(),
            velocity_tracker: LateralVelocityTracker::new(),
            trajectory_analyzer: TrajectoryAnalyzer::new(),
            is_in_curve: false,
            curve_compensation_factor: 1.0,
            curve_intensity: 0.0,
            pending_change_direction: Direction::Unknown,
            pending_max_offset: 0.0,
        }
    }

    pub fn current_state(&self) -> &str {
        self.state.as_str()
    }

    pub fn update_curve_detector(&mut self, lanes: &[crate::types::Lane]) -> bool {
        // Use both curve detectors
        let basic_curve = self.curve_detector.is_in_curve(lanes);
        let enhanced_curve = self.enhanced_curve_analyzer.update(lanes);

        self.is_in_curve = basic_curve || enhanced_curve;
        self.curve_intensity = self.enhanced_curve_analyzer.curve_intensity;

        self.curve_compensation_factor = if self.is_in_curve {
            CURVE_COMPENSATION_FACTOR + self.curve_intensity * 5.0
        } else {
            1.0
        };

        if self.is_in_curve {
            debug!(
                "🔄 Curve detected: intensity={:.4}, direction={:?}",
                self.curve_intensity, self.enhanced_curve_analyzer.curve_direction
            );
        }

        self.is_in_curve
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

        // Handle timeouts
        if let Some(event) = self.handle_timeouts(frame_id, timestamp_ms) {
            return Some(event);
        }

        if !vehicle_state.is_valid() {
            return None;
        }

        let lane_width = vehicle_state.lane_width.unwrap();
        let raw_offset = vehicle_state.lateral_offset / lane_width;
        let normalized_offset = self.position_filter.update(raw_offset);

        let lateral_velocity = self
            .velocity_tracker
            .get_velocity(vehicle_state.lateral_offset, timestamp_ms);

        // Update history buffers
        self.update_histories(normalized_offset, lateral_velocity, timestamp_ms);

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
        let current_direction = Direction::from_offset(signed_deviation);

        // Update trajectory analyzer
        self.trajectory_analyzer
            .add_sample(normalized_offset, timestamp_ms, lateral_velocity);

        // Track max offset during change
        self.track_max_offset(deviation);

        // Update sample buffers
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
            "F{}: off={:.1}%, base={:.1}%{}, dev={:.1}%, max={:.1}%, curve={:.4}, state={:?}→{:?}",
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
            self.curve_intensity,
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

    fn handle_timeouts(&mut self, frame_id: u64, timestamp_ms: f64) -> Option<LaneChangeEvent> {
        if self.state == LaneChangeState::Drifting {
            if let Some(start_time) = self.change_start_time {
                let elapsed = timestamp_ms - start_time;

                if elapsed > MAX_DRIFTING_MS {
                    if self.max_offset_in_change >= self.config.crossing_threshold {
                        info!(
                            "⏰ Long DRIFTING ({:.0}ms) with good offset ({:.1}%) - auto-completing",
                            elapsed,
                            self.max_offset_in_change * 100.0
                        );
                        return self.force_complete(frame_id, timestamp_ms);
                    }
                }

                if elapsed > self.config.max_duration_ms {
                    if self.max_offset_in_change >= self.config.crossing_threshold {
                        info!("⏰ Timeout but good offset - completing");
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
                    info!("⏰ Timeout in CROSSING - completing anyway");
                    return self.force_complete(frame_id, timestamp_ms);
                }
            }
        }

        None
    }

    fn update_histories(
        &mut self,
        normalized_offset: f32,
        lateral_velocity: f32,
        timestamp_ms: f64,
    ) {
        self.velocity_history.push_back(lateral_velocity);
        if self.velocity_history.len() > 30 {
            self.velocity_history.pop_front();
        }

        self.offset_history.push(normalized_offset);
        if self.offset_history.len() > 60 {
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

        // Get trajectory analysis
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
            self.curve_intensity,
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
            "✅ FORCE CONFIRMED: {} at {:.2}s, dur={:.0}ms, max={:.1}%, conf={:.2}, path={:?}",
            event.direction_name(),
            start_time / 1000.0,
            duration,
            self.max_offset_in_change * 100.0,
            confidence,
            self.change_detection_path
        );

        self.finalize_lane_change();
        Some(event)
    }

    /// Comprehensive validation using multiple criteria
    fn validate_lane_change(
        &self,
        duration: f64,
        net_displacement: f32,
        trajectory_analysis: &OvertakeAnalysis,
    ) -> bool {
        // ============================================================
        // VALIDATION RULE 1: Curve rejection
        // ============================================================
        if self.is_in_curve && self.max_offset_in_change < CURVE_REJECTION_OFFSET {
            warn!(
                "❌ Rejected: curve detected (intensity={:.4}) with insufficient offset ({:.1}% < {:.1}%)",
                self.curve_intensity,
                self.max_offset_in_change * 100.0,
                CURVE_REJECTION_OFFSET * 100.0
            );
            return false;
        }

        // ============================================================
        // VALIDATION RULE 2: Very high offset = auto-accept
        // ============================================================
        if self.max_offset_in_change >= HIGH_OFFSET_THRESHOLD {
            // Even with high offset, check trajectory if we have enough data
            if trajectory_analysis.is_valid_overtake || self.max_offset_in_change >= 0.75 {
                info!(
                    "✅ Valid: max offset {:.1}% >= {:.1}% (definitely crossed lane)",
                    self.max_offset_in_change * 100.0,
                    HIGH_OFFSET_THRESHOLD * 100.0
                );
                return true;
            }
        }

        // ============================================================
        // VALIDATION RULE 3: Dynamic threshold based on context
        // ============================================================
        let avg_velocity =
            self.velocity_history.iter().sum::<f32>() / self.velocity_history.len().max(1) as f32;

        let dynamic_offset_threshold =
            DynamicThresholds::get_offset_threshold(self.curve_intensity, avg_velocity);

        let dynamic_duration_threshold =
            DynamicThresholds::get_duration_threshold(self.max_offset_in_change);

        // ============================================================
        // VALIDATION RULE 4: Medium-high offset with duration check
        // ============================================================
        if self.max_offset_in_change >= dynamic_offset_threshold {
            if duration >= dynamic_duration_threshold {
                // Additional check: trajectory should look like an overtake
                if trajectory_analysis.shape_score >= 0.5 || trajectory_analysis.returned_to_start {
                    info!(
                        "✅ Valid: max={:.1}% >= {:.1}%, dur={:.0}ms >= {:.0}ms, shape={:.2}",
                        self.max_offset_in_change * 100.0,
                        dynamic_offset_threshold * 100.0,
                        duration,
                        dynamic_duration_threshold,
                        trajectory_analysis.shape_score
                    );
                    return true;
                } else {
                    warn!(
                        "❌ Rejected: trajectory shape invalid (score={:.2}, returned={})",
                        trajectory_analysis.shape_score, trajectory_analysis.returned_to_start
                    );
                    return false;
                }
            } else {
                warn!(
                    "❌ Rejected: max={:.1}% but dur={:.0}ms < {:.0}ms",
                    self.max_offset_in_change * 100.0,
                    duration,
                    dynamic_duration_threshold
                );
                return false;
            }
        }

        // ============================================================
        // VALIDATION RULE 5: Lower offset needs strong evidence
        // ============================================================
        if self.max_offset_in_change >= LOW_OFFSET_THRESHOLD {
            let has_sufficient_duration = duration >= MIN_DURATION_FOR_VALIDATION;
            let has_sufficient_displacement = net_displacement >= MIN_NET_DISPLACEMENT;
            let has_valid_trajectory = trajectory_analysis.is_valid_overtake;

            if has_sufficient_duration && has_sufficient_displacement && has_valid_trajectory {
                info!(
                    "✅ Valid: max={:.1}%, dur={:.0}ms, net={:.1}%, trajectory=valid",
                    self.max_offset_in_change * 100.0,
                    duration,
                    net_displacement * 100.0
                );
                return true;
            } else {
                warn!(
                    "❌ Rejected: max={:.1}%, dur={:.0}ms, net={:.1}%, trajectory={}",
                    self.max_offset_in_change * 100.0,
                    duration,
                    net_displacement * 100.0,
                    if has_valid_trajectory {
                        "valid"
                    } else {
                        "invalid"
                    }
                );
                return false;
            }
        }

        // ============================================================
        // VALIDATION RULE 6: Too low offset = reject
        // ============================================================
        warn!(
            "❌ Rejected: low max offset {:.1}% < {:.1}%",
            self.max_offset_in_change * 100.0,
            LOW_OFFSET_THRESHOLD * 100.0
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

    fn calculate_window_metrics(&self, _current_time_ms: f64, lane_width: f32) -> WindowMetrics {
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

        // TLC calculation (improved)
        if metrics.avg_velocity.abs() > 5.0 {
            let distance_to_boundary = (0.5 - last.deviation.abs()) * lane_width;
            if distance_to_boundary > 0.0 {
                metrics.tlc_estimate =
                    Some((distance_to_boundary / metrics.avg_velocity.abs()) as f64);
            }
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
        let drift_threshold = self.config.drift_threshold * self.curve_compensation_factor;
        let crossing_threshold = self.config.crossing_threshold * self.curve_compensation_factor;

        let vel_fast = MIN_VELOCITY_FAST * self.curve_compensation_factor;
        let vel_medium = MIN_VELOCITY_MEDIUM * self.curve_compensation_factor;

        match self.state {
            LaneChangeState::Centered => {
                // PATH 1: BOUNDARY CROSSING (most reliable)
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

                // PATH 3: VELOCITY SPIKE (sudden movement)
                if lateral_velocity.abs() > VELOCITY_SPIKE_THRESHOLD {
                    if self.is_velocity_sustained(vel_fast) {
                        let position_change = self.get_recent_position_change();
                        if position_change >= POSITION_CHANGE_THRESHOLD {
                            self.change_detection_path = Some(DetectionPath::VelocitySpike);
                            info!(
                                "🚨 [VELOCITY-SPIKE] vel={:.1}px/s, pos_change={:.1}%, dev={:.1}%",
                                lateral_velocity,
                                position_change * 100.0,
                                deviation * 100.0
                            );
                            return LaneChangeState::Drifting;
                        }
                    }
                }

                // PATH 4: TLC-BASED (predictive)
                if let Some(tlc) = metrics.tlc_estimate {
                    if tlc < TLC_WARNING_THRESHOLD
                        && deviation >= drift_threshold
                        && metrics.is_sustained_movement
                        && !self.is_in_curve
                    // Don't use TLC in curves
                    {
                        self.change_detection_path = Some(DetectionPath::TLCBased);
                        info!("🚨 [TLC] TLC={:.2}s, dev={:.1}%", tlc, deviation * 100.0);
                        return LaneChangeState::Drifting;
                    }
                }

                // PATH 5: MEDIUM SPEED + HIGH DEVIATION
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

                // PATH 6: GRADUAL CHANGE
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

                // PATH 7: LARGE DEVIATION
                if deviation >= DEVIATION_LANE_CENTER {
                    if self.is_deviation_sustained(drift_threshold) {
                        self.change_detection_path = Some(DetectionPath::LargeDeviation);
                        info!("🚨 [LARGE] dev={:.1}%", deviation * 100.0);
                        return LaneChangeState::Drifting;
                    }
                }

                // PATH 8: CUMULATIVE (catch-all)
                if metrics.max_deviation >= DEVIATION_SIGNIFICANT
                    && metrics.direction_consistency >= DIRECTION_CONSISTENCY_THRESHOLD
                    && metrics.time_span_ms >= 2500.0
                    && !self.is_in_curve
                // Critical: don't trigger in curves
                {
                    self.change_detection_path = Some(DetectionPath::CumulativeDisplacement);
                    info!(
                        "🚨 [CUMULATIVE] max={:.1}%, span={:.1}s",
                        metrics.max_deviation * 100.0,
                        metrics.time_span_ms / 1000.0
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

                if self.frames_in_state > 30 && self.max_offset_in_change >= drift_threshold + 0.08
                {
                    if self.is_deviation_stable() {
                        info!(
                            "✅ DRIFTING complete: stabilized with max={:.1}%",
                            self.max_offset_in_change * 100.0
                        );
                        return LaneChangeState::Completed;
                    }
                }

                let cancel_threshold = drift_threshold * 0.5;
                if deviation < cancel_threshold && self.max_offset_in_change < drift_threshold {
                    warn!(
                        "❌ Cancelled: max={:.1}% < drift={:.1}%",
                        self.max_offset_in_change * 100.0,
                        drift_threshold * 100.0
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

                if self.is_deviation_stable() && deviation < 0.35 {
                    info!("✅ Completing: stabilized at {:.1}%", deviation * 100.0);
                    return LaneChangeState::Completed;
                }

                let return_threshold = self.config.drift_threshold * HYSTERESIS_EXIT;
                if deviation < return_threshold {
                    info!("✅ Completing: returned to center");
                    return LaneChangeState::Completed;
                }

                if self.stable_deviation_frames >= 30 && deviation < 0.45 {
                    info!(
                        "✅ Completing: stable for {} frames",
                        self.stable_deviation_frames
                    );
                    return LaneChangeState::Completed;
                }

                // Direction reversal check
                if self.max_offset_in_change >= self.config.crossing_threshold
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
                    if reversal_count >= 7 {
                        info!("✅ Completing: direction reversed");
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
            >= 5
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
            >= 12
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
            >= 4
    }

    fn is_deviation_stable(&self) -> bool {
        if self.recent_deviations.len() < 15 {
            return false;
        }
        let recent = &self.recent_deviations[self.recent_deviations.len() - 15..];
        let max = recent.iter().fold(f32::MIN, |a, &b| a.max(b));
        let min = recent.iter().fold(f32::MAX, |a, &b| a.min(b));
        if max - min > 0.08 {
            return false;
        }
        recent
            .windows(2)
            .filter(|w| (w[1] - w[0]).abs() > 0.03)
            .count()
            <= 2
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
        self.adaptive_baseline.reset();
        self.position_filter.reset();
        self.post_lane_change_grace = POST_CHANGE_GRACE_FRAMES;
        self.offset_samples.clear();
        self.cooldown_remaining = self.config.cooldown_frames;
        info!("🔄 Baseline reset - will adapt to new position");
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

        // Freeze baseline when potential lane change detected
        if target_state == LaneChangeState::Drifting && self.state == LaneChangeState::Centered {
            if self.pending_state != Some(LaneChangeState::Drifting) {
                self.adaptive_baseline.freeze();
                self.initial_position_frozen = Some(normalized_offset);
                self.pending_change_direction = direction;
                self.pending_max_offset = current_deviation;
                info!(
                    "🧊 Baseline frozen at {:.1}%, initial position saved: {:.1}%",
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

        // Unfreeze if not transitioning
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
                "🚗 Lane change started: {} at {:.2}s via {:?} (max so far: {:.1}%)",
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

            // Get trajectory analysis
            let trajectory_analysis = self.trajectory_analyzer.analyze_overtake_pattern(
                self.initial_position_frozen.unwrap_or(0.0),
                final_position,
                self.max_offset_in_change,
            );

            info!(
                "📊 Trajectory analysis: excursion={}, returned={}, smooth={}, reversal={}, shape={:.2}",
                trajectory_analysis.excursion_sufficient,
                trajectory_analysis.returned_to_start,
                trajectory_analysis.is_smooth,
                trajectory_analysis.has_reversal,
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
                self.curve_intensity,
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

            // Add comprehensive metadata
            event.metadata.insert(
                "max_offset_normalized".to_string(),
                serde_json::json!(self.max_offset_in_change),
            );
            event.metadata.insert(
                "curve_intensity".to_string(),
                serde_json::json!(self.curve_intensity),
            );
            event.metadata.insert(
                "trajectory_shape_score".to_string(),
                serde_json::json!(trajectory_analysis.shape_score),
            );
            event.metadata.insert(
                "trajectory_smoothness".to_string(),
                serde_json::json!(trajectory_analysis.smoothness),
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
                "✅ CONFIRMED: {} at {:.2}s, dur={:.0}ms, max={:.1}%, conf={:.2}, path={:?}",
                event.direction_name(),
                start_time / 1000.0,
                duration,
                self.max_offset_in_change * 100.0,
                confidence,
                self.change_detection_path
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
        self.enhanced_curve_analyzer.reset();
        self.velocity_tracker.reset();
        self.offset_samples.clear();
        self.direction_samples.clear();
        self.trajectory_analyzer.clear();
        self.is_in_curve = false;
        self.curve_compensation_factor = 1.0;
        self.curve_intensity = 0.0;
    }

    pub fn set_source_id(&mut self, source_id: String) {
        self.source_id = source_id;
    }
}
