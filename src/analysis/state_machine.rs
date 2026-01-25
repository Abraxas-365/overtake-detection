// src/analysis/state_machine.rs
//
// STATE-OF-THE-ART LANE CHANGE DETECTION v2.0
//
// Based on Academic Research:
// - Kalman Filter for position smoothing (CMU, Tokyo University)
// - EWMA for adaptive baseline (Streaming algorithms research)
// - Multi-path detection (PMC8786501, PMC6020696)
// - Variance-based stability (Lane departure warning systems)
//

use super::boundary_detector::CrossingType;
use super::curve_detector::CurveDetector;
use super::velocity_tracker::LateralVelocityTracker;
use crate::types::{Direction, LaneChangeConfig, LaneChangeEvent, LaneChangeState, VehicleState};
use std::collections::VecDeque;
use tracing::{debug, info, warn};

// ============================================================================
// RESEARCH-BASED CONSTANTS
// ============================================================================

// Velocity thresholds (from NGSIM research: typical lane change = 0.31 m/s)
const MIN_VELOCITY_FAST: f32 = 120.0;
const MIN_VELOCITY_MEDIUM: f32 = 60.0;
const MIN_VELOCITY_SLOW: f32 = 20.0;

// TLC threshold (NHTSA standard)
const TLC_WARNING_THRESHOLD: f64 = 1.5;

// Analysis window
const ANALYSIS_WINDOW_MS: f64 = 4000.0;

// Deviation thresholds
const DEVIATION_DRIFT_START: f32 = 0.20;
const DEVIATION_CROSSING: f32 = 0.30;
const DEVIATION_LANE_CENTER: f32 = 0.50;
const DEVIATION_SIGNIFICANT: f32 = 0.40;

// Hysteresis
const HYSTERESIS_EXIT: f32 = 0.6;

// Direction consistency
const DIRECTION_CONSISTENCY_THRESHOLD: f32 = 0.65;

// Grace period
const POST_CHANGE_GRACE_FRAMES: u32 = 90;

// ============================================================================
// 🆕 KALMAN FILTER PARAMETERS (from academic research)
// ============================================================================

/// Process noise - how much we expect position to change naturally
const KALMAN_PROCESS_NOISE: f32 = 0.001;

/// Measurement noise - how noisy our lane detection is
const KALMAN_MEASUREMENT_NOISE: f32 = 0.01;

// ============================================================================
// 🆕 ADAPTIVE EWMA BASELINE PARAMETERS
// ============================================================================

/// EWMA alpha for baseline tracking (0.02 = slow adaptation, 0.1 = fast)
/// Lower = more stable baseline, Higher = faster adaptation to new position
const EWMA_ALPHA_STABLE: f32 = 0.02; // When position is stable
const EWMA_ALPHA_ADAPTING: f32 = 0.08; // When actively adapting to new position

/// Minimum frames before EWMA baseline is considered valid
const EWMA_MIN_SAMPLES: u32 = 30;

/// Variance threshold for considering position "stable" (from research)
const STABILITY_VARIANCE_THRESHOLD: f32 = 0.005; // Tighter than before

/// Maximum variance before forcing baseline reset
const INSTABILITY_VARIANCE_THRESHOLD: f32 = 0.05;

// ============================================================================
// 🆕 SIMPLE KALMAN FILTER FOR POSITION SMOOTHING
// ============================================================================

#[derive(Clone)]
struct SimpleKalmanFilter {
    /// Current state estimate
    x: f32,
    /// Current estimate uncertainty
    p: f32,
    /// Process noise
    q: f32,
    /// Measurement noise
    r: f32,
    /// Is filter initialized?
    initialized: bool,
}

impl SimpleKalmanFilter {
    fn new() -> Self {
        Self {
            x: 0.0,
            p: 1.0, // High initial uncertainty
            q: KALMAN_PROCESS_NOISE,
            r: KALMAN_MEASUREMENT_NOISE,
            initialized: false,
        }
    }

    /// Update filter with new measurement, return smoothed estimate
    fn update(&mut self, measurement: f32) -> f32 {
        if !self.initialized {
            self.x = measurement;
            self.p = self.r;
            self.initialized = true;
            return measurement;
        }

        // Predict step (assume position stays same)
        let p_pred = self.p + self.q;

        // Update step
        let k = p_pred / (p_pred + self.r); // Kalman gain
        self.x = self.x + k * (measurement - self.x);
        self.p = (1.0 - k) * p_pred;

        self.x
    }

    fn reset(&mut self) {
        self.x = 0.0;
        self.p = 1.0;
        self.initialized = false;
    }

    fn current_estimate(&self) -> f32 {
        self.x
    }
}

// ============================================================================
// 🆕 ADAPTIVE EWMA BASELINE TRACKER
// ============================================================================

#[derive(Clone)]
struct AdaptiveBaseline {
    /// Current baseline estimate
    value: f32,
    /// EWMA variance estimate
    variance: f32,
    /// Number of samples seen
    sample_count: u32,
    /// Is baseline valid/reliable?
    is_valid: bool,
    /// Recent samples for variance calculation
    recent_samples: VecDeque<f32>,
    /// Is currently adapting to new position?
    is_adapting: bool,
    /// Frames since last significant change
    stable_frames: u32,
}

impl AdaptiveBaseline {
    fn new() -> Self {
        Self {
            value: 0.0,
            variance: 1.0,
            sample_count: 0,
            is_valid: false,
            recent_samples: VecDeque::with_capacity(30),
            is_adapting: true,
            stable_frames: 0,
        }
    }

    /// Update baseline with new measurement
    fn update(&mut self, measurement: f32) -> f32 {
        self.sample_count += 1;

        // Track recent samples for variance calculation
        self.recent_samples.push_back(measurement);
        if self.recent_samples.len() > 30 {
            self.recent_samples.pop_front();
        }

        // Calculate current variance
        if self.recent_samples.len() >= 10 {
            let samples: Vec<f32> = self.recent_samples.iter().copied().collect();
            let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
            self.variance =
                samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / samples.len() as f32;
        }

        // Determine if we're stable or adapting
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

        // Choose alpha based on state
        let alpha = if self.is_adapting || !self.is_valid {
            EWMA_ALPHA_ADAPTING
        } else {
            EWMA_ALPHA_STABLE
        };

        // EWMA update
        if self.sample_count == 1 {
            self.value = measurement;
        } else {
            self.value = alpha * measurement + (1.0 - alpha) * self.value;
        }

        // Check validity
        if self.sample_count >= EWMA_MIN_SAMPLES && self.variance < INSTABILITY_VARIANCE_THRESHOLD {
            self.is_valid = true;
        }

        self.value
    }

    fn reset(&mut self) {
        self.value = 0.0;
        self.variance = 1.0;
        self.sample_count = 0;
        self.is_valid = false;
        self.recent_samples.clear();
        self.is_adapting = true;
        self.stable_frames = 0;
    }

    fn is_stable(&self) -> bool {
        self.variance < STABILITY_VARIANCE_THRESHOLD && self.stable_frames > 20
    }

    fn get_deviation(&self, measurement: f32) -> f32 {
        (measurement - self.value).abs()
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
}

// ============================================================================
// MAIN STATE MACHINE
// ============================================================================

pub struct LaneChangeStateMachine {
    config: LaneChangeConfig,
    source_id: String,

    // Core state
    state: LaneChangeState,
    frames_in_state: u32,
    pending_state: Option<LaneChangeState>,
    pending_frames: u32,

    // Lane change tracking
    change_direction: Direction,
    change_start_frame: Option<u64>,
    change_start_time: Option<f64>,
    change_detection_path: Option<DetectionPath>,
    max_offset_in_change: f32,

    // Timing
    cooldown_remaining: u32,
    total_frames_processed: u64,
    post_lane_change_grace: u32,

    // 🆕 Kalman filter for smoothing
    position_filter: SimpleKalmanFilter,

    // 🆕 Adaptive baseline
    adaptive_baseline: AdaptiveBaseline,

    // History buffers
    offset_history: Vec<f32>,
    velocity_history: VecDeque<f32>,
    offset_samples: VecDeque<OffsetSample>,
    direction_samples: VecDeque<Direction>,
    recent_deviations: Vec<f32>,

    // Stabilization
    stable_deviation_frames: u32,
    last_deviation: f32,

    // Peaks
    peak_deviation_in_window: f32,
    peak_velocity_in_window: f32,
    peak_direction: Direction,

    // Enhanced detectors
    curve_detector: CurveDetector,
    velocity_tracker: LateralVelocityTracker,

    // Curve handling
    is_in_curve: bool,
    curve_compensation_factor: f32,
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
            velocity_tracker: LateralVelocityTracker::new(),

            is_in_curve: false,
            curve_compensation_factor: 1.0,
        }
    }

    pub fn current_state(&self) -> &str {
        self.state.as_str()
    }

    pub fn update_curve_detector(&mut self, lanes: &[crate::types::Lane]) -> bool {
        self.is_in_curve = self.curve_detector.is_in_curve(lanes);
        self.curve_compensation_factor = if self.is_in_curve { 1.3 } else { 1.0 };
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

        // Skip initial frames
        if self.total_frames_processed < self.config.skip_initial_frames {
            return None;
        }

        // Handle cooldown
        if self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
            if self.cooldown_remaining == 0 {
                self.state = LaneChangeState::Centered;
                self.frames_in_state = 0;
            }
            return None;
        }

        // Handle grace period
        if self.post_lane_change_grace > 0 {
            self.post_lane_change_grace -= 1;
        }

        // Check timeout
        if self.state == LaneChangeState::Drifting || self.state == LaneChangeState::Crossing {
            if let Some(start_time) = self.change_start_time {
                let elapsed = timestamp_ms - start_time;
                if elapsed > self.config.max_duration_ms {
                    warn!("⏰ Timeout after {:.0}ms", elapsed);
                    self.reset_lane_change();
                    self.cooldown_remaining = 30;
                    return None;
                }
            }
        }

        if !vehicle_state.is_valid() {
            return None;
        }

        let lane_width = vehicle_state.lane_width.unwrap();
        let raw_offset = vehicle_state.lateral_offset / lane_width;

        // =====================================================================
        // 🆕 KALMAN FILTER: Smooth the noisy position measurement
        // =====================================================================
        let normalized_offset = self.position_filter.update(raw_offset);

        // Get velocity
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

        // =====================================================================
        // 🆕 ADAPTIVE BASELINE: Continuously track "normal" position
        // =====================================================================

        // During grace period, still collect samples but don't detect
        if self.post_lane_change_grace > 0 {
            self.adaptive_baseline.update(normalized_offset);
            return None;
        }

        // Update adaptive baseline
        let baseline = self.adaptive_baseline.update(normalized_offset);

        // Wait for baseline to become valid
        if !self.adaptive_baseline.is_valid {
            if self.adaptive_baseline.sample_count % 30 == 0 {
                debug!(
                    "Baseline forming: {:.1}% (samples={}, var={:.4})",
                    baseline * 100.0,
                    self.adaptive_baseline.sample_count,
                    self.adaptive_baseline.variance
                );
            }
            return None;
        }

        // Log when baseline becomes valid or changes significantly
        if self.adaptive_baseline.sample_count == EWMA_MIN_SAMPLES {
            info!(
                "✅ Adaptive baseline ready: {:.1}% at frame {} ({:.1}s)",
                baseline * 100.0,
                frame_id,
                timestamp_ms / 1000.0
            );
        }

        // =====================================================================
        // CALCULATE DEVIATION FROM ADAPTIVE BASELINE
        // =====================================================================

        let signed_deviation = normalized_offset - baseline;
        let deviation = signed_deviation.abs();
        let current_direction = Direction::from_offset(signed_deviation);

        // Track max offset during lane change
        if self.state == LaneChangeState::Drifting || self.state == LaneChangeState::Crossing {
            if deviation > self.max_offset_in_change {
                self.max_offset_in_change = deviation;
            }
        }

        // Update recent deviations
        self.recent_deviations.push(deviation);
        if self.recent_deviations.len() > 30 {
            self.recent_deviations.remove(0);
        }

        // =====================================================================
        // TIME-WINDOW ANALYSIS
        // =====================================================================

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

        // Track peaks
        if deviation > self.peak_deviation_in_window {
            self.peak_deviation_in_window = deviation;
            self.peak_direction = current_direction;
        }
        if lateral_velocity.abs() > self.peak_velocity_in_window {
            self.peak_velocity_in_window = lateral_velocity.abs();
        }

        let window_metrics = self.calculate_window_metrics(timestamp_ms, lane_width);

        // =====================================================================
        // STATE DETERMINATION
        // =====================================================================

        let target_state = self.determine_target_state(
            deviation,
            crossing_type,
            lateral_velocity,
            current_direction,
            &window_metrics,
        );

        debug!(
            "F{}: raw={:.1}%, smooth={:.1}%, base={:.1}%, dev={:.1}%, state={:?}→{:?}",
            frame_id,
            raw_offset * 100.0,
            normalized_offset * 100.0,
            baseline * 100.0,
            deviation * 100.0,
            self.state,
            target_state
        );

        self.check_transition(target_state, current_direction, frame_id, timestamp_ms)
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

        if metrics.avg_velocity.abs() > 5.0 {
            let distance_to_boundary = (0.5 - last.deviation.abs()) * lane_width;
            if distance_to_boundary > 0.0 {
                let tlc = distance_to_boundary / metrics.avg_velocity.abs();
                metrics.tlc_estimate = Some(tlc as f64);
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

                // PATH 2: HIGH VELOCITY
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

                // PATH 3: TLC-BASED
                if let Some(tlc) = metrics.tlc_estimate {
                    if tlc < TLC_WARNING_THRESHOLD && deviation >= drift_threshold {
                        if metrics.is_sustained_movement {
                            self.change_detection_path = Some(DetectionPath::TLCBased);
                            info!("🚨 [TLC] TLC={:.2}s, dev={:.1}%", tlc, deviation * 100.0);
                            return LaneChangeState::Drifting;
                        }
                    }
                }

                // PATH 4: MEDIUM SPEED
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

                // PATH 7: CUMULATIVE
                if metrics.max_deviation >= DEVIATION_SIGNIFICANT
                    && metrics.direction_consistency >= DIRECTION_CONSISTENCY_THRESHOLD
                    && metrics.time_span_ms >= 2500.0
                    && !self.is_in_curve
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

                let cancel_threshold = drift_threshold * HYSTERESIS_EXIT;
                if deviation < cancel_threshold {
                    if self.max_offset_in_change >= crossing_threshold {
                        return LaneChangeState::Completed;
                    } else {
                        warn!(
                            "❌ Cancelled: max={:.1}% < {:.1}%",
                            self.max_offset_in_change * 100.0,
                            crossing_threshold * 100.0
                        );
                        return LaneChangeState::Centered;
                    }
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

                // CRITERION 1: Stabilized
                if self.is_deviation_stable() && deviation < 0.35 {
                    info!("✅ Completing: stabilized at {:.1}%", deviation * 100.0);
                    return LaneChangeState::Completed;
                }

                // CRITERION 2: Returned to center
                let return_threshold = drift_threshold * HYSTERESIS_EXIT;
                if deviation < return_threshold {
                    info!("✅ Completing: returned to center");
                    return LaneChangeState::Completed;
                }

                // CRITERION 3: Prolonged stability
                if self.stable_deviation_frames >= 30 && deviation < 0.45 {
                    info!(
                        "✅ Completing: stable for {} frames",
                        self.stable_deviation_frames
                    );
                    return LaneChangeState::Completed;
                }

                // CRITERION 4: Direction reversal
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
        let baseline = self.adaptive_baseline.value;
        let count = self
            .offset_history
            .iter()
            .rev()
            .take(6)
            .filter(|o| (*o - baseline).abs() >= threshold)
            .count();
        count >= 5
    }

    fn is_deviation_sustained_long(&self, threshold: f32) -> bool {
        if self.offset_history.len() < 20 {
            return false;
        }
        let baseline = self.adaptive_baseline.value;
        let count = self
            .offset_history
            .iter()
            .rev()
            .take(15)
            .filter(|o| (*o - baseline).abs() >= threshold)
            .count();
        count >= 12
    }

    fn is_velocity_sustained(&self, threshold: f32) -> bool {
        if self.velocity_history.len() < 5 {
            return false;
        }
        let count = self
            .velocity_history
            .iter()
            .rev()
            .take(5)
            .filter(|v| v.abs() >= threshold)
            .count();
        count >= 4
    }

    fn is_deviation_stable(&self) -> bool {
        if self.recent_deviations.len() < 15 {
            return false;
        }

        let recent = &self.recent_deviations[self.recent_deviations.len() - 15..];
        let max = recent.iter().fold(f32::MIN, |a, &b| a.max(b));
        let min = recent.iter().fold(f32::MAX, |a, &b| a.min(b));
        let range = max - min;

        if range > 0.08 {
            return false;
        }

        let large_changes = recent
            .windows(2)
            .filter(|w| (w[1] - w[0]).abs() > 0.03)
            .count();

        large_changes <= 2
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
        self.stable_deviation_frames = 0;
        self.last_deviation = 0.0;
        self.recent_deviations.clear();
        self.peak_deviation_in_window = 0.0;
        self.peak_velocity_in_window = 0.0;
        self.peak_direction = Direction::Unknown;
    }

    fn check_transition(
        &mut self,
        target_state: LaneChangeState,
        direction: Direction,
        frame_id: u64,
        timestamp_ms: f64,
    ) -> Option<LaneChangeEvent> {
        if target_state == self.state {
            self.pending_state = None;
            self.pending_frames = 0;
            self.frames_in_state += 1;
            return None;
        }

        if self.pending_state == Some(target_state) {
            self.pending_frames += 1;
        } else {
            self.pending_state = Some(target_state);
            self.pending_frames = 1;
        }

        if self.pending_frames < self.config.min_frames_confirm {
            return None;
        }

        self.execute_transition(target_state, direction, frame_id, timestamp_ms)
    }

    fn execute_transition(
        &mut self,
        target_state: LaneChangeState,
        direction: Direction,
        frame_id: u64,
        timestamp_ms: f64,
    ) -> Option<LaneChangeEvent> {
        let from_state = self.state;

        info!(
            "State: {:?} → {:?} at frame {} ({:.2}s)",
            from_state,
            target_state,
            frame_id,
            timestamp_ms / 1000.0
        );

        // Starting lane change
        if target_state == LaneChangeState::Drifting && from_state == LaneChangeState::Centered {
            self.change_direction = direction;
            self.change_start_frame = Some(frame_id);
            self.change_start_time = Some(timestamp_ms);
            self.max_offset_in_change = 0.0;
            self.stable_deviation_frames = 0;
            self.last_deviation = 0.0;

            info!(
                "🚗 Lane change started: {} at {:.2}s via {:?}",
                direction.as_str(),
                timestamp_ms / 1000.0,
                self.change_detection_path
            );
        }

        // Cancellation
        if target_state == LaneChangeState::Centered && from_state == LaneChangeState::Drifting {
            info!("↩️ Cancelled");
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

        // Handle completion
        if target_state == LaneChangeState::Completed {
            // Duration validation
            if let Some(dur) = duration_ms {
                if dur < self.config.min_duration_ms
                    && self.max_offset_in_change < DEVIATION_SIGNIFICANT
                {
                    warn!("❌ Rejected: short + low dev");
                    self.reset_lane_change();
                    self.cooldown_remaining = 60;
                    return None;
                }
            }

            // Crossing threshold validation
            if self.max_offset_in_change < self.config.crossing_threshold {
                warn!(
                    "❌ Rejected: max={:.1}% < threshold",
                    self.max_offset_in_change * 100.0
                );
                self.reset_lane_change();
                self.cooldown_remaining = 60;
                return None;
            }

            self.cooldown_remaining = self.config.cooldown_frames;

            let start_frame = self.change_start_frame.unwrap_or(frame_id);
            let start_time = self.change_start_time.unwrap_or(timestamp_ms);
            let confidence = self.calculate_confidence(duration_ms);

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
                "✅ CONFIRMED: {} at {:.2}s, dur={:.0}ms, max={:.1}%, path={:?}",
                event.direction_name(),
                start_time / 1000.0,
                duration_ms.unwrap_or(0.0),
                self.max_offset_in_change * 100.0,
                self.change_detection_path
            );

            // 🆕 Reset adaptive baseline with grace period
            // The baseline will automatically adapt to new position
            self.adaptive_baseline.reset();
            self.position_filter.reset();
            self.post_lane_change_grace = POST_CHANGE_GRACE_FRAMES;
            self.offset_samples.clear();

            info!("🔄 Baseline reset - will adapt to new position");

            self.reset_lane_change();
            return Some(event);
        }

        None
    }

    fn calculate_confidence(&self, duration_ms: Option<f64>) -> f32 {
        let mut confidence: f32 = 0.5;

        if self.max_offset_in_change > 0.60 {
            confidence += 0.25;
        } else if self.max_offset_in_change > 0.50 {
            confidence += 0.20;
        } else if self.max_offset_in_change > 0.40 {
            confidence += 0.15;
        } else {
            confidence += 0.05;
        }

        if let Some(dur) = duration_ms {
            if dur > 1000.0 && dur < 6000.0 {
                confidence += 0.15;
            } else if dur > 500.0 && dur < 10000.0 {
                confidence += 0.10;
            } else {
                confidence += 0.05;
            }
        }

        if let Some(path) = &self.change_detection_path {
            match path {
                DetectionPath::BoundaryCrossing | DetectionPath::TLCBased => confidence += 0.05,
                DetectionPath::HighVelocity => confidence += 0.03,
                _ => {}
            }
        }

        confidence.min(0.95)
    }

    pub fn reset(&mut self) {
        self.reset_lane_change();
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
        self.is_in_curve = false;
        self.curve_compensation_factor = 1.0;
    }

    pub fn set_source_id(&mut self, source_id: String) {
        self.source_id = source_id;
    }
}
