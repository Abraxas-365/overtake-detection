// src/analysis/state_machine.rs
//
// STATE-OF-THE-ART LANE CHANGE DETECTION v2.2
//
// FIX: Freeze baseline when PENDING, not when CONFIRMED
// FIX: Track max offset during pending phase
// FIX: Allow completion from DRIFTING state
//

use super::boundary_detector::CrossingType;
use super::curve_detector::CurveDetector;
use super::velocity_tracker::LateralVelocityTracker;
use crate::types::{Direction, LaneChangeConfig, LaneChangeEvent, LaneChangeState, VehicleState};
use std::collections::VecDeque;
use tracing::{debug, info, warn};

// ============================================================================
// CONSTANTS
// ============================================================================

const MIN_VELOCITY_FAST: f32 = 120.0;
const MIN_VELOCITY_MEDIUM: f32 = 60.0;
const MIN_VELOCITY_SLOW: f32 = 20.0;
const TLC_WARNING_THRESHOLD: f64 = 1.5;
const ANALYSIS_WINDOW_MS: f64 = 4000.0;

const DEVIATION_DRIFT_START: f32 = 0.20;
const DEVIATION_CROSSING: f32 = 0.30;
const DEVIATION_LANE_CENTER: f32 = 0.50;
const DEVIATION_SIGNIFICANT: f32 = 0.40;

const HYSTERESIS_EXIT: f32 = 0.6;
const DIRECTION_CONSISTENCY_THRESHOLD: f32 = 0.65;
const POST_CHANGE_GRACE_FRAMES: u32 = 90;

const KALMAN_PROCESS_NOISE: f32 = 0.001;
const KALMAN_MEASUREMENT_NOISE: f32 = 0.01;

const EWMA_ALPHA_STABLE: f32 = 0.02;
const EWMA_ALPHA_ADAPTING: f32 = 0.08;
const EWMA_MIN_SAMPLES: u32 = 30;
const STABILITY_VARIANCE_THRESHOLD: f32 = 0.005;
const INSTABILITY_VARIANCE_THRESHOLD: f32 = 0.05;

const CURVE_COMPENSATION_FACTOR: f32 = 1.15;

// 🆕 Max time in DRIFTING before auto-completing if we have enough deviation
const MAX_DRIFTING_MS: f64 = 8000.0;

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
}

// ============================================================================
// ADAPTIVE BASELINE
// ============================================================================

#[derive(Clone)]
struct AdaptiveBaseline {
    value: f32,
    variance: f32,
    sample_count: u32,
    is_valid: bool,
    recent_samples: VecDeque<f32>,
    is_adapting: bool,
    stable_frames: u32,
    is_frozen: bool,
    frozen_value: f32,
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
            is_frozen: false,
            frozen_value: 0.0,
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

        let alpha = if self.is_adapting || !self.is_valid {
            EWMA_ALPHA_ADAPTING
        } else {
            EWMA_ALPHA_STABLE
        };

        if self.sample_count == 1 {
            self.value = measurement;
        } else {
            self.value = alpha * measurement + (1.0 - alpha) * self.value;
        }

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
        self.is_frozen = false;
        self.frozen_value = 0.0;
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

    state: LaneChangeState,
    frames_in_state: u32,
    pending_state: Option<LaneChangeState>,
    pending_frames: u32,

    change_direction: Direction,
    change_start_frame: Option<u64>,
    change_start_time: Option<f64>,
    change_detection_path: Option<DetectionPath>,
    max_offset_in_change: f32,

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

    is_in_curve: bool,
    curve_compensation_factor: f32,

    // 🆕 Track when we first detected potential lane change (for early freeze)
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
            pending_change_direction: Direction::Unknown,
            pending_max_offset: 0.0,
        }
    }

    pub fn current_state(&self) -> &str {
        self.state.as_str()
    }

    pub fn update_curve_detector(&mut self, lanes: &[crate::types::Lane]) -> bool {
        self.is_in_curve = self.curve_detector.is_in_curve(lanes);
        self.curve_compensation_factor = if self.is_in_curve {
            CURVE_COMPENSATION_FACTOR
        } else {
            1.0
        };
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

        // 🆕 IMPROVED TIMEOUT: Only timeout from DRIFTING, and allow completion if we have enough data
        if self.state == LaneChangeState::Drifting {
            if let Some(start_time) = self.change_start_time {
                let elapsed = timestamp_ms - start_time;

                // If we've been drifting a long time but have good data, try to complete
                if elapsed > MAX_DRIFTING_MS {
                    if self.max_offset_in_change >= self.config.crossing_threshold {
                        info!("⏰ Long DRIFTING ({:.0}ms) with good offset ({:.1}%) - auto-completing", 
                              elapsed, self.max_offset_in_change * 100.0);
                        // Force completion
                        return self.force_complete(frame_id, timestamp_ms);
                    }
                }

                // Full timeout
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
                    return None;
                }
            }
        }

        // Timeout for CROSSING state (shouldn't take too long)
        if self.state == LaneChangeState::Crossing {
            if let Some(start_time) = self.change_start_time {
                let elapsed = timestamp_ms - start_time;
                if elapsed > self.config.max_duration_ms {
                    // Complete anyway since we reached CROSSING
                    info!("⏰ Timeout in CROSSING - completing anyway");
                    return self.force_complete(frame_id, timestamp_ms);
                }
            }
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

        self.velocity_history.push_back(lateral_velocity);
        if self.velocity_history.len() > 30 {
            self.velocity_history.pop_front();
        }

        self.offset_history.push(normalized_offset);
        if self.offset_history.len() > 60 {
            self.offset_history.remove(0);
        }

        if self.post_lane_change_grace > 0 {
            self.adaptive_baseline.update(normalized_offset);
            return None;
        }

        self.adaptive_baseline.update(normalized_offset);

        if !self.adaptive_baseline.is_valid {
            if self.adaptive_baseline.sample_count % 30 == 0 {
                debug!(
                    "Baseline forming: {:.1}%",
                    self.adaptive_baseline.value * 100.0
                );
            }
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

        // 🆕 Track max offset during PENDING phase too!
        if self.pending_state == Some(LaneChangeState::Drifting)
            || self.state == LaneChangeState::Drifting
            || self.state == LaneChangeState::Crossing
        {
            if deviation > self.max_offset_in_change {
                self.max_offset_in_change = deviation;
            }
            // Also track pending max
            if deviation > self.pending_max_offset {
                self.pending_max_offset = deviation;
            }
        }

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

        let window_metrics = self.calculate_window_metrics(timestamp_ms, lane_width);

        let target_state = self.determine_target_state(
            deviation,
            crossing_type,
            lateral_velocity,
            current_direction,
            &window_metrics,
        );

        debug!(
            "F{}: off={:.1}%, base={:.1}%{}, dev={:.1}%, max={:.1}%, pend_max={:.1}%, state={:?}→{:?}",
            frame_id, normalized_offset * 100.0, baseline * 100.0,
            if self.adaptive_baseline.is_frozen { "🧊" } else { "" },
            deviation * 100.0, self.max_offset_in_change * 100.0, self.pending_max_offset * 100.0,
            self.state, target_state
        );

        self.check_transition(
            target_state,
            current_direction,
            frame_id,
            timestamp_ms,
            deviation,
        )
    }

    // 🆕 Force complete a lane change
    fn force_complete(&mut self, frame_id: u64, timestamp_ms: f64) -> Option<LaneChangeEvent> {
        let start_frame = self.change_start_frame.unwrap_or(frame_id);
        let start_time = self.change_start_time.unwrap_or(timestamp_ms);
        let duration_ms = Some(timestamp_ms - start_time);
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
            "✅ FORCE CONFIRMED: {} at {:.2}s, dur={:.0}ms, max={:.1}%, path={:?}",
            event.direction_name(),
            start_time / 1000.0,
            duration_ms.unwrap_or(0.0),
            self.max_offset_in_change * 100.0,
            self.change_detection_path
        );

        self.adaptive_baseline.reset();
        self.position_filter.reset();
        self.post_lane_change_grace = POST_CHANGE_GRACE_FRAMES;
        self.offset_samples.clear();
        self.cooldown_remaining = self.config.cooldown_frames;

        info!("🔄 Baseline reset - will adapt to new position");
        self.reset_lane_change();

        Some(event)
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
                    if tlc < TLC_WARNING_THRESHOLD
                        && deviation >= drift_threshold
                        && metrics.is_sustained_movement
                    {
                        self.change_detection_path = Some(DetectionPath::TLCBased);
                        info!("🚨 [TLC] TLC={:.2}s, dev={:.1}%", tlc, deviation * 100.0);
                        return LaneChangeState::Drifting;
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
                // Check for crossing threshold
                if deviation >= crossing_threshold {
                    return LaneChangeState::Crossing;
                }

                // 🆕 Also check max_offset (might have already crossed even if current deviation is lower)
                if self.max_offset_in_change >= crossing_threshold {
                    return LaneChangeState::Crossing;
                }

                // 🆕 Check if we should complete based on sustained high deviation
                // (vehicle may have settled at new position without hitting crossing threshold)
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

                // Cancellation check - only if max offset is very low
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
        self.stable_deviation_frames = 0;
        self.last_deviation = 0.0;
        self.recent_deviations.clear();
        self.peak_deviation_in_window = 0.0;
        self.peak_velocity_in_window = 0.0;
        self.peak_direction = Direction::Unknown;
        self.pending_change_direction = Direction::Unknown;
        self.pending_max_offset = 0.0;
    }

    fn check_transition(
        &mut self,
        target_state: LaneChangeState,
        direction: Direction,
        frame_id: u64,
        timestamp_ms: f64,
        current_deviation: f32,
    ) -> Option<LaneChangeEvent> {
        if target_state == self.state {
            self.pending_state = None;
            self.pending_frames = 0;
            self.frames_in_state += 1;
            return None;
        }

        // 🆕 FREEZE BASELINE IMMEDIATELY when we first detect potential lane change!
        if target_state == LaneChangeState::Drifting && self.state == LaneChangeState::Centered {
            if self.pending_state != Some(LaneChangeState::Drifting) {
                // First frame of potential lane change - freeze NOW!
                self.adaptive_baseline.freeze();
                self.pending_change_direction = direction;
                self.pending_max_offset = current_deviation;
                info!(
                    "🧊 Early freeze: baseline at {:.1}%, initial dev={:.1}%",
                    self.adaptive_baseline.effective_value() * 100.0,
                    current_deviation * 100.0
                );
            }
        }

        if self.pending_state == Some(target_state) {
            self.pending_frames += 1;
        } else {
            self.pending_state = Some(target_state);
            self.pending_frames = 1;
        }

        // 🆕 Unfreeze if we're NOT going to transition after all
        if self.pending_frames >= self.config.min_frames_confirm {
            // We're going to transition
        } else if target_state != LaneChangeState::Drifting
            && self.adaptive_baseline.is_frozen
            && self.state == LaneChangeState::Centered
        {
            // We detected something but it's not DRIFTING anymore - unfreeze
            self.adaptive_baseline.unfreeze();
            self.pending_max_offset = 0.0;
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

        if target_state == LaneChangeState::Drifting && from_state == LaneChangeState::Centered {
            self.change_direction = if self.pending_change_direction != Direction::Unknown {
                self.pending_change_direction
            } else {
                direction
            };
            self.change_start_frame = Some(frame_id);
            self.change_start_time = Some(timestamp_ms);
            // 🆕 Use pending_max_offset which was tracked during confirmation frames
            self.max_offset_in_change = self.pending_max_offset;
            self.stable_deviation_frames = 0;
            self.last_deviation = 0.0;

            // Baseline should already be frozen from check_transition
            if !self.adaptive_baseline.is_frozen {
                self.adaptive_baseline.freeze();
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
            if let Some(dur) = duration_ms {
                if dur < self.config.min_duration_ms
                    && self.max_offset_in_change < DEVIATION_SIGNIFICANT
                {
                    warn!("❌ Rejected: short + low dev");
                    self.adaptive_baseline.unfreeze();
                    self.reset_lane_change();
                    self.cooldown_remaining = 60;
                    return None;
                }
            }

            // 🆕 Lower threshold for completion validation (was crossing_threshold)
            let min_offset_for_valid = self.config.drift_threshold + 0.05; // 25% instead of 28%
            if self.max_offset_in_change < min_offset_for_valid {
                warn!(
                    "❌ Rejected: max={:.1}% < {:.1}%",
                    self.max_offset_in_change * 100.0,
                    min_offset_for_valid * 100.0
                );
                self.adaptive_baseline.unfreeze();
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
        self.is_in_curve = false;
        self.curve_compensation_factor = 1.0;
    }

    pub fn set_source_id(&mut self, source_id: String) {
        self.source_id = source_id;
    }
}
