// src/analysis/state_machine.rs
//
// STATE-OF-THE-ART LANE CHANGE DETECTION
// Based on academic research and industry best practices
//
// References:
// [1] NGSIM Trajectory Analysis - Kesting et al. (2008)
// [2] Learning-Based Lane-Change Detection - PMC8786501
// [3] Time-to-Lane-Crossing - NHTSA Research
// [4] Multi-Model Fusion Approach - PMC8786501
// [5] Naturalistic Driving Video Analysis - PMC6020696
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

/// Minimum lateral velocity for fast lane changes (pixels/second)
/// Based on NHTSA research: typical lane change = 0.31 m/s ≈ 150-200 px/s
const MIN_VELOCITY_FAST: f32 = 120.0;

/// Minimum lateral velocity for medium-speed lane changes
const MIN_VELOCITY_MEDIUM: f32 = 60.0;

/// Minimum lateral velocity for slow/gradual lane changes
const MIN_VELOCITY_SLOW: f32 = 20.0;

/// Time-to-Lane-Crossing warning threshold (seconds)
/// Based on NHTSA: TLC < 1.5s indicates lane departure
const TLC_WARNING_THRESHOLD: f64 = 1.5;

/// Typical lane change duration range (seconds) - from NGSIM analysis
/// Mean: 6.28s for single lane changes, range: 2-10s
const MIN_LANE_CHANGE_DURATION_S: f64 = 0.4; // Fast aggressive changes
const MAX_LANE_CHANGE_DURATION_S: f64 = 12.0; // Slow gradual changes

/// Time window for cumulative analysis (milliseconds)
/// Based on research: lane changes take 2-8 seconds typically
const ANALYSIS_WINDOW_MS: f64 = 4000.0;

/// Deviation thresholds (percentage of lane width)
/// Derived from empirical NGSIM data analysis
const DEVIATION_DRIFT_START: f32 = 0.20; // 20% - start noticing
const DEVIATION_CROSSING: f32 = 0.30; // 30% - definitely crossing
const DEVIATION_LANE_CENTER: f32 = 0.50; // 50% - at lane boundary
const DEVIATION_SIGNIFICANT: f32 = 0.40; // 40% - significant displacement

/// Hysteresis factor for state transitions
/// Different thresholds for entering vs exiting states
const HYSTERESIS_ENTER: f32 = 1.0;
const HYSTERESIS_EXIT: f32 = 0.6;

/// Direction consistency threshold (0.0-1.0)
/// Percentage of samples that must agree on direction
const DIRECTION_CONSISTENCY_THRESHOLD: f32 = 0.65;

/// Minimum frames for baseline re-establishment after lane change
/// ~3 seconds at 30fps to allow vehicle to stabilize
const POST_CHANGE_GRACE_FRAMES: u32 = 90;

/// Frames required for stable baseline establishment
const BASELINE_STABLE_FRAMES: usize = 90;

/// Maximum offset for considering vehicle "centered" during baseline
const BASELINE_CENTER_THRESHOLD: f32 = 0.12;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Sample for time-window analysis
#[derive(Clone, Copy, Debug)]
struct OffsetSample {
    normalized_offset: f32,
    deviation: f32,
    timestamp_ms: f64,
    lateral_velocity: f32,
    direction: Direction,
}

/// Metrics calculated over sliding time window
#[derive(Debug, Default)]
struct WindowMetrics {
    /// Total displacement from window start to end
    total_displacement: f32,
    /// Maximum deviation seen in window
    max_deviation: f32,
    /// Average lateral velocity over window
    avg_velocity: f32,
    /// Peak lateral velocity in window
    peak_velocity: f32,
    /// Direction consistency score (0.0-1.0)
    direction_consistency: f32,
    /// Time span of samples in window (ms)
    time_span_ms: f64,
    /// Estimated Time-to-Lane-Crossing (if applicable)
    tlc_estimate: Option<f64>,
    /// Whether this looks like an intentional lane change
    is_intentional_change: bool,
    /// Whether movement is sustained (not oscillation)
    is_sustained_movement: bool,
}

/// Detection path that triggered the lane change
#[derive(Debug, Clone, Copy, PartialEq)]
enum DetectionPath {
    BoundaryCrossing,       // Fast: crossed lane boundary with high velocity
    HighVelocity,           // Fast: high lateral velocity detected
    MediumDeviation,        // Medium: sustained deviation with moderate velocity
    GradualChange,          // Slow: gradual movement over time window
    LargeDeviation,         // Large: significant deviation regardless of speed
    TLCBased,               // TLC: time-to-lane-crossing threshold exceeded
    CumulativeDisplacement, // Cumulative: total displacement over time
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

    // Timing and cooldown
    cooldown_remaining: u32,
    total_frames_processed: u64,
    post_lane_change_grace: u32,

    // Baseline tracking
    baseline_offset: f32,
    baseline_samples: Vec<f32>,
    is_baseline_established: bool,
    frames_since_baseline: u32,
    stable_centered_frames: u32,

    // History buffers
    offset_history: Vec<f32>,
    velocity_history: VecDeque<f32>,
    offset_samples: VecDeque<OffsetSample>,
    direction_samples: VecDeque<Direction>,
    recent_deviations: Vec<f32>,

    // Stabilization detection
    stable_deviation_frames: u32,
    last_deviation: f32,

    // Peak tracking
    peak_deviation_in_window: f32,
    peak_velocity_in_window: f32,
    peak_direction: Direction,

    // Enhanced detectors
    curve_detector: CurveDetector,
    velocity_tracker: LateralVelocityTracker,

    // Curve compensation
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

            baseline_offset: 0.0,
            baseline_samples: Vec::with_capacity(BASELINE_STABLE_FRAMES + 30),
            is_baseline_established: false,
            frames_since_baseline: 0,
            stable_centered_frames: 0,

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

        // Adjust compensation factor based on curve severity
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

        // Skip initial frames for calibration
        if self.total_frames_processed < self.config.skip_initial_frames {
            return None;
        }

        // Handle cooldown period
        if self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
            if self.cooldown_remaining == 0 {
                self.state = LaneChangeState::Centered;
                self.frames_in_state = 0;
            }
            return None;
        }

        // Handle post-lane-change grace period
        if self.post_lane_change_grace > 0 {
            self.post_lane_change_grace -= 1;
        }

        // Check for timeout during active detection
        if self.state == LaneChangeState::Drifting || self.state == LaneChangeState::Crossing {
            if let Some(start_time) = self.change_start_time {
                let elapsed = timestamp_ms - start_time;
                if elapsed > self.config.max_duration_ms {
                    warn!("⏰ Lane change timeout after {:.0}ms", elapsed);
                    self.reset_lane_change();
                    self.cooldown_remaining = 30;
                    return None;
                }
            }
        }

        // Validate vehicle state
        if !vehicle_state.is_valid() {
            return None;
        }

        let lane_width = vehicle_state.lane_width.unwrap();
        let normalized_offset = vehicle_state.lateral_offset / lane_width;
        let abs_offset = normalized_offset.abs();

        // Get lateral velocity
        let lateral_velocity = self
            .velocity_tracker
            .get_velocity(vehicle_state.lateral_offset, timestamp_ms);

        // Update velocity history
        self.velocity_history.push_back(lateral_velocity);
        if self.velocity_history.len() > 30 {
            self.velocity_history.pop_front();
        }

        // Update offset history
        self.offset_history.push(normalized_offset);
        if self.offset_history.len() > 60 {
            self.offset_history.remove(0);
        }

        // =====================================================================
        // PHASE 1: BASELINE ESTABLISHMENT
        // =====================================================================
        if !self.is_baseline_established {
            if self.post_lane_change_grace > 0 {
                return None;
            }

            if abs_offset < BASELINE_CENTER_THRESHOLD {
                self.baseline_samples.push(normalized_offset);
                self.stable_centered_frames += 1;
            } else {
                if self.stable_centered_frames < 30 {
                    self.baseline_samples.clear();
                    self.stable_centered_frames = 0;
                }
            }

            if self.baseline_samples.len() >= BASELINE_STABLE_FRAMES
                && self.stable_centered_frames >= BASELINE_STABLE_FRAMES as u32
            {
                let mut sorted = self.baseline_samples.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                self.baseline_offset = sorted[sorted.len() / 2];
                self.is_baseline_established = true;
                self.frames_since_baseline = 0;

                // Clear analysis buffers
                self.offset_samples.clear();
                self.peak_deviation_in_window = 0.0;
                self.peak_velocity_in_window = 0.0;

                info!(
                    "✅ Baseline established: {:.3} ({:.1}%) at frame {} ({:.1}s)",
                    self.baseline_offset,
                    self.baseline_offset * 100.0,
                    frame_id,
                    timestamp_ms / 1000.0
                );
            }
            return None;
        }

        self.frames_since_baseline += 1;

        // Wait for baseline to stabilize
        if self.frames_since_baseline < 15 {
            return None;
        }

        // =====================================================================
        // PHASE 2: CALCULATE METRICS
        // =====================================================================

        let signed_deviation = normalized_offset - self.baseline_offset;
        let deviation = signed_deviation.abs();
        let current_direction = Direction::from_offset(signed_deviation);

        // Track max offset during active lane change
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

        // Track stable centered frames
        if deviation < 0.10 {
            self.stable_centered_frames += 1;
        } else {
            self.stable_centered_frames = 0;
        }

        // =====================================================================
        // PHASE 3: TIME-WINDOW ANALYSIS
        // =====================================================================

        let sample = OffsetSample {
            normalized_offset,
            deviation,
            timestamp_ms,
            lateral_velocity,
            direction: current_direction,
        };
        self.offset_samples.push_back(sample);

        // Remove old samples
        while let Some(oldest) = self.offset_samples.front() {
            if timestamp_ms - oldest.timestamp_ms > ANALYSIS_WINDOW_MS {
                self.offset_samples.pop_front();
            } else {
                break;
            }
        }

        // Track direction consistency
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

        // Calculate comprehensive window metrics
        let window_metrics = self.calculate_window_metrics(timestamp_ms, lane_width);

        // =====================================================================
        // PHASE 4: STATE DETERMINATION
        // =====================================================================

        let target_state = self.determine_target_state(
            deviation,
            crossing_type,
            lateral_velocity,
            current_direction,
            &window_metrics,
        );

        debug!(
            "F{}: off={:.1}%, dev={:.1}%, vel={:.1}px/s, TLC={:.2}s, curve={}, state={:?}→{:?}",
            frame_id,
            normalized_offset * 100.0,
            deviation * 100.0,
            lateral_velocity,
            window_metrics.tlc_estimate.unwrap_or(99.0),
            self.is_in_curve,
            self.state,
            target_state
        );

        self.check_transition(target_state, current_direction, frame_id, timestamp_ms)
    }

    // =========================================================================
    // TIME-WINDOW METRICS CALCULATION
    // =========================================================================

    fn calculate_window_metrics(&self, current_time_ms: f64, lane_width: f32) -> WindowMetrics {
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

        // Total displacement
        metrics.total_displacement = (last.deviation - first.deviation).abs();

        // Max deviation in window
        metrics.max_deviation = self
            .offset_samples
            .iter()
            .map(|s| s.deviation)
            .fold(0.0f32, |a, b| a.max(b));

        // Velocity analysis
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

        // Direction consistency
        if !self.direction_samples.is_empty() {
            let target_dir = self.peak_direction;
            let consistent = self
                .direction_samples
                .iter()
                .filter(|&&d| d == target_dir)
                .count();
            metrics.direction_consistency = consistent as f32 / self.direction_samples.len() as f32;
        }

        // Time-to-Lane-Crossing (TLC) estimation
        // TLC = Distance_to_boundary / lateral_velocity
        if metrics.avg_velocity.abs() > 5.0 {
            let distance_to_boundary = (0.5 - last.deviation.abs()) * lane_width;
            if distance_to_boundary > 0.0 && metrics.avg_velocity.abs() > 0.0 {
                let tlc = distance_to_boundary / metrics.avg_velocity.abs();
                metrics.tlc_estimate = Some(tlc as f64);
            }
        }

        // Determine if this is an intentional lane change
        // Criteria: sustained movement in one direction with reasonable velocity
        metrics.is_sustained_movement = metrics.direction_consistency
            >= DIRECTION_CONSISTENCY_THRESHOLD
            && metrics.time_span_ms >= 1000.0;

        metrics.is_intentional_change = metrics.max_deviation >= DEVIATION_DRIFT_START
            && metrics.is_sustained_movement
            && (metrics.avg_velocity.abs() > MIN_VELOCITY_SLOW || metrics.time_span_ms >= 2000.0);

        metrics
    }

    // =========================================================================
    // STATE DETERMINATION (Multi-Path Detection)
    // =========================================================================

    fn determine_target_state(
        &mut self,
        deviation: f32,
        crossing_type: CrossingType,
        lateral_velocity: f32,
        current_direction: Direction,
        metrics: &WindowMetrics,
    ) -> LaneChangeState {
        // Apply curve compensation to thresholds
        let drift_threshold = self.config.drift_threshold * self.curve_compensation_factor;
        let crossing_threshold = self.config.crossing_threshold * self.curve_compensation_factor;

        // Velocity thresholds (higher in curves to reduce false positives)
        let vel_fast = MIN_VELOCITY_FAST * self.curve_compensation_factor;
        let vel_medium = MIN_VELOCITY_MEDIUM * self.curve_compensation_factor;

        match self.state {
            LaneChangeState::Centered => {
                // =========================================================
                // PATH 1: BOUNDARY CROSSING (Fastest detection)
                // When vehicle actually crosses lane boundary with velocity
                // =========================================================
                if crossing_type != CrossingType::None && lateral_velocity.abs() > vel_fast {
                    if self.is_deviation_sustained(drift_threshold * 0.9) {
                        self.change_detection_path = Some(DetectionPath::BoundaryCrossing);
                        info!(
                            "🚨 [BOUNDARY] Lane change: {:?}, vel={:.1}px/s",
                            crossing_type, lateral_velocity
                        );
                        return LaneChangeState::Drifting;
                    }
                }

                // =========================================================
                // PATH 2: HIGH VELOCITY DETECTION
                // Fast lateral movement even without boundary crossing
                // =========================================================
                if lateral_velocity.abs() > vel_fast && deviation >= drift_threshold {
                    if self.is_velocity_sustained(vel_medium) {
                        self.change_detection_path = Some(DetectionPath::HighVelocity);
                        info!(
                            "🚨 [HIGH-VEL] Lane change: vel={:.1}px/s, dev={:.1}%",
                            lateral_velocity,
                            deviation * 100.0
                        );
                        return LaneChangeState::Drifting;
                    }
                }

                // =========================================================
                // PATH 3: TLC-BASED DETECTION
                // Time-to-Lane-Crossing below threshold
                // =========================================================
                if let Some(tlc) = metrics.tlc_estimate {
                    if tlc < TLC_WARNING_THRESHOLD && deviation >= drift_threshold {
                        if metrics.is_sustained_movement {
                            self.change_detection_path = Some(DetectionPath::TLCBased);
                            info!(
                                "🚨 [TLC] Lane change: TLC={:.2}s, dev={:.1}%",
                                tlc,
                                deviation * 100.0
                            );
                            return LaneChangeState::Drifting;
                        }
                    }
                }

                // =========================================================
                // PATH 4: MEDIUM SPEED DETECTION
                // Sustained deviation with moderate velocity
                // =========================================================
                if deviation >= drift_threshold + 0.10 && lateral_velocity.abs() > vel_medium {
                    if self.is_deviation_sustained(drift_threshold) {
                        self.change_detection_path = Some(DetectionPath::MediumDeviation);
                        info!(
                            "🚨 [MEDIUM] Lane change: dev={:.1}%, vel={:.1}px/s",
                            deviation * 100.0,
                            lateral_velocity
                        );
                        return LaneChangeState::Drifting;
                    }
                }

                // =========================================================
                // PATH 5: GRADUAL/SLOW LANE CHANGE
                // Uses time-window analysis for slow maneuvers
                // =========================================================
                if metrics.is_intentional_change && metrics.max_deviation >= DEVIATION_SIGNIFICANT {
                    if self.is_deviation_sustained_long(DEVIATION_DRIFT_START) {
                        self.change_detection_path = Some(DetectionPath::GradualChange);
                        info!(
                            "🚨 [GRADUAL] Lane change: max_dev={:.1}%, span={:.1}s, consistency={:.0}%",
                            metrics.max_deviation * 100.0,
                            metrics.time_span_ms / 1000.0,
                            metrics.direction_consistency * 100.0
                        );
                        return LaneChangeState::Drifting;
                    }
                }

                // =========================================================
                // PATH 6: LARGE DEVIATION (Emergency/Aggressive)
                // Very large deviation regardless of velocity
                // =========================================================
                if deviation >= DEVIATION_LANE_CENTER {
                    if self.is_deviation_sustained(drift_threshold) {
                        self.change_detection_path = Some(DetectionPath::LargeDeviation);
                        info!(
                            "🚨 [LARGE] Lane change: dev={:.1}% (exceeds 50%)",
                            deviation * 100.0
                        );
                        return LaneChangeState::Drifting;
                    }
                }

                // =========================================================
                // PATH 7: CUMULATIVE DISPLACEMENT
                // Catch-all for slow changes that accumulate over time
                // =========================================================
                if metrics.max_deviation >= DEVIATION_SIGNIFICANT
                    && metrics.direction_consistency >= DIRECTION_CONSISTENCY_THRESHOLD
                    && metrics.time_span_ms >= 2500.0
                    && !self.is_in_curve
                // More strict in curves
                {
                    self.change_detection_path = Some(DetectionPath::CumulativeDisplacement);
                    info!(
                        "🚨 [CUMULATIVE] Lane change: max_dev={:.1}%, span={:.1}s",
                        metrics.max_deviation * 100.0,
                        metrics.time_span_ms / 1000.0
                    );
                    return LaneChangeState::Drifting;
                }

                LaneChangeState::Centered
            }

            LaneChangeState::Drifting => {
                // Transition to CROSSING
                if deviation >= crossing_threshold {
                    return LaneChangeState::Crossing;
                }

                // Check for cancellation (with hysteresis)
                let cancel_threshold = drift_threshold * HYSTERESIS_EXIT;
                if deviation < cancel_threshold {
                    if self.max_offset_in_change >= crossing_threshold {
                        return LaneChangeState::Completed;
                    } else {
                        warn!(
                            "❌ Cancelled: max_dev={:.1}% < threshold={:.1}%",
                            self.max_offset_in_change * 100.0,
                            crossing_threshold * 100.0
                        );
                        return LaneChangeState::Centered;
                    }
                }

                LaneChangeState::Drifting
            }

            LaneChangeState::Crossing => {
                // Update stabilization tracking
                let deviation_change = (deviation - self.last_deviation).abs();
                if deviation_change < 0.03 {
                    self.stable_deviation_frames += 1;
                } else {
                    self.stable_deviation_frames = 0;
                }
                self.last_deviation = deviation;

                // COMPLETION CRITERION 1: Stabilized at new position
                if self.is_deviation_stable() && deviation < 0.35 {
                    info!(
                        "✅ Completing: stabilized at {:.1}% ({} frames stable)",
                        deviation * 100.0,
                        self.stable_deviation_frames
                    );
                    return LaneChangeState::Completed;
                }

                // COMPLETION CRITERION 2: Returned to center
                let return_threshold = drift_threshold * HYSTERESIS_EXIT;
                if deviation < return_threshold {
                    info!(
                        "✅ Completing: returned to center ({:.1}%)",
                        deviation * 100.0
                    );
                    return LaneChangeState::Completed;
                }

                // COMPLETION CRITERION 3: Prolonged stability at offset
                if self.stable_deviation_frames >= 30 && deviation < 0.45 {
                    info!(
                        "✅ Completing: prolonged stability ({} frames) at {:.1}%",
                        self.stable_deviation_frames,
                        deviation * 100.0
                    );
                    return LaneChangeState::Completed;
                }

                // COMPLETION CRITERION 4: Direction reversal
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
                        info!(
                            "✅ Completing: direction reversed ({} opposite frames)",
                            reversal_count
                        );
                        return LaneChangeState::Completed;
                    }
                }

                LaneChangeState::Crossing
            }

            LaneChangeState::Completed => LaneChangeState::Centered,
        }
    }

    // =========================================================================
    // HELPER FUNCTIONS
    // =========================================================================

    fn is_deviation_sustained(&self, threshold: f32) -> bool {
        if self.offset_history.len() < 8 {
            return false;
        }
        let count = self
            .offset_history
            .iter()
            .rev()
            .take(6)
            .filter(|o| (*o - self.baseline_offset).abs() >= threshold)
            .count();
        count >= 5
    }

    fn is_deviation_sustained_long(&self, threshold: f32) -> bool {
        if self.offset_history.len() < 20 {
            return false;
        }
        let count = self
            .offset_history
            .iter()
            .rev()
            .take(15)
            .filter(|o| (*o - self.baseline_offset).abs() >= threshold)
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
            info!("↩️ Lane change cancelled");
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
            // Duration validation (lenient for high-deviation cases)
            if let Some(dur) = duration_ms {
                let min_duration = self.config.min_duration_ms;
                if dur < min_duration {
                    // Accept if deviation was very high (clear lane change)
                    if self.max_offset_in_change < DEVIATION_SIGNIFICANT {
                        warn!(
                            "❌ Rejected: too short ({:.0}ms) with low deviation ({:.1}%)",
                            dur,
                            self.max_offset_in_change * 100.0
                        );
                        self.reset_lane_change();
                        self.cooldown_remaining = 60;
                        return None;
                    }
                    info!(
                        "⚠️ Short ({:.0}ms) but high deviation ({:.1}%) - accepting",
                        dur,
                        self.max_offset_in_change * 100.0
                    );
                }
            }

            // Crossing threshold validation
            if self.max_offset_in_change < self.config.crossing_threshold {
                warn!(
                    "❌ Rejected: max_dev={:.1}% < threshold={:.1}%",
                    self.max_offset_in_change * 100.0,
                    self.config.crossing_threshold * 100.0
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
                "✅ CONFIRMED: {} at {:.2}s, duration={:.0}ms, max_dev={:.1}%, path={:?}",
                event.direction_name(),
                start_time / 1000.0,
                duration_ms.unwrap_or(0.0),
                self.max_offset_in_change * 100.0,
                self.change_detection_path
            );

            // Post-completion: clear baseline with grace period
            self.baseline_samples.clear();
            self.is_baseline_established = false;
            self.stable_centered_frames = 0;
            self.frames_since_baseline = 0;
            self.post_lane_change_grace = POST_CHANGE_GRACE_FRAMES;
            self.offset_samples.clear();

            info!(
                "🔄 Baseline cleared - grace period of {} frames",
                POST_CHANGE_GRACE_FRAMES
            );

            self.reset_lane_change();
            return Some(event);
        }

        None
    }

    fn calculate_confidence(&self, duration_ms: Option<f64>) -> f32 {
        let mut confidence: f32 = 0.5;

        // Deviation-based confidence
        if self.max_offset_in_change > 0.60 {
            confidence += 0.25;
        } else if self.max_offset_in_change > 0.50 {
            confidence += 0.20;
        } else if self.max_offset_in_change > 0.40 {
            confidence += 0.15;
        } else {
            confidence += 0.05;
        }

        // Duration-based confidence
        if let Some(dur) = duration_ms {
            if dur > 1000.0 && dur < 6000.0 {
                confidence += 0.15;
            } else if dur > 500.0 && dur < 10000.0 {
                confidence += 0.10;
            } else {
                confidence += 0.05;
            }
        }

        // Detection path bonus
        if let Some(path) = &self.change_detection_path {
            match path {
                DetectionPath::BoundaryCrossing => confidence += 0.05,
                DetectionPath::TLCBased => confidence += 0.05,
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
        self.offset_history.clear();
        self.velocity_history.clear();
        self.baseline_offset = 0.0;
        self.baseline_samples.clear();
        self.is_baseline_established = false;
        self.frames_since_baseline = 0;
        self.stable_centered_frames = 0;
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
