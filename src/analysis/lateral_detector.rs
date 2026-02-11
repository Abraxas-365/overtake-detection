use std::collections::VecDeque;
use tracing::{debug, info, warn};

// ============================================================================
// CONFIGURATION
// ============================================================================

#[derive(Debug, Clone)]
pub struct LateralDetectorConfig {
    /// Minimum lane detection confidence to accept a measurement
    pub min_lane_confidence: f32,
    /// Consecutive valid frames needed before baseline is trusted
    pub baseline_warmup_frames: u32,
    /// EWMA alpha for baseline during stable driving
    pub baseline_alpha_stable: f32,
    /// EWMA alpha for baseline during fast recovery after occlusion
    pub baseline_alpha_recovery: f32,
    /// Normalized offset (|offset / lane_width|) threshold to start a shift
    pub shift_start_threshold: f32,
    /// Normalized offset threshold to confirm a shift (must exceed this at peak)
    pub shift_confirm_threshold: f32,
    /// Offset below this → shift ended (return to center)
    pub shift_end_threshold: f32,
    /// Frames the shift must persist to be reported
    pub min_shift_frames: u32,
    /// Maximum shift duration before auto-cancel (probably not a real shift)
    pub max_shift_frames: u32,
    /// After occlusion of this many frames (with no ego motion), reset baseline
    pub occlusion_reset_frames: u32,
    /// After baseline reset, freeze for this many frames before detecting
    pub post_reset_freeze_frames: u32,

    // ── v4.4: Ego-motion fusion ─────────────────────────────────
    /// Minimum ego lateral velocity (px/frame) to consider as lateral motion
    pub ego_motion_min_velocity: f32,
    /// Consecutive frames of above-threshold ego motion to start an ego-only shift
    pub ego_shift_start_frames: u32,
    /// Max frames to bridge lane dropout using ego motion during an active shift
    /// Beyond this, the ego-motion-only estimate degrades too much
    pub ego_bridge_max_frames: u32,
    /// During ego bridging: estimated px per normalized offset unit
    /// (used to convert integrated ego px to approximate normalized offset)
    pub ego_px_per_norm_unit: f32,
    /// Confidence penalty for ego-motion-only portions of a shift
    pub ego_only_confidence_penalty: f32,
    /// Max shift duration for ego-started shifts (shorter than lane-started)
    pub ego_shift_max_frames: u32,

    // ── v4.10: Ego-preempt tuning ───────────────────────────────
    /// Minimum ego lateral velocity to trigger ego-preempt when lanes ARE present.
    /// Higher than ego_motion_min_velocity because we're overriding lane data and
    /// need stronger evidence. Set to 0.0 to disable ego-preempt entirely.
    pub ego_preempt_min_velocity: f32,
    /// Minimum frames a shift must be active before shift_end_threshold is checked.
    /// Prevents ego-preempt shifts from being immediately killed before the lane
    /// offset has had time to develop (camera and lanes move together initially).
    pub shift_end_grace_frames: u32,
    /// Extended hold period for ego-preempt shifts: while ego is still active,
    /// shift_end is suppressed for up to this many frames. Only after ego stops
    /// AND this many frames have elapsed does the normal shift_end logic apply.
    /// This prevents the ego-preempt oscillation bug where shifts get rejected
    /// every 2 frames because lane offset is small.
    pub ego_preempt_hold_frames: u32,
    /// After an ego-preempt shift is rejected, suppress further ego-preempts
    /// for this many frames to prevent oscillation.
    pub ego_preempt_cooldown_frames: u32,

    // ── v4.10: Lane measurement caching ─────────────────────────
    /// Max frames to reuse the last valid lane measurement during brief dropouts.
    /// The cached measurement has its offset adjusted by ego motion each frame
    /// and its confidence decayed, so it degrades gracefully.
    pub lane_cache_max_frames: u32,
    /// Confidence decay per cached frame (multiplied each frame)
    pub lane_cache_confidence_decay: f32,
}

impl Default for LateralDetectorConfig {
    fn default() -> Self {
        Self {
            min_lane_confidence: 0.25,
            baseline_warmup_frames: 20,
            baseline_alpha_stable: 0.005,
            baseline_alpha_recovery: 0.03,
            shift_start_threshold: 0.22,
            shift_confirm_threshold: 0.30,
            shift_end_threshold: 0.12,
            min_shift_frames: 10,
            max_shift_frames: 300,        // 10s at 30fps
            occlusion_reset_frames: 45,   // 1.5s
            post_reset_freeze_frames: 30, // 1s

            // v4.4 ego-motion defaults
            ego_motion_min_velocity: 1.5, // px/frame — clear lateral motion
            ego_shift_start_frames: 8,    // ~270ms sustained motion
            ego_bridge_max_frames: 120,   // 4s max bridge
            ego_px_per_norm_unit: 600.0,  // ~lane_width, rough conversion
            ego_only_confidence_penalty: 0.2,
            ego_shift_max_frames: 180, // 6s max for ego-started shifts

            // v4.10 ego-preempt tuning
            ego_preempt_min_velocity: 3.5, // px/frame — stronger than ego_motion_min_velocity
            shift_end_grace_frames: 15,    // ~500ms at 30fps — lane offset needs time to develop
            ego_preempt_hold_frames: 75,   // ~2.5s at 30fps — keep shift alive while ego is active
            ego_preempt_cooldown_frames: 60, // ~2s cooldown after rejection

            // v4.10 lane cache defaults
            lane_cache_max_frames: 4,          // ~133ms at 30fps
            lane_cache_confidence_decay: 0.75, // 0.8 → 0.6 → 0.45 → 0.34
        }
    }
}

// ============================================================================
// TYPES
// ============================================================================

/// Direction of lateral shift
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftDirection {
    Left,
    Right,
}

impl ShiftDirection {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Left => "LEFT",
            Self::Right => "RIGHT",
        }
    }

    pub fn opposite(&self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }
}

/// A detected lateral shift event
#[derive(Debug, Clone)]
pub struct LateralShiftEvent {
    pub direction: ShiftDirection,
    /// Peak normalized offset from baseline (0.0 = no shift, 1.0 = full lane width)
    pub peak_offset: f32,
    /// Timestamp when shift started (ms)
    pub start_ms: f64,
    /// Timestamp when shift ended or was reported (ms)
    pub end_ms: f64,
    /// Frame IDs
    pub start_frame: u64,
    pub end_frame: u64,
    /// Duration in ms
    pub duration_ms: f64,
    /// Confidence based on lane detection quality during the shift
    pub confidence: f32,
    /// Was the shift confirmed (peak exceeded confirm threshold)?
    pub confirmed: bool,
}

/// Input from ego-motion estimator (optical flow based)
#[derive(Debug, Clone, Copy, Default)]
pub struct EgoMotionInput {
    /// Lateral velocity in pixels/frame (negative = leftward, positive = rightward)
    pub lateral_velocity: f32,
    /// Confidence of the ego-motion estimate [0, 1]
    pub confidence: f32,
}

/// Current state of the detector
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Waiting for lanes / building baseline
    Initializing,
    /// Baseline established, watching for shifts
    Stable,
    /// Shift in progress (lane-based, ego-bridged, or ego-started)
    Shifting,
    /// Lanes lost AND no ego motion — no output
    Occluded,
    /// Just recovered from occlusion — rebuilding baseline
    Recovering,
}

/// How the current shift is being tracked
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShiftSource {
    /// Shift started and tracked via lane position
    LaneBased,
    /// Shift started via lanes but currently bridging through dropout via ego motion
    EgoBridged,
    /// Shift started entirely from ego motion (lanes were never available)
    EgoStarted,
}

/// Input measurement from lane detection
#[derive(Debug, Clone, Copy)]
pub struct LaneMeasurement {
    /// Raw lateral offset in pixels
    pub lateral_offset_px: f32,
    /// Detected lane width in pixels
    pub lane_width_px: f32,
    /// Detection confidence [0, 1]
    pub confidence: f32,
    /// Are both lane boundaries detected?
    pub both_lanes: bool,
}

// ============================================================================
// DETECTOR
// ============================================================================

pub struct LateralShiftDetector {
    config: LateralDetectorConfig,
    state: State,

    // Baseline tracking
    baseline: f32,           // EWMA of normalized offset
    baseline_samples: u32,   // Frames contributing to baseline
    freeze_remaining: u32,   // Post-reset detection freeze
    last_lane_width_px: f32, // Last known lane width for ego→norm conversion

    // Occlusion tracking
    frames_without_lanes: u32,

    // Active shift tracking
    shift_direction: Option<ShiftDirection>,
    shift_source: ShiftSource,
    shift_start_ms: f64,
    shift_start_frame: u64,
    shift_peak_offset: f32,
    shift_frames: u32,
    shift_confidence_sum: f32,
    shift_lane_frames: u32, // frames with actual lane data during this shift

    // v4.4: Ego-motion tracking
    ego_cumulative_px: f32, // integrated ego lateral displacement during shift
    ego_active_frames: u32, // consecutive frames with strong ego motion (for ego-start)
    ego_bridge_frames: u32, // consecutive frames of ego-only bridging
    ego_estimated_offset: f32, // interpolated normalized offset during bridge
    ego_last_velocity: f32, // for logging

    // History for smoothing
    offset_history: VecDeque<f32>,

    // v4.10: Lane measurement cache for brief dropouts
    cached_measurement: Option<LaneMeasurement>,
    cached_measurement_age: u32,

    // v4.10: Ego-preempt anti-oscillation
    ego_preempt_originated: bool, // whether current shift started via ego-preempt
    ego_preempt_cooldown: u32,    // remaining cooldown frames after rejection
    ego_cumulative_peak_px: f32, // peak |ego_cumulative_px| during shift (for direction validation)
}

impl LateralShiftDetector {
    pub fn new(config: LateralDetectorConfig) -> Self {
        Self {
            config,
            state: State::Initializing,
            baseline: 0.0,
            baseline_samples: 0,
            freeze_remaining: 0,
            last_lane_width_px: 600.0,
            frames_without_lanes: 0,
            shift_direction: None,
            shift_source: ShiftSource::LaneBased,
            shift_start_ms: 0.0,
            shift_start_frame: 0,
            shift_peak_offset: 0.0,
            shift_frames: 0,
            shift_confidence_sum: 0.0,
            shift_lane_frames: 0,
            ego_cumulative_px: 0.0,
            ego_active_frames: 0,
            ego_bridge_frames: 0,
            ego_estimated_offset: 0.0,
            ego_last_velocity: 0.0,
            offset_history: VecDeque::with_capacity(30),
            cached_measurement: None,
            cached_measurement_age: 0,
            ego_preempt_originated: false,
            ego_preempt_cooldown: 0,
            ego_cumulative_peak_px: 0.0,
        }
    }

    /// Process one frame. Returns a LateralShiftEvent if a shift just completed.
    ///
    /// v4.4: Now accepts optional ego-motion input for fusion.
    /// Pass `None` for ego_motion if not available — detector falls back to
    /// lane-only behavior identical to v4.3.
    pub fn update(
        &mut self,
        measurement: Option<LaneMeasurement>,
        ego_motion: Option<EgoMotionInput>,
        timestamp_ms: f64,
        frame_id: u64,
    ) -> Option<LateralShiftEvent> {
        let ego = ego_motion.unwrap_or_default();
        self.ego_last_velocity = ego.lateral_velocity;

        // v4.10: Decrement ego-preempt cooldown
        if self.ego_preempt_cooldown > 0 {
            self.ego_preempt_cooldown -= 1;
        }

        // Track sustained ego motion for ego-start detection
        if ego.confidence > 0.3 && ego.lateral_velocity.abs() >= self.config.ego_motion_min_velocity
        {
            self.ego_active_frames += 1;
        } else {
            self.ego_active_frames = 0;
        }

        // ── VALID LANE MEASUREMENT? ─────────────────────────────
        let valid_meas = measurement
            .filter(|m| m.confidence >= self.config.min_lane_confidence && m.lane_width_px > 50.0);

        if let Some(m) = &valid_meas {
            self.frames_without_lanes = 0;
            self.last_lane_width_px = m.lane_width_px;
            self.ego_bridge_frames = 0; // lanes back, bridge ends
                                        // v4.10: Update cache with fresh measurement
            self.cached_measurement = Some(*m);
            self.cached_measurement_age = 0;
        } else {
            self.frames_without_lanes += 1;
            // v4.10: Age the cache
            self.cached_measurement_age += 1;
        }

        // ── v4.10: LANE CACHE — bridge brief dropouts ───────────
        // If no fresh lanes but we have a recent cached measurement,
        // synthesize a measurement with ego-compensated offset and
        // decayed confidence. This keeps the lane-based path active
        // through 1-4 frame dropouts instead of falling into
        // handle_no_lanes which loses lane context.
        let effective_meas = if valid_meas.is_some() {
            valid_meas
        } else if let Some(ref cached) = self.cached_measurement {
            if self.cached_measurement_age <= self.config.lane_cache_max_frames {
                // Ego-compensate: shift the cached offset by accumulated ego motion
                let ego_offset_delta = ego.lateral_velocity * self.cached_measurement_age as f32;
                let compensated_offset = cached.lateral_offset_px + ego_offset_delta;
                let decayed_confidence = cached.confidence
                    * self
                        .config
                        .lane_cache_confidence_decay
                        .powi(self.cached_measurement_age as i32);

                if self.cached_measurement_age == 1 {
                    debug!(
                        "📋 Using cached lane measurement (age={}f): offset={:.1}px → {:.1}px (ego_comp={:.1}px) | conf={:.2} → {:.2}",
                        self.cached_measurement_age,
                        cached.lateral_offset_px,
                        compensated_offset,
                        ego_offset_delta,
                        cached.confidence,
                        decayed_confidence,
                    );
                }

                Some(LaneMeasurement {
                    lateral_offset_px: compensated_offset,
                    lane_width_px: cached.lane_width_px,
                    confidence: decayed_confidence,
                    both_lanes: cached.both_lanes,
                })
            } else {
                // Cache expired — invalidate and fall through to no-lanes
                None
            }
        } else {
            None
        };

        // ── NO LANES PATH (neither fresh nor cached) ────────────
        if effective_meas.is_none() {
            return self.handle_no_lanes(ego, timestamp_ms, frame_id);
        }

        let meas = effective_meas.unwrap();

        // ── NORMALIZE ───────────────────────────────────────────
        let normalized = meas.lateral_offset_px / meas.lane_width_px;

        self.offset_history.push_back(normalized);
        if self.offset_history.len() > 20 {
            self.offset_history.pop_front();
        }

        // ── STATE MACHINE ───────────────────────────────────────
        match self.state {
            State::Occluded => {
                info!(
                    "🔄 Lateral detector: recovering from occlusion at offset={:.1}%",
                    normalized * 100.0
                );
                self.baseline = normalized;
                self.baseline_samples = 1;
                self.freeze_remaining = self.config.post_reset_freeze_frames;
                self.state = State::Recovering;
                None
            }

            State::Initializing | State::Recovering => {
                self.update_baseline(normalized);

                if self.freeze_remaining > 0 {
                    self.freeze_remaining -= 1;
                    return None;
                }

                if self.baseline_samples >= self.config.baseline_warmup_frames {
                    info!(
                        "✅ Lateral baseline established at {:.1}%",
                        self.baseline * 100.0
                    );
                    self.state = State::Stable;
                }
                None
            }

            State::Stable => {
                let deviation = normalized - self.baseline;
                let abs_dev = deviation.abs();

                if abs_dev < self.config.shift_start_threshold {
                    self.update_baseline(normalized);
                }

                if abs_dev >= self.config.shift_start_threshold {
                    let direction = if deviation < 0.0 {
                        ShiftDirection::Left
                    } else {
                        ShiftDirection::Right
                    };

                    self.start_shift(
                        direction,
                        ShiftSource::LaneBased,
                        abs_dev,
                        meas.confidence,
                        timestamp_ms,
                        frame_id,
                    );

                    info!(
                        "🔀 Lateral shift started: {} | dev={:.1}% | baseline={:.1}% | ego={:.2}px/f",
                        direction.as_str(),
                        abs_dev * 100.0,
                        self.baseline * 100.0,
                        ego.lateral_velocity,
                    );
                }
                // ── v4.10: Ego-motion pre-empt with lanes present ───
                // When lanes are present but offset hasn't crossed the
                // threshold yet, strong sustained ego motion should still
                // trigger a shift. This handles the case where the camera
                // and lane markings move together initially during a lane
                // change, keeping the lane offset small while the ego is
                // clearly moving laterally.
                // Uses a higher velocity threshold than no-lanes ego-start
                // because we're overriding lane data — need stronger evidence.
                else if self.ego_preempt_cooldown == 0
                    && self.config.ego_preempt_min_velocity > 0.0
                    && ego.lateral_velocity.abs() >= self.config.ego_preempt_min_velocity
                    && self.ego_active_frames >= self.config.ego_shift_start_frames
                {
                    let direction = if ego.lateral_velocity < 0.0 {
                        ShiftDirection::Left
                    } else {
                        ShiftDirection::Right
                    };

                    // Use ego-estimated deviation as initial offset since
                    // lane offset hasn't caught up yet
                    let est_dev = ego.lateral_velocity.abs() / self.last_lane_width_px
                        * self.ego_active_frames as f32;

                    self.start_shift(
                        direction,
                        ShiftSource::EgoStarted,
                        est_dev.max(abs_dev),
                        meas.confidence,
                        timestamp_ms,
                        frame_id,
                    );
                    self.ego_preempt_originated = true;

                    info!(
                        "🔀🚀 Ego-preempt shift (lanes present): {} | lane_dev={:.1}% < threshold {:.1}% \
                         | ego={:.2}px/f sustained {}f | est_dev={:.1}%",
                        direction.as_str(),
                        abs_dev * 100.0,
                        self.config.shift_start_threshold * 100.0,
                        ego.lateral_velocity,
                        self.ego_active_frames,
                        est_dev * 100.0,
                    );
                }

                None
            }

            State::Shifting => {
                // Lanes are back during a shift — this is the primary tracking path.
                // If we were ego-bridging, upgrade back to lane-based.
                if self.shift_source == ShiftSource::EgoBridged {
                    info!(
                        "🔄 Lanes recovered during shift — resuming lane-based tracking \
                         (bridged {} frames, ego_cum={:.1}px)",
                        self.ego_bridge_frames, self.ego_cumulative_px
                    );
                    self.shift_source = ShiftSource::LaneBased;
                }

                // For ego-started shifts that now have lanes: upgrade
                if self.shift_source == ShiftSource::EgoStarted {
                    info!("🔄 Lanes appeared for ego-started shift — upgrading to lane-based");
                    self.shift_source = ShiftSource::LaneBased;
                }

                self.shift_lane_frames += 1;
                self.update_shift_with_lane(
                    normalized,
                    meas.confidence,
                    ego,
                    timestamp_ms,
                    frame_id,
                )
            }
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // NO-LANES HANDLER (v4.4 core addition)
    // ════════════════════════════════════════════════════════════════════

    fn handle_no_lanes(
        &mut self,
        ego: EgoMotionInput,
        timestamp_ms: f64,
        frame_id: u64,
    ) -> Option<LateralShiftEvent> {
        let has_ego = ego.confidence > 0.3
            && ego.lateral_velocity.abs() >= self.config.ego_motion_min_velocity;

        match self.state {
            State::Shifting => {
                // ── ACTIVE SHIFT + NO LANES: Bridge with ego motion ──
                if has_ego && self.ego_bridge_frames < self.config.ego_bridge_max_frames {
                    self.ego_bridge_frames += 1;
                    self.shift_frames += 1;

                    if self.shift_source == ShiftSource::LaneBased {
                        self.shift_source = ShiftSource::EgoBridged;
                        // Initialize estimated offset from last known lane position
                        self.ego_estimated_offset =
                            self.offset_history.back().copied().unwrap_or(self.baseline)
                                - self.baseline;

                        info!(
                            "🌉 Ego-bridging shift (lanes lost): starting from est_offset={:.1}% | ego={:.2}px/f",
                            self.ego_estimated_offset * 100.0,
                            ego.lateral_velocity,
                        );
                    }

                    // Integrate ego motion into estimated offset
                    let ego_norm_delta = ego.lateral_velocity / self.last_lane_width_px;
                    self.ego_estimated_offset += ego_norm_delta;
                    self.ego_cumulative_px += ego.lateral_velocity;

                    let abs_est = self.ego_estimated_offset.abs();
                    if abs_est > self.shift_peak_offset {
                        self.shift_peak_offset = abs_est;
                    }

                    // Reduced confidence for ego-only frames
                    self.shift_confidence_sum += (ego.confidence * 0.5).min(0.4);

                    // Check if ego motion suggests we've settled (velocity dropped)
                    if self.ego_bridge_frames > 30
                        && ego.lateral_velocity.abs() < self.config.ego_motion_min_velocity * 0.5
                    {
                        return self.settle_shift_ego(timestamp_ms, frame_id);
                    }

                    return None;
                }

                // No ego motion OR bridge exhausted → go occluded, cancel shift
                if self.frames_without_lanes >= self.config.occlusion_reset_frames {
                    warn!(
                        "🌫️  Shift lost: lanes gone and {} (bridge_frames={}/{})",
                        if has_ego {
                            "bridge exhausted"
                        } else {
                            "no ego motion"
                        },
                        self.ego_bridge_frames,
                        self.config.ego_bridge_max_frames,
                    );
                    self.state = State::Occluded;
                    self.reset_shift();
                }
                None
            }

            State::Stable => {
                // ── NO LANES + STRONG EGO MOTION: Start ego-only shift ──
                if has_ego && self.ego_active_frames >= self.config.ego_shift_start_frames {
                    let direction = if ego.lateral_velocity < 0.0 {
                        ShiftDirection::Left
                    } else {
                        ShiftDirection::Right
                    };

                    let est_dev = ego.lateral_velocity.abs() / self.last_lane_width_px
                        * self.ego_active_frames as f32;

                    self.start_shift(
                        direction,
                        ShiftSource::EgoStarted,
                        est_dev,
                        0.3,
                        timestamp_ms,
                        frame_id,
                    );
                    self.ego_estimated_offset = est_dev
                        * if ego.lateral_velocity < 0.0 {
                            -1.0
                        } else {
                            1.0
                        };
                    self.ego_bridge_frames = 1;

                    info!(
                        "🔀🏗️ Ego-motion shift started: {} | ego={:.2}px/f sustained {}f | est_dev={:.1}%",
                        direction.as_str(),
                        ego.lateral_velocity,
                        self.ego_active_frames,
                        est_dev * 100.0,
                    );

                    return None;
                }

                // Just normal lane dropout, check for occlusion
                if self.frames_without_lanes >= self.config.occlusion_reset_frames {
                    if self.state != State::Occluded {
                        warn!(
                            "🌫️  Lateral detector: occluded ({:.1}s without lanes, ego={:.2}px/f)",
                            self.frames_without_lanes as f64 / 30.0,
                            ego.lateral_velocity,
                        );
                        self.state = State::Occluded;
                    }
                }
                None
            }

            _ => {
                // Occluded, Initializing, Recovering — just wait
                if self.frames_without_lanes >= self.config.occlusion_reset_frames
                    && self.state != State::Occluded
                {
                    self.state = State::Occluded;
                    self.reset_shift();
                }
                None
            }
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // SHIFTING STATE — LANE-BASED UPDATE
    // ════════════════════════════════════════════════════════════════════

    fn update_shift_with_lane(
        &mut self,
        normalized: f32,
        lane_confidence: f32,
        ego: EgoMotionInput,
        timestamp_ms: f64,
        frame_id: u64,
    ) -> Option<LateralShiftEvent> {
        let deviation = normalized - self.baseline;
        let abs_dev = deviation.abs();
        self.shift_frames += 1;
        self.shift_confidence_sum += lane_confidence;
        self.ego_cumulative_px += ego.lateral_velocity;

        // v4.10: Track peak cumulative for direction validation at emit time
        if self.ego_cumulative_px.abs() > self.ego_cumulative_peak_px.abs() {
            self.ego_cumulative_peak_px = self.ego_cumulative_px;
        }

        if abs_dev > self.shift_peak_offset {
            self.shift_peak_offset = abs_dev;
        }

        // ── DIRECTION VALIDATION (v4.4) ─────────────────────────
        // If lane position says one direction but cumulative ego motion
        // strongly disagrees, trust ego for direction.
        // v4.10: Only allow correction in the first 30 frames — late flips
        // are caused by the return-to-lane phase inflating cumulative.
        let lane_direction = if deviation < 0.0 {
            ShiftDirection::Left
        } else {
            ShiftDirection::Right
        };

        if let Some(current_dir) = self.shift_direction {
            if self.shift_frames <= 30
                && lane_direction != current_dir
                && abs_dev > self.config.shift_confirm_threshold
            {
                // Lane says opposite direction. Check ego.
                let ego_direction = if self.ego_cumulative_px < 0.0 {
                    ShiftDirection::Left
                } else {
                    ShiftDirection::Right
                };

                if ego_direction == lane_direction && self.ego_cumulative_px.abs() > 20.0 {
                    // Both lane AND ego disagree with initial direction → flip
                    warn!(
                        "🔄 Direction corrected: {} → {} (lane_dev={:.1}%, ego_cum={:.1}px)",
                        current_dir.as_str(),
                        lane_direction.as_str(),
                        deviation * 100.0,
                        self.ego_cumulative_px,
                    );
                    self.shift_direction = Some(lane_direction);
                }
            }
        }

        // ── MAX DURATION ────────────────────────────────────────
        let max_frames = match self.shift_source {
            ShiftSource::EgoStarted => self.config.ego_shift_max_frames,
            _ => self.config.max_shift_frames,
        };

        if self.shift_frames > max_frames {
            warn!(
                "❌ Lateral shift timeout after {} frames — settling into new baseline",
                self.shift_frames
            );
            return self.force_settle(normalized, timestamp_ms, frame_id);
        }

        // ── SHIFT END (returned toward baseline) ────────────────
        // v4.10: Grace period — don't check shift_end until the shift has been
        // active for enough frames. This prevents ego-preempt shifts from being
        // immediately killed before the lane offset has time to develop.
        //
        // For ego-preempt shifts, use extended hold period AND keep the shift
        // alive while ego is still active (the whole point of ego-preempt is
        // that lane offset lags behind ego motion).
        let grace = if self.ego_preempt_originated {
            self.config.ego_preempt_hold_frames
        } else {
            self.config.shift_end_grace_frames
        };
        let ego_holding = self.ego_preempt_originated
            && self.ego_active_frames > 0
            && self.shift_frames <= self.config.ego_preempt_hold_frames;

        if !ego_holding && self.shift_frames > grace && abs_dev < self.config.shift_end_threshold {
            let confirmed = self.shift_peak_offset >= self.config.shift_confirm_threshold
                && self.shift_frames >= self.config.min_shift_frames;

            return if confirmed {
                Some(self.emit_shift_event(timestamp_ms, frame_id))
            } else {
                debug!(
                    "❌ Lateral shift rejected: peak={:.1}% (need {:.1}%), frames={} (need {})",
                    self.shift_peak_offset * 100.0,
                    self.config.shift_confirm_threshold * 100.0,
                    self.shift_frames,
                    self.config.min_shift_frames
                );
                // v4.10: Cooldown after ego-preempt rejection to prevent oscillation
                if self.ego_preempt_originated {
                    self.ego_preempt_cooldown = self.config.ego_preempt_cooldown_frames;
                    debug!(
                        "⏸️  Ego-preempt cooldown: {} frames after rejected shift",
                        self.ego_preempt_cooldown,
                    );
                }
                self.update_baseline(normalized);
                self.state = State::Stable;
                self.reset_shift();
                None
            };
        }

        // ── SETTLED IN NEW LANE ─────────────────────────────────
        if self.shift_frames > 60 && self.is_deviation_stable() {
            let confirmed = self.shift_peak_offset >= self.config.shift_confirm_threshold;

            return if confirmed {
                let evt = self.emit_shift_event(timestamp_ms, frame_id);
                self.baseline = normalized;
                self.baseline_samples = 1;
                Some(evt)
            } else {
                self.baseline = normalized;
                self.baseline_samples = 1;
                self.state = State::Stable;
                self.reset_shift();
                None
            };
        }

        None
    }

    // ════════════════════════════════════════════════════════════════════
    // SHIFT LIFECYCLE HELPERS
    // ════════════════════════════════════════════════════════════════════

    fn start_shift(
        &mut self,
        direction: ShiftDirection,
        source: ShiftSource,
        initial_dev: f32,
        initial_confidence: f32,
        timestamp_ms: f64,
        frame_id: u64,
    ) {
        self.state = State::Shifting;
        self.shift_direction = Some(direction);
        self.shift_source = source;
        self.shift_start_ms = timestamp_ms;
        self.shift_start_frame = frame_id;
        self.shift_peak_offset = initial_dev;
        self.shift_frames = 1;
        self.shift_confidence_sum = initial_confidence;
        self.shift_lane_frames = if source == ShiftSource::EgoStarted {
            0
        } else {
            1
        };
        self.ego_cumulative_px = 0.0;
        self.ego_bridge_frames = 0;
        self.ego_estimated_offset = 0.0;
    }

    /// Settle an ego-bridged shift when ego velocity drops (vehicle stopped moving laterally).
    fn settle_shift_ego(&mut self, timestamp_ms: f64, frame_id: u64) -> Option<LateralShiftEvent> {
        let confirmed = self.shift_peak_offset >= self.config.shift_confirm_threshold
            && self.shift_frames >= self.config.min_shift_frames;

        if confirmed {
            info!(
                "✅ Lateral shift settled via ego-motion: peak_est={:.1}% | ego_cum={:.1}px | dur={:.1}s",
                self.shift_peak_offset * 100.0,
                self.ego_cumulative_px,
                (timestamp_ms - self.shift_start_ms) / 1000.0,
            );
            let evt = self.emit_shift_event(timestamp_ms, frame_id);
            // Reset baseline — we don't know exact position, so keep it and let
            // recovery re-establish when lanes come back.
            self.baseline_samples = 0;
            self.freeze_remaining = self.config.post_reset_freeze_frames;
            self.state = State::Recovering;
            return Some(evt);
        }

        // Not enough evidence to confirm — just reset
        debug!(
            "❌ Ego-bridged shift rejected: peak={:.1}%, frames={}",
            self.shift_peak_offset * 100.0,
            self.shift_frames,
        );
        self.state = State::Stable;
        self.reset_shift();
        None
    }

    /// Force-settle a shift that hit max duration.
    fn force_settle(
        &mut self,
        current_normalized: f32,
        timestamp_ms: f64,
        frame_id: u64,
    ) -> Option<LateralShiftEvent> {
        let confirmed = self.shift_peak_offset >= self.config.shift_confirm_threshold;

        let result = if confirmed {
            Some(self.emit_shift_event(timestamp_ms, frame_id))
        } else {
            None
        };

        self.baseline = current_normalized;
        self.baseline_samples = 1;
        self.state = State::Stable;
        self.reset_shift();
        result
    }

    /// Build and emit the shift event, applying direction validation.
    fn emit_shift_event(&mut self, end_ms: f64, end_frame: u64) -> LateralShiftEvent {
        // ── FINAL DIRECTION VALIDATION (v4.4) ───────────────────
        // Ego cumulative displacement is the ground truth for direction.
        // If ego clearly moved one way, trust that over initial lane reading.
        let validated_direction = self.validate_final_direction();

        let avg_confidence = if self.shift_frames > 0 {
            self.shift_confidence_sum / self.shift_frames as f32
        } else {
            0.3
        };

        let mut confidence = self.compute_confidence(avg_confidence);

        // Apply ego-only penalty proportional to how much of the shift was ego-only
        if self.shift_frames > 0 {
            let ego_only_ratio = if self.shift_lane_frames > 0 {
                1.0 - (self.shift_lane_frames as f32 / self.shift_frames as f32)
            } else {
                1.0
            };
            confidence -= self.config.ego_only_confidence_penalty * ego_only_ratio;
            confidence = confidence.max(0.20);
        }

        let evt = LateralShiftEvent {
            direction: validated_direction,
            peak_offset: self.shift_peak_offset,
            start_ms: self.shift_start_ms,
            end_ms,
            start_frame: self.shift_start_frame,
            end_frame,
            duration_ms: end_ms - self.shift_start_ms,
            confidence,
            confirmed: true,
        };

        info!(
            "✅ Lateral shift completed: {} | peak={:.1}% | dur={:.1}s | conf={:.2} | \
             source={:?} | ego_cum={:.1}px (peak={:.1}px) | lane_frames={}/{}",
            evt.direction.as_str(),
            evt.peak_offset * 100.0,
            evt.duration_ms / 1000.0,
            evt.confidence,
            self.shift_source,
            self.ego_cumulative_px,
            self.ego_cumulative_peak_px,
            self.shift_lane_frames,
            self.shift_frames,
        );

        self.state = State::Stable;
        self.reset_shift();
        evt
    }

    /// Validate shift direction using cumulative ego motion.
    /// Returns the validated direction.
    /// v4.10: Uses peak cumulative (max absolute value during shift) instead of
    /// final cumulative, because by shift end the vehicle has returned to its lane
    /// and the cumulative has swung back toward zero or even the opposite direction.
    fn validate_final_direction(&self) -> ShiftDirection {
        let initial_dir = self.shift_direction.unwrap_or(ShiftDirection::Left);

        // Use peak cumulative — this captures the direction when ego motion
        // was strongest (during the actual lane change), not after return.
        let cum = if self.ego_cumulative_peak_px.abs() > self.ego_cumulative_px.abs() {
            self.ego_cumulative_peak_px
        } else {
            self.ego_cumulative_px
        };

        // If we don't have enough ego data, trust the lane-based direction
        if cum.abs() < 15.0 {
            return initial_dir;
        }

        let ego_dir = if cum < 0.0 {
            ShiftDirection::Left
        } else {
            ShiftDirection::Right
        };

        if ego_dir != initial_dir {
            warn!(
                "🔄 Direction validation: initial={} ego={} (peak_cum={:.1}px, final_cum={:.1}px) → using {}",
                initial_dir.as_str(),
                ego_dir.as_str(),
                self.ego_cumulative_peak_px,
                self.ego_cumulative_px,
                ego_dir.as_str(),
            );
            ego_dir
        } else {
            initial_dir
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // PUBLIC QUERIES
    // ════════════════════════════════════════════════════════════════════

    pub fn is_shifting(&self) -> bool {
        self.state == State::Shifting
    }

    pub fn state_str(&self) -> &str {
        match self.state {
            State::Initializing => "INITIALIZING",
            State::Stable => "STABLE",
            State::Shifting => match self.shift_source {
                ShiftSource::LaneBased => "SHIFTING",
                ShiftSource::EgoBridged => "SHIFTING(ego-bridge)",
                ShiftSource::EgoStarted => "SHIFTING(ego-start)",
            },
            State::Occluded => "OCCLUDED",
            State::Recovering => "RECOVERING",
        }
    }

    pub fn baseline(&self) -> f32 {
        self.baseline
    }

    pub fn reset(&mut self) {
        self.state = State::Initializing;
        self.baseline = 0.0;
        self.baseline_samples = 0;
        self.freeze_remaining = 0;
        self.last_lane_width_px = 600.0;
        self.frames_without_lanes = 0;
        self.ego_active_frames = 0;
        self.ego_last_velocity = 0.0;
        self.offset_history.clear();
        self.cached_measurement = None;
        self.cached_measurement_age = 0;
        self.ego_preempt_cooldown = 0;
        self.reset_shift();
    }

    // ════════════════════════════════════════════════════════════════════
    // PRIVATE HELPERS
    // ════════════════════════════════════════════════════════════════════

    fn update_baseline(&mut self, normalized: f32) {
        let alpha = if self.state == State::Recovering {
            self.config.baseline_alpha_recovery
        } else {
            self.config.baseline_alpha_stable
        };

        if self.baseline_samples == 0 {
            self.baseline = normalized;
        } else {
            self.baseline = alpha * normalized + (1.0 - alpha) * self.baseline;
        }
        self.baseline_samples += 1;
    }

    fn reset_shift(&mut self) {
        self.shift_direction = None;
        self.shift_source = ShiftSource::LaneBased;
        self.shift_start_ms = 0.0;
        self.shift_start_frame = 0;
        self.shift_peak_offset = 0.0;
        self.shift_frames = 0;
        self.shift_confidence_sum = 0.0;
        self.shift_lane_frames = 0;
        self.ego_cumulative_px = 0.0;
        self.ego_cumulative_peak_px = 0.0;
        self.ego_bridge_frames = 0;
        self.ego_estimated_offset = 0.0;
        self.ego_preempt_originated = false;
    }

    fn is_deviation_stable(&self) -> bool {
        if self.offset_history.len() < 10 {
            return false;
        }
        let recent: Vec<f32> = self.offset_history.iter().rev().take(10).copied().collect();
        let mean = recent.iter().sum::<f32>() / recent.len() as f32;
        let var = recent.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / recent.len() as f32;
        var < 0.002
    }

    fn compute_confidence(&self, avg_lane_confidence: f32) -> f32 {
        let mut conf: f32 = 0.5;

        if avg_lane_confidence > 0.7 {
            conf += 0.20;
        } else if avg_lane_confidence > 0.5 {
            conf += 0.10;
        }

        if self.shift_peak_offset > 0.50 {
            conf += 0.15;
        } else if self.shift_peak_offset > 0.35 {
            conf += 0.10;
        }

        let dur_s = self.shift_frames as f32 / 30.0;
        if dur_s >= 1.0 && dur_s <= 8.0 {
            conf += 0.10;
        }

        conf.min(0.95)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn meas(offset_px: f32, lane_w: f32, conf: f32) -> Option<LaneMeasurement> {
        Some(LaneMeasurement {
            lateral_offset_px: offset_px,
            lane_width_px: lane_w,
            confidence: conf,
            both_lanes: true,
        })
    }

    fn ego(vel: f32) -> Option<EgoMotionInput> {
        Some(EgoMotionInput {
            lateral_velocity: vel,
            confidence: 0.9,
        })
    }

    fn no_ego() -> Option<EgoMotionInput> {
        None
    }

    #[test]
    fn test_baseline_warmup() {
        let cfg = LateralDetectorConfig {
            baseline_warmup_frames: 10,
            ..Default::default()
        };
        let mut det = LateralShiftDetector::new(cfg);

        for i in 0..10 {
            let result = det.update(meas(0.0, 400.0, 0.8), no_ego(), i as f64 * 33.3, i);
            assert!(result.is_none());
        }

        assert_eq!(det.state, State::Stable);
        assert!(det.baseline().abs() < 0.01);
    }

    #[test]
    fn test_lateral_shift_detection() {
        let cfg = LateralDetectorConfig {
            baseline_warmup_frames: 10,
            min_shift_frames: 5,
            ..Default::default()
        };
        let mut det = LateralShiftDetector::new(cfg);
        let w = 400.0;

        // Warmup
        for i in 0..20 {
            det.update(meas(0.0, w, 0.8), no_ego(), i as f64 * 33.3, i);
        }

        // Shift left
        for i in 20..40 {
            det.update(meas(-140.0, w, 0.8), ego(-3.0), i as f64 * 33.3, i);
        }

        // Return to center
        let mut events = Vec::new();
        for i in 40..50 {
            if let Some(e) = det.update(meas(0.0, w, 0.8), ego(0.0), i as f64 * 33.3, i) {
                events.push(e);
            }
        }

        assert!(!events.is_empty(), "Should detect the lateral shift");
        assert_eq!(events[0].direction, ShiftDirection::Left);
        assert!(events[0].confirmed);
    }

    #[test]
    fn test_occlusion_suppression() {
        let cfg = LateralDetectorConfig {
            baseline_warmup_frames: 10,
            occlusion_reset_frames: 15,
            ..Default::default()
        };
        let mut det = LateralShiftDetector::new(cfg);

        for i in 0..20 {
            det.update(meas(0.0, 400.0, 0.8), no_ego(), i as f64 * 33.3, i);
        }

        // Occlusion with no ego motion
        for i in 20..40 {
            let result = det.update(None, no_ego(), i as f64 * 33.3, i);
            assert!(result.is_none());
        }

        assert_eq!(det.state, State::Occluded);
    }

    // ── v4.4 TESTS ──────────────────────────────────────────────

    #[test]
    fn test_ego_motion_starts_shift_during_lane_dropout() {
        let cfg = LateralDetectorConfig {
            baseline_warmup_frames: 10,
            ego_motion_min_velocity: 1.5,
            ego_shift_start_frames: 8,
            ego_shift_max_frames: 120,
            min_shift_frames: 5,
            ..Default::default()
        };
        let mut det = LateralShiftDetector::new(cfg);
        let w = 800.0;

        // Warmup with lanes
        for i in 0..20 {
            det.update(meas(0.0, w, 0.8), ego(0.0), i as f64 * 33.3, i);
        }
        assert_eq!(det.state, State::Stable);

        // Lanes drop out, strong leftward ego motion
        for i in 20..35 {
            det.update(None, ego(-5.0), i as f64 * 33.3, i);
        }

        // Should have started an ego-motion shift
        assert_eq!(det.state, State::Shifting);
        assert_eq!(det.shift_source, ShiftSource::EgoStarted);
        assert_eq!(det.shift_direction, Some(ShiftDirection::Left));
    }

    #[test]
    fn test_ego_bridge_maintains_shift_through_dropout() {
        let cfg = LateralDetectorConfig {
            baseline_warmup_frames: 10,
            min_shift_frames: 5,
            ego_motion_min_velocity: 1.5,
            ego_bridge_max_frames: 60,
            ..Default::default()
        };
        let mut det = LateralShiftDetector::new(cfg);
        let w = 400.0;

        // Warmup
        for i in 0..20 {
            det.update(meas(0.0, w, 0.8), ego(0.0), i as f64 * 33.3, i);
        }

        // Start shift with lanes, then lanes drop
        for i in 20..25 {
            det.update(meas(-120.0, w, 0.8), ego(-4.0), i as f64 * 33.3, i);
        }
        assert_eq!(det.state, State::Shifting);
        assert_eq!(det.shift_source, ShiftSource::LaneBased);

        // Lanes drop, ego motion continues
        for i in 25..50 {
            det.update(None, ego(-4.0), i as f64 * 33.3, i);
        }

        // Should be ego-bridged, NOT occluded
        assert_eq!(det.state, State::Shifting);
        assert_eq!(det.shift_source, ShiftSource::EgoBridged);
        assert!(
            det.ego_cumulative_px < -50.0,
            "Should have accumulated leftward displacement"
        );
    }

    #[test]
    fn test_direction_validation_corrects_wrong_initial() {
        let cfg = LateralDetectorConfig {
            baseline_warmup_frames: 10,
            min_shift_frames: 5,
            shift_start_threshold: 0.15,
            shift_confirm_threshold: 0.25,
            ..Default::default()
        };
        let mut det = LateralShiftDetector::new(cfg);
        let w = 400.0;

        // Warmup
        for i in 0..20 {
            det.update(meas(0.0, w, 0.8), ego(0.0), i as f64 * 33.3, i);
        }

        // Noisy measurement triggers RIGHT shift, but ego is going LEFT
        det.update(meas(80.0, w, 0.6), ego(-3.0), 20.0 * 33.3, 20);
        assert_eq!(det.shift_direction, Some(ShiftDirection::Right));

        // Continued: lanes correct to LEFT, ego also LEFT
        for i in 21..50 {
            det.update(meas(-160.0, w, 0.8), ego(-4.0), i as f64 * 33.3, i);
        }

        // Settle in new lane
        let mut events = Vec::new();
        for i in 50..80 {
            if let Some(e) = det.update(meas(-160.0, w, 0.8), ego(0.0), i as f64 * 33.3, i) {
                events.push(e);
            }
        }

        assert!(!events.is_empty(), "Should detect shift");
        assert_eq!(
            events[0].direction,
            ShiftDirection::Left,
            "Direction should be corrected to LEFT via ego validation"
        );
    }

    #[test]
    fn test_ego_settle_when_velocity_drops() {
        let cfg = LateralDetectorConfig {
            baseline_warmup_frames: 10,
            min_shift_frames: 5,
            ego_motion_min_velocity: 1.5,
            ego_shift_start_frames: 8,
            ego_bridge_max_frames: 120,
            shift_confirm_threshold: 0.10, // lower for ego-only
            ..Default::default()
        };
        let mut det = LateralShiftDetector::new(cfg);
        let w = 600.0;

        // Warmup
        for i in 0..20 {
            det.update(meas(0.0, w, 0.8), ego(0.0), i as f64 * 33.3, i);
        }

        // Lanes drop + strong ego motion → ego shift starts
        for i in 20..40 {
            det.update(None, ego(-5.0), i as f64 * 33.3, i);
        }
        assert_eq!(det.state, State::Shifting);

        // Continue bridging, then ego velocity drops (settled)
        let mut events = Vec::new();
        for i in 40..55 {
            det.update(None, ego(-5.0), i as f64 * 33.3, i);
        }
        for i in 55..90 {
            if let Some(e) = det.update(None, ego(-0.3), i as f64 * 33.3, i) {
                events.push(e);
            }
        }

        assert!(!events.is_empty(), "Should settle when ego velocity drops");
        assert_eq!(events[0].direction, ShiftDirection::Left);
    }

    #[test]
    fn test_no_ego_shift_with_weak_motion() {
        let cfg = LateralDetectorConfig {
            baseline_warmup_frames: 10,
            ego_motion_min_velocity: 1.5,
            ego_shift_start_frames: 8,
            ..Default::default()
        };
        let mut det = LateralShiftDetector::new(cfg);

        for i in 0..20 {
            det.update(meas(0.0, 400.0, 0.8), ego(0.0), i as f64 * 33.3, i);
        }

        // Lanes drop, weak ego motion (below threshold) → should NOT start shift
        for i in 20..50 {
            det.update(None, ego(-0.5), i as f64 * 33.3, i);
        }

        // Should go occluded, not shifting
        assert_ne!(det.state, State::Shifting);
    }

    // ── v4.10 TESTS ─────────────────────────────────────────────

    #[test]
    fn test_ego_preempt_with_lanes_present() {
        // v4.10: Strong ego motion should start a shift even when lanes
        // are present but offset hasn't crossed the threshold yet.
        // Requires ego velocity >= ego_preempt_min_velocity (higher bar).
        let cfg = LateralDetectorConfig {
            baseline_warmup_frames: 10,
            ego_motion_min_velocity: 1.5,
            ego_preempt_min_velocity: 3.5,
            ego_shift_start_frames: 8,
            shift_start_threshold: 0.35, // mining threshold
            min_shift_frames: 5,
            ..Default::default()
        };
        let mut det = LateralShiftDetector::new(cfg);
        let w = 800.0;

        // Warmup
        for i in 0..20 {
            det.update(meas(0.0, w, 0.8), ego(0.0), i as f64 * 33.3, i);
        }
        assert_eq!(det.state, State::Stable);

        // Strong ego motion (above preempt threshold) with lanes present but small offset
        // Offset = -80px / 800px = 10% — below 35% threshold
        for i in 20..35 {
            det.update(meas(-80.0, w, 0.8), ego(-7.0), i as f64 * 33.3, i);
        }

        // Should be shifting via ego-preempt despite offset < threshold
        assert_eq!(det.state, State::Shifting);
        assert_eq!(det.shift_direction, Some(ShiftDirection::Left));
    }

    #[test]
    fn test_ego_preempt_rejected_weak_motion() {
        // v4.10: Moderate ego motion below ego_preempt_min_velocity should
        // NOT trigger ego-preempt when lanes are present. This prevents
        // false triggers from road vibration / mild corrections.
        let cfg = LateralDetectorConfig {
            baseline_warmup_frames: 10,
            ego_motion_min_velocity: 1.5,
            ego_preempt_min_velocity: 3.5,
            ego_shift_start_frames: 8,
            shift_start_threshold: 0.35,
            ..Default::default()
        };
        let mut det = LateralShiftDetector::new(cfg);
        let w = 800.0;

        // Warmup
        for i in 0..20 {
            det.update(meas(0.0, w, 0.8), ego(0.0), i as f64 * 33.3, i);
        }

        // Moderate ego motion (above ego_motion_min but below preempt threshold)
        for i in 20..40 {
            det.update(meas(-50.0, w, 0.8), ego(-2.5), i as f64 * 33.3, i);
        }

        // Should NOT be shifting — ego too weak for preempt
        assert_eq!(det.state, State::Stable);
    }

    #[test]
    fn test_shift_end_grace_period() {
        // v4.10: Shift should not be killed by shift_end_threshold during
        // the grace period, giving lane offset time to develop after ego-preempt.
        let cfg = LateralDetectorConfig {
            baseline_warmup_frames: 10,
            ego_motion_min_velocity: 1.5,
            ego_preempt_min_velocity: 3.5,
            ego_shift_start_frames: 8,
            shift_start_threshold: 0.35,
            shift_end_threshold: 0.12,
            shift_end_grace_frames: 15,
            ego_preempt_hold_frames: 75,
            min_shift_frames: 5,
            ..Default::default()
        };
        let mut det = LateralShiftDetector::new(cfg);
        let w = 800.0;

        // Warmup
        for i in 0..20 {
            det.update(meas(0.0, w, 0.8), ego(0.0), i as f64 * 33.3, i);
        }

        // Ego-preempt starts shift
        for i in 20..30 {
            det.update(meas(-50.0, w, 0.8), ego(-5.0), i as f64 * 33.3, i);
        }
        assert_eq!(det.state, State::Shifting);

        // Lane offset is small (6.25% < shift_end 12%) but within grace period
        // — shift should survive
        for i in 30..38 {
            det.update(meas(-50.0, w, 0.8), ego(-5.0), i as f64 * 33.3, i);
        }
        assert_eq!(
            det.state,
            State::Shifting,
            "Shift should survive during grace period"
        );
    }

    #[test]
    fn test_ego_preempt_hold_keeps_shift_alive() {
        // v4.10: Even after shift_end_grace expires, ego-preempt shift stays
        // alive while ego is still active (ego_active_frames > 0) within the
        // hold period. The shift only ends when ego stops OR hold expires.
        let cfg = LateralDetectorConfig {
            baseline_warmup_frames: 10,
            ego_motion_min_velocity: 1.5,
            ego_preempt_min_velocity: 3.5,
            ego_shift_start_frames: 8,
            shift_start_threshold: 0.35,
            shift_end_threshold: 0.12,
            shift_end_grace_frames: 15,
            ego_preempt_hold_frames: 75,
            min_shift_frames: 5,
            ..Default::default()
        };
        let mut det = LateralShiftDetector::new(cfg);
        let w = 800.0;

        // Warmup
        for i in 0..20 {
            det.update(meas(0.0, w, 0.8), ego(0.0), i as f64 * 33.3, i);
        }

        // Ego-preempt starts shift (8 frames sustained at -7.0)
        for i in 20..30 {
            det.update(meas(-50.0, w, 0.8), ego(-7.0), i as f64 * 33.3, i);
        }
        assert_eq!(det.state, State::Shifting);
        assert!(det.ego_preempt_originated);

        // Continue for 50 more frames with ego active + small lane offset
        // This is PAST the 15-frame grace but WITHIN the 75-frame hold
        // and ego is still active → shift must survive
        for i in 30..80 {
            det.update(meas(-60.0, w, 0.8), ego(-6.0), i as f64 * 33.3, i);
        }
        assert_eq!(
            det.state,
            State::Shifting,
            "Ego-preempt hold should keep shift alive while ego is active"
        );
    }

    #[test]
    fn test_ego_preempt_cooldown_prevents_oscillation() {
        // v4.10: After an ego-preempt shift is rejected, cooldown prevents
        // immediate re-triggering (the oscillation bug).
        let cfg = LateralDetectorConfig {
            baseline_warmup_frames: 10,
            ego_motion_min_velocity: 1.5,
            ego_preempt_min_velocity: 3.5,
            ego_shift_start_frames: 8,
            shift_start_threshold: 0.35,
            shift_end_threshold: 0.12,
            ego_preempt_hold_frames: 10, // Short hold for test
            ego_preempt_cooldown_frames: 30,
            min_shift_frames: 5,
            shift_confirm_threshold: 0.30,
            ..Default::default()
        };
        let mut det = LateralShiftDetector::new(cfg);
        let w = 800.0;

        // Warmup
        for i in 0..20 {
            det.update(meas(0.0, w, 0.8), ego(0.0), i as f64 * 33.3, i);
        }

        // Ego-preempt starts shift
        for i in 20..30 {
            det.update(meas(-30.0, w, 0.8), ego(-5.0), i as f64 * 33.3, i);
        }
        assert_eq!(det.state, State::Shifting);

        // Ego stops → hold expires → shift rejected (peak < confirm threshold)
        for i in 30..50 {
            det.update(meas(-30.0, w, 0.8), ego(0.0), i as f64 * 33.3, i);
        }
        assert_eq!(det.state, State::Stable, "Shift should be rejected");
        assert!(det.ego_preempt_cooldown > 0, "Cooldown should be active");

        // Ego motion resumes but cooldown is active → should NOT re-trigger
        for i in 50..65 {
            det.update(meas(-30.0, w, 0.8), ego(-5.0), i as f64 * 33.3, i);
        }
        assert_eq!(
            det.state,
            State::Stable,
            "Ego-preempt should not fire during cooldown"
        );
    }

    #[test]
    fn test_lane_cache_bridges_brief_dropout() {
        // v4.10: Brief lane dropout should use cached measurement
        // instead of falling into handle_no_lanes.
        let cfg = LateralDetectorConfig {
            baseline_warmup_frames: 10,
            min_shift_frames: 5,
            shift_start_threshold: 0.22,
            lane_cache_max_frames: 4,
            ..Default::default()
        };
        let mut det = LateralShiftDetector::new(cfg);
        let w = 800.0;

        // Warmup
        for i in 0..20 {
            det.update(meas(0.0, w, 0.8), ego(0.0), i as f64 * 33.3, i);
        }
        assert_eq!(det.state, State::Stable);

        // Start a shift with a big offset
        det.update(meas(-200.0, w, 0.8), ego(-4.0), 20.0 * 33.3, 20);
        assert_eq!(det.state, State::Shifting);

        // Lanes drop for 3 frames (within cache window)
        for i in 21..24 {
            det.update(None, ego(-4.0), i as f64 * 33.3, i);
        }

        // Should still be shifting (NOT ego-bridged), because cached
        // measurement kept the lane-based path active
        assert_eq!(det.state, State::Shifting);
        // After lanes come back, should be LaneBased (cache kept it from
        // ever transitioning to EgoBridged for a 3-frame dropout)
        det.update(meas(-250.0, w, 0.8), ego(-4.0), 24.0 * 33.3, 24);
        assert_eq!(det.state, State::Shifting);
        assert_eq!(det.shift_source, ShiftSource::LaneBased);
    }

    #[test]
    fn test_lane_cache_expires_after_max_frames() {
        // v4.10: Cache should expire and fall through to handle_no_lanes
        // after lane_cache_max_frames.
        let cfg = LateralDetectorConfig {
            baseline_warmup_frames: 10,
            lane_cache_max_frames: 3,
            occlusion_reset_frames: 10,
            ..Default::default()
        };
        let mut det = LateralShiftDetector::new(cfg);
        let w = 800.0;

        // Warmup
        for i in 0..20 {
            det.update(meas(0.0, w, 0.8), ego(0.0), i as f64 * 33.3, i);
        }
        assert_eq!(det.state, State::Stable);

        // Lanes drop for longer than cache window
        for i in 20..35 {
            det.update(None, no_ego(), i as f64 * 33.3, i);
        }

        // Should be occluded (cache expired, no ego motion)
        assert_eq!(det.state, State::Occluded);
    }
}

