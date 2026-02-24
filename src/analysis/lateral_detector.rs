use std::collections::VecDeque;
use tracing::{debug, info, warn};

use super::curvature_estimator::CurvatureEstimate;
use super::polynomial_tracker::GeometricLaneChangeSignals;

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

    // ── v4.11: Curve-aware false positive suppression ───────────
    /// Boundary coherence above this → curve mode activates (after sustained frames).
    /// Range: [0, 1]. Higher = more conservative (requires stronger co-movement).
    pub curve_coherence_threshold: f32,
    /// Multiplier applied to shift_start_threshold and shift_confirm_threshold
    /// while in curve mode. E.g. 1.8 raises thresholds by 80%.
    pub curve_shift_threshold_multiplier: f32,
    /// Consecutive frames of above-threshold coherence needed to activate curve mode.
    pub curve_min_sustained_frames: u32,
    /// v4.12: Multiplier applied to ego velocity thresholds during curve mode.
    /// On curves, optical flow shows genuine lateral pixel displacement (road
    /// surface rotates in camera frame), but it's rotational, not translational.
    /// Raising the bar prevents ego-preempt and ego-start from triggering on curves.
    pub curve_ego_velocity_multiplier: f32,
    /// v4.12: Frames after curve mode deactivates during which ego-only shift
    /// starts remain suppressed. Curves often cause brief coherence drops when
    /// one boundary is temporarily lost, so we carry the suppression forward.
    pub curve_ego_cooldown_frames: u32,

    // ── v4.11: Adaptive baseline alpha (smooth drift tracking) ──
    /// Maximum baseline alpha when smooth drift is detected.
    /// The actual alpha interpolates between baseline_alpha_stable and this value
    /// based on how drift-like the offset history is.
    pub adaptive_baseline_alpha_max: f32,
    /// Maximum variance of recent offsets to qualify as "smooth drift".
    /// Low variance + consistent drift = curve. High variance = noise/lane change.
    pub adaptive_baseline_max_variance: f32,
    /// Minimum absolute drift rate (normalized offset per frame) to boost alpha.
    /// Below this, the offset is essentially stable and base alpha is fine.
    pub adaptive_baseline_min_drift: f32,
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

            // v4.11 curve suppression defaults
            curve_coherence_threshold: 0.65,
            curve_shift_threshold_multiplier: 1.8,
            curve_min_sustained_frames: 5,
            curve_ego_velocity_multiplier: 2.0, // v4.12: double ego thresholds on curves
            curve_ego_cooldown_frames: 15,      // v4.12: ~500ms at 30fps

            // v4.11 adaptive baseline defaults
            adaptive_baseline_alpha_max: 0.04,
            adaptive_baseline_max_variance: 0.0015,
            adaptive_baseline_min_drift: 0.002,
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
    /// v4.13: Whether curve mode was active during this shift.
    pub curve_mode: bool,
    /// v4.13: Peak absolute ego cumulative displacement during shift (px).
    /// Near zero = no physical lateral motion (perspective artifact if LaneBased).
    pub ego_cumulative_peak_px: f32,
    /// v4.13: Shift detection source (LaneBased / EgoBridged / EgoStarted).
    pub source_label: &'static str,
    /// v7.0: Snapshot of geometric lane change signals at time of shift emission.
    /// Populated by the pipeline from the polynomial tracker; None when created.
    pub geometric_signals: Option<GeometricLaneChangeSignals>,
}

/// v7.0: Early notification when a shift is confirmed in-progress.
/// Fires once per shift when peak exceeds confirm threshold, min frames met,
/// and (on curves) ego motion confirms lateral displacement.
/// Used to emit an early LANE_CHANGE before the shift completes.
#[derive(Debug, Clone)]
pub struct ShiftConfirmedNotification {
    pub direction: ShiftDirection,
    pub start_ms: f64,
    pub start_frame: u64,
    /// Peak offset at confirmation time
    pub peak_offset: f32,
    /// Confidence estimate at confirmation time
    pub confidence: f32,
    /// Was curve mode active?
    pub curve_mode: bool,
    /// Frame when confirmation occurred
    pub confirmed_frame: u64,
    pub confirmed_ms: f64,
    /// Geometric signals snapshot (populated by pipeline)
    pub geometric_signals: Option<GeometricLaneChangeSignals>,
}

/// Result from a single frame update of the lateral shift detector.
pub struct LateralUpdateResult {
    /// A shift that completed this frame (returned to baseline / settled / timed out).
    pub completed_shift: Option<LateralShiftEvent>,
    /// A shift that was just confirmed in-progress (fires once per shift).
    /// Used to emit an early LANE_CHANGE before the shift completes.
    pub confirmed_in_progress: Option<ShiftConfirmedNotification>,
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
#[derive(Debug, Clone)]
pub struct LaneMeasurement {
    /// Raw lateral offset in pixels
    pub lateral_offset_px: f32,
    /// Detected lane width in pixels
    pub lane_width_px: f32,
    /// Detection confidence [0, 1]
    pub confidence: f32,
    /// Are both lane boundaries detected?
    pub both_lanes: bool,
    /// v4.11: Boundary coherence from lane_legality detector.
    /// +1.0 = boundaries co-moving (curve), -1.0 = diverging (lane change),
    /// 0.0 = inconclusive, < -0.5 = no data.
    pub boundary_coherence: f32,
    /// v4.13: Polynomial curvature estimate from YOLO-seg mask geometry.
    /// Direct geometric measurement — replaces boundary coherence as primary
    /// curve signal when available. Boundary coherence remains as fallback.
    pub curvature: Option<CurvatureEstimate>,
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
    // v4.13b: Ego confidence accumulation during active shift.
    // Tracks how many frames had confident ego readings (conf > 0.3).
    // If too few frames had confident readings, the ego "no motion"
    // conclusion is unreliable (estimator failure, not vehicle stillness)
    // and should not be used to veto lane changes.
    ego_shift_confident_frames: u32,

    // History for smoothing
    offset_history: VecDeque<f32>,

    // v4.10: Lane measurement cache for brief dropouts
    cached_measurement: Option<LaneMeasurement>,
    cached_measurement_age: u32,

    // v4.10: Ego-preempt anti-oscillation
    ego_preempt_originated: bool, // whether current shift started via ego-preempt
    ego_preempt_cooldown: u32,    // remaining cooldown frames after rejection
    ego_cumulative_peak_px: f32, // peak |ego_cumulative_px| during shift (for direction validation)

    // v4.11: Curve-aware suppression state
    coherence_history: VecDeque<f32>, // recent boundary coherence values
    curve_sustained_frames: u32,      // consecutive frames above coherence threshold
    in_curve_mode: bool,              // whether thresholds are currently raised
    // v4.12: Curve ego suppression
    frames_since_curve_mode: u32, // frames since curve mode was last active
    // v4.13b: Whether curve mode was active at ANY point during the current shift.
    // Catches false positives that start/complete during brief curve-mode gaps.
    shift_saw_curve_mode: bool,

    // v4.13b: Whether the current shift was allowed through via return-window
    // bypass. If true, the resulting LC must NOT set a new return expectation,
    // preventing cascading false positives (RIGHT→LEFT→RIGHT→...).
    shift_used_return_bypass: bool,
    /// v7.0: Whether early "shift confirmed" notification has fired for this shift.
    shift_confirmed_notified: bool,

    // v4.13b: Post-lane-change return expectation.
    //
    // After a confirmed lane change (e.g. LEFT), the return in the opposite
    // direction (RIGHT) is overwhelmingly likely within ~30s. On curvy roads
    // the curve ego veto can incorrectly block this return because:
    //   1. Ego flow is unreliable on low-texture terrain (desert) — reports
    //      high confidence but near-zero velocity during real lateral motion
    //   2. The return happens during sustained curve mode, keeping thresholds
    //      elevated and vetoes active
    //
    // When a return is expected, the curve ego veto is bypassed for the
    // expected direction. This is safe because:
    //   - A confirmed LC establishes strong prior for the return
    //   - The lane detection itself (offset crossing threshold) is evidence
    //   - False positives from curves are typically in the SAME direction
    //     as the curve distortion, not alternating LEFT→RIGHT→LEFT
    pending_return_direction: Option<ShiftDirection>,
    pending_return_deadline_ms: f64, // timestamp after which expectation expires
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
            ego_shift_confident_frames: 0,
            offset_history: VecDeque::with_capacity(30),
            cached_measurement: None,
            cached_measurement_age: 0,
            ego_preempt_originated: false,
            ego_preempt_cooldown: 0,
            ego_cumulative_peak_px: 0.0,
            coherence_history: VecDeque::with_capacity(20),
            curve_sustained_frames: 0,
            in_curve_mode: false,
            frames_since_curve_mode: u32::MAX, // start with no recent curve
            shift_saw_curve_mode: false,
            shift_used_return_bypass: false,
            shift_confirmed_notified: false,
            pending_return_direction: None,
            pending_return_deadline_ms: 0.0,
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // v4.11/v4.13: CURVE STATE MANAGEMENT
    // ════════════════════════════════════════════════════════════════════

    /// Update curve detection state from measurement data.
    ///
    /// v4.13: Uses polynomial curvature as PRIMARY signal when available.
    /// Polynomial curvature is a direct geometric measurement from mask shapes —
    /// it works on a single frame with zero temporal lag. Boundary coherence
    /// (v4.11) is retained as FALLBACK when masks are too noisy for polynomial fit.
    fn update_curve_state(&mut self, coherence: f32, curvature: Option<&CurvatureEstimate>) {
        // ── v4.13: Polynomial curvature (primary signal) ──────────
        //
        // The curvature estimator fits x = ay² + by + c to each boundary's
        // mask spine. If both boundaries have similar `a` coefficients
        // (curvature_agreement > threshold), it's a road curve.
        //
        // This is INSTANT — no sustained-frame requirement. One good frame
        // with both boundaries visible is enough to confirm a curve.
        //
        let curvature_says_curve = curvature
            .map(|c| c.is_curve && c.confidence > 0.3)
            .unwrap_or(false);

        // ── v4.11: Boundary coherence (fallback signal) ──────────
        let coherence_says_curve = if coherence < -0.5 {
            None // no data
        } else {
            self.coherence_history.push_back(coherence);
            if self.coherence_history.len() > 15 {
                self.coherence_history.pop_front();
            }
            Some(coherence >= self.config.curve_coherence_threshold)
        };

        // ── Combined decision ────────────────────────────────────
        //
        // Curvature is authoritative when available (geometric, instant).
        // Coherence fills in when curvature can't compute (poor masks).
        // We still use the sustained-frame counter for coherence-only mode.
        //
        let frame_is_curve = if curvature.is_some() && curvature.unwrap().confidence > 0.2 {
            // Curvature estimator had enough data to form an opinion
            curvature_says_curve
        } else {
            // Fall back to coherence-based detection
            coherence_says_curve.unwrap_or(false)
        };

        if frame_is_curve {
            self.curve_sustained_frames += 1;
        } else if coherence_says_curve == Some(false) && !curvature_says_curve {
            // Both signals say not-curve — faster decay
            self.curve_sustained_frames = self.curve_sustained_frames.saturating_sub(2);
        } else {
            // Ambiguous (no data) — slow decay
            if self.curve_sustained_frames > 0 {
                self.curve_sustained_frames -= 1;
            }
        }

        let was_curve = self.in_curve_mode;

        // v4.14: Both poly curvature and coherence use the sustained-frame
        // gate. Instant activation caused chattering on straight roads where
        // a single noisy frame would activate curve mode, then deactivate
        // the next frame, keeping the curve VETO tainted.
        self.in_curve_mode = self.curve_sustained_frames >= self.config.curve_min_sustained_frames;

        // v4.12: Track how recently curve mode was active
        if self.in_curve_mode {
            self.frames_since_curve_mode = 0;
        } else {
            self.frames_since_curve_mode = self.frames_since_curve_mode.saturating_add(1);
        }

        if self.in_curve_mode && !was_curve {
            let source = if curvature_says_curve {
                "poly_curvature"
            } else {
                "coherence"
            };
            let curv_info = curvature
                .map(|c| {
                    format!(
                        "agree={:.2} mean_a={:.6} dir={}",
                        c.curvature_agreement,
                        c.mean_curvature,
                        c.curve_direction.as_str()
                    )
                })
                .unwrap_or_else(|| "N/A".to_string());
            info!(
                "🔄 Curve mode ACTIVATED [{}]: coherence={:.2} sustained={}f | curv=[{}] | thresholds ×{:.1} | ego ×{:.1}",
                source,
                coherence,
                self.curve_sustained_frames,
                curv_info,
                self.config.curve_shift_threshold_multiplier,
                self.config.curve_ego_velocity_multiplier,
            );
        } else if !self.in_curve_mode && was_curve {
            info!(
                "🔄 Curve mode DEACTIVATED: coherence={:.2} sustained={}f curvature_curve={}",
                coherence, self.curve_sustained_frames, curvature_says_curve,
            );
        }
    }

    /// v4.12: Whether curve state should suppress ego-motion-initiated shifts.
    /// True when in curve mode OR within the cooldown period after it.
    /// On curves, optical flow shows genuine lateral displacement (road rotates),
    /// but it's NOT a lane change — ego-start/preempt must be suppressed.
    fn curve_suppresses_ego(&self) -> bool {
        self.in_curve_mode || self.frames_since_curve_mode <= self.config.curve_ego_cooldown_frames
    }

    /// Effective shift_start_threshold accounting for curve mode.
    fn effective_shift_start_threshold(&self) -> f32 {
        if self.in_curve_mode {
            self.config.shift_start_threshold * self.config.curve_shift_threshold_multiplier
        } else {
            self.config.shift_start_threshold
        }
    }

    /// Effective shift_confirm_threshold accounting for curve mode.
    /// v4.13b: Uses base threshold (no curve multiplier) when the current
    /// shift direction matches the expected return direction.
    fn effective_shift_confirm_threshold(&self) -> f32 {
        // During return window, use base threshold for confirm too
        if self.is_in_return_window() {
            return self.config.shift_confirm_threshold;
        }
        if self.in_curve_mode {
            self.config.shift_confirm_threshold * self.config.curve_shift_threshold_multiplier
        } else {
            self.config.shift_confirm_threshold
        }
    }

    /// v4.13b: Whether the current shift is in the expected return direction
    /// and within the return time window.
    fn is_in_return_window(&self) -> bool {
        if let (Some(pending_dir), Some(shift_dir)) =
            (self.pending_return_direction, self.shift_direction)
        {
            // Note: We can't check timestamp here (no parameter), but the
            // pending_return_direction is cleared when expired or consumed,
            // so if it's set, we're still in the window.
            shift_dir == pending_dir
        } else {
            false
        }
    }

    /// Whether the detector is currently in curve suppression mode.
    pub fn in_curve_mode(&self) -> bool {
        self.in_curve_mode
    }

    /// Average boundary coherence over recent history (for diagnostics).
    pub fn avg_boundary_coherence(&self) -> f32 {
        if self.coherence_history.is_empty() {
            return -1.0;
        }
        let sum: f32 = self.coherence_history.iter().sum();
        sum / self.coherence_history.len() as f32
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
    ) -> LateralUpdateResult {
        let ego = ego_motion.unwrap_or_default();
        self.ego_last_velocity = ego.lateral_velocity;

        // v4.10: Decrement ego-preempt cooldown
        if self.ego_preempt_cooldown > 0 {
            self.ego_preempt_cooldown -= 1;
        }

        // v4.13b: Expire pending return expectation
        if self.pending_return_direction.is_some() && timestamp_ms > self.pending_return_deadline_ms
        {
            debug!("🔄 Return expectation expired (no return detected within window)");
            self.pending_return_direction = None;
        }

        // Track sustained ego motion for ego-start detection
        if ego.confidence > 0.3 && ego.lateral_velocity.abs() >= self.config.ego_motion_min_velocity
        {
            self.ego_active_frames += 1;
        } else {
            self.ego_active_frames = 0;
        }

        // ── v4.11/v4.13: UPDATE CURVE STATE ─────────────────────
        if let Some(ref m) = measurement {
            self.update_curve_state(m.boundary_coherence, m.curvature.as_ref());
        }

        // ── VALID LANE MEASUREMENT? ─────────────────────────────
        let valid_meas = measurement
            .filter(|m| m.confidence >= self.config.min_lane_confidence && m.lane_width_px > 50.0);

        if let Some(m) = &valid_meas {
            self.frames_without_lanes = 0;
            self.last_lane_width_px = m.lane_width_px;
            self.ego_bridge_frames = 0; // lanes back, bridge ends
                                        // v4.10: Update cache with fresh measurement
            self.cached_measurement = Some(m.clone());
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
                    boundary_coherence: cached.boundary_coherence, // v4.11: preserve from cache
                    curvature: cached.curvature.clone(), // v4.13: preserve curvature from cache
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
            let completed_shift = self.handle_no_lanes(ego, timestamp_ms, frame_id);
            return LateralUpdateResult {
                completed_shift,
                confirmed_in_progress: None,
            };
        }

        let meas = effective_meas.unwrap();

        // ── NORMALIZE ───────────────────────────────────────────
        let normalized = meas.lateral_offset_px / meas.lane_width_px;

        self.offset_history.push_back(normalized);
        if self.offset_history.len() > 20 {
            self.offset_history.pop_front();
        }

        // ── STATE MACHINE ───────────────────────────────────────
        let completed_shift = match self.state {
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
                    return LateralUpdateResult {
                        completed_shift: None,
                        confirmed_in_progress: None,
                    };
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

                // v4.11: Use effective thresholds that account for curve mode
                let mut eff_start = self.effective_shift_start_threshold();

                // v4.13b: If a return is expected and the deviation is in the
                // return direction, use base threshold (ignore curve multiplier).
                // The return after a confirmed LC is overwhelmingly likely to be
                // real, and the raised curve threshold blocks gradual returns.
                let dev_direction = if deviation < 0.0 {
                    ShiftDirection::Left
                } else {
                    ShiftDirection::Right
                };
                let in_return_window = self.pending_return_direction == Some(dev_direction)
                    && timestamp_ms <= self.pending_return_deadline_ms;

                if in_return_window && self.in_curve_mode {
                    // Use base threshold for the return direction — curve
                    // multiplier would block the gradual return.
                    eff_start = self.config.shift_start_threshold;
                }

                if abs_dev < eff_start {
                    // v4.13b: During return window, slow baseline adaptation by 4x
                    // so the return deviation can build up against a more stable
                    // baseline. Normal alpha chases the offset too aggressively
                    // for gradual returns on curves.
                    if in_return_window {
                        if frame_id % 4 == 0 {
                            self.update_baseline(normalized);
                        }
                    } else {
                        self.update_baseline(normalized);
                    }
                }

                if abs_dev >= eff_start {
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

                    if in_return_window {
                        info!(
                            "🔀 Lateral shift started: {} | dev={:.1}% | baseline={:.1}% | ego={:.2}px/f | curve_mode={} eff_thresh={:.1}% [RETURN WINDOW: base threshold]",
                            direction.as_str(),
                            abs_dev * 100.0,
                            self.baseline * 100.0,
                            ego.lateral_velocity,
                            self.in_curve_mode,
                            eff_start * 100.0,
                        );
                    } else {
                        info!(
                            "🔀 Lateral shift started: {} | dev={:.1}% | baseline={:.1}% | ego={:.2}px/f | curve_mode={} eff_thresh={:.1}%",
                            direction.as_str(),
                            abs_dev * 100.0,
                            self.baseline * 100.0,
                            ego.lateral_velocity,
                            self.in_curve_mode,
                            eff_start * 100.0,
                        );
                    }
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
                //
                // v4.12: On curves, optical flow shows genuine lateral displacement
                // (road surface rotates), but it's NOT a lane change. Multiply the
                // velocity threshold to prevent curve-induced ego-preempt triggers.
                else if self.ego_preempt_cooldown == 0
                    && self.config.ego_preempt_min_velocity > 0.0
                    && !self.curve_suppresses_ego()
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
                        eff_start * 100.0,
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
        };

        // v7.0: Check for early shift confirmation notification.
        // Only fires when: still in Shifting state (shift didn't complete this frame),
        // not already notified for this shift, peak meets confirm threshold,
        // min frames met, and (not on curve OR ego confirms lateral displacement).
        let confirmed_in_progress = if completed_shift.is_none()
            && self.state == State::Shifting
            && !self.shift_confirmed_notified
        {
            self.check_shift_confirmed(timestamp_ms, frame_id)
        } else {
            None
        };

        LateralUpdateResult {
            completed_shift,
            confirmed_in_progress,
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
                    self.shift_saw_curve_mode |= self.in_curve_mode; // v4.13b
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
                    // v4.13b: Track ego confidence during bridge frames.
                    if ego.confidence > 0.3 {
                        self.ego_shift_confident_frames += 1;
                    }

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
                // v4.12: Suppress if curve mode was recently active — lanes just
                // dropped out on a curve, ego flow is rotational, not a lane change.
                if has_ego
                    && !self.curve_suppresses_ego()
                    && self.ego_active_frames >= self.config.ego_shift_start_frames
                {
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

        // v4.13b: Track ego confidence during shift for veto reliability.
        if ego.confidence > 0.3 {
            self.ego_shift_confident_frames += 1;
        }
        if self.ego_cumulative_px.abs() > self.ego_cumulative_peak_px.abs() {
            self.ego_cumulative_peak_px = self.ego_cumulative_px;
        }

        // v4.13b: Latch curve mode — if curve activates at ANY point during
        // the shift, the flag stays true even if curve_mode deactivates later.
        self.shift_saw_curve_mode |= self.in_curve_mode;

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

        // v4.11: Use effective confirm threshold for all checks in this method
        let eff_confirm = self.effective_shift_confirm_threshold();

        if let Some(current_dir) = self.shift_direction {
            if self.shift_frames <= 30 && lane_direction != current_dir && abs_dev > eff_confirm {
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
            let confirmed = self.shift_peak_offset >= eff_confirm
                && self.shift_frames >= self.config.min_shift_frames;

            return if confirmed {
                self.emit_shift_event(timestamp_ms, frame_id)
            } else {
                debug!(
                    "❌ Lateral shift rejected: peak={:.1}% (need {:.1}%), frames={} (need {})",
                    self.shift_peak_offset * 100.0,
                    eff_confirm * 100.0,
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
            let confirmed = self.shift_peak_offset >= eff_confirm;

            return if confirmed {
                let evt = self.emit_shift_event(timestamp_ms, frame_id);
                // Update baseline whether event was emitted or vetoed —
                // the vehicle HAS settled in a new position regardless.
                self.baseline = normalized;
                self.baseline_samples = 1;
                evt
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

    /// v7.0: Check if the in-progress shift should fire an early "confirmed" notification.
    ///
    /// Fires ONCE per shift when:
    ///   1. Peak offset >= effective confirm threshold
    ///   2. Shift frames >= min_shift_frames
    ///   3. Either NOT curve-tainted, OR ego cumulative exceeds dynamic threshold
    ///
    /// This allows the classifier to emit an early LANE_CHANGE before the shift
    /// completes (returns to baseline), so the entry LC of an overtake fires in
    /// near-real-time instead of being delayed until the shift ends.
    fn check_shift_confirmed(
        &mut self,
        timestamp_ms: f64,
        frame_id: u64,
    ) -> Option<ShiftConfirmedNotification> {
        let eff_confirm = self.effective_shift_confirm_threshold();

        // 1. Peak must meet confirm threshold
        if self.shift_peak_offset < eff_confirm {
            return None;
        }

        // 2. Minimum frames must be met
        if self.shift_frames < self.config.min_shift_frames {
            return None;
        }

        // 3. On curves, require ego motion confirmation (same logic as emit_shift_event veto)
        if self.shift_saw_curve_mode && self.shift_source == ShiftSource::LaneBased {
            let duration_s = ((timestamp_ms - self.shift_start_ms) / 1000.0) as f32;
            let ego_threshold = 10.0_f32.max(duration_s * 15.0);

            let ego_confidence_ratio = if self.shift_frames > 0 {
                self.ego_shift_confident_frames as f32 / self.shift_frames as f32
            } else {
                0.0
            };
            let ego_trustworthy = ego_confidence_ratio >= 0.30;

            // Directional ego: must confirm the shift direction
            let shift_sign = match self.shift_direction {
                Some(ShiftDirection::Right) => 1.0f32,
                Some(ShiftDirection::Left) => -1.0f32,
                None => 1.0f32,
            };
            let directional_ego = self.ego_cumulative_peak_px * shift_sign;

            // Gate A: Ego trustworthy + insufficient confirming motion → suppress
            if ego_trustworthy && directional_ego < ego_threshold {
                return None; // Curve artifact — ego doesn't confirm
            }

            // Gate B: Direction disagreement — even with low ego confidence,
            // if ego has accumulated meaningful displacement OPPOSING the shift
            // direction, the lane-based direction is likely a perspective artifact.
            // This mirrors the direction-correction veto in emit_shift_event().
            // Threshold: 5px is well above noise, but catches clear opposing motion.
            let ego_opposing = self.ego_cumulative_px * shift_sign;
            if ego_opposing < -5.0 {
                debug!(
                    "🔔❌ Early LC suppressed: ego opposes shift on curve | ego_cum={:.1}px vs shift={} | ego_conf={:.0}%",
                    self.ego_cumulative_px,
                    self.shift_direction.unwrap_or(ShiftDirection::Left).as_str(),
                    ego_confidence_ratio * 100.0,
                );
                return None;
            }
        }

        // All gates passed — fire the notification
        self.shift_confirmed_notified = true;

        let direction = self.shift_direction.unwrap_or(ShiftDirection::Left);
        let avg_confidence = if self.shift_frames > 0 {
            self.shift_confidence_sum / self.shift_frames as f32
        } else {
            0.3
        };
        let confidence = self.compute_confidence(avg_confidence);

        info!(
            "🔔 Shift confirmed in-progress: {} | peak={:.1}% | frames={} | conf={:.2} | \
             curve={} | ego_cum={:.1}px | frame={}",
            direction.as_str(),
            self.shift_peak_offset * 100.0,
            self.shift_frames,
            confidence,
            self.shift_saw_curve_mode,
            self.ego_cumulative_px,
            frame_id,
        );

        Some(ShiftConfirmedNotification {
            direction,
            start_ms: self.shift_start_ms,
            start_frame: self.shift_start_frame,
            peak_offset: self.shift_peak_offset,
            confidence,
            curve_mode: self.shift_saw_curve_mode,
            confirmed_frame: frame_id,
            confirmed_ms: timestamp_ms,
            geometric_signals: None, // Populated by pipeline
        })
    }

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
        self.ego_cumulative_peak_px = 0.0;
        self.ego_shift_confident_frames = 0;
        self.ego_bridge_frames = 0;
        self.ego_estimated_offset = 0.0;
        // v4.13b: Latch curve mode at shift start. Will also be latched
        // if curve_mode activates later during the shift.
        self.shift_saw_curve_mode = self.in_curve_mode;
        self.shift_used_return_bypass = false;
    }

    /// Settle an ego-bridged shift when ego velocity drops (vehicle stopped moving laterally).
    fn settle_shift_ego(&mut self, timestamp_ms: f64, frame_id: u64) -> Option<LateralShiftEvent> {
        // v4.11: Use effective confirm threshold
        let eff_confirm = self.effective_shift_confirm_threshold();
        let confirmed = self.shift_peak_offset >= eff_confirm
            && self.shift_frames >= self.config.min_shift_frames;

        if confirmed {
            info!(
                "✅ Lateral shift settled via ego-motion: peak_est={:.1}% | ego_cum={:.1}px | dur={:.1}s",
                self.shift_peak_offset * 100.0,
                self.ego_cumulative_px,
                (timestamp_ms - self.shift_start_ms) / 1000.0,
            );
            let evt = self.emit_shift_event(timestamp_ms, frame_id);
            if evt.is_some() {
                // Only transition to Recovering if the event was actually emitted
                // (not vetoed by ego-direction disagreement).
                self.baseline_samples = 0;
                self.freeze_remaining = self.config.post_reset_freeze_frames;
                self.state = State::Recovering;
            }
            return evt;
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
        // v4.11: Use effective confirm threshold
        let confirmed = self.shift_peak_offset >= self.effective_shift_confirm_threshold();

        let result = if confirmed {
            // emit_shift_event handles state reset internally
            self.emit_shift_event(timestamp_ms, frame_id)
        } else {
            self.state = State::Stable;
            self.reset_shift();
            None
        };

        // Update baseline to current position regardless of outcome
        self.baseline = current_normalized;
        self.baseline_samples = 1;
        result
    }

    /// Build and emit the shift event, applying direction validation.
    /// Returns None if the shift was vetoed by ego-direction disagreement.
    fn emit_shift_event(&mut self, end_ms: f64, end_frame: u64) -> Option<LateralShiftEvent> {
        let duration_ms = end_ms - self.shift_start_ms;

        // ══════════════════════════════════════════════════════════
        // v4.13 FIX (Bug 2): Curve-mode ego cross-validation gate.
        //
        // On sharp curves, perspective distortion can produce arbitrarily
        // large normalized offsets (162.9% seen in production). No scalar
        // multiplier on shift thresholds can catch this because the
        // distortion scales with curvature, not with a fixed factor.
        //
        // Gate: If curve_mode was active during this shift AND the shift
        // was LaneBased (not ego-started), require minimum ego cumulative
        // displacement to confirm the vehicle actually moved laterally.
        // If the car didn't move (ego_cum ≈ 0), the "shift" is pure
        // perspective artifact from the curve.
        //
        // Threshold: Duration-scaled ego displacement.
        //
        // A flat 10px threshold is too low for longer shifts — on curves,
        // optical flow accumulates at ~5-15px/s from road rotation alone.
        // A real lane change produces ≥15px/s of true lateral displacement
        // regardless of duration (3.5m lateral / ~3s ≈ 50-150px total).
        //
        // Formula: max(10.0, duration_s × 15.0)
        //   0.5s → 10px (floor)     |  LC2 prod: 1.1px / 1.1s → 16.5px min → VETOED
        //   2.5s → 37.5px           |  LC1 prod: 31.5px / 2.5s → 37.5px min → VETOED
        //   3.0s → 45px             |  LC3 prod: 116.8px / 3.0s → 45px min → PASSES
        //
        // v4.13b: Use shift_saw_curve_mode (latched during shift lifetime)
        // instead of only in_curve_mode at emit time. Curve mode chatters
        // on curvy roads — shifts that start/complete during brief off-gaps
        // escaped the gate. Latching catches them.
        // ══════════════════════════════════════════════════════════
        const CURVE_EGO_MIN_FLOOR_PX: f32 = 10.0;
        const CURVE_EGO_MIN_RATE_PX_PER_S: f32 = 15.0;
        // v4.13b: Minimum ratio of confident ego frames to trust the veto.
        // If the ego flow estimator had poor confidence during most of the shift
        // (low-texture terrain like desert), "ego says zero" is unreliable —
        // it means the estimator failed, not that the vehicle didn't move.
        const CURVE_EGO_MIN_CONFIDENCE_RATIO: f32 = 0.30;

        let duration_s = (duration_ms / 1000.0) as f32;
        let curve_ego_threshold =
            CURVE_EGO_MIN_FLOOR_PX.max(duration_s * CURVE_EGO_MIN_RATE_PX_PER_S);

        let curve_tainted = self.in_curve_mode || self.shift_saw_curve_mode;

        let ego_confidence_ratio = if self.shift_frames > 0 {
            self.ego_shift_confident_frames as f32 / self.shift_frames as f32
        } else {
            0.0
        };
        let ego_trustworthy = ego_confidence_ratio >= CURVE_EGO_MIN_CONFIDENCE_RATIO;

        // v6.1: Use DIRECTIONAL ego displacement for curve veto.
        //
        // On curves, rotational flow accumulates in one direction. A shift
        // RIGHT with ego moving LEFT means ego OPPOSES the shift — this is
        // stronger evidence of perspective distortion, not weaker.
        //
        // directional_ego > 0: ego confirms shift direction
        // directional_ego <= 0: ego opposes → always veto
        // directional_ego < threshold: insufficient confirming motion → veto
        let shift_sign = match self.shift_direction {
            Some(ShiftDirection::Right) => 1.0f32,
            Some(ShiftDirection::Left) => -1.0f32,
            None => 1.0f32,
        };
        let directional_ego = self.ego_cumulative_peak_px * shift_sign;

        // v4.13b: Check if this shift matches the expected return direction.
        // After a confirmed LC, the return is overwhelmingly likely to be real,
        // not a curve artifact. Bypass the veto for expected returns.
        let is_expected_return = if let (Some(pending_dir), Some(shift_dir)) =
            (self.pending_return_direction, self.shift_direction)
        {
            shift_dir == pending_dir && end_ms <= self.pending_return_deadline_ms
        } else {
            false
        };

        // IMMEDIATELY consume the return expectation on first direction match.
        // This is the critical one-shot guard: without it, every subsequent
        // shift in the return direction also bypasses the veto, creating an
        // infinite L→R→L→R cascade of false positives.
        if is_expected_return {
            self.pending_return_direction = None;
            self.shift_used_return_bypass = true;
        }

        if curve_tainted
            && self.shift_source == ShiftSource::LaneBased
            && ego_trustworthy
            && directional_ego < curve_ego_threshold
            && !is_expected_return
        {
            warn!(
                "❌ Curve ego cross-validation VETO: LaneBased shift {} rejected | \
                 peak={:.1}% | ego_cum={:.1}px (peak={:.1}px, dir={:.1}px) < {:.0}px min (floor={:.0} + {:.1}s×{:.0}) | \
                 dur={:.1}s | curve_now={} curve_saw={} | ego_conf={}/{} ({:.0}%) → perspective distortion, not lane change",
                self.shift_direction
                    .unwrap_or(ShiftDirection::Left)
                    .as_str(),
                self.shift_peak_offset * 100.0,
                self.ego_cumulative_px,
                self.ego_cumulative_peak_px,
                directional_ego,
                curve_ego_threshold,
                CURVE_EGO_MIN_FLOOR_PX,
                duration_s,
                CURVE_EGO_MIN_RATE_PX_PER_S,
                duration_ms / 1000.0,
                self.in_curve_mode,
                self.shift_saw_curve_mode,
                self.ego_shift_confident_frames,
                self.shift_frames,
                ego_confidence_ratio * 100.0,
            );
            self.state = State::Stable;
            self.reset_shift();
            return None;
        }

        // v4.13b: Diagnostic logs for when veto WOULD have fired but was bypassed.
        if curve_tainted
            && self.shift_source == ShiftSource::LaneBased
            && directional_ego < curve_ego_threshold
        {
            if is_expected_return {
                warn!(
                    "⚠️ Curve veto BYPASSED (expected return): shift {} | \
                     peak={:.1}% | ego_cum={:.1}px (peak={:.1}px, dir={:.1}px) < {:.0}px threshold | \
                     return expected after prior LC → allowing (one-shot, consumed)",
                    self.shift_direction
                        .unwrap_or(ShiftDirection::Left)
                        .as_str(),
                    self.shift_peak_offset * 100.0,
                    self.ego_cumulative_px,
                    self.ego_cumulative_peak_px,
                    directional_ego,
                    curve_ego_threshold,
                );
            } else if !ego_trustworthy {
                warn!(
                    "⚠️ Curve veto BYPASSED (ego untrustworthy): shift {} | \
                     peak={:.1}% | ego_cum={:.1}px (peak={:.1}px, dir={:.1}px) < {:.0}px threshold | \
                     ego_conf={}/{} ({:.0}%) < {:.0}% min → allowing despite curve",
                    self.shift_direction
                        .unwrap_or(ShiftDirection::Left)
                        .as_str(),
                    self.shift_peak_offset * 100.0,
                    self.ego_cumulative_px,
                    self.ego_cumulative_peak_px,
                    directional_ego,
                    curve_ego_threshold,
                    self.ego_shift_confident_frames,
                    self.shift_frames,
                    ego_confidence_ratio * 100.0,
                    CURVE_EGO_MIN_CONFIDENCE_RATIO * 100.0,
                );
            }
        }

        // ── FINAL DIRECTION VALIDATION (v4.4, v4.10b) ────────────
        // Ego cumulative displacement is the ground truth for direction.
        // If ego clearly moved one way, trust that over initial lane reading.
        // v4.10b: Can return None to veto short shifts where ego disagrees.
        let validated_direction = match self.validate_final_direction(duration_ms) {
            Some(dir) => dir,
            None => {
                // Vetoed — ego disagrees with lane for a short shift.
                // Reset and return to stable without emitting an event.
                self.state = State::Stable;
                self.reset_shift();
                return None;
            }
        };

        // v6.1: Direction-correction veto for curve-tainted shifts.
        //
        // When lane says direction A but ego says direction B during a curve,
        // the lane-detected shift in direction A is perspective distortion.
        // The ego motion in direction B may be real, but it should be detected
        // as its OWN shift starting natively in direction B — not as a
        // direction-corrected version of the A-direction shift.
        //
        // Evidence from production:
        //   F574: RIGHT→LEFT corrected, peak=266.8% (RIGHT measurement, meaningless for LEFT)
        //   F609: RIGHT→LEFT corrected, peak=635.1% (same issue)
        //   F701: Native LEFT, peak=152.4%, ego=-114.4px → real lane change
        //
        // Without this veto, each corrected shift triggers overtake classification
        // and creates duplicate LC events. The real motion always gets detected
        // natively (F701), so the corrected shifts are pure duplicates.
        let initial_dir = self.shift_direction.unwrap_or(ShiftDirection::Left);
        if curve_tainted && validated_direction != initial_dir {
            warn!(
                "❌ Curve direction-correction VETO: shift {} corrected to {} during curve | \
                 peak={:.1}% | ego_cum={:.1}px | dur={:.1}s → perspective distortion + real motion \
                 in opposite direction; will be detected natively",
                initial_dir.as_str(),
                validated_direction.as_str(),
                self.shift_peak_offset * 100.0,
                self.ego_cumulative_px,
                duration_ms / 1000.0,
            );
            self.state = State::Stable;
            self.reset_shift();
            return None;
        }

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

        let source_label = match self.shift_source {
            ShiftSource::LaneBased => "LaneBased",
            ShiftSource::EgoBridged => "EgoBridged",
            ShiftSource::EgoStarted => "EgoStarted",
        };

        let evt = LateralShiftEvent {
            direction: validated_direction,
            peak_offset: self.shift_peak_offset,
            start_ms: self.shift_start_ms,
            end_ms,
            start_frame: self.shift_start_frame,
            end_frame: end_frame,
            duration_ms,
            confidence,
            confirmed: true,
            curve_mode: self.in_curve_mode || self.shift_saw_curve_mode,
            ego_cumulative_peak_px: self.ego_cumulative_peak_px,
            source_label,
            geometric_signals: None, // v7.0: Populated by pipeline
        };

        info!(
            "✅ Lateral shift completed: {} | peak={:.1}% | dur={:.1}s | conf={:.2} | \
             source={:?} | ego_cum={:.1}px (peak={:.1}px) | lane_frames={}/{} | curve_mode={} | ego_conf={}/{} ({:.0}%)",
            evt.direction.as_str(),
            evt.peak_offset * 100.0,
            evt.duration_ms / 1000.0,
            evt.confidence,
            self.shift_source,
            self.ego_cumulative_px,
            self.ego_cumulative_peak_px,
            self.shift_lane_frames,
            self.shift_frames,
            self.in_curve_mode,
            self.ego_shift_confident_frames,
            self.shift_frames,
            if self.shift_frames > 0 {
                self.ego_shift_confident_frames as f32 / self.shift_frames as f32 * 100.0
            } else {
                0.0
            },
        );

        // v4.13b: Set return expectation — after a confirmed lane change,
        // expect the opposite direction within 30s. This allows the curve
        // veto to be bypassed for the return on roads where ego flow is
        // unreliable (desert, low-texture terrain).
        //
        // CRITICAL: Only set from shifts that were NOT themselves return-
        // window bypasses. Otherwise each return sets another expectation,
        // creating an infinite L→R→L→R chain of false positives.
        // A return-bypassed shift has is_expected_return=true (computed above).
        if !is_expected_return && !self.shift_used_return_bypass {
            const RETURN_WINDOW_MS: f64 = 30_000.0;
            let return_dir = match evt.direction {
                ShiftDirection::Left => ShiftDirection::Right,
                ShiftDirection::Right => ShiftDirection::Left,
            };
            self.pending_return_direction = Some(return_dir);
            self.pending_return_deadline_ms = end_ms + RETURN_WINDOW_MS;
            debug!(
                "🔄 Return expected: {} within {:.0}s (deadline {:.1}s)",
                return_dir.as_str(),
                RETURN_WINDOW_MS / 1000.0,
                self.pending_return_deadline_ms / 1000.0,
            );
        }

        self.state = State::Stable;
        self.reset_shift();
        Some(evt)
    }

    /// Validate shift direction using cumulative ego motion.
    /// Returns the validated direction, or None if the shift should be vetoed
    /// (ego strongly disagrees with lane direction for a short shift).
    ///
    /// v4.10: Uses peak cumulative (max absolute value during shift) instead of
    /// final cumulative, because by shift end the vehicle has returned to its lane
    /// and the cumulative has swung back toward zero or even the opposite direction.
    ///
    /// v4.10b: Duration-scaled override threshold. Short shifts (< 1.5s) need
    /// far less ego evidence to override lane direction, because lane transients
    /// are common (detector jumps, boundary misidentification) but ego motion
    /// doesn't lie about which direction the vehicle moved.
    fn validate_final_direction(&self, duration_ms: f64) -> Option<ShiftDirection> {
        let initial_dir = self.shift_direction.unwrap_or(ShiftDirection::Left);

        // Use peak cumulative — this captures the direction when ego motion
        // was strongest (during the actual lane change), not after return.
        let cum = if self.ego_cumulative_peak_px.abs() > self.ego_cumulative_px.abs() {
            self.ego_cumulative_peak_px
        } else {
            self.ego_cumulative_px
        };

        // ── v4.10b: Duration-scaled ego threshold ────────────────
        // For short shifts, lane data alone is unreliable (could be a
        // detector transient). Lower the ego override bar proportionally.
        //   < 1.0s: 3px  (very easy to override — short shifts are suspect)
        //   1.0-2.0s: 5-10px (moderate)
        //   > 3.0s: 15px (high bar — long shifts with consistent lane data are trustworthy)
        let duration_s = (duration_ms / 1000.0) as f32;
        let ego_override_threshold = if duration_s < 1.0 {
            3.0
        } else if duration_s < 2.0 {
            3.0 + (duration_s - 1.0) * 7.0 // 3→10 linearly
        } else if duration_s < 3.0 {
            10.0 + (duration_s - 2.0) * 5.0 // 10→15 linearly
        } else {
            15.0
        };

        // If we don't have enough ego data, trust the lane-based direction
        if cum.abs() < ego_override_threshold {
            // ── v4.10b: Ego disagreement veto for very short shifts ──
            // Even if ego cumulative is below the override threshold,
            // if it's in the OPPOSITE direction for a very short shift,
            // the shift is almost certainly a lane detector transient.
            // Veto it entirely rather than emitting a wrong direction.
            if duration_s < 1.5 && cum.abs() > 1.5 {
                let ego_dir = if cum < 0.0 {
                    ShiftDirection::Left
                } else {
                    ShiftDirection::Right
                };
                if ego_dir != initial_dir {
                    warn!(
                        "❌ Ego-direction veto: shift {} dur={:.1}s but ego_cum={:.1}px says {} \
                         (below override threshold {:.0}px but disagrees) → REJECTING shift",
                        initial_dir.as_str(),
                        duration_s,
                        cum,
                        ego_dir.as_str(),
                        ego_override_threshold,
                    );
                    return None; // Veto — don't emit this shift
                }
            }
            return Some(initial_dir);
        }

        let ego_dir = if cum < 0.0 {
            ShiftDirection::Left
        } else {
            ShiftDirection::Right
        };

        if ego_dir != initial_dir {
            warn!(
                "🔄 Direction validation: initial={} ego={} (peak_cum={:.1}px, final_cum={:.1}px, \
                 threshold={:.0}px, dur={:.1}s) → using {}",
                initial_dir.as_str(),
                ego_dir.as_str(),
                self.ego_cumulative_peak_px,
                self.ego_cumulative_px,
                ego_override_threshold,
                duration_s,
                ego_dir.as_str(),
            );
            Some(ego_dir)
        } else {
            Some(initial_dir)
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
            State::Stable => {
                if self.in_curve_mode {
                    "STABLE(curve)"
                } else {
                    "STABLE"
                }
            }
            State::Shifting => match self.shift_source {
                ShiftSource::LaneBased => {
                    if self.in_curve_mode {
                        "SHIFTING(curve)"
                    } else {
                        "SHIFTING"
                    }
                }
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
        self.coherence_history.clear();
        self.curve_sustained_frames = 0;
        self.in_curve_mode = false;
        self.frames_since_curve_mode = u32::MAX;
        self.reset_shift();
    }

    // ════════════════════════════════════════════════════════════════════
    // PRIVATE HELPERS
    // ════════════════════════════════════════════════════════════════════

    /// v4.11 (Fix 2): Adaptive baseline alpha.
    ///
    /// Default alpha (baseline_alpha_stable) is intentionally slow to avoid
    /// chasing noise. But on curves, the normalized offset drifts smoothly
    /// and consistently — the baseline must track it or deviation accumulates
    /// and triggers false lane-change detections.
    ///
    /// Detection of smooth drift:
    ///   1. Compute variance of recent offset_history (last 12 frames)
    ///   2. Compute drift rate (average per-frame delta from first differences)
    ///   3. If variance < max AND drift > min → curve-like drift → boost alpha
    ///   4. Interpolate alpha between base and adaptive_max
    fn update_baseline(&mut self, normalized: f32) {
        let base_alpha = if self.state == State::Recovering {
            self.config.baseline_alpha_recovery
        } else {
            self.config.baseline_alpha_stable
        };

        // ── Compute adaptive boost (only in Stable state) ───────
        let alpha = if self.state == State::Stable && self.offset_history.len() >= 12 {
            let recent: Vec<f32> = self.offset_history.iter().rev().take(12).copied().collect();

            // Variance of recent offsets
            let mean = recent.iter().sum::<f32>() / recent.len() as f32;
            let variance =
                recent.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / recent.len() as f32;

            // Drift rate: average absolute first-difference
            let deltas: Vec<f32> = recent.windows(2).map(|w| (w[0] - w[1]).abs()).collect();
            let avg_drift = if !deltas.is_empty() {
                deltas.iter().sum::<f32>() / deltas.len() as f32
            } else {
                0.0
            };

            // Drift direction consistency: what fraction of deltas have the same sign?
            // (smooth curve drift is monotonic; noise flips sign constantly)
            let signed_deltas: Vec<f32> = recent.windows(2).map(|w| w[0] - w[1]).collect();
            let positive_count = signed_deltas.iter().filter(|d| **d > 0.0).count();
            let negative_count = signed_deltas.iter().filter(|d| **d < 0.0).count();
            let direction_consistency =
                positive_count.max(negative_count) as f32 / signed_deltas.len().max(1) as f32;

            if variance < self.config.adaptive_baseline_max_variance
                && avg_drift > self.config.adaptive_baseline_min_drift
                && direction_consistency > 0.6
            {
                // Smooth, consistent drift detected — boost alpha
                let drift_factor = ((avg_drift - self.config.adaptive_baseline_min_drift)
                    / (self.config.adaptive_baseline_min_drift * 5.0))
                    .min(1.0);
                let variance_factor =
                    (1.0 - variance / self.config.adaptive_baseline_max_variance).max(0.0);
                let consistency_factor = ((direction_consistency - 0.6) / 0.4).min(1.0);

                let boost = drift_factor * variance_factor * consistency_factor;
                let adaptive_alpha =
                    base_alpha + boost * (self.config.adaptive_baseline_alpha_max - base_alpha);

                if boost > 0.3 {
                    debug!(
                        "📈 Adaptive baseline: alpha={:.4} (boost={:.2}) | var={:.5} drift={:.4} consistency={:.2}",
                        adaptive_alpha, boost, variance, avg_drift, direction_consistency,
                    );
                }

                adaptive_alpha
            } else {
                base_alpha
            }
        } else {
            base_alpha
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
        self.shift_saw_curve_mode = false;
        self.shift_used_return_bypass = false;
        self.shift_confirmed_notified = false;
        self.ego_shift_confident_frames = 0;
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

    fn default_mining_config() -> LateralDetectorConfig {
        LateralDetectorConfig {
            min_lane_confidence: 0.20,
            shift_start_threshold: 0.35,
            shift_confirm_threshold: 0.50,
            shift_end_threshold: 0.20,
            min_shift_frames: 15,
            baseline_alpha_stable: 0.002,
            baseline_warmup_frames: 25,
            // v4.11 mining curve tuning
            curve_coherence_threshold: 0.60,
            curve_shift_threshold_multiplier: 2.0,
            curve_min_sustained_frames: 4,
            adaptive_baseline_alpha_max: 0.05,
            adaptive_baseline_max_variance: 0.002,
            adaptive_baseline_min_drift: 0.0015,
            ..LateralDetectorConfig::default()
        }
    }

    fn make_measurement(offset_px: f32, lane_width: f32, coherence: f32) -> LaneMeasurement {
        LaneMeasurement {
            lateral_offset_px: offset_px,
            lane_width_px: lane_width,
            confidence: 0.8,
            both_lanes: true,
            boundary_coherence: coherence,
            curvature: None, // tests use coherence directly
        }
    }

    /// On a curve with high boundary coherence, a moderate offset swing
    /// (e.g. -43.8% of lane width over 0.6s) should NOT trigger a shift
    /// because thresholds are raised by the curve multiplier.
    #[test]
    fn test_curve_suppresses_false_shift() {
        let config = default_mining_config();
        let mut det = LateralShiftDetector::new(config);

        let lane_w = 467.0;

        // Warmup: establish baseline at ~0% offset
        for i in 0..30 {
            let m = make_measurement(0.0, lane_w, -1.0);
            det.update(Some(m), None, i as f64 * 33.3, i);
        }
        assert_eq!(det.state, State::Stable);

        // Simulate curve: high coherence sustained for several frames
        // while offset drifts to -43.8%
        let peak_offset_px = -204.4; // -43.8% of 467px
        let num_frames = 18; // ~0.6s at 30fps
        for i in 0..num_frames {
            let t = i as f32 / num_frames as f32;
            let offset_px = t * peak_offset_px;
            // High coherence throughout (boundaries co-moving)
            let m = make_measurement(offset_px, lane_w, 0.85);
            let frame = 30 + i as u64;
            let result = det.update(Some(m), None, frame as f64 * 33.3, frame);
            // Should NOT produce a shift event
            assert!(
                result.completed_shift.is_none(),
                "Frame {}: unexpected shift event during curve! offset={:.1}%",
                frame,
                (offset_px / lane_w) * 100.0,
            );
        }

        // Verify curve mode was activated
        assert!(det.in_curve_mode, "Curve mode should be active");
        // With mining config: base threshold 35% × 2.0 = 70%.
        // Peak was 43.8% which is below 70%, so no shift should have started.
        assert_eq!(
            det.state,
            State::Stable,
            "Should remain Stable, not Shifting"
        );
    }

    /// A real lane change (2-3s, boundaries diverging = low/negative coherence)
    /// should still be detected even when curve suppression is available.
    #[test]
    fn test_real_lane_change_not_suppressed() {
        let config = default_mining_config();
        let mut det = LateralShiftDetector::new(config);
        let lane_w = 467.0;

        // Warmup
        for i in 0..30 {
            let m = make_measurement(0.0, lane_w, -1.0);
            det.update(Some(m), None, i as f64 * 33.3, i);
        }

        // Real lane change: LOW coherence (boundaries diverging),
        // offset grows to 60% over 2.5 seconds
        let peak_pct = 0.60;
        let change_frames = 75; // 2.5s
        let mut shift_started = false;
        for i in 0..change_frames {
            let t = (i as f32 / change_frames as f32).min(1.0);
            let offset_px = t * peak_pct * lane_w;
            // Low/negative coherence — boundaries diverging
            let coherence = -0.3 + (0.2 * (i as f32 / 10.0).sin()); // oscillates around -0.3
            let m = make_measurement(offset_px, lane_w, coherence);
            let frame = 30 + i as u64;
            det.update(Some(m), None, frame as f64 * 33.3, frame);
            if det.state == State::Shifting {
                shift_started = true;
            }
        }

        assert!(
            shift_started,
            "Real lane change should have entered Shifting state"
        );
        assert!(
            !det.in_curve_mode,
            "Curve mode should NOT be active during a real lane change"
        );
    }

    /// Adaptive baseline should track smooth drift, preventing deviation
    /// buildup that would trigger a false shift.
    #[test]
    fn test_adaptive_baseline_tracks_smooth_drift() {
        let config = default_mining_config();
        let mut det = LateralShiftDetector::new(config);
        let lane_w = 500.0;

        // Warmup
        for i in 0..30 {
            let m = make_measurement(0.0, lane_w, -1.0);
            det.update(Some(m), None, i as f64 * 33.3, i);
        }
        let baseline_after_warmup = det.baseline;

        // Smooth drift: offset increases by ~0.15% per frame for 100 frames
        // Total drift = 15% of lane width. Without adaptive alpha, baseline
        // barely moves (alpha=0.002). With adaptive alpha, it should track.
        let drift_rate_px = 0.75; // px/frame → 0.15% of 500px lane
        for i in 0..100 {
            let offset_px = drift_rate_px * i as f32;
            // Moderate coherence (boundaries co-moving during curve)
            let m = make_measurement(offset_px, lane_w, 0.5);
            let frame = 30 + i as u64;
            let result = det.update(Some(m), None, frame as f64 * 33.3, frame);
            assert!(
                result.completed_shift.is_none(),
                "Frame {}: smooth drift should not produce shift event",
                frame
            );
        }

        // After 100 frames of drift, the normalized offset is 0.15 (15%).
        // With base alpha only (0.002), baseline would barely move from 0.
        // With adaptive alpha, baseline should have tracked significantly.
        let final_offset_norm = (drift_rate_px * 100.0) / lane_w;
        let baseline_drift = det.baseline - baseline_after_warmup;

        // Baseline should have tracked at least 40% of the actual drift
        assert!(
            baseline_drift > final_offset_norm * 0.4,
            "Adaptive baseline should track smooth drift: baseline_drift={:.4} vs expected>{:.4}",
            baseline_drift,
            final_offset_norm * 0.4,
        );

        // And the deviation from baseline should be small (< shift_start)
        let current_normalized = (drift_rate_px * 100.0) / lane_w;
        let deviation = (current_normalized - det.baseline).abs();
        assert!(
            deviation < det.effective_shift_start_threshold(),
            "Deviation ({:.3}) should be below shift threshold ({:.3}) thanks to adaptive baseline",
            deviation,
            det.effective_shift_start_threshold(),
        );
    }

    /// v4.13: On a sharp curve, perspective distortion can produce offsets
    /// exceeding even the raised curve threshold (e.g. 162% peak). If ego
    /// cumulative displacement is near zero, the shift must be vetoed as a
    /// perspective artifact, not emitted as a lane change.
    ///
    /// Reproduces production Bug 2:
    ///   curve_mode=true, source=LaneBased, ego_cum=-3.3px, peak=162.9%
    #[test]
    fn test_curve_ego_crossvalidation_veto() {
        let config = default_mining_config();
        let mut det = LateralShiftDetector::new(config);
        let lane_w = 248.0; // From production bug: narrow lane

        // Warmup: establish baseline
        for i in 0..30 {
            let m = make_measurement(0.0, lane_w, -1.0);
            det.update(Some(m), None, i as f64 * 33.3, i);
        }
        assert_eq!(det.state, State::Stable);

        // Phase 1: Activate curve mode with sustained high coherence
        for i in 0..8 {
            let m = make_measurement(0.0, lane_w, 0.85);
            let frame = 30 + i as u64;
            det.update(Some(m), None, frame as f64 * 33.3, frame);
        }
        assert!(det.in_curve_mode, "Curve mode should be active");

        // Phase 2: Sharp curve drives offset to 160%+ (beyond raised threshold)
        // with zero ego motion — pure perspective distortion.
        // Mining effective threshold = 35% × 2.0 = 70%.
        // We ramp through 70% (shift starts) up to 160%+.
        let ramp_frames = 30; // 1.0s
        for i in 0..ramp_frames {
            let t = i as f32 / ramp_frames as f32;
            let offset_px = t * 1.63 * lane_w; // ramp to ~163%
            let m = make_measurement(offset_px, lane_w, 0.80);
            // Very small ego motion (matching production: -0.30 px/frame)
            let ego = EgoMotionInput {
                lateral_velocity: -0.30,
                confidence: 0.5,
            };
            let frame = 38 + i as u64;
            let result = det.update(Some(m), Some(ego), frame as f64 * 33.3, frame);
            // Shift should NOT be emitted yet (still ramping or still active)
            // But it might start the shift state
            assert!(
                result.completed_shift.is_none(),
                "Frame {}: unexpected shift emission during ramp",
                frame
            );
        }

        // Phase 3: Offset returns toward baseline — shift should try to complete.
        // The ego cross-validation should VETO it since ego_cumulative_peak is tiny.
        let return_frames = 25;
        let mut any_shift_emitted = false;
        for i in 0..return_frames {
            let t = i as f32 / return_frames as f32;
            let offset_px = (1.0 - t) * 1.63 * lane_w; // returning to 0
            let m = make_measurement(offset_px, lane_w, 0.70);
            let ego = EgoMotionInput {
                lateral_velocity: -0.10,
                confidence: 0.4,
            };
            let frame = 68 + i as u64;
            if det
                .update(Some(m), Some(ego), frame as f64 * 33.3, frame)
                .completed_shift
                .is_some()
            {
                any_shift_emitted = true;
            }
        }

        assert!(
            !any_shift_emitted,
            "Curve perspective distortion with no ego motion should be VETOED, not emitted"
        );
    }

    /// v4.13b: Curve mode chatters on curvy roads (activating for a few frames,
    /// deactivating for a few). A shift that starts during a brief off-gap and
    /// completes while curve is off again must STILL be vetoed if curve was
    /// active at any point during the shift (shift_saw_curve_mode).
    ///
    /// Reproduces production false positives LC1 & LC2:
    ///   LC2: peak=58.3%, dur=1.1s, ego_cum=-1.1px, curve_mode=false (at emit)
    ///   LC1: peak=99.0%, dur=2.5s, ego_cum=-31.5px, curve_mode=false (at emit)
    #[test]
    fn test_curve_chatter_veto_via_saw_curve_mode() {
        let config = default_mining_config();
        let mut det = LateralShiftDetector::new(config);
        let lane_w = 400.0;

        // Warmup: establish baseline
        for i in 0..30 {
            let m = make_measurement(0.0, lane_w, -1.0);
            det.update(Some(m), None, i as f64 * 33.3, i);
        }
        assert_eq!(det.state, State::Stable);

        // Phase 1: Brief curve mode activation, then deactivation (chatter)
        for i in 0..6 {
            let m = make_measurement(0.0, lane_w, 0.85);
            let frame = 30 + i as u64;
            det.update(Some(m), None, frame as f64 * 33.3, frame);
        }
        assert!(det.in_curve_mode, "Curve mode should activate");

        // Deactivate curve mode (low coherence)
        for i in 0..4 {
            let m = make_measurement(0.0, lane_w, 0.1);
            let frame = 36 + i as u64;
            det.update(Some(m), None, frame as f64 * 33.3, frame);
        }
        assert!(!det.in_curve_mode, "Curve mode should have deactivated");

        // Phase 2: Shift starts while curve_mode=false (the gap).
        // Offset ramps to 60% — above base 35% threshold but below raised 70%.
        let ramp_frames = 20;
        for i in 0..ramp_frames {
            let t = i as f32 / ramp_frames as f32;
            let offset_px = t * 0.60 * lane_w;
            // Low coherence — curve mode stays off
            let m = make_measurement(offset_px, lane_w, 0.2);
            let ego = EgoMotionInput {
                lateral_velocity: -0.15, // tiny, curve-induced
                confidence: 0.4,
            };
            let frame = 40 + i as u64;
            det.update(Some(m), Some(ego), frame as f64 * 33.3, frame);
        }

        // Phase 3: Curve mode re-activates briefly during the shift
        for i in 0..5 {
            let offset_px = 0.55 * lane_w; // holding near peak
            let m = make_measurement(offset_px, lane_w, 0.85);
            let ego = EgoMotionInput {
                lateral_velocity: -0.10,
                confidence: 0.5, // ego working but measuring near-zero → curve artifact
            };
            let frame = 60 + i as u64;
            det.update(Some(m), Some(ego), frame as f64 * 33.3, frame);
        }
        // shift_saw_curve_mode should now be latched true

        // Phase 4: Curve mode off again, offset returns to baseline.
        // Shift should try to complete but be vetoed by saw_curve_mode + low ego.
        let return_frames = 25;
        let mut any_shift_emitted = false;
        for i in 0..return_frames {
            let t = i as f32 / return_frames as f32;
            let offset_px = (1.0 - t) * 0.55 * lane_w;
            let m = make_measurement(offset_px, lane_w, 0.1); // curve mode off
            let ego = EgoMotionInput {
                lateral_velocity: 0.05,
                confidence: 0.5, // ego working but measuring near-zero → curve artifact
            };
            let frame = 65 + i as u64;
            if det
                .update(Some(m), Some(ego), frame as f64 * 33.3, frame)
                .completed_shift
                .is_some()
            {
                any_shift_emitted = true;
            }
        }

        assert!(
            !det.in_curve_mode,
            "Curve mode should be OFF at emit time (reproducing the chatter gap)"
        );
        assert!(
            !any_shift_emitted,
            "Shift during curve chatter gap with near-zero ego should be VETOED via shift_saw_curve_mode"
        );
    }
}
