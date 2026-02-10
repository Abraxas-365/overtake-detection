// src/analysis/lateral_detector.rs
//
// Simplified lateral shift detector using lane markings.
//
// Unlike the old 1200-line state machine, this module has ONE job:
//   "Did the ego vehicle move laterally by a significant amount?"
//
// It does NOT try to classify overtakes. It reports:
//   - Direction (left/right)
//   - Magnitude (normalized offset from baseline)
//   - Confidence (based on lane detection quality)
//   - Timing (start/end)
//
// The fusion layer combines this with vehicle tracking to distinguish
// overtakes from lane changes.
//
// Design principles:
//   - Only produces signal when lanes are actually visible
//   - Automatically suppresses during occlusion (outputs Nothing)
//   - Simple EWMA baseline, no mining profiles, no six detection paths
//   - ~200 lines instead of ~1200

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
    /// After occlusion of this many frames, reset baseline entirely
    pub occlusion_reset_frames: u32,
    /// After baseline reset, freeze for this many frames before detecting
    pub post_reset_freeze_frames: u32,
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

/// Current state of the detector
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Waiting for lanes / building baseline
    Initializing,
    /// Baseline established, watching for shifts
    Stable,
    /// Shift in progress
    Shifting,
    /// Lanes lost — no output
    Occluded,
    /// Just recovered from occlusion — rebuilding baseline
    Recovering,
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
    baseline: f32,         // EWMA of normalized offset
    baseline_samples: u32, // Frames contributing to baseline
    freeze_remaining: u32, // Post-reset detection freeze

    // Occlusion tracking
    frames_without_lanes: u32,

    // Active shift tracking
    shift_direction: Option<ShiftDirection>,
    shift_start_ms: f64,
    shift_start_frame: u64,
    shift_peak_offset: f32,
    shift_frames: u32,
    shift_confidence_sum: f32,

    // History for smoothing
    offset_history: VecDeque<f32>,
}

impl LateralShiftDetector {
    pub fn new(config: LateralDetectorConfig) -> Self {
        Self {
            config,
            state: State::Initializing,
            baseline: 0.0,
            baseline_samples: 0,
            freeze_remaining: 0,
            frames_without_lanes: 0,
            shift_direction: None,
            shift_start_ms: 0.0,
            shift_start_frame: 0,
            shift_peak_offset: 0.0,
            shift_frames: 0,
            shift_confidence_sum: 0.0,
            offset_history: VecDeque::with_capacity(30),
        }
    }

    /// Process one frame. Returns a LateralShiftEvent if a shift just completed.
    /// Returns None most frames — shifts are reported on completion.
    pub fn update(
        &mut self,
        measurement: Option<LaneMeasurement>,
        timestamp_ms: f64,
        frame_id: u64,
    ) -> Option<LateralShiftEvent> {
        // ── NO LANES ────────────────────────────────────────────
        let meas = match measurement {
            Some(m)
                if m.confidence >= self.config.min_lane_confidence && m.lane_width_px > 50.0 =>
            {
                self.frames_without_lanes = 0;
                m
            }
            _ => {
                self.frames_without_lanes += 1;

                if self.frames_without_lanes >= self.config.occlusion_reset_frames {
                    if self.state != State::Occluded {
                        warn!(
                            "🌫️  Lateral detector: occluded ({:.1}s without lanes)",
                            self.frames_without_lanes as f64 / 30.0
                        );
                        // If shift was in progress, cancel it — we can't track through occlusion
                        if self.state == State::Shifting {
                            warn!("❌ Shift cancelled due to occlusion");
                        }
                        self.state = State::Occluded;
                        self.reset_shift();
                    }
                }
                return None; // No signal during occlusion
            }
        };

        // ── NORMALIZE ───────────────────────────────────────────
        let normalized = meas.lateral_offset_px / meas.lane_width_px;

        self.offset_history.push_back(normalized);
        if self.offset_history.len() > 20 {
            self.offset_history.pop_front();
        }

        // ── STATE MACHINE ───────────────────────────────────────
        match self.state {
            State::Occluded => {
                // Lanes just recovered — start rebuilding baseline
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

                // Update baseline during stable driving
                if abs_dev < self.config.shift_start_threshold {
                    self.update_baseline(normalized);
                }

                // Check for shift start
                if abs_dev >= self.config.shift_start_threshold {
                    let direction = if deviation < 0.0 {
                        ShiftDirection::Left
                    } else {
                        ShiftDirection::Right
                    };

                    self.state = State::Shifting;
                    self.shift_direction = Some(direction);
                    self.shift_start_ms = timestamp_ms;
                    self.shift_start_frame = frame_id;
                    self.shift_peak_offset = abs_dev;
                    self.shift_frames = 1;
                    self.shift_confidence_sum = meas.confidence;

                    info!(
                        "🔀 Lateral shift started: {} | dev={:.1}% | baseline={:.1}%",
                        direction.as_str(),
                        abs_dev * 100.0,
                        self.baseline * 100.0
                    );
                }

                None
            }

            State::Shifting => {
                let deviation = normalized - self.baseline;
                let abs_dev = deviation.abs();
                self.shift_frames += 1;
                self.shift_confidence_sum += meas.confidence;

                if abs_dev > self.shift_peak_offset {
                    self.shift_peak_offset = abs_dev;
                }

                // Timeout — too long, probably not a real shift
                if self.shift_frames > self.config.max_shift_frames {
                    warn!(
                        "❌ Lateral shift timeout after {} frames — resetting baseline",
                        self.shift_frames
                    );
                    // The vehicle probably changed lanes permanently — reset baseline
                    self.baseline = normalized;
                    self.baseline_samples = 1;
                    self.state = State::Stable;
                    self.reset_shift();
                    return None;
                }

                // Check for shift end (returned toward baseline)
                if abs_dev < self.config.shift_end_threshold {
                    let confirmed = self.shift_peak_offset >= self.config.shift_confirm_threshold
                        && self.shift_frames >= self.config.min_shift_frames;

                    let avg_confidence = self.shift_confidence_sum / self.shift_frames as f32;

                    let event = if confirmed {
                        let evt = LateralShiftEvent {
                            direction: self.shift_direction.unwrap_or(ShiftDirection::Left),
                            peak_offset: self.shift_peak_offset,
                            start_ms: self.shift_start_ms,
                            end_ms: timestamp_ms,
                            start_frame: self.shift_start_frame,
                            end_frame: frame_id,
                            duration_ms: timestamp_ms - self.shift_start_ms,
                            confidence: self.compute_confidence(avg_confidence),
                            confirmed: true,
                        };

                        info!(
                            "✅ Lateral shift completed: {} | peak={:.1}% | dur={:.1}s | conf={:.2}",
                            evt.direction.as_str(),
                            evt.peak_offset * 100.0,
                            evt.duration_ms / 1000.0,
                            evt.confidence
                        );

                        Some(evt)
                    } else {
                        debug!(
                            "❌ Lateral shift rejected: peak={:.1}% (need {:.1}%), frames={} (need {})",
                            self.shift_peak_offset * 100.0,
                            self.config.shift_confirm_threshold * 100.0,
                            self.shift_frames,
                            self.config.min_shift_frames
                        );
                        None
                    };

                    // Update baseline to current position (vehicle may have shifted permanently)
                    self.update_baseline(normalized);
                    self.state = State::Stable;
                    self.reset_shift();

                    return event;
                }

                // Check if we've settled in a new lane (deviation stable but high)
                if self.shift_frames > 60 && self.is_deviation_stable() {
                    let confirmed = self.shift_peak_offset >= self.config.shift_confirm_threshold;
                    let avg_confidence = self.shift_confidence_sum / self.shift_frames as f32;

                    let event = if confirmed {
                        let evt = LateralShiftEvent {
                            direction: self.shift_direction.unwrap_or(ShiftDirection::Left),
                            peak_offset: self.shift_peak_offset,
                            start_ms: self.shift_start_ms,
                            end_ms: timestamp_ms,
                            start_frame: self.shift_start_frame,
                            end_frame: frame_id,
                            duration_ms: timestamp_ms - self.shift_start_ms,
                            confidence: self.compute_confidence(avg_confidence),
                            confirmed: true,
                        };

                        info!(
                            "✅ Lateral shift settled (new lane): {} | peak={:.1}% | dur={:.1}s",
                            evt.direction.as_str(),
                            evt.peak_offset * 100.0,
                            evt.duration_ms / 1000.0,
                        );

                        Some(evt)
                    } else {
                        None
                    };

                    // Reset baseline to new position
                    self.baseline = normalized;
                    self.baseline_samples = 1;
                    self.state = State::Stable;
                    self.reset_shift();

                    return event;
                }

                None
            }
        }
    }

    /// Is the ego vehicle currently shifting?
    pub fn is_shifting(&self) -> bool {
        self.state == State::Shifting
    }

    /// Current state as string (for diagnostics)
    pub fn state_str(&self) -> &str {
        match self.state {
            State::Initializing => "INITIALIZING",
            State::Stable => "STABLE",
            State::Shifting => "SHIFTING",
            State::Occluded => "OCCLUDED",
            State::Recovering => "RECOVERING",
        }
    }

    /// Current baseline value
    pub fn baseline(&self) -> f32 {
        self.baseline
    }

    pub fn reset(&mut self) {
        self.state = State::Initializing;
        self.baseline = 0.0;
        self.baseline_samples = 0;
        self.freeze_remaining = 0;
        self.frames_without_lanes = 0;
        self.offset_history.clear();
        self.reset_shift();
    }

    // ── PRIVATE ─────────────────────────────────────────────────

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
        self.shift_start_ms = 0.0;
        self.shift_start_frame = 0;
        self.shift_peak_offset = 0.0;
        self.shift_frames = 0;
        self.shift_confidence_sum = 0.0;
    }

    fn is_deviation_stable(&self) -> bool {
        if self.offset_history.len() < 10 {
            return false;
        }
        let recent: Vec<f32> = self.offset_history.iter().rev().take(10).copied().collect();
        let mean = recent.iter().sum::<f32>() / recent.len() as f32;
        let var = recent.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / recent.len() as f32;
        var < 0.002 // Very stable
    }

    fn compute_confidence(&self, avg_lane_confidence: f32) -> f32 {
        let mut conf: f32 = 0.5;

        // Lane detection quality
        if avg_lane_confidence > 0.7 {
            conf += 0.20;
        } else if avg_lane_confidence > 0.5 {
            conf += 0.10;
        }

        // Peak offset magnitude
        if self.shift_peak_offset > 0.50 {
            conf += 0.15;
        } else if self.shift_peak_offset > 0.35 {
            conf += 0.10;
        }

        // Duration in reasonable range (1-8s)
        let dur_s = self.shift_frames as f32 / 30.0;
        if dur_s >= 1.0 && dur_s <= 8.0 {
            conf += 0.10;
        }

        conf.min(0.95)
    }
}

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

    #[test]
    fn test_baseline_warmup() {
        let cfg = LateralDetectorConfig {
            baseline_warmup_frames: 10,
            ..Default::default()
        };
        let mut det = LateralShiftDetector::new(cfg);

        // Feed 10 frames at center
        for i in 0..10 {
            let result = det.update(meas(0.0, 400.0, 0.8), i as f64 * 33.3, i);
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

        // Warmup: center position
        for i in 0..20 {
            det.update(meas(0.0, w, 0.8), i as f64 * 33.3, i);
        }

        // Shift left: offset goes negative to -140px (35% of 400)
        for i in 20..40 {
            det.update(meas(-140.0, w, 0.8), i as f64 * 33.3, i);
        }

        // Return to center
        let mut events = Vec::new();
        for i in 40..50 {
            if let Some(e) = det.update(meas(0.0, w, 0.8), i as f64 * 33.3, i) {
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

        // Warmup
        for i in 0..20 {
            det.update(meas(0.0, 400.0, 0.8), i as f64 * 33.3, i);
        }

        // Occlusion: no lanes for 20 frames
        for i in 20..40 {
            let result = det.update(None, i as f64 * 33.3, i);
            assert!(
                result.is_none(),
                "Should not detect anything during occlusion"
            );
        }

        assert_eq!(det.state, State::Occluded);
    }
}
