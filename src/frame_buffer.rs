// src/frame_buffer.rs
//
// Strategic Frame Buffer for LLM Analysis
//
// Captures frames at strategically important moments during maneuvers
// to give the LLM vision model the best possible images for line analysis.
//
// Frame Selection Strategy:
//   1. PRE-CROSSING: Frame just before the vehicle crosses a line
//      (line is most visible, unobstructed, from original lane)
//   2. AT-CROSSING: Frame when the vehicle is crossing the line
//      (shows actual contact with the line marking)
//   3. PEAK-OFFSET: Frame at maximum lateral displacement
//      (shows the line from the opposing lane — different perspective)
//   4. RETURN-CROSSING: Frame when returning to original lane
//      (second view of the line, confirms line type)
//   5. POST-MANEUVER: Frame after maneuver completes
//      (final clean view of lane markings)
//
// Additionally captures:
//   - CURVE-CONTEXT: When curvature is detected, captures wide-angle view
//   - VEHICLE-CONTEXT: Frames showing overtaken vehicles
//
// Each captured frame stores per-frame metadata (lane confidence,
// offset percentage, marking detections) for the LLM context.

use std::collections::VecDeque;
use tracing::debug;

/// Maximum frames to buffer per maneuver
const MAX_FRAMES_PER_MANEUVER: usize = 20;

/// Maximum JPEG size per frame (to avoid blowing up HTTP payloads)
const MAX_JPEG_BYTES: usize = 400_000;

/// Why this particular frame was captured
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureReason {
    /// Periodic capture during active maneuver
    Periodic,
    /// Just before crossing started (clean line view)
    PreCrossing,
    /// During line crossing (contact with marking)
    AtCrossing,
    /// Maximum lateral displacement (opposing perspective)
    PeakOffset,
    /// During return to original lane
    ReturnCrossing,
    /// After maneuver completes
    PostManeuver,
    /// Curve context frame
    CurveContext,
    /// Frame showing overtaken vehicle
    VehicleContext,
}

impl CaptureReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Periodic => "periodic",
            Self::PreCrossing => "pre_crossing",
            Self::AtCrossing => "at_crossing",
            Self::PeakOffset => "peak_offset",
            Self::ReturnCrossing => "return_crossing",
            Self::PostManeuver => "post_maneuver",
            Self::CurveContext => "curve_context",
            Self::VehicleContext => "vehicle_context",
        }
    }

    /// Priority for frame selection — higher = more likely to be sent to LLM
    pub fn priority(&self) -> u8 {
        match self {
            Self::AtCrossing => 10,
            Self::PreCrossing => 9,
            Self::PeakOffset => 8,
            Self::ReturnCrossing => 7,
            Self::CurveContext => 6,
            Self::VehicleContext => 5,
            Self::PostManeuver => 4,
            Self::Periodic => 1,
        }
    }
}

/// Metadata attached to each captured frame
#[derive(Debug, Clone)]
pub struct CapturedFrameMeta {
    pub frame_id: u64,
    pub timestamp_ms: f64,
    pub width: usize,
    pub height: usize,
    pub reason: CaptureReason,

    // Per-frame detection context
    pub lane_confidence: Option<f32>,
    pub offset_percentage: Option<f32>,
    pub left_marking_class: Option<String>,
    pub right_marking_class: Option<String>,
    pub curve_detected: bool,
    pub curve_angle_degrees: f32,
    pub vehicles_visible: u32,
}

/// A captured frame with its JPEG data and metadata
#[derive(Debug, Clone)]
pub struct CapturedFrame {
    pub meta: CapturedFrameMeta,
    /// JPEG-encoded image data (base64 will be computed at send time)
    pub jpeg_data: Vec<u8>,
}

/// State machine for tracking maneuver phase for strategic capture
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ManeuverPhase {
    Idle,
    /// Detected lateral shift starting — capture pre-crossing
    ShiftStarting,
    /// Vehicle is crossing/has crossed line — capture at-crossing
    Crossing,
    /// Vehicle at peak offset — capture from opposing perspective
    PeakOffset,
    /// Vehicle returning — capture return crossing
    Returning,
    /// Maneuver complete — capture post frame
    Complete,
}

/// Ring buffer that captures frames strategically during maneuvers.
///
/// Usage:
///   1. Call `update()` every frame with current pipeline state
///   2. When a maneuver is detected, call `collect_for_maneuver()` to get
///      the best frames for LLM analysis
///   3. Call `reset()` after sending to prepare for next maneuver
pub struct StrategicFrameBuffer {
    /// All captured frames for current maneuver
    frames: VecDeque<CapturedFrame>,
    /// Current maneuver phase
    phase: ManeuverPhase,
    /// Frame counter for periodic captures
    frames_since_last_capture: u32,
    /// Capture interval during active maneuver (frames)
    periodic_interval: u32,
    /// Peak offset seen so far in current maneuver
    peak_offset: f32,
    /// Frame ID at peak offset
    peak_offset_frame: u64,
    /// Whether we already captured the peak
    peak_captured: bool,
    /// Whether currently in an active shift
    shift_active: bool,
    /// Direction of current shift (-1 = left, 1 = right, 0 = none)
    shift_direction: i8,
    /// Last N frames in a rolling window (for pre-crossing lookback)
    lookback_buffer: VecDeque<CapturedFrame>,
    /// Max lookback frames
    max_lookback: usize,
}

impl StrategicFrameBuffer {
    pub fn new() -> Self {
        Self {
            frames: VecDeque::with_capacity(MAX_FRAMES_PER_MANEUVER),
            phase: ManeuverPhase::Idle,
            frames_since_last_capture: 0,
            periodic_interval: 10, // Every 10 frames during active maneuver
            peak_offset: 0.0,
            peak_offset_frame: 0,
            peak_captured: false,
            shift_active: false,
            shift_direction: 0,
            lookback_buffer: VecDeque::with_capacity(15),
            max_lookback: 10,
        }
    }

    /// Feed a frame into the lookback buffer (called every frame).
    /// This does NOT mean the frame will be sent — it just keeps a rolling
    /// window so we can look back when a maneuver starts.
    pub fn feed_frame(
        &mut self,
        frame_rgb: &[u8],
        width: usize,
        height: usize,
        frame_id: u64,
        timestamp_ms: f64,
        lane_confidence: Option<f32>,
        offset_percentage: Option<f32>,
        left_marking: Option<&str>,
        right_marking: Option<&str>,
        curve_detected: bool,
        curve_angle: f32,
        vehicles_visible: u32,
    ) {
        let meta = CapturedFrameMeta {
            frame_id,
            timestamp_ms,
            width,
            height,
            reason: CaptureReason::Periodic,
            lane_confidence,
            offset_percentage,
            left_marking_class: left_marking.map(|s| s.to_string()),
            right_marking_class: right_marking.map(|s| s.to_string()),
            curve_detected,
            curve_angle_degrees: curve_angle,
            vehicles_visible,
        };

        // Encode to JPEG for the lookback buffer
        if let Some(jpeg) = encode_rgb_to_jpeg(frame_rgb, width, height, 85) {
            if jpeg.len() <= MAX_JPEG_BYTES {
                let frame = CapturedFrame {
                    meta,
                    jpeg_data: jpeg,
                };
                self.lookback_buffer.push_back(frame);
                if self.lookback_buffer.len() > self.max_lookback {
                    self.lookback_buffer.pop_front();
                }
            }
        }

        // Track peak offset
        if let Some(offset) = offset_percentage {
            if offset.abs() > self.peak_offset.abs() {
                self.peak_offset = offset;
                self.peak_offset_frame = frame_id;
                self.peak_captured = false;
            }
        }

        self.frames_since_last_capture += 1;
    }

    /// Signal that a lateral shift has started (from LateralShiftDetector)
    pub fn notify_shift_start(&mut self, direction_left: bool) {
        if self.phase == ManeuverPhase::Idle {
            self.phase = ManeuverPhase::ShiftStarting;
            self.shift_active = true;
            self.shift_direction = if direction_left { -1 } else { 1 };
            self.peak_offset = 0.0;
            self.peak_captured = false;

            // Grab pre-crossing frames from lookback
            // Pick the best 2 from the lookback (highest lane confidence)
            let mut lookback_sorted: Vec<_> = self.lookback_buffer.iter().cloned().collect();
            lookback_sorted
                .sort_by(|a, b| {
                    let conf_a = a.meta.lane_confidence.unwrap_or(0.0);
                    let conf_b = b.meta.lane_confidence.unwrap_or(0.0);
                    conf_b.partial_cmp(&conf_a).unwrap_or(std::cmp::Ordering::Equal)
                });

            for mut frame in lookback_sorted.into_iter().take(2) {
                frame.meta.reason = CaptureReason::PreCrossing;
                self.add_frame(frame);
            }

            debug!(
                "📸 StrategicFrameBuffer: shift started ({}), captured {} pre-crossing frames",
                if direction_left { "LEFT" } else { "RIGHT" },
                self.frames.len()
            );
        }
    }

    /// Signal that the vehicle is actively crossing a line
    pub fn notify_crossing(&mut self) {
        if self.phase == ManeuverPhase::ShiftStarting || self.phase == ManeuverPhase::Crossing {
            self.phase = ManeuverPhase::Crossing;

            // Capture the most recent frame from lookback as at-crossing
            if let Some(mut frame) = self.lookback_buffer.back().cloned() {
                frame.meta.reason = CaptureReason::AtCrossing;
                self.add_frame(frame);
            }
        }
    }

    /// Signal that the vehicle has reached peak offset
    pub fn notify_peak_offset(&mut self) {
        if !self.peak_captured {
            self.phase = ManeuverPhase::PeakOffset;
            self.peak_captured = true;

            if let Some(mut frame) = self.lookback_buffer.back().cloned() {
                frame.meta.reason = CaptureReason::PeakOffset;
                self.add_frame(frame);
            }
        }
    }

    /// Signal that the vehicle is returning to original lane
    pub fn notify_return(&mut self) {
        self.phase = ManeuverPhase::Returning;

        if let Some(mut frame) = self.lookback_buffer.back().cloned() {
            frame.meta.reason = CaptureReason::ReturnCrossing;
            self.add_frame(frame);
        }
    }

    /// Signal that a curve was detected — capture a context frame
    pub fn notify_curve_detected(&mut self) {
        if self.shift_active {
            if let Some(mut frame) = self.lookback_buffer.back().cloned() {
                frame.meta.reason = CaptureReason::CurveContext;
                self.add_frame(frame);
            }
        }
    }

    /// Capture a periodic frame during active maneuver
    pub fn maybe_capture_periodic(&mut self) {
        if self.shift_active && self.frames_since_last_capture >= self.periodic_interval {
            if let Some(mut frame) = self.lookback_buffer.back().cloned() {
                frame.meta.reason = CaptureReason::Periodic;
                self.add_frame(frame);
                self.frames_since_last_capture = 0;
            }
        }
    }

    /// Signal that the maneuver is complete — capture post frame and return all
    pub fn notify_maneuver_complete(&mut self) {
        self.phase = ManeuverPhase::Complete;
        self.shift_active = false;

        if let Some(mut frame) = self.lookback_buffer.back().cloned() {
            frame.meta.reason = CaptureReason::PostManeuver;
            self.add_frame(frame);
        }
    }

    /// Collect the best frames for LLM analysis.
    /// Selects up to `max_frames` using priority-based selection to ensure
    /// diversity of capture reasons.
    pub fn collect_best_frames(&self, max_frames: usize) -> Vec<CapturedFrame> {
        if self.frames.is_empty() {
            return Vec::new();
        }

        let mut all_frames: Vec<&CapturedFrame> = self.frames.iter().collect();

        // Sort by priority (highest first), then by frame_id (temporal order)
        all_frames.sort_by(|a, b| {
            let pri_cmp = b.meta.reason.priority().cmp(&a.meta.reason.priority());
            if pri_cmp == std::cmp::Ordering::Equal {
                a.meta.frame_id.cmp(&b.meta.frame_id)
            } else {
                pri_cmp
            }
        });

        // Ensure diversity: pick at most 1 from each capture reason, then fill remaining
        let mut selected: Vec<CapturedFrame> = Vec::with_capacity(max_frames);
        let mut seen_reasons: Vec<CaptureReason> = Vec::new();

        // First pass: one from each unique reason
        for frame in &all_frames {
            if selected.len() >= max_frames {
                break;
            }
            if !seen_reasons.contains(&frame.meta.reason) {
                selected.push((*frame).clone());
                seen_reasons.push(frame.meta.reason);
            }
        }

        // Second pass: fill remaining slots with highest-priority duplicates
        for frame in &all_frames {
            if selected.len() >= max_frames {
                break;
            }
            if !selected.iter().any(|s| s.meta.frame_id == frame.meta.frame_id) {
                selected.push((*frame).clone());
            }
        }

        // Sort final selection by temporal order
        selected.sort_by_key(|f| f.meta.frame_id);
        selected
    }

    /// Reset the buffer for the next maneuver
    pub fn reset(&mut self) {
        self.frames.clear();
        self.phase = ManeuverPhase::Idle;
        self.frames_since_last_capture = 0;
        self.peak_offset = 0.0;
        self.peak_offset_frame = 0;
        self.peak_captured = false;
        self.shift_active = false;
        self.shift_direction = 0;
        // Don't clear lookback — we want continuous rolling window
    }

    /// Check if buffer has any captured frames
    pub fn has_frames(&self) -> bool {
        !self.frames.is_empty()
    }

    /// Number of captured frames
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    fn add_frame(&mut self, frame: CapturedFrame) {
        if self.frames.len() < MAX_FRAMES_PER_MANEUVER {
            self.frames.push_back(frame);
        }
    }
}

/// Encode RGB frame data to JPEG bytes using the `image` crate.
fn encode_rgb_to_jpeg(rgb_data: &[u8], width: usize, height: usize, quality: u8) -> Option<Vec<u8>> {
    use image::{ImageBuffer, RgbImage};
    use std::io::Cursor;

    let expected_len = width * height * 3;
    if rgb_data.len() < expected_len {
        return None;
    }

    let img: RgbImage = ImageBuffer::from_raw(width as u32, height as u32, rgb_data[..expected_len].to_vec())?;

    let mut buf = Cursor::new(Vec::new());
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality);
    if img.write_with_encoder(encoder).is_ok() {
        Some(buf.into_inner())
    } else {
        None
    }
}
