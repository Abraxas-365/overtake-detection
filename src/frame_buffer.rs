// src/frame_buffer.rs

use crate::types::{CurveInfo, CurveType, Frame, LaneChangeEvent};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;
use tracing::{error, info, warn};

/// Buffer to capture frames during a lane change event with pre-buffering
pub struct LaneChangeFrameBuffer {
    frames: Vec<Frame>,
    max_frames: usize,
    is_capturing: bool,
    capture_start_frame_id: Option<u64>,
    /// Pre-buffer: keeps last N frames before lane change starts
    pre_buffer: VecDeque<Frame>,
    pre_buffer_size: usize,
}

impl LaneChangeFrameBuffer {
    pub fn new(max_frames: usize) -> Self {
        // INCREASED pre-buffer to 60 frames (2 seconds) to ensure we have
        // enough context for end-of-video analysis
        let pre_buffer_size = 60;

        Self {
            frames: Vec::with_capacity(max_frames),
            max_frames,
            is_capturing: false,
            capture_start_frame_id: None,
            pre_buffer: VecDeque::with_capacity(pre_buffer_size),
            pre_buffer_size,
        }
    }

    /// Add frame to pre-buffer (called continuously while in CENTERED state)
    pub fn add_to_pre_buffer(&mut self, frame: Frame) {
        self.pre_buffer.push_back(frame);
        if self.pre_buffer.len() > self.pre_buffer_size {
            self.pre_buffer.pop_front();
        }
    }

    pub fn start_capture(&mut self, frame_id: u64) {
        self.frames.clear();
        self.is_capturing = true;
        self.capture_start_frame_id = Some(frame_id);

        // Transfer pre-buffer frames to main buffer
        let pre_buffer_count = self.pre_buffer.len();
        for frame in self.pre_buffer.drain(..) {
            if self.frames.len() < self.max_frames {
                self.frames.push(frame);
            }
        }

        info!(
            "📹 Started capturing at frame {} (included {} pre-buffered frames)",
            frame_id, pre_buffer_count
        );
    }

    pub fn add_frame(&mut self, frame: Frame) {
        if self.is_capturing && self.frames.len() < self.max_frames {
            self.frames.push(frame);
        }
    }

    pub fn stop_capture(&mut self) -> (Vec<Frame>, u64) {
        let start_id = self.capture_start_frame_id.unwrap_or(0);
        // Estimate start ID of pre-buffered frames
        let effective_start = start_id.saturating_sub(self.pre_buffer_size as u64);

        self.is_capturing = false;
        self.capture_start_frame_id = None;
        let frames = std::mem::take(&mut self.frames);
        (frames, effective_start)
    }

    pub fn force_flush(&mut self) -> (Vec<Frame>, u64) {
        let start_id = self.capture_start_frame_id.unwrap_or(0);
        let effective_start = start_id.saturating_sub(self.pre_buffer_size as u64);

        self.is_capturing = false;
        self.capture_start_frame_id = None;

        let mut result = std::mem::take(&mut self.frames);
        if result.is_empty() && !self.pre_buffer.is_empty() {
            for f in &self.pre_buffer {
                result.push(f.clone());
            }
        }
        (result, effective_start)
    }

    pub fn is_capturing(&self) -> bool {
        self.is_capturing
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    pub fn cancel_capture(&mut self) {
        self.frames.clear();
        self.is_capturing = false;
        self.capture_start_frame_id = None;
    }

    pub fn pre_buffer_count(&self) -> usize {
        self.pre_buffer.len()
    }
}
// ============================================================================
// ENHANCED API STRUCTURES WITH DETECTION METADATA
// ============================================================================

/// Detection quality metadata to help AI make better decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionMetadata {
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // EXISTING FIELDS (keep these)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    pub detection_confidence: f32,
    pub max_offset_normalized: f32,
    pub avg_lane_confidence: f32,
    pub both_lanes_ratio: f32,
    pub video_resolution: String,
    pub fps: f32,
    pub region: String,
    pub avg_lane_width_px: Option<f32>,

    // Curve info
    pub curve_detected: bool,
    pub curve_angle_degrees: f32,
    pub curve_confidence: f32,
    pub curve_type: String,

    // Shadow overtake
    pub shadow_overtake_detected: bool,
    pub shadow_overtake_count: u32,
    pub shadow_worst_severity: String,
    pub shadow_blocking_vehicles: Vec<String>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 🆕 NEW FIELDS - Comprehensive Context
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    // ─── LANE LINE LEGALITY ───
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line_crossing_info: Option<LineCrossingInfo>,

    // ─── VEHICLES OVERTAKEN ───
    pub vehicles_overtaken_count: u32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub overtaken_vehicle_types: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub overtaken_vehicle_ids: Vec<u32>,

    // ─── MANEUVER CHARACTERISTICS ───
    pub maneuver_type: String, // "complete_overtake", "incomplete_overtake", "simple_lane_change"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incomplete_reason: Option<String>,

    // ─── TRAJECTORY ANALYSIS ───
    pub trajectory_info: TrajectoryInfo,

    // ─── VELOCITY & DYNAMICS ───
    pub velocity_info: VelocityInfo,

    // ─── POSITIONING DETAILS ───
    pub positioning_info: PositioningInfo,

    // ─── DETECTION METHOD ───
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detection_path: Option<String>, // How was this detected?

    // ─── TEMPORAL ANALYSIS ───
    pub temporal_info: TemporalInfo,
}

/// Enhanced payload for the legality analysis API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneChangeLegalityRequest {
    pub event_id: String,
    pub direction: String,
    pub start_frame_id: u64,
    pub end_frame_id: u64,
    pub video_timestamp_ms: f64,
    pub duration_ms: Option<f64>,
    pub source_id: String,
    pub frames: Vec<FrameData>,
    /// Detection quality metadata (includes curve + shadow info)
    pub detection_metadata: DetectionMetadata,
}

/// Enhanced frame data with per-frame metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameData {
    pub frame_index: usize,
    pub timestamp_ms: f64,
    pub width: usize,
    pub height: usize,
    pub base64_image: String,
    /// Lane detection confidence for this frame (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lane_confidence: Option<f32>,
    /// Lateral offset as percentage of lane width (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset_percentage: Option<f32>,
}

/// Response from the legality API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneChangeLegalityResponse {
    pub event_id: String,
    pub status: String,
    pub message: String,
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Extract key frames emphasizing CONTEXT and PROGRESSION
pub fn extract_key_frames_for_lane_change(frames: &[Frame], count: usize) -> Vec<&Frame> {
    if frames.is_empty() || count == 0 {
        return vec![];
    }

    if frames.len() <= count {
        return frames.iter().collect();
    }

    let mut selected_indices = Vec::with_capacity(count);

    // Strategy: Include frames from BEFORE, DURING, and AFTER the maneuver
    // This gives the AI context about what was "normal" before the change

    match count {
        5 => {
            // Frame 0: Start (should be from pre-buffer, BEFORE maneuver)
            // Frame 1: Early (25%)
            // Frame 2: Middle (50%)
            // Frame 3: Late (75%)
            // Frame 4: End (100%)
            selected_indices.push(0);
            selected_indices.push(frames.len() / 4);
            selected_indices.push(frames.len() / 2);
            selected_indices.push((frames.len() * 3) / 4);
            selected_indices.push(frames.len() - 1);
        }
        7 => {
            // More granular: every ~16%
            selected_indices.push(0); // 0%
            selected_indices.push(frames.len() / 6); // 16%
            selected_indices.push(frames.len() / 3); // 33%
            selected_indices.push(frames.len() / 2); // 50%
            selected_indices.push((frames.len() * 2) / 3); // 66%
            selected_indices.push((frames.len() * 5) / 6); // 83%
            selected_indices.push(frames.len() - 1); // 100%
        }
        _ => {
            // Evenly distributed
            let step = (frames.len() - 1) as f32 / (count - 1) as f32;
            for i in 0..count {
                let index = (i as f32 * step).round() as usize;
                selected_indices.push(index.min(frames.len() - 1));
            }
        }
    }

    selected_indices.into_iter().map(|i| &frames[i]).collect()
}

/// Convert frame to base64 JPEG with quality optimization
pub fn frame_to_base64(frame: &Frame) -> Result<String, anyhow::Error> {
    use image::{codecs::jpeg::JpegEncoder, ImageBuffer, Rgb};
    use std::io::Cursor;

    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(frame.width as u32, frame.height as u32, frame.data.clone())
            .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;

    let mut buffer = Cursor::new(Vec::new());

    // Use explicit JPEG encoder with quality setting
    let mut encoder = JpegEncoder::new_with_quality(&mut buffer, 85);
    encoder.encode(
        img.as_raw(),
        img.width(),
        img.height(),
        image::ExtendedColorType::Rgb8,
    )?;

    Ok(STANDARD.encode(buffer.into_inner()))
}

/// Build the API request payload with enhanced metadata
/// Includes curve_info and shadow overtake info (extracted from event metadata)
pub fn build_legality_request(
    event: &LaneChangeEvent,
    captured_frames: &[Frame],
    num_frames_to_send: usize,
    curve_info: CurveInfo,
) -> Result<LaneChangeLegalityRequest, anyhow::Error> {
    let key_frames = extract_key_frames_for_lane_change(captured_frames, num_frames_to_send);

    if key_frames.is_empty() {
        anyhow::bail!("No frames to send");
    }

    let mut frame_data_list = Vec::with_capacity(key_frames.len());

    // Log which frames were selected
    let selected_indices: Vec<usize> = key_frames
        .iter()
        .filter_map(|kf| captured_frames.iter().position(|f| std::ptr::eq(*kf, f)))
        .collect();

    info!(
        "📸 Selected {} frames from {} total: indices {:?}",
        key_frames.len(),
        captured_frames.len(),
        selected_indices
    );

    // Process each frame
    for (i, frame) in key_frames.iter().enumerate() {
        let base64_image = frame_to_base64(frame)?;

        let lane_confidence = None;
        let offset_percentage = None;

        frame_data_list.push(FrameData {
            frame_index: i,
            timestamp_ms: frame.timestamp_ms,
            width: frame.width,
            height: frame.height,
            base64_image,
            lane_confidence,
            offset_percentage,
        });
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // CALCULATE VIDEO METADATA
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    let video_resolution = format!(
        "{}x{}",
        key_frames.first().unwrap().width,
        key_frames.first().unwrap().height
    );

    // Calculate FPS from frame timestamps
    let fps = if key_frames.len() >= 2 {
        let time_span =
            key_frames.last().unwrap().timestamp_ms - key_frames.first().unwrap().timestamp_ms;
        if time_span > 0.0 {
            ((key_frames.len() - 1) as f64 / (time_span / 1000.0)) as f32
        } else {
            25.0
        }
    } else {
        25.0
    };

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // EXTRACT BASIC DETECTION QUALITY METADATA
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    let max_offset_normalized = event
        .metadata
        .get("max_offset_normalized")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(0.5);

    let both_lanes_ratio = event
        .metadata
        .get("both_lanes_ratio")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(0.7);

    let avg_lane_confidence = event
        .metadata
        .get("avg_lane_confidence")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(0.6);

    let avg_lane_width_px = event
        .metadata
        .get("avg_lane_width_px")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 🆕 EXTRACT LINE CROSSING INFORMATION
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    let line_crossing_info = if let Some(legality) = &event.legality {
        Some(LineCrossingInfo {
            line_crossed: true,
            line_type: legality.lane_line_type.clone(),
            is_legal: legality.is_legal,
            severity: if legality.is_legal {
                "LEGAL".to_string()
            } else {
                event
                    .metadata
                    .get("line_legality_verdict")
                    .and_then(|v| v.as_str())
                    .unwrap_or("ILLEGAL")
                    .to_string()
            },
            line_detection_confidence: legality.confidence,
            crossed_at_frame: event
                .metadata
                .get("line_legality_frame")
                .and_then(|v| v.as_u64())
                .unwrap_or(event.start_frame_id),
            additional_lines_crossed: vec![], // Could be enhanced to track multiple
            analysis_details: legality.analysis_details.clone(),
        })
    } else {
        None
    };

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 🆕 EXTRACT VEHICLES OVERTAKEN
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    let vehicles_overtaken_count = event
        .metadata
        .get("vehicles_overtaken")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;

    let overtaken_vehicle_types: Vec<String> = event
        .metadata
        .get("overtaken_vehicle_types")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let overtaken_vehicle_ids: Vec<u32> = event
        .metadata
        .get("overtaken_vehicle_ids")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect()
        })
        .unwrap_or_default();

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 🆕 EXTRACT MANEUVER TYPE & STATUS
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    let maneuver_type = event
        .metadata
        .get("maneuver_type")
        .and_then(|v| v.as_str())
        .unwrap_or("simple_lane_change")
        .to_string();

    let incomplete_reason = event
        .metadata
        .get("incomplete_reason")
        .and_then(|v| v.as_str())
        .map(String::from);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 🆕 EXTRACT TRAJECTORY INFORMATION
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    let initial_position = event
        .metadata
        .get("initial_position")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;

    let final_position = event
        .metadata
        .get("final_position")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;

    let net_displacement = event
        .metadata
        .get("net_displacement")
        .and_then(|v| v.as_f64())
        .unwrap_or((final_position - initial_position).abs() as f64)
        as f32;

    let trajectory_info = TrajectoryInfo {
        initial_position,
        final_position,
        net_displacement,
        returned_to_start: net_displacement.abs() < 0.20,
        excursion_sufficient: max_offset_normalized >= 0.30,
        shape_score: event
            .metadata
            .get("trajectory_shape_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32,
        smoothness: event
            .metadata
            .get("trajectory_smoothness")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32,
        has_direction_reversal: event
            .metadata
            .get("has_direction_reversal")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    };

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 🆕 EXTRACT VELOCITY INFORMATION
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    let peak_lateral_velocity = event
        .metadata
        .get("peak_lateral_velocity")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;

    let avg_lateral_velocity = event
        .metadata
        .get("avg_lateral_velocity")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;

    let velocity_pattern = event
        .metadata
        .get("velocity_pattern")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_else(|| {
            // Infer pattern from available data
            if peak_lateral_velocity > 180.0 {
                "spike".to_string()
            } else if peak_lateral_velocity > 100.0 && avg_lateral_velocity > 60.0 {
                "sustained".to_string()
            } else if peak_lateral_velocity > 0.0 {
                "moderate".to_string()
            } else {
                "unknown".to_string()
            }
        });

    let velocity_info = VelocityInfo {
        peak_lateral_velocity,
        avg_lateral_velocity,
        velocity_pattern,
        max_acceleration: event
            .metadata
            .get("max_acceleration")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32),
    };

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 🆕 EXTRACT POSITIONING INFORMATION
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    let avg_width = avg_lane_width_px.unwrap_or(400.0);

    let lane_width_min = event
        .metadata
        .get("lane_width_min")
        .and_then(|v| v.as_f64())
        .unwrap_or(avg_width as f64) as f32;

    let lane_width_max = event
        .metadata
        .get("lane_width_max")
        .and_then(|v| v.as_f64())
        .unwrap_or(avg_width as f64) as f32;

    let lane_width_variation = lane_width_max - lane_width_min;
    let lane_width_stable = lane_width_variation < (avg_width * 0.15); // < 15% variation

    let positioning_info = PositioningInfo {
        lane_width_min,
        lane_width_max,
        lane_width_avg: avg_width,
        lane_width_stable,
        adjacent_lane_penetration: max_offset_normalized,
        baseline_offset: event
            .metadata
            .get("baseline_offset")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32,
        baseline_frozen: event
            .metadata
            .get("baseline_frozen")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    };

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 🆕 EXTRACT DETECTION PATH
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    let detection_path = event
        .metadata
        .get("detection_path")
        .and_then(|v| v.as_str())
        .map(String::from);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 🆕 EXTRACT TEMPORAL INFORMATION
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    let total_duration = event.duration_ms.unwrap_or(0.0);

    let time_drifting_ms = event
        .metadata
        .get("time_drifting_ms")
        .and_then(|v| v.as_f64());

    let time_crossing_ms = event
        .metadata
        .get("time_crossing_ms")
        .and_then(|v| v.as_f64());

    // Duration is plausible if between 800ms and 10s
    let duration_plausible = total_duration >= 800.0 && total_duration <= 10000.0;

    let temporal_info = TemporalInfo {
        time_drifting_ms,
        time_crossing_ms,
        total_maneuver_duration_ms: total_duration,
        duration_plausible,
        state_progression: vec![], // Could be enhanced with detailed state transitions
    };

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // CURVE INFORMATION (existing)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    let curve_type_str = match curve_info.curve_type {
        CurveType::None => "NONE",
        CurveType::Moderate => "MODERATE",
        CurveType::Sharp => "SHARP",
    };

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // SHADOW OVERTAKE INFORMATION (existing)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    let shadow_detected = event
        .metadata
        .get("shadow_overtake_detected")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let shadow_count = event
        .metadata
        .get("shadow_overtake_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;

    let shadow_severity = event
        .metadata
        .get("shadow_worst_severity")
        .and_then(|v| v.as_str())
        .unwrap_or("NONE")
        .to_string();

    let shadow_vehicles: Vec<String> = event
        .metadata
        .get("shadow_blocking_vehicles")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BUILD COMPREHENSIVE METADATA STRUCTURE
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    let metadata = DetectionMetadata {
        // Basic detection quality
        detection_confidence: event.confidence,
        max_offset_normalized,
        avg_lane_confidence,
        both_lanes_ratio,
        video_resolution,
        fps,
        region: "PE".to_string(),
        avg_lane_width_px,

        // Curve information
        curve_detected: curve_info.is_curve,
        curve_angle_degrees: curve_info.angle_degrees,
        curve_confidence: curve_info.confidence,
        curve_type: curve_type_str.to_string(),

        // Shadow overtake
        shadow_overtake_detected: shadow_detected,
        shadow_overtake_count: shadow_count,
        shadow_worst_severity: shadow_severity,
        shadow_blocking_vehicles: shadow_vehicles,

        // 🆕 NEW COMPREHENSIVE FIELDS
        line_crossing_info,
        vehicles_overtaken_count,
        overtaken_vehicle_types,
        overtaken_vehicle_ids,
        maneuver_type,
        incomplete_reason,
        trajectory_info,
        velocity_info,
        positioning_info,
        detection_path,
        temporal_info,
    };

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // LOGGING
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    info!(
        "📊 Detection quality: conf={:.1}%, max_offset={:.1}%, lanes={:.0}%",
        metadata.detection_confidence * 100.0,
        metadata.max_offset_normalized * 100.0,
        metadata.both_lanes_ratio * 100.0
    );

    if curve_info.is_curve {
        info!(
            "🌀 Curve detected: type={}, angle={:.1}°, confidence={:.0}%",
            curve_type_str,
            curve_info.angle_degrees,
            curve_info.confidence * 100.0
        );
    }

    if shadow_detected {
        warn!(
            "⚫ Shadow overtake: count={}, severity={}, vehicles={:?}",
            metadata.shadow_overtake_count,
            metadata.shadow_worst_severity,
            metadata.shadow_blocking_vehicles
        );
    }

    if vehicles_overtaken_count > 0 {
        info!(
            "🎯 Vehicles overtaken: {} ({:?})",
            vehicles_overtaken_count, metadata.overtaken_vehicle_types
        );
    }

    if let Some(ref line_info) = metadata.line_crossing_info {
        let legality_emoji = if line_info.is_legal { "✅" } else { "⚠️" };
        info!(
            "{} Line crossing: {} [{}] (conf: {:.0}%)",
            legality_emoji,
            line_info.line_type,
            line_info.severity,
            line_info.line_detection_confidence * 100.0
        );
    }

    info!(
        "🏃 Velocity: peak={:.0}px/s, avg={:.0}px/s, pattern={}",
        peak_lateral_velocity, avg_lateral_velocity, metadata.velocity_info.velocity_pattern
    );

    info!(
        "📐 Trajectory: shape={:.2}, smooth={:.2}, reversal={}",
        metadata.trajectory_info.shape_score,
        metadata.trajectory_info.smoothness,
        metadata.trajectory_info.has_direction_reversal
    );

    if let Some(ref path) = metadata.detection_path {
        info!("🔍 Detected via: {}", path);
    }
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BUILD AND RETURN REQUEST
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Ok(LaneChangeLegalityRequest {
        event_id: event.event_id.clone(),
        direction: event.direction_name().to_string(),
        start_frame_id: event.start_frame_id,
        end_frame_id: event.end_frame_id,
        video_timestamp_ms: event.video_timestamp_ms,
        duration_ms: event.duration_ms,
        source_id: event.source_id.clone(),
        frames: frame_data_list,
        detection_metadata: metadata,
    })
}

/// Send the request to the legality analysis API
pub async fn send_to_legality_api(
    request: &LaneChangeLegalityRequest,
    api_url: &str,
) -> Result<LaneChangeLegalityResponse, anyhow::Error> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()?;

    info!(
        "📤 Sending event {} to legality API at {}",
        request.event_id, api_url
    );

    let response = client.post(api_url).json(request).send().await;

    match response {
        Ok(resp) => {
            let status = resp.status();

            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                anyhow::bail!("API error {}: {}", status, body);
            }

            let result = resp.json::<LaneChangeLegalityResponse>().await?;

            info!(
                "📨 API response for {}: {} - {}",
                result.event_id, result.status, result.message
            );

            Ok(result)
        }
        Err(e) => {
            error!("❌ Failed to connect to API: {}", e);
            anyhow::bail!("Failed to connect to API: {}", e)
        }
    }
}

/// Print the request payload to console with enhanced metadata
/// Includes curve information and shadow overtake display
pub fn print_legality_request(request: &LaneChangeLegalityRequest) {
    println!("\n============================================================");
    println!("🚗 LANE CHANGE LEGALITY CHECK REQUEST");
    println!("============================================================");
    println!("Event ID: {}", request.event_id);
    println!("Direction: {}", request.direction);
    println!(
        "Frames: {} -> {}",
        request.start_frame_id, request.end_frame_id
    );
    println!("Timestamp: {:.2}s", request.video_timestamp_ms / 1000.0);
    if let Some(duration) = request.duration_ms {
        println!("Duration: {:.0}ms", duration);
    }
    println!("Source: {}", request.source_id);

    println!("\n📊 Detection Quality:");
    println!(
        "  • Confidence:       {:.0}%",
        request.detection_metadata.detection_confidence * 100.0
    );
    println!(
        "  • Max offset:       {:.0}%",
        request.detection_metadata.max_offset_normalized * 100.0
    );
    println!(
        "  • Lane confidence:  {:.0}%",
        request.detection_metadata.avg_lane_confidence * 100.0
    );
    println!(
        "  • Both lanes ratio: {:.0}%",
        request.detection_metadata.both_lanes_ratio * 100.0
    );
    println!(
        "  • Resolution:       {}",
        request.detection_metadata.video_resolution
    );
    println!(
        "  • FPS:              {:.1}",
        request.detection_metadata.fps
    );
    if let Some(width) = request.detection_metadata.avg_lane_width_px {
        println!("  • Avg lane width:   {:.0}px", width);
    }

    // CURVE INFORMATION
    if request.detection_metadata.curve_detected {
        println!("\n🌀 Curve Detection:");
        println!(
            "  • Type:             {}",
            request.detection_metadata.curve_type
        );
        println!(
            "  • Angle:            {:.1}°",
            request.detection_metadata.curve_angle_degrees
        );
        println!(
            "  • Confidence:       {:.0}%",
            request.detection_metadata.curve_confidence * 100.0
        );

        // Warning if overtaking in curve
        if request.detection_metadata.curve_type != "NONE" {
            println!("  ⚠️  WARNING: Overtaking in curve is ILLEGAL in Peru (DS 016-2009-MTC)");
        }
    } else {
        println!("\n🌀 Curve Detection: No curve detected (straight road)");
    }

    // SHADOW OVERTAKE INFORMATION
    if request.detection_metadata.shadow_overtake_detected {
        println!("\n⚫ Shadow Overtake Detection:");
        println!(
            "  • Blocking vehicles: {}",
            request.detection_metadata.shadow_overtake_count
        );
        println!(
            "  • Worst severity:    {}",
            request.detection_metadata.shadow_worst_severity
        );
        for vehicle in &request.detection_metadata.shadow_blocking_vehicles {
            println!("  • Vehicle:           {}", vehicle);
        }
        println!(
            "  ⚠️  WARNING: Shadow overtake — visibility blocked by vehicle ahead in overtaking lane"
        );
        println!("  ⚠️  This is EXTREMELY DANGEROUS and ILLEGAL (DS 016-2009-MTC Art. 90)");
        println!(
            "  ⚠️  Driver cannot see oncoming traffic due to {} blocking vehicle(s)",
            request.detection_metadata.shadow_overtake_count
        );
    } else {
        println!("\n⚫ Shadow Overtake Detection: None (clear forward visibility)");
    }

    println!("\n🎬 Frames for analysis: {}", request.frames.len());
    for frame_data in &request.frames {
        print!(
            "  Frame {}: {}x{} @ {:.2}s | base64: {} chars",
            frame_data.frame_index,
            frame_data.width,
            frame_data.height,
            frame_data.timestamp_ms / 1000.0,
            frame_data.base64_image.len()
        );
        if let Some(conf) = frame_data.lane_confidence {
            print!(" | lane_conf: {:.0}%", conf * 100.0);
        }
        if let Some(offset) = frame_data.offset_percentage {
            print!(" | offset: {:.0}%", offset * 100.0);
        }
        println!();
    }
    println!("============================================================\n");
}

/// Save the request to a JSON file
pub fn save_legality_request_to_file(
    request: &LaneChangeLegalityRequest,
    output_dir: &str,
) -> Result<std::path::PathBuf, anyhow::Error> {
    let dir = Path::new(output_dir).join("legality_requests");
    std::fs::create_dir_all(&dir)?;

    let filename = format!("{}_legality_request.json", request.event_id);
    let filepath = dir.join(&filename);

    let json = serde_json::to_string_pretty(request)?;
    std::fs::write(&filepath, json)?;

    info!("💾 Saved legality request to: {}", filepath.display());
    Ok(filepath)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_key_frames_for_lane_change() {
        let frames: Vec<Frame> = (0..10)
            .map(|i| Frame {
                data: vec![],
                width: 1280,
                height: 720,
                timestamp_ms: i as f64 * 100.0,
            })
            .collect();
        let key_frames = extract_key_frames_for_lane_change(&frames, 5);
        assert_eq!(key_frames.len(), 5);

        assert_eq!(key_frames[0].timestamp_ms, 0.0);
        assert_eq!(key_frames[4].timestamp_ms, 900.0);
    }

    #[test]
    fn test_curve_type_conversion() {
        use crate::types::CurveType;

        let curve_types = vec![
            (CurveType::None, "NONE"),
            (CurveType::Moderate, "MODERATE"),
            (CurveType::Sharp, "SHARP"),
        ];

        for (curve_type, expected_str) in curve_types {
            let curve_str = match curve_type {
                CurveType::None => "NONE",
                CurveType::Moderate => "MODERATE",
                CurveType::Sharp => "SHARP",
            };
            assert_eq!(curve_str, expected_str);
        }
    }

    #[test]
    fn test_shadow_metadata_defaults_when_absent() {
        // When event has no shadow metadata, defaults should be safe
        use crate::types::{Direction, LaneChangeEvent};

        let event = LaneChangeEvent::new(1000.0, 10, 50, Direction::Left, 0.8);

        // Simulate what build_legality_request does
        let shadow_detected = event
            .metadata
            .get("shadow_overtake_detected")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let shadow_count = event
            .metadata
            .get("shadow_overtake_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        let shadow_severity = event
            .metadata
            .get("shadow_worst_severity")
            .and_then(|v| v.as_str())
            .unwrap_or("NONE")
            .to_string();

        let shadow_vehicles: Vec<String> = event
            .metadata
            .get("shadow_blocking_vehicles")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        assert!(!shadow_detected);
        assert_eq!(shadow_count, 0);
        assert_eq!(shadow_severity, "NONE");
        assert!(shadow_vehicles.is_empty());
    }

    #[test]
    fn test_shadow_metadata_present() {
        use crate::types::{Direction, LaneChangeEvent};

        let mut event = LaneChangeEvent::new(1000.0, 10, 50, Direction::Left, 0.8);

        // Simulate attaching shadow metadata
        event.metadata.insert(
            "shadow_overtake_detected".to_string(),
            serde_json::json!(true),
        );
        event
            .metadata
            .insert("shadow_overtake_count".to_string(), serde_json::json!(2));
        event.metadata.insert(
            "shadow_worst_severity".to_string(),
            serde_json::json!("CRITICAL"),
        );
        event.metadata.insert(
            "shadow_blocking_vehicles".to_string(),
            serde_json::json!(["car (ID #3)", "truck (ID #7)"]),
        );

        let shadow_detected = event
            .metadata
            .get("shadow_overtake_detected")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let shadow_count = event
            .metadata
            .get("shadow_overtake_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        let shadow_severity = event
            .metadata
            .get("shadow_worst_severity")
            .and_then(|v| v.as_str())
            .unwrap_or("NONE")
            .to_string();

        let shadow_vehicles: Vec<String> = event
            .metadata
            .get("shadow_blocking_vehicles")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        assert!(shadow_detected);
        assert_eq!(shadow_count, 2);
        assert_eq!(shadow_severity, "CRITICAL");
        assert_eq!(shadow_vehicles.len(), 2);
        assert_eq!(shadow_vehicles[0], "car (ID #3)");
        assert_eq!(shadow_vehicles[1], "truck (ID #7)");
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineCrossingInfo {
    /// Was a line crossed during the maneuver?
    pub line_crossed: bool,

    /// Type of line crossed (e.g., "solid_single_yellow", "dashed_single_white")
    pub line_type: String,

    /// Is crossing this line legal?
    pub is_legal: bool,

    /// Severity level
    pub severity: String, // "LEGAL", "ILLEGAL", "CRITICAL_ILLEGAL"

    /// Confidence of the line detection
    pub line_detection_confidence: f32,

    /// Frame ID where the line was crossed
    pub crossed_at_frame: u64,

    /// Multiple lines crossed?
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub additional_lines_crossed: Vec<String>,

    /// Legality analysis details from the segmentation model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub analysis_details: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryInfo {
    /// Initial normalized position (before maneuver)
    pub initial_position: f32,

    /// Final normalized position (after maneuver)
    pub final_position: f32,

    /// Net displacement (final - initial)
    pub net_displacement: f32,

    /// Did the vehicle return close to starting position?
    pub returned_to_start: bool,

    /// Was excursion sufficient for a real lane change?
    pub excursion_sufficient: bool,

    /// Trajectory shape score (0.0-1.0, higher = cleaner arc)
    pub shape_score: f32,

    /// Smoothness score (lower = smoother, < 0.25 is good)
    pub smoothness: f32,

    /// Was there a clear direction reversal?
    pub has_direction_reversal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VelocityInfo {
    /// Peak lateral velocity (px/s)
    pub peak_lateral_velocity: f32,

    /// Average lateral velocity during maneuver
    pub avg_lateral_velocity: f32,

    /// Was velocity sustained or erratic?
    pub velocity_pattern: String, // "sustained", "erratic", "spike"

    /// Maximum acceleration (change in velocity)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_acceleration: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositioningInfo {
    /// Lane width statistics
    pub lane_width_min: f32,
    pub lane_width_max: f32,
    pub lane_width_avg: f32,
    pub lane_width_stable: bool, // Did width vary significantly?

    /// How far into adjacent lane did vehicle go? (0.0 = just crossed, 1.0 = centered in new lane)
    pub adjacent_lane_penetration: f32,

    /// Baseline offset the system was tracking from
    pub baseline_offset: f32,

    /// Was baseline frozen during detection?
    pub baseline_frozen: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalInfo {
    /// Time spent in DRIFTING state (ms)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_drifting_ms: Option<f64>,

    /// Time spent in CROSSING state (ms)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_crossing_ms: Option<f64>,

    /// Total duration from first drift to completion
    pub total_maneuver_duration_ms: f64,

    /// Was duration within expected range for this type?
    pub duration_plausible: bool,

    /// State progression summary
    pub state_progression: Vec<StateTransition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    pub from_state: String,
    pub to_state: String,
    pub frame_id: u64,
    pub timestamp_ms: f64,
}
