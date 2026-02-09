// src/frame_buffer.rs

use crate::types::{CurveInfo, CurveType, Frame, LaneChangeEvent};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, VecDeque};
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
        // INCREASED pre-buffer to 60 frames (2 seconds)
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

    /// Stop capturing and return frames + the start_frame_id of the buffer
    /// Returns: (captured_frames, buffer_start_frame_id)
    pub fn stop_capture(&mut self) -> (Vec<Frame>, u64) {
        let start_id = self.capture_start_frame_id.unwrap_or(0);
        let effective_start = start_id.saturating_sub(self.pre_buffer_size as u64);

        self.is_capturing = false;
        self.capture_start_frame_id = None;
        let frames = std::mem::take(&mut self.frames);

        info!("📹 Stopped capturing. Total frames: {}", frames.len());
        (frames, effective_start)
    }

    /// Force return frames, including pre-buffer if main buffer is empty.
    /// Returns: (captured_frames, buffer_start_frame_id)
    pub fn force_flush(&mut self) -> (Vec<Frame>, u64) {
        let start_id = self.capture_start_frame_id.unwrap_or(0);
        let effective_start = start_id.saturating_sub(self.pre_buffer_size as u64);

        self.is_capturing = false;
        self.capture_start_frame_id = None;

        let mut result = std::mem::take(&mut self.frames);

        if result.is_empty() && !self.pre_buffer.is_empty() {
            info!(
                "📹 Active buffer empty, using {} pre-buffered frames for evidence",
                self.pre_buffer.len()
            );
            for f in &self.pre_buffer {
                result.push(f.clone());
            }
        } else {
            info!("📹 Flushed capture. Total frames: {}", result.len());
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
// API STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionMetadata {
    pub detection_confidence: f32,
    pub max_offset_normalized: f32,
    pub avg_lane_confidence: f32,
    pub both_lanes_ratio: f32,
    pub video_resolution: String,
    pub fps: f32,
    pub region: String,
    pub avg_lane_width_px: Option<f32>,

    pub curve_detected: bool,
    pub curve_angle_degrees: f32,
    pub curve_confidence: f32,
    pub curve_type: String,

    pub shadow_overtake_detected: bool,
    pub shadow_overtake_count: u32,
    pub shadow_worst_severity: String,
    pub shadow_blocking_vehicles: Vec<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub line_crossing_info: Option<LineCrossingInfo>,

    pub vehicles_overtaken_count: u32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub overtaken_vehicle_types: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub overtaken_vehicle_ids: Vec<u32>,

    pub maneuver_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incomplete_reason: Option<String>,

    pub trajectory_info: TrajectoryInfo,
    pub velocity_info: VelocityInfo,
    pub positioning_info: PositioningInfo,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub detection_path: Option<String>,

    pub temporal_info: TemporalInfo,
}

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
    pub detection_metadata: DetectionMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameData {
    pub frame_index: usize,
    pub timestamp_ms: f64,
    pub width: usize,
    pub height: usize,
    pub base64_image: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lane_confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset_percentage: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneChangeLegalityResponse {
    pub event_id: String,
    pub status: String,
    pub message: String,
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Convert frame to base64 JPEG with quality optimization
pub fn frame_to_base64(frame: &Frame) -> Result<String, anyhow::Error> {
    use image::{codecs::jpeg::JpegEncoder, ImageBuffer, Rgb};
    use std::io::Cursor;

    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(frame.width as u32, frame.height as u32, frame.data.clone())
            .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;

    let mut buffer = Cursor::new(Vec::new());
    let mut encoder = JpegEncoder::new_with_quality(&mut buffer, 85);
    encoder.encode(
        img.as_raw(),
        img.width(),
        img.height(),
        image::ExtendedColorType::Rgb8,
    )?;

    Ok(STANDARD.encode(buffer.into_inner()))
}

/// SMART Frame Extraction
pub fn extract_smart_frames(
    frames: &[Frame],
    count: usize,
    critical_index: Option<usize>,
) -> Vec<&Frame> {
    if frames.is_empty() || count == 0 {
        return vec![];
    }
    if frames.len() <= count {
        return frames.iter().collect();
    }

    let last_idx = frames.len() - 1;
    let mut indices = BTreeSet::new();

    indices.insert(0);
    indices.insert(last_idx);

    let focus = critical_index
        .unwrap_or(frames.len() / 2)
        .clamp(0, last_idx);
    indices.insert(focus);

    while indices.len() < count {
        let mut best_candidate = 0;
        let mut max_dist = 0;

        let sorted_indices: Vec<usize> = indices.iter().cloned().collect();
        for window in sorted_indices.windows(2) {
            let start = window[0];
            let end = window[1];
            let gap = end - start;

            if gap > 1 {
                let mid = start + gap / 2;
                let dist_to_focus = (mid as isize - focus as isize).abs();
                let score = gap as isize * 100 - dist_to_focus;

                if score > max_dist {
                    max_dist = score;
                    best_candidate = mid;
                }
            }
        }

        if max_dist == 0 {
            break;
        }
        indices.insert(best_candidate);
    }

    indices.into_iter().map(|i| &frames[i]).collect()
}

/// Build the API request payload with enhanced metadata
pub fn build_legality_request(
    event: &LaneChangeEvent,
    captured_frames: &[Frame],
    num_frames_to_send: usize,
    curve_info: CurveInfo,
    critical_frame_index: Option<usize>,
) -> Result<LaneChangeLegalityRequest, anyhow::Error> {
    let key_frames =
        extract_smart_frames(captured_frames, num_frames_to_send, critical_frame_index);

    if key_frames.is_empty() {
        anyhow::bail!("No frames to send");
    }

    let mut frame_data_list = Vec::with_capacity(key_frames.len());

    let selected_indices: Vec<usize> = key_frames
        .iter()
        .filter_map(|kf| captured_frames.iter().position(|f| std::ptr::eq(*kf, f)))
        .collect();

    info!(
        "📸 Smart Select: {} frames (Target: {:?}) -> indices {:?}",
        key_frames.len(),
        critical_frame_index,
        selected_indices
    );

    for (i, frame) in key_frames.iter().enumerate() {
        let base64_image = frame_to_base64(frame)?;
        frame_data_list.push(FrameData {
            frame_index: i,
            timestamp_ms: frame.timestamp_ms,
            width: frame.width,
            height: frame.height,
            base64_image,
            lane_confidence: None,
            offset_percentage: None,
        });
    }

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

    let metadata = DetectionMetadata {
        detection_confidence: event.confidence,
        max_offset_normalized: event
            .metadata
            .get("max_offset_normalized")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.5),
        avg_lane_confidence: event
            .metadata
            .get("avg_lane_confidence")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.6),
        both_lanes_ratio: event
            .metadata
            .get("both_lanes_ratio")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.7),
        video_resolution: format!(
            "{}x{}",
            key_frames.first().unwrap().width,
            key_frames.first().unwrap().height
        ),
        fps,
        region: "PE".to_string(),
        avg_lane_width_px: event
            .metadata
            .get("avg_lane_width_px")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32),

        curve_detected: curve_info.is_curve,
        curve_angle_degrees: curve_info.angle_degrees,
        curve_confidence: curve_info.confidence,
        curve_type: match curve_info.curve_type {
            CurveType::None => "NONE",
            CurveType::Moderate => "MODERATE",
            CurveType::Sharp => "SHARP",
        }
        .to_string(),

        shadow_overtake_detected: event
            .metadata
            .get("shadow_overtake_detected")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        shadow_overtake_count: event
            .metadata
            .get("shadow_overtake_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32,
        shadow_worst_severity: event
            .metadata
            .get("shadow_worst_severity")
            .and_then(|v| v.as_str())
            .unwrap_or("NONE")
            .to_string(),
        shadow_blocking_vehicles: event
            .metadata
            .get("shadow_blocking_vehicles")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default(),

        line_crossing_info: event.legality.as_ref().map(|legality| LineCrossingInfo {
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
            additional_lines_crossed: vec![],
            analysis_details: legality.analysis_details.clone(),
        }),

        vehicles_overtaken_count: event
            .metadata
            .get("vehicles_overtaken")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32,
        overtaken_vehicle_types: event
            .metadata
            .get("overtaken_vehicle_types")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default(),
        overtaken_vehicle_ids: event
            .metadata
            .get("overtaken_vehicle_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as u32))
                    .collect()
            })
            .unwrap_or_default(),

        maneuver_type: event
            .metadata
            .get("maneuver_type")
            .and_then(|v| v.as_str())
            .unwrap_or("simple_lane_change")
            .to_string(),
        incomplete_reason: event
            .metadata
            .get("incomplete_reason")
            .and_then(|v| v.as_str())
            .map(String::from),

        trajectory_info: TrajectoryInfo {
            initial_position: event
                .metadata
                .get("initial_position")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32,
            final_position: event
                .metadata
                .get("final_position")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32,
            net_displacement: event
                .metadata
                .get("net_displacement")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32,
            returned_to_start: false,
            excursion_sufficient: true,
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
        },

        velocity_info: VelocityInfo {
            peak_lateral_velocity: event
                .metadata
                .get("peak_lateral_velocity")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32,
            avg_lateral_velocity: 0.0,
            velocity_pattern: "unknown".to_string(),
            max_acceleration: None,
        },

        positioning_info: PositioningInfo {
            lane_width_min: 0.0,
            lane_width_max: 0.0,
            lane_width_avg: 0.0,
            lane_width_stable: true,
            adjacent_lane_penetration: 0.0,
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
        },

        detection_path: event
            .metadata
            .get("detection_path")
            .and_then(|v| v.as_str())
            .map(String::from),

        temporal_info: TemporalInfo {
            time_drifting_ms: None,
            time_crossing_ms: None,
            total_maneuver_duration_ms: event.duration_ms.unwrap_or(0.0),
            duration_plausible: true,
            state_progression: vec![],
        },
    };

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

/// Print the request payload to console
pub fn print_legality_request(request: &LaneChangeLegalityRequest) {
    println!("\n============================================================");
    println!("🚗 LANE CHANGE LEGALITY CHECK REQUEST");
    println!("============================================================");
    println!("Event ID: {}", request.event_id);
    println!("Direction: {}", request.direction);
    println!(
        "Confidence: {:.0}%",
        request.detection_metadata.detection_confidence * 100.0
    );
    println!("Frames: {}", request.frames.len());
    println!("============================================================\n");
}

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

// ─── HELPER STRUCTS FOR SERIALIZATION ───

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineCrossingInfo {
    pub line_crossed: bool,
    pub line_type: String,
    pub is_legal: bool,
    pub severity: String,
    pub line_detection_confidence: f32,
    pub crossed_at_frame: u64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub additional_lines_crossed: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub analysis_details: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryInfo {
    pub initial_position: f32,
    pub final_position: f32,
    pub net_displacement: f32,
    pub returned_to_start: bool,
    pub excursion_sufficient: bool,
    pub shape_score: f32,
    pub smoothness: f32,
    pub has_direction_reversal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VelocityInfo {
    pub peak_lateral_velocity: f32,
    pub avg_lateral_velocity: f32,
    pub velocity_pattern: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_acceleration: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositioningInfo {
    pub lane_width_min: f32,
    pub lane_width_max: f32,
    pub lane_width_avg: f32,
    pub lane_width_stable: bool,
    pub adjacent_lane_penetration: f32,
    pub baseline_offset: f32,
    pub baseline_frozen: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_drifting_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_crossing_ms: Option<f64>,
    pub total_maneuver_duration_ms: f64,
    pub duration_plausible: bool,
    pub state_progression: Vec<StateTransition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    pub from_state: String,
    pub to_state: String,
    pub frame_id: u64,
    pub timestamp_ms: f64,
}

