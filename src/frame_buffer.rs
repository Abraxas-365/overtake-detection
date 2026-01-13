// src/frame_buffer.rs

use crate::types::{Frame, LaneChangeEvent};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{error, info};

/// Buffer to capture frames during a lane change event
pub struct LaneChangeFrameBuffer {
    frames: Vec<Frame>,
    max_frames: usize,
    is_capturing: bool,
    capture_start_frame_id: Option<u64>,
}

impl LaneChangeFrameBuffer {
    pub fn new(max_frames: usize) -> Self {
        Self {
            frames: Vec::with_capacity(max_frames),
            max_frames,
            is_capturing: false,
            capture_start_frame_id: None,
        }
    }

    pub fn start_capture(&mut self, frame_id: u64) {
        self.frames.clear();
        self.is_capturing = true;
        self.capture_start_frame_id = Some(frame_id);
        info!("📹 Started capturing frames at frame {}", frame_id);
    }

    pub fn add_frame(&mut self, frame: Frame) {
        if self.is_capturing && self.frames.len() < self.max_frames {
            self.frames.push(frame);
        }
    }

    pub fn stop_capture(&mut self) -> Vec<Frame> {
        self.is_capturing = false;
        self.capture_start_frame_id = None;
        let frames = std::mem::take(&mut self.frames);
        info!("📹 Stopped capturing. Total frames: {}", frames.len());
        frames
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
}

// ============================================================================
// ENHANCED API STRUCTURES WITH DETECTION METADATA
// ============================================================================

/// Detection quality metadata to help AI make better decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionMetadata {
    /// Confidence of the lane change detection (0.0-1.0)
    pub detection_confidence: f32,
    /// Maximum lateral offset reached during maneuver (normalized, 0.0-1.0)
    pub max_offset_normalized: f32,
    /// Average lane detection confidence across frames (0.0-1.0)
    pub avg_lane_confidence: f32,
    /// Percentage of frames where both lanes were detected (0.0-1.0)
    pub both_lanes_ratio: f32,
    /// Video resolution (e.g., "1280x720")
    pub video_resolution: String,
    /// Frames per second
    pub fps: f32,
    /// Country/region for traffic rules (e.g., "PE" for Peru)
    pub region: String,
    /// Average lane width in pixels during the maneuver
    pub avg_lane_width_px: Option<f32>,
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
    /// NEW: Detection quality metadata
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
    /// NEW: Lane detection confidence for this frame (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lane_confidence: Option<f32>,
    /// NEW: Lateral offset as percentage of lane width (if available)
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

/// Extract evenly spaced key frames
pub fn extract_key_frames_for_lane_change(frames: &[Frame], count: usize) -> Vec<&Frame> {
    if frames.is_empty() || count == 0 {
        return vec![];
    }

    if frames.len() <= count {
        return frames.iter().collect();
    }

    let mut selected_indices = Vec::with_capacity(count);

    // Strategy: More frames from START and MIDDLE of maneuver
    // This helps AI see the "before" context

    match count {
        1 => {
            // Just middle frame
            selected_indices.push(frames.len() / 2);
        }
        2 => {
            // Start and end
            selected_indices.push(0);
            selected_indices.push(frames.len() - 1);
        }
        3 => {
            // Start, middle, end
            selected_indices.push(0);
            selected_indices.push(frames.len() / 2);
            selected_indices.push(frames.len() - 1);
        }
        4 => {
            // Start, early-middle, late-middle, end
            selected_indices.push(0);
            selected_indices.push(frames.len() / 3);
            selected_indices.push((frames.len() * 2) / 3);
            selected_indices.push(frames.len() - 1);
        }
        5 => {
            // Start, early-mid, middle, late-mid, end
            // Bias toward earlier frames to show context
            selected_indices.push(0);
            selected_indices.push(frames.len() / 4);
            selected_indices.push(frames.len() / 2);
            selected_indices.push((frames.len() * 3) / 4);
            selected_indices.push(frames.len() - 1);
        }
        _ => {
            // For more frames, distribute evenly
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
    let encoder = JpegEncoder::new_with_quality(&mut buffer, 85);
    encoder.encode(
        img.as_raw(),
        img.width(),
        img.height(),
        image::ColorType::Rgb8,
    )?;

    Ok(STANDARD.encode(buffer.into_inner()))
}

/// Build the API request payload with enhanced metadata
pub fn build_legality_request(
    event: &LaneChangeEvent,
    captured_frames: &[Frame],
    num_frames_to_send: usize,
) -> Result<LaneChangeLegalityRequest, anyhow::Error> {
    let key_frames = extract_key_frames(captured_frames, num_frames_to_send);

    if key_frames.is_empty() {
        anyhow::bail!("No frames to send");
    }

    let mut frame_data_list = Vec::with_capacity(key_frames.len());

    // Process each frame
    for (i, frame) in key_frames.iter().enumerate() {
        let base64_image = frame_to_base64(frame)?;

        // TODO: Extract per-frame metadata if available in Frame struct
        // For now, these are None - you can enhance Frame struct to include these
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

    // Calculate video metadata
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
            25.0 // Default fallback
        }
    } else {
        25.0 // Default fallback
    };

    // Extract max_offset from event metadata if available
    let max_offset_normalized = event
        .metadata
        .get("max_offset_normalized")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(0.5); // Default if not available

    // Extract both_lanes_ratio from event metadata if available
    let both_lanes_ratio = event
        .metadata
        .get("both_lanes_ratio")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(0.7); // Assume decent ratio if not tracked

    // Extract average lane confidence from event metadata if available
    let avg_lane_confidence = event
        .metadata
        .get("avg_lane_confidence")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(0.6); // Default moderate confidence

    // Extract average lane width if available
    let avg_lane_width_px = event
        .metadata
        .get("avg_lane_width_px")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32);

    // Build detection metadata
    let metadata = DetectionMetadata {
        detection_confidence: event.confidence,
        max_offset_normalized,
        avg_lane_confidence,
        both_lanes_ratio,
        video_resolution,
        fps,
        region: "PE".to_string(), // Peru
        avg_lane_width_px,
    };

    info!(
        "📊 Detection quality: conf={:.1}%, max_offset={:.1}%, lanes={:.0}%",
        metadata.detection_confidence * 100.0,
        metadata.max_offset_normalized * 100.0,
        metadata.both_lanes_ratio * 100.0
    );

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
        .timeout(std::time::Duration::from_secs(60)) // Increased timeout
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

    // Print detection metadata
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
    fn test_extract_key_frames() {
        let frames: Vec<Frame> = (0..10)
            .map(|i| Frame {
                data: vec![],
                width: 1280,
                height: 720,
                timestamp_ms: i as f64 * 100.0,
            })
            .collect();

        let key_frames = extract_key_frames(&frames, 5);
        assert_eq!(key_frames.len(), 5);

        // Should include first, last, and evenly distributed middle frames
        assert_eq!(key_frames[0].timestamp_ms, 0.0);
        assert_eq!(key_frames[4].timestamp_ms, 900.0);
    }

    #[test]
    fn test_extract_key_frames_fewer_than_requested() {
        let frames: Vec<Frame> = (0..3)
            .map(|i| Frame {
                data: vec![],
                width: 1280,
                height: 720,
                timestamp_ms: i as f64 * 100.0,
            })
            .collect();

        let key_frames = extract_key_frames(&frames, 5);
        assert_eq!(key_frames.len(), 3); // Should return all available frames
    }
}
