use crate::types::{Direction, Frame, LaneChangeEvent};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::info;

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

/// Payload for the legality analysis API
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameData {
    pub frame_index: usize,
    pub timestamp_ms: f64,
    pub width: usize,
    pub height: usize,
    pub base64_image: String,
}

/// Extract evenly spaced key frames
pub fn extract_key_frames(frames: &[Frame], count: usize) -> Vec<&Frame> {
    if frames.is_empty() || count == 0 {
        return vec![];
    }

    if frames.len() <= count {
        return frames.iter().collect();
    }

    let step = (frames.len() - 1) as f32 / (count - 1) as f32;

    (0..count)
        .map(|i| {
            let index = (i as f32 * step).round() as usize;
            &frames[index.min(frames.len() - 1)]
        })
        .collect()
}

/// Convert frame to base64 JPEG
pub fn frame_to_base64(frame: &Frame) -> Result<String, anyhow::Error> {
    use image::{ImageBuffer, Rgb};
    use std::io::Cursor;

    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(frame.width as u32, frame.height as u32, frame.data.clone())
            .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;

    let mut buffer = Cursor::new(Vec::new());
    img.write_to(&mut buffer, image::ImageFormat::Jpeg)?;

    Ok(STANDARD.encode(buffer.into_inner()))
}

/// Build the API request payload
pub fn build_legality_request(
    event: &LaneChangeEvent,
    captured_frames: &[Frame],
    num_frames_to_send: usize,
) -> Result<LaneChangeLegalityRequest, anyhow::Error> {
    let key_frames = extract_key_frames(captured_frames, num_frames_to_send);

    let mut frame_data_list = Vec::with_capacity(key_frames.len());

    for (i, frame) in key_frames.iter().enumerate() {
        let base64_image = frame_to_base64(frame)?;

        frame_data_list.push(FrameData {
            frame_index: i,
            timestamp_ms: frame.timestamp_ms,
            width: frame.width,
            height: frame.height,
            base64_image,
        });
    }

    Ok(LaneChangeLegalityRequest {
        event_id: event.event_id.clone(),
        direction: event.direction_name().to_string(),
        start_frame_id: event.start_frame_id,
        end_frame_id: event.end_frame_id,
        video_timestamp_ms: event.video_timestamp_ms,
        duration_ms: event.duration_ms,
        source_id: event.source_id.clone(),
        frames: frame_data_list,
    })
}

/// Print the request payload to console
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
    println!("Number of frames for analysis: {}", request.frames.len());
    println!("\n--- FRAME DATA ---");

    for frame_data in &request.frames {
        println!(
            "Frame {}: {}x{} @ {:.2}s | base64 length: {} chars",
            frame_data.frame_index,
            frame_data.width,
            frame_data.height,
            frame_data.timestamp_ms / 1000.0,
            frame_data.base64_image.len()
        );

        // Print first 100 chars of base64 for verification
        let preview: String = frame_data.base64_image.chars().take(100).collect();
        println!("  Preview: {}...", preview);
    }

    println!("\n--- FULL JSON PAYLOAD ---");
    match serde_json::to_string_pretty(&request) {
        Ok(json) => {
            if json.len() > 5000 {
                println!(
                    "{}...\n[Truncated - total {} bytes]",
                    &json[..5000],
                    json.len()
                );
            } else {
                println!("{}", json);
            }
        }
        Err(e) => println!("Error serializing to JSON: {}", e),
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
