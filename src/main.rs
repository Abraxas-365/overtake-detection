// src/main.rs

mod analysis;
mod frame_buffer;
mod inference;
mod lane_detection;
mod preprocessing;
mod types;
mod video_processor;

use analysis::LaneChangeAnalyzer;
use anyhow::Result;
use frame_buffer::{
    build_legality_request, print_legality_request, save_legality_request_to_file,
    send_to_legality_api, LaneChangeFrameBuffer,
};
use std::path::Path;
use tracing::{debug, error, info, warn};
use types::{DetectedLane, Frame, Lane, LaneChangeConfig, LaneChangeEvent};

/// Configuration for legality analysis
struct LegalityAnalysisConfig {
    /// Number of frames to extract and send for analysis
    num_frames_to_analyze: usize,
    /// Maximum frames to buffer during lane change
    max_buffer_frames: usize,
    /// Whether to save the request payload to a file
    save_to_file: bool,
    /// Whether to print the request to console
    print_to_console: bool,
    /// Whether to send to the API
    send_to_api: bool,
    /// API URL for legality analysis
    api_url: String,
}

impl Default for LegalityAnalysisConfig {
    fn default() -> Self {
        Self {
            num_frames_to_analyze: 7,
            max_buffer_frames: 90,
            save_to_file: false,
            print_to_console: true,
            send_to_api: true,
            api_url: "http://localhost:3000/api/analyze".to_string(),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("overtake_detection=info,ort=warn")
        .init();

    info!("🚗 Lane Change Detection System Starting");

    let config = types::Config::load("config.yaml")?;
    info!("✓ Configuration loaded");

    // Log key detection parameters
    info!(
        "Detection thresholds: drift={:.2}, crossing={:.2}, confirm_frames={}",
        config.detection.drift_threshold,
        config.detection.crossing_threshold,
        config.detection.confirm_frames
    );

    let mut inference_engine = inference::InferenceEngine::new(config.clone())?;
    info!("✓ Inference engine ready");

    let video_processor = video_processor::VideoProcessor::new(config.clone());

    let video_files = video_processor.find_video_files()?;

    if video_files.is_empty() {
        error!("No video files found in {}", config.video.input_dir);
        return Ok(());
    }

    info!("Found {} video file(s) to process", video_files.len());

    // Legality analysis configuration
    let legality_config = LegalityAnalysisConfig {
        num_frames_to_analyze: 5,
        max_buffer_frames: 90,
        save_to_file: false,
        print_to_console: true,
        send_to_api: true,
        api_url: std::env::var("LEGALITY_API_URL")
            .unwrap_or_else(|_| "http://localhost:3000/api/analyze".to_string()),
    };

    info!("📡 Legality API URL: {}", legality_config.api_url);
    info!("🎨 Contrast enhancement: ENABLED (CLAHE + Lane Boost)");

    for (idx, video_path) in video_files.iter().enumerate() {
        info!("\n========================================");
        info!(
            "Processing video {}/{}: {}",
            idx + 1,
            video_files.len(),
            video_path.display()
        );
        info!("========================================\n");

        match process_video(
            video_path,
            &mut inference_engine,
            &video_processor,
            &config,
            &legality_config,
        )
        .await
        {
            Ok(stats) => {
                info!("\n✓ Video processed successfully!");
                info!("  Total frames: {}", stats.total_frames);
                info!(
                    "  Valid position frames: {} ({:.1}%)",
                    stats.frames_with_position,
                    100.0 * stats.frames_with_position as f64 / stats.total_frames as f64
                );
                info!("  Lane changes detected: {}", stats.lane_changes_detected);
                info!("  Events sent to API: {}", stats.events_sent_to_api);
            }
            Err(e) => {
                error!("Failed to process video: {}", e);
            }
        }
    }

    Ok(())
}

struct ProcessingStats {
    total_frames: u64,
    frames_with_position: u64,
    lane_changes_detected: usize,
    events_sent_to_api: usize,
    #[allow(dead_code)]
    duration_secs: f64,
    #[allow(dead_code)]
    avg_fps: f64,
}

async fn process_video(
    video_path: &Path,
    inference_engine: &mut inference::InferenceEngine,
    video_processor: &video_processor::VideoProcessor,
    config: &types::Config,
    legality_config: &LegalityAnalysisConfig,
) -> Result<ProcessingStats> {
    use std::time::Instant;

    let start_time = Instant::now();

    let mut reader = video_processor.open_video(video_path)?;

    let mut writer =
        video_processor.create_writer(video_path, reader.width, reader.height, reader.fps)?;

    // *** USE CONFIG VALUES INSTEAD OF HARDCODED ***
    let lane_change_config = LaneChangeConfig::from_detection_config(&config.detection);

    info!(
        "Lane change config: drift={:.2}, crossing={:.2}, confirm={}, cooldown={}, hysteresis={:.2}",
        lane_change_config.drift_threshold,
        lane_change_config.crossing_threshold,
        lane_change_config.min_frames_confirm,
        lane_change_config.cooldown_frames,
        lane_change_config.hysteresis_factor
    );

    let mut analyzer = LaneChangeAnalyzer::new(lane_change_config);
    analyzer.set_source_id(video_path.to_string_lossy().to_string());

    let mut lane_changes: Vec<LaneChangeEvent> = Vec::new();
    let mut frame_count: u64 = 0;
    let mut frames_with_valid_position: u64 = 0;
    let mut events_sent_to_api: usize = 0;

    let mut previous_state = "CENTERED".to_string();

    // Frame buffer for capturing lane change frames
    let mut frame_buffer = LaneChangeFrameBuffer::new(legality_config.max_buffer_frames);

    while let Some(frame) = reader.read_frame()? {
        frame_count += 1;
        let timestamp_ms = frame.timestamp_ms;

        if frame_count % 50 == 0 {
            info!(
            "Progress: {:.1}% ({}/{}) | State: {} | Lane changes: {} | Buffered: {} | Pre-buffered: {}",
            reader.progress(),
            reader.current_frame,
            reader.total_frames,
            analyzer.current_state(),
            lane_changes.len(),
            frame_buffer.frame_count(),
            frame_buffer.pre_buffer_count()
        );
        }

        // 🆕 Use enhanced preprocessing for better lane detection
        match process_frame_enhanced(
            &frame,
            inference_engine,
            config,
            config.detection.min_lane_confidence,
        )
        .await
        {
            Ok(detected_lanes) => {
                let analysis_lanes: Vec<Lane> = detected_lanes
                    .iter()
                    .enumerate()
                    .map(|(i, dl)| Lane::from_detected(i, dl))
                    .collect();

                // ✅ IMPORTANTE: Agregar al pre-buffer ANTES de analizar (usa previous_state)
                if previous_state == "CENTERED" {
                    frame_buffer.add_to_pre_buffer(frame.clone());
                }

                // Check if lane change completed
                if let Some(mut event) = analyzer.analyze(
                    &analysis_lanes,
                    frame.width as u32,
                    frame.height as u32,
                    frame_count,
                    timestamp_ms,
                ) {
                    info!(
                        "🚀 LANE CHANGE DETECTED: {} at {:.2}s (frame {})",
                        event.direction_name(),
                        event.video_timestamp_ms / 1000.0,
                        event.end_frame_id
                    );

                    // Get captured frames (includes pre-buffer)
                    let captured_frames = frame_buffer.stop_capture();

                    info!(
                        "📹 Captured {} frames total (includes pre-buffer context)",
                        captured_frames.len()
                    );

                    // Build and send legality request
                    if !captured_frames.is_empty() {
                        match build_legality_request(
                            &event,
                            &captured_frames,
                            legality_config.num_frames_to_analyze,
                        ) {
                            Ok(request) => {
                                if legality_config.print_to_console {
                                    print_legality_request(&request);
                                }

                                if legality_config.save_to_file {
                                    if let Err(e) = save_legality_request_to_file(
                                        &request,
                                        &config.video.output_dir,
                                    ) {
                                        warn!("Failed to save legality request: {}", e);
                                    }
                                }

                                if legality_config.send_to_api {
                                    match send_to_legality_api(&request, &legality_config.api_url)
                                        .await
                                    {
                                        Ok(response) => {
                                            info!(
                                                "✅ Event {} sent to API: {} - {}",
                                                response.event_id,
                                                response.status,
                                                response.message
                                            );
                                            events_sent_to_api += 1;
                                        }
                                        Err(e) => {
                                            error!("❌ Failed to send event to API: {}", e);
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("Failed to build legality request: {}", e);
                            }
                        }
                    } else {
                        warn!("No frames captured for lane change event");
                    }

                    // Save evidence images (use first captured frame as start)
                    let video_stem = video_path.file_stem().unwrap().to_str().unwrap();
                    let start_filename =
                        format!("{}_event_{}_start.jpg", video_stem, event.event_id);
                    let end_filename = format!("{}_event_{}_end.jpg", video_stem, event.event_id);

                    let mut start_path_str = String::new();
                    let mut end_path_str = String::new();

                    // Use first captured frame (from pre-buffer) as start
                    if !captured_frames.is_empty() {
                        if let Ok(path) =
                            video_processor.save_frame_to_disk(&captured_frames[0], &start_filename)
                        {
                            start_path_str = path.to_string_lossy().to_string();
                        }
                    }

                    if let Ok(path) = video_processor.save_frame_to_disk(&frame, &end_filename) {
                        end_path_str = path.to_string_lossy().to_string();
                    }

                    event.evidence_images = Some(types::EvidencePaths {
                        start_image_path: start_path_str,
                        end_image_path: end_path_str,
                    });

                    lane_changes.push(event);
                }

                // ✅ Obtener current_state DESPUÉS del análisis
                let current_state = analyzer.current_state().to_string();

                // Start capturing when CENTERED -> DRIFTING
                if previous_state == "CENTERED" && current_state == "DRIFTING" {
                    frame_buffer.start_capture(frame_count);
                    debug!(
                        "📸 Started capturing at frame {} (with pre-buffer)",
                        frame_count
                    );
                }

                // Continue capturing during lane change
                if frame_buffer.is_capturing() {
                    frame_buffer.add_frame(frame.clone());
                }

                // Cancel if returned to CENTERED without completing
                if current_state == "CENTERED" && frame_buffer.is_capturing() {
                    frame_buffer.cancel_capture();
                    debug!("❌ Lane change cancelled");
                }

                // ✅ Actualizar previous_state al final
                previous_state = current_state;

                if frame_count % 50 == 0 {
                    if let Some(vs) = analyzer.last_vehicle_state() {
                        if vs.is_valid() {
                            let normalized = vs.normalized_offset().unwrap_or(0.0);
                            let width = vs.lane_width.unwrap_or(0.0);
                            if normalized.abs() > 0.1 {
                                info!(
                                "Frame {}: State={} | Offset: {:.1}px ({:.1}%) | Width: {:.0}px",
                                frame_count,
                                analyzer.current_state(),
                                vs.lateral_offset,
                                normalized * 100.0,
                                width
                            );
                            }
                        }
                    }
                }

                if analyzer
                    .last_vehicle_state()
                    .map_or(false, |s| s.is_valid())
                {
                    frames_with_valid_position += 1;
                }

                if let Some(ref mut w) = writer {
                    if let Ok(annotated) = video_processor::draw_lanes_with_state(
                        &frame.data,
                        reader.width,
                        reader.height,
                        &detected_lanes,
                        analyzer.current_state(),
                        analyzer.last_vehicle_state(),
                    ) {
                        use opencv::videoio::VideoWriterTrait;
                        w.write(&annotated)?;
                    }
                }
            }
            Err(e) => error!("Frame {} failed: {}", frame_count, e),
        }
    }

    let duration = start_time.elapsed();
    let avg_fps = frame_count as f64 / duration.as_secs_f64();

    info!("\n📊 Final Report:");
    info!("  Total Lane Changes: {}", lane_changes.len());
    info!("  Events Sent to API: {}", events_sent_to_api);
    info!("  Processing Speed: {:.1} FPS", avg_fps);

    for (i, event) in lane_changes.iter().enumerate() {
        info!(
            "  {}. {} at {:.2}s (confidence: {:.2})",
            i + 1,
            event.direction_name(),
            event.video_timestamp_ms / 1000.0,
            event.confidence
        );
    }

    save_results(video_path, &lane_changes, config)?;

    Ok(ProcessingStats {
        total_frames: frame_count,
        frames_with_position: frames_with_valid_position,
        lane_changes_detected: lane_changes.len(),
        events_sent_to_api,
        duration_secs: duration.as_secs_f64(),
        avg_fps,
    })
}

/// 🆕 Process frame WITH contrast enhancement for better lane detection
async fn process_frame_enhanced(
    frame: &Frame,
    inference_engine: &mut inference::InferenceEngine,
    config: &types::Config,
    confidence_threshold: f32,
) -> Result<Vec<DetectedLane>> {
    // Use the enhanced preprocessing with CLAHE and lane marking boost
    let preprocessed = preprocessing::preprocess_with_enhancement(
        &frame.data,
        frame.width,
        frame.height,
        config.model.input_width,
        config.model.input_height,
    )?;

    let output = inference_engine.infer(&preprocessed)?;

    let lane_detection = lane_detection::parse_lanes(
        &output,
        frame.width as f32,
        frame.height as f32,
        config,
        frame.timestamp_ms,
    )?;

    // Use config threshold instead of hardcoded
    let high_confidence_lanes: Vec<DetectedLane> = lane_detection
        .lanes
        .into_iter()
        .filter(|lane| {
            lane.confidence > confidence_threshold
                && lane.points.len() >= config.detection.min_points_per_lane
        })
        .collect();

    Ok(high_confidence_lanes)
}

/// Original process frame WITHOUT enhancement (kept for reference/comparison)
#[allow(dead_code)]
async fn process_frame(
    frame: &Frame,
    inference_engine: &mut inference::InferenceEngine,
    config: &types::Config,
    confidence_threshold: f32,
) -> Result<Vec<DetectedLane>> {
    let preprocessed = preprocessing::preprocess(
        &frame.data,
        frame.width,
        frame.height,
        config.model.input_width,
        config.model.input_height,
    )?;

    let output = inference_engine.infer(&preprocessed)?;

    let lane_detection = lane_detection::parse_lanes(
        &output,
        frame.width as f32,
        frame.height as f32,
        config,
        frame.timestamp_ms,
    )?;

    // Use config threshold instead of hardcoded
    let high_confidence_lanes: Vec<DetectedLane> = lane_detection
        .lanes
        .into_iter()
        .filter(|lane| {
            lane.confidence > confidence_threshold
                && lane.points.len() >= config.detection.min_points_per_lane
        })
        .collect();

    Ok(high_confidence_lanes)
}

fn save_results(
    video_path: &Path,
    lane_changes: &[LaneChangeEvent],
    config: &types::Config,
) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    std::fs::create_dir_all(&config.video.output_dir)?;
    let video_name = video_path.file_stem().unwrap().to_str().unwrap();
    let jsonl_path =
        Path::new(&config.video.output_dir).join(format!("{}_lane_changes.jsonl", video_name));

    let mut file = File::create(&jsonl_path)?;
    for event in lane_changes {
        let json_line = serde_json::to_string(&event.to_json())?;
        writeln!(file, "{}", json_line)?;
    }
    info!("💾 Saved to: {}", jsonl_path.display());
    Ok(())
}
