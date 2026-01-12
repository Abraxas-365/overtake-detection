// src/main.rs

mod analysis;
mod inference;
mod lane_detection;
mod preprocessing;
mod types;
mod video_processor;

use analysis::LaneChangeAnalyzer;
use anyhow::Result;
use std::path::Path;
use tracing::{debug, error, info};
use types::{DetectedLane, Lane, LaneChangeConfig, LaneChangeEvent};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("overtake_detection=info,ort=warn")
        .init();

    info!("🚗 Lane Change Detection System Starting");

    let config = types::Config::load("config.yaml")?;
    info!("✓ Configuration loaded");

    let mut inference_engine = inference::InferenceEngine::new(config.clone())?;
    info!("✓ Inference engine ready");

    let video_processor = video_processor::VideoProcessor::new(config.clone());

    let video_files = video_processor.find_video_files()?;

    if video_files.is_empty() {
        error!("No video files found in {}", config.video.input_dir);
        return Ok(());
    }

    info!("Found {} video file(s) to process", video_files.len());

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
            &video_path,
            &mut inference_engine,
            &video_processor,
            &config,
        )
        .await
        {
            Ok(stats) => {
                info!("\n✓ Video processed successfully!");
                info!("  Total frames: {}", stats.total_frames);
                info!(
                    "  Frames with valid position: {} ({:.1}%)",
                    stats.frames_with_position,
                    (stats.frames_with_position as f32 / stats.total_frames as f32) * 100.0
                );
                info!("  Lane changes detected: {}", stats.lane_changes_detected);
                info!("  Processing time: {:.2}s", stats.duration_secs);
                info!("  Average FPS: {:.2}", stats.avg_fps);
            }
            Err(e) => {
                error!("Failed to process video: {}", e);
            }
        }
    }

    info!("\n🎉 All videos processed!");
    Ok(())
}

struct ProcessingStats {
    total_frames: u64,
    frames_with_position: u64,
    lane_changes_detected: usize,
    duration_secs: f64,
    avg_fps: f64,
}

async fn process_video(
    video_path: &Path,
    inference_engine: &mut inference::InferenceEngine,
    video_processor: &video_processor::VideoProcessor,
    config: &types::Config,
) -> Result<ProcessingStats> {
    use std::time::Instant;

    let start_time = Instant::now();

    let mut reader = video_processor.open_video(video_path)?;

    let mut writer =
        video_processor.create_writer(video_path, reader.width, reader.height, reader.fps)?;

    let lane_change_config = LaneChangeConfig {
        drift_threshold: 0.2,
        crossing_threshold: 0.4,
        min_frames_confirm: config.detection.confirm_frames,
        cooldown_frames: 30,
        smoothing_alpha: 0.3,
        reference_y_ratio: 0.8,
    };

    let mut analyzer = LaneChangeAnalyzer::new(lane_change_config);
    analyzer.set_source_id(video_path.to_string_lossy().to_string());

    info!("📊 Lane Change Analyzer Configuration:");
    info!(
        "   Drift threshold: {:.0}% of lane width",
        analyzer.config().drift_threshold * 100.0
    );
    info!(
        "   Crossing threshold: {:.0}% of lane width",
        analyzer.config().crossing_threshold * 100.0
    );
    info!(
        "   Min frames to confirm: {}",
        analyzer.config().min_frames_confirm
    );
    info!("   Cooldown frames: {}", analyzer.config().cooldown_frames);

    let mut lane_changes: Vec<LaneChangeEvent> = Vec::new();
    let mut frame_count: u64 = 0;
    let mut frames_with_valid_position: u64 = 0;

    while let Some(frame) = reader.read_frame()? {
        frame_count += 1;
        let timestamp_ms = frame.timestamp_ms;

        if frame_count % 30 == 0 {
            info!(
                "Progress: {:.1}% ({}/{}) | State: {} | Lane changes: {}",
                reader.progress(),
                reader.current_frame,
                reader.total_frames,
                analyzer.current_state(),
                lane_changes.len()
            );
        }

        match process_frame(&frame, inference_engine, config).await {
            Ok(detected_lanes) => {
                // Convert DetectedLane to Lane for analysis
                let analysis_lanes: Vec<Lane> = detected_lanes
                    .iter()
                    .enumerate()
                    .map(|(i, dl)| Lane::from_detected(i, dl))
                    .collect();

                // Run lane change analysis
                if let Some(event) = analyzer.analyze(
                    &analysis_lanes,
                    frame.width as u32,
                    frame.height as u32,
                    frame_count,
                    timestamp_ms,
                ) {
                    lane_changes.push(event.clone());
                    info!(
                        "🔄 LANE CHANGE #{}: {} at {:.2}s (frame {}) - Duration: {:.0}ms",
                        lane_changes.len(),
                        event.direction_name(),
                        event.video_timestamp_ms / 1000.0,
                        event.frame_id,
                        event.duration_ms.unwrap_or(0.0)
                    );
                }

                // Track valid positions
                if analyzer
                    .last_vehicle_state()
                    .map_or(false, |s| s.is_valid())
                {
                    frames_with_valid_position += 1;
                }

                // Debug logging
                if frame_count % 30 == 0 {
                    if let Some(vs) = analyzer.last_vehicle_state() {
                        if vs.is_valid() {
                            let normalized = vs.normalized_offset().unwrap_or(0.0);
                            debug!(
                                "Frame {}: State={}, Offset={:.1}px ({:.1}%), LaneWidth={:.0}px",
                                frame_count,
                                analyzer.current_state(),
                                vs.lateral_offset,
                                normalized * 100.0,
                                vs.lane_width.unwrap_or(0.0)
                            );
                        }
                    }
                }

                // Write annotated frame
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
            Err(e) => {
                error!("Frame {} processing failed: {}", frame_count, e);
            }
        }
    }

    let duration = start_time.elapsed();
    let avg_fps = frame_count as f64 / duration.as_secs_f64();

    info!("\n📊 Processing Summary:");
    info!(
        "  Frames with valid position: {}/{} ({:.1}%)",
        frames_with_valid_position,
        frame_count,
        (frames_with_valid_position as f32 / frame_count as f32) * 100.0
    );
    info!("  Lane changes detected: {}", lane_changes.len());

    if !lane_changes.is_empty() {
        info!("\n  📋 Lane change events:");
        for (i, event) in lane_changes.iter().enumerate() {
            info!(
                "    {}. {} at {:.2}s (frame {}) - duration: {:.0}ms",
                i + 1,
                event.direction_name(),
                event.video_timestamp_ms / 1000.0,
                event.frame_id,
                event.duration_ms.unwrap_or(0.0)
            );
        }
    }

    save_results(video_path, &lane_changes, config)?;

    Ok(ProcessingStats {
        total_frames: frame_count,
        frames_with_position: frames_with_valid_position,
        lane_changes_detected: lane_changes.len(),
        duration_secs: duration.as_secs_f64(),
        avg_fps,
    })
}

async fn process_frame(
    frame: &types::Frame,
    inference_engine: &mut inference::InferenceEngine,
    config: &types::Config,
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

    let high_confidence_lanes: Vec<DetectedLane> = lane_detection
        .lanes
        .into_iter()
        .filter(|lane| lane.confidence > config.detection.min_lane_confidence)
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

    // Save as JSON Lines
    let jsonl_path =
        Path::new(&config.video.output_dir).join(format!("{}_lane_changes.jsonl", video_name));
    let mut file = File::create(&jsonl_path)?;
    for event in lane_changes {
        let json_line = serde_json::to_string(&event.to_json())?;
        writeln!(file, "{}", json_line)?;
    }
    info!("💾 Lane changes saved to: {}", jsonl_path.display());

    // Save as pretty JSON
    let json_path =
        Path::new(&config.video.output_dir).join(format!("{}_lane_changes.json", video_name));
    let events_json: Vec<serde_json::Value> = lane_changes.iter().map(|e| e.to_json()).collect();
    let json = serde_json::to_string_pretty(&events_json)?;
    let mut file = File::create(&json_path)?;
    file.write_all(json.as_bytes())?;

    // Save summary
    let summary = serde_json::json!({
        "video": video_name,
        "total_lane_changes": lane_changes.len(),
        "events": lane_changes.iter().map(|e| e.to_json()).collect::<Vec<_>>(),
    });

    let summary_path =
        Path::new(&config.video.output_dir).join(format!("{}_summary.json", video_name));
    let json = serde_json::to_string_pretty(&summary)?;
    let mut file = File::create(&summary_path)?;
    file.write_all(json.as_bytes())?;
    info!("💾 Summary saved to: {}", summary_path.display());

    Ok(())
}
