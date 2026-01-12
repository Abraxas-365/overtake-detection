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
use tracing::{debug, error, info, warn};
use types::{DetectedLane, Lane, LaneChangeConfig, LaneChangeEvent};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("overtake_detection=debug,ort=warn") // Changed to debug level
        .init();

    info!("🚗 Lane Change Detection System Starting");

    let config = types::Config::load("config.yaml")?;
    info!("✓ Configuration loaded");

    // Log important config values
    info!("  Model: {}", config.model.path);
    info!("  griding_num: {}", config.model.griding_num);
    info!("  num_anchors: {}", config.model.num_anchors);
    info!("  num_lanes: {}", config.model.num_lanes);
    info!(
        "  min_lane_confidence: {}",
        config.detection.min_lane_confidence
    );
    info!(
        "  confidence_threshold: {}",
        config.detection.confidence_threshold
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
                    "  Frames with valid position: {}",
                    stats.frames_with_position
                );
                info!("  Lane changes detected: {}", stats.lane_changes_detected);
                info!("  Duration: {:.1}s", stats.duration_secs);
                info!("  Average FPS: {:.1}", stats.avg_fps);
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
        min_frames_confirm: 3,
        cooldown_frames: 30,
        smoothing_alpha: 0.3,
        reference_y_ratio: 0.8,
    };

    let mut analyzer = LaneChangeAnalyzer::new(lane_change_config);
    analyzer.set_source_id(video_path.to_string_lossy().to_string());

    let mut lane_changes: Vec<LaneChangeEvent> = Vec::new();
    let mut frame_count: u64 = 0;
    let mut frames_with_valid_position: u64 = 0;

    // Statistics for debugging
    let mut total_lanes_detected: u64 = 0;
    let mut frames_with_lanes: u64 = 0;
    let mut frames_with_left_lane: u64 = 0;
    let mut frames_with_right_lane: u64 = 0;
    let mut frames_with_both_lanes: u64 = 0;

    while let Some(frame) = reader.read_frame()? {
        frame_count += 1;
        let timestamp_ms = frame.timestamp_ms;
        let vehicle_x = frame.width as f32 / 2.0;

        match process_frame(&frame, inference_engine, config).await {
            Ok(detected_lanes) => {
                // 🔍 DEBUG: Log lane detection details every 30 frames
                if frame_count % 30 == 0 || frame_count <= 5 {
                    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    info!(
                        "Frame {}: Detected {} raw lanes",
                        frame_count,
                        detected_lanes.len()
                    );

                    let mut has_left = false;
                    let mut has_right = false;

                    for (i, lane) in detected_lanes.iter().enumerate() {
                        if !lane.points.is_empty() {
                            let avg_x: f32 = lane.points.iter().map(|p| p.0).sum::<f32>()
                                / lane.points.len() as f32;
                            let min_x: f32 = lane
                                .points
                                .iter()
                                .map(|p| p.0)
                                .fold(f32::INFINITY, f32::min);
                            let max_x: f32 = lane
                                .points
                                .iter()
                                .map(|p| p.0)
                                .fold(f32::NEG_INFINITY, f32::max);
                            let side = if avg_x < vehicle_x { "LEFT" } else { "RIGHT" };

                            if avg_x < vehicle_x {
                                has_left = true;
                            } else {
                                has_right = true;
                            }

                            info!(
                                "  Lane {}: {} pts | conf={:.3} | x=[{:.0}..{:.0}] avg={:.0} | {}",
                                i,
                                lane.points.len(),
                                lane.confidence,
                                min_x,
                                max_x,
                                avg_x,
                                side
                            );
                        } else {
                            info!("  Lane {}: EMPTY (0 points)", i);
                        }
                    }

                    info!("  Vehicle center X: {:.0}", vehicle_x);
                    info!(
                        "  Has LEFT lane: {} | Has RIGHT lane: {}",
                        has_left, has_right
                    );

                    if has_left && has_right {
                        info!("  ✓ Can calculate lane width");
                    } else {
                        warn!("  ✗ Cannot calculate lane width - need lanes on BOTH sides");
                    }
                }

                // Update statistics
                if !detected_lanes.is_empty() {
                    frames_with_lanes += 1;
                    total_lanes_detected += detected_lanes.len() as u64;

                    let has_left = detected_lanes.iter().any(|l| {
                        if l.points.is_empty() {
                            return false;
                        }
                        let avg_x: f32 =
                            l.points.iter().map(|p| p.0).sum::<f32>() / l.points.len() as f32;
                        avg_x < vehicle_x
                    });
                    let has_right = detected_lanes.iter().any(|l| {
                        if l.points.is_empty() {
                            return false;
                        }
                        let avg_x: f32 =
                            l.points.iter().map(|p| p.0).sum::<f32>() / l.points.len() as f32;
                        avg_x >= vehicle_x
                    });

                    if has_left {
                        frames_with_left_lane += 1;
                    }
                    if has_right {
                        frames_with_right_lane += 1;
                    }
                    if has_left && has_right {
                        frames_with_both_lanes += 1;
                    }
                }

                // Convert to analysis lanes
                let analysis_lanes: Vec<Lane> = detected_lanes
                    .iter()
                    .enumerate()
                    .map(|(i, dl)| Lane::from_detected(i, dl))
                    .collect();

                // Run analysis
                if let Some(event) = analyzer.analyze(
                    &analysis_lanes,
                    frame.width as u32,
                    frame.height as u32,
                    frame_count,
                    timestamp_ms,
                ) {
                    lane_changes.push(event.clone());
                    info!(
                        "🚀 LANE CHANGE DETECTED: {} at {:.2}s (frame {})",
                        event.direction_name(),
                        event.video_timestamp_ms / 1000.0,
                        event.frame_id
                    );
                }

                // Debug log vehicle state every 30 frames
                if frame_count % 30 == 0 {
                    if let Some(vs) = analyzer.last_vehicle_state() {
                        if vs.is_valid() {
                            let normalized = vs.normalized_offset().unwrap_or(0.0);
                            let width = vs.lane_width.unwrap_or(0.0);
                            info!(
                                "  VehicleState: offset={:.1}px ({:.1}%) | width={:.0}px | state={}",
                                vs.lateral_offset,
                                normalized * 100.0,
                                width,
                                analyzer.current_state()
                            );
                            frames_with_valid_position += 1;
                        } else {
                            warn!(
                                "  VehicleState: INVALID (no lane width) | state={}",
                                analyzer.current_state()
                            );
                        }
                    } else {
                        warn!("  VehicleState: None");
                    }
                    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                }

                if analyzer
                    .last_vehicle_state()
                    .map_or(false, |s| s.is_valid())
                {
                    frames_with_valid_position += 1;
                }

                // Write annotated video
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

        // Progress indicator every 100 frames
        if frame_count % 100 == 0 {
            let progress = (frame_count as f32 / reader.total_frames as f32) * 100.0;
            info!(
                "Progress: {:.1}% ({}/{})",
                progress, frame_count, reader.total_frames
            );
        }
    }

    let duration = start_time.elapsed();
    let avg_fps = frame_count as f64 / duration.as_secs_f64();

    // Print detailed statistics
    info!("\n📊 Detection Statistics:");
    info!("  Total frames processed: {}", frame_count);
    info!(
        "  Frames with any lanes: {} ({:.1}%)",
        frames_with_lanes,
        (frames_with_lanes as f64 / frame_count as f64) * 100.0
    );
    info!(
        "  Frames with LEFT lane: {} ({:.1}%)",
        frames_with_left_lane,
        (frames_with_left_lane as f64 / frame_count as f64) * 100.0
    );
    info!(
        "  Frames with RIGHT lane: {} ({:.1}%)",
        frames_with_right_lane,
        (frames_with_right_lane as f64 / frame_count as f64) * 100.0
    );
    info!(
        "  Frames with BOTH lanes: {} ({:.1}%)",
        frames_with_both_lanes,
        (frames_with_both_lanes as f64 / frame_count as f64) * 100.0
    );
    info!(
        "  Average lanes per frame: {:.2}",
        total_lanes_detected as f64 / frame_count as f64
    );
    info!(
        "  Frames with valid position: {} ({:.1}%)",
        frames_with_valid_position,
        (frames_with_valid_position as f64 / frame_count as f64) * 100.0
    );

    info!("\n📊 Lane Change Report:");
    info!("  Total Lane Changes: {}", lane_changes.len());

    for (i, event) in lane_changes.iter().enumerate() {
        info!(
            "  {}. {} at {:.2}s (frame {})",
            i + 1,
            event.direction_name(),
            event.video_timestamp_ms / 1000.0,
            event.frame_id
        );
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

    // Filter by confidence
    let high_confidence_lanes: Vec<DetectedLane> = lane_detection
        .lanes
        .into_iter()
        .filter(|lane| {
            let dominated = lane.confidence > config.detection.min_lane_confidence;
            if !dominated && lane.points.len() >= config.detection.min_points_per_lane {
                debug!(
                    "Lane filtered: {} points, conf={:.3} < threshold {:.3}",
                    lane.points.len(),
                    lane.confidence,
                    config.detection.min_lane_confidence
                );
            }
            passed
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
