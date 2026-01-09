// src/main.rs

mod config;
mod inference;
mod lane_detection;
mod overtake_detector;
mod preprocessing;
mod smoother;
mod types;
mod video_processor; // ← NEW MODULE

use anyhow::Result;
use std::path::Path;
use tracing::{debug, error, info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("overtake_detection=info,ort=warn")
        .init();

    info!("🚗 Overtake Detection System Starting");

    // Load configuration
    let config = types::Config::load("config.yaml")?;
    info!("✓ Configuration loaded");
    info!(
        "  Smoother window: {} frames",
        config.detection.smoother_window_size
    );
    info!(
        "  Calibration frames: {}",
        config.detection.calibration_frames
    );
    info!("  Debounce frames: {}", config.detection.debounce_frames);
    info!("  Confirm frames: {}", config.detection.confirm_frames);

    // Initialize inference engine
    let mut inference_engine = inference::InferenceEngine::new(config.clone())?;
    info!("✓ Inference engine ready");

    // Initialize video processor
    let video_processor = video_processor::VideoProcessor::new(config.clone());

    // Find all video files
    let video_files = video_processor.find_video_files()?;

    if video_files.is_empty() {
        error!("No video files found in {}", config.video.input_dir);
        return Ok(());
    }

    // Process each video
    for (idx, video_path) in video_files.iter().enumerate() {
        info!("\n========================================");
        info!(
            "Processing video {}/{}: {}",
            idx + 1,
            video_files.len(),
            video_path.display()
        );
        info!("========================================\n");

        match process_video(video_path, &mut inference_engine, &video_processor, &config).await {
            Ok(stats) => {
                info!("\n✓ Video processed successfully!");
                info!("  Total frames: {}", stats.total_frames);
                info!(
                    "  Frames with valid position: {} ({:.1}%)",
                    stats.frames_with_position,
                    (stats.frames_with_position as f32 / stats.total_frames as f32) * 100.0
                );
                info!("  Lane changes detected: {}", stats.lane_changes_detected);
                info!("  Overtakes detected: {}", stats.overtakes_detected);
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
    total_frames: i32,
    frames_with_position: i32,
    lane_changes_detected: usize,
    overtakes_detected: usize,
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

    // Open video
    let mut reader = video_processor.open_video(video_path)?;

    // Create video writer for annotated output
    let mut writer =
        video_processor.create_writer(video_path, reader.width, reader.height, reader.fps)?;

    // Initialize components with smoother! ⭐
    let mut smoother = smoother::LanePositionSmoother::new(config.detection.smoother_window_size);
    let mut overtake_detector = overtake_detector::OvertakeDetector::new(config.clone());

    // Results storage
    let mut overtakes = Vec::new();
    let mut lane_changes = Vec::new();
    let mut frame_count = 0;

    // Track stats for summary
    let mut frames_with_lanes = 0;
    let mut frames_with_valid_position = 0;
    let mut calibration_complete = false;

    info!(
        "🔄 Starting calibration phase ({} frames)...",
        config.detection.calibration_frames
    );

    // Process frames
    while let Some(frame) = reader.read_frame()? {
        frame_count += 1;

        // Show progress
        if frame_count % 30 == 0 {
            let progress_msg = if calibration_complete {
                format!(
                    "Progress: {:.1}% ({}/{}) - Valid pos: {}/{} ({:.1}%) - Lane changes: {} - Overtakes: {}",
                    reader.progress(),
                    reader.current_frame,
                    reader.total_frames,
                    frames_with_valid_position,
                    frame_count,
                    (frames_with_valid_position as f32 / frame_count as f32) * 100.0,
                    lane_changes.len(),
                    overtakes.len()
                )
            } else {
                format!(
                    "🔄 Calibrating: {}/{} frames",
                    frame_count, config.detection.calibration_frames
                )
            };
            info!("{}", progress_msg);
        }

        // Process frame with smoother ⭐
        match process_frame(
            &frame,
            inference_engine,
            &mut smoother,
            &mut overtake_detector,
            config,
            frame_count,
            calibration_complete,
        )
        .await
        {
            Ok(result) => {
                // Update stats
                if !result.lanes.is_empty() {
                    frames_with_lanes += 1;
                }

                if result.had_valid_position {
                    frames_with_valid_position += 1;
                }

                // Check if calibration just completed
                if !calibration_complete && result.calibration_complete {
                    calibration_complete = true;
                    info!(
                        "✅ Calibration complete! Baseline lane: {}",
                        result.baseline_lane.unwrap_or(-1)
                    );
                }

                // Save lane change event
                if let Some(lane_change) = result.lane_change {
                    lane_changes.push(lane_change.clone());
                    info!(
                        "🔄 Lane change #{}: {:?} (lane {} → {}) at {:.2}s",
                        lane_changes.len(),
                        lane_change.direction,
                        lane_change.from_lane,
                        lane_change.to_lane,
                        lane_change.timestamp
                    );
                }

                // Save overtake event
                if let Some(overtake) = result.overtake {
                    overtakes.push(overtake.clone());
                    info!(
                        "🎯 OVERTAKE #{} detected at {:.2}s",
                        overtakes.len(),
                        overtake.end_timestamp
                    );
                    info!(
                        "   Direction: {:?} → {:?}",
                        overtake.first_direction, overtake.second_direction
                    );
                    info!(
                        "   Lanes: {} → {} → {}",
                        overtake.start_lane,
                        if overtake.first_direction == types::Direction::Right {
                            overtake.start_lane + 1
                        } else {
                            overtake.start_lane - 1
                        },
                        overtake.end_lane
                    );
                    info!(
                        "   Duration: {:.2}s",
                        overtake.end_timestamp - overtake.start_timestamp
                    );
                    info!(
                        "   Complete: {} | Confidence: {:.2}",
                        overtake.is_complete, overtake.confidence
                    );
                }

                // Draw lanes on frame and write to output video
                if let Some(ref mut w) = writer {
                    if let Ok(annotated) = video_processor::draw_lanes_with_info(
                        &frame.data,
                        reader.width,
                        reader.height,
                        &result.lanes,
                        result.smoothed_position.as_ref(),
                        calibration_complete,
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

    // Print detailed summary
    info!("\n📊 Processing Summary:");
    info!(
        "  Frames with lanes: {}/{} ({:.1}%)",
        frames_with_lanes,
        frame_count,
        (frames_with_lanes as f32 / frame_count as f32) * 100.0
    );
    info!(
        "  Frames with valid position: {}/{} ({:.1}%)",
        frames_with_valid_position,
        frame_count,
        (frames_with_valid_position as f32 / frame_count as f32) * 100.0
    );
    info!("  Lane changes detected: {}", lane_changes.len());
    info!("  Overtakes detected: {}", overtakes.len());

    if !overtakes.is_empty() {
        let complete_overtakes = overtakes.iter().filter(|o| o.is_complete).count();
        info!("    └─ Complete overtakes: {}", complete_overtakes);
        info!(
            "    └─ Incomplete: {}",
            overtakes.len() - complete_overtakes
        );
    }

    // Save results to JSON
    save_results(video_path, &overtakes, &lane_changes, config)?;

    Ok(ProcessingStats {
        total_frames: frame_count,
        frames_with_position: frames_with_valid_position,
        lane_changes_detected: lane_changes.len(),
        overtakes_detected: overtakes.len(),
        duration_secs: duration.as_secs_f64(),
        avg_fps,
    })
}

struct FrameResult {
    lanes: Vec<types::Lane>,
    raw_position: Option<types::VehiclePosition>,
    smoothed_position: Option<types::VehiclePosition>,
    lane_change: Option<types::LaneChangeEvent>,
    overtake: Option<types::OvertakeEvent>,
    had_valid_position: bool,
    calibration_complete: bool,
    baseline_lane: Option<i32>,
}

async fn process_frame(
    frame: &types::Frame,
    inference_engine: &mut inference::InferenceEngine,
    smoother: &mut smoother::LanePositionSmoother,
    overtake_detector: &mut overtake_detector::OvertakeDetector,
    config: &types::Config,
    frame_count: i32,
    calibration_complete: bool,
) -> Result<FrameResult> {
    // 1. Preprocess
    let preprocessed = preprocessing::preprocess(
        &frame.data,
        frame.width,
        frame.height,
        config.model.input_width,
        config.model.input_height,
    )?;

    // 2. Run inference
    let output = inference_engine.infer(&preprocessed)?;

    // 3. Parse lanes
    let lane_detection = lane_detection::parse_lanes(
        &output,
        frame.width as f32,
        frame.height as f32,
        config,
        frame.timestamp,
    )?;

    // Filter lanes by confidence ⭐
    let high_confidence_lanes: Vec<types::Lane> = lane_detection
        .lanes
        .into_iter()
        .filter(|lane| lane.confidence > config.detection.min_lane_confidence)
        .collect();

    // Log lane detection every 30 frames
    if frame_count % 30 == 0 {
        debug!(
            "Frame {}: {}/{} lanes above confidence threshold",
            frame_count,
            high_confidence_lanes.len(),
            lane_detection.lanes.len()
        );
    }

    // 4. Calculate raw vehicle position
    let raw_position = if let Some((lane_idx, lateral_offset, confidence)) =
        lane_detection::find_vehicle_lane_with_confidence(
            &high_confidence_lanes,
            frame.width as f32,
        ) {
        Some(types::VehiclePosition {
            lane_index: lane_idx as i32,
            lateral_offset,
            confidence,
            timestamp: frame.timestamp,
        })
    } else {
        None
    };

    // 5. Smooth position ⭐ (CRITICAL STEP!)
    let smoothed_position = if let Some(raw_pos) = raw_position {
        // Only process if confidence is high enough
        if raw_pos.confidence >= config.detection.min_position_confidence {
            Some(smoother.smooth(raw_pos))
        } else {
            if frame_count % 30 == 0 {
                debug!(
                    "Frame {}: Skipping low confidence position ({:.2})",
                    frame_count, raw_pos.confidence
                );
            }
            None
        }
    } else {
        None
    };

    // 6. Update overtake detector with smoothed position ⭐
    let mut lane_change = None;
    let mut overtake = None;
    let mut new_calibration_complete = calibration_complete;
    let mut baseline_lane = None;

    if let Some(smooth_pos) = smoothed_position {
        // Update detector
        let result = overtake_detector.update_with_position(smooth_pos);

        // Check calibration status
        if !calibration_complete && overtake_detector.is_calibrated() {
            new_calibration_complete = true;
            baseline_lane = overtake_detector.get_baseline_lane();
        }

        // Extract events
        lane_change = result.lane_change;
        overtake = result.overtake;

        // Debug logging
        if frame_count % 30 == 0 {
            debug!(
                "Frame {}: Lane={}, Offset={:.2}, Conf={:.2}",
                frame_count,
                smooth_pos.lane_index,
                smooth_pos.lateral_offset,
                smooth_pos.confidence
            );
        }
    }

    Ok(FrameResult {
        lanes: high_confidence_lanes,
        raw_position,
        smoothed_position,
        lane_change,
        overtake,
        had_valid_position: smoothed_position.is_some(),
        calibration_complete: new_calibration_complete,
        baseline_lane,
    })
}

fn save_results(
    video_path: &Path,
    overtakes: &[types::OvertakeEvent],
    lane_changes: &[types::LaneChangeEvent],
    config: &types::Config,
) -> Result<()> {
    use serde_json;
    use std::fs::File;
    use std::io::Write;

    let video_name = video_path.file_stem().unwrap().to_str().unwrap();

    // Save overtakes
    let overtakes_path =
        Path::new(&config.video.output_dir).join(format!("{}_overtakes.json", video_name));
    let json = serde_json::to_string_pretty(overtakes)?;
    let mut file = File::create(&overtakes_path)?;
    file.write_all(json.as_bytes())?;
    info!("💾 Overtakes saved to: {}", overtakes_path.display());

    // Save lane changes
    let lane_changes_path =
        Path::new(&config.video.output_dir).join(format!("{}_lane_changes.json", video_name));
    let json = serde_json::to_string_pretty(lane_changes)?;
    let mut file = File::create(&lane_changes_path)?;
    file.write_all(json.as_bytes())?;
    info!("💾 Lane changes saved to: {}", lane_changes_path.display());

    // Save summary
    let summary = serde_json::json!({
        "video": video_name,
        "total_lane_changes": lane_changes.len(),
        "total_overtakes": overtakes.len(),
        "complete_overtakes": overtakes.iter().filter(|o| o.is_complete).count(),
        "overtakes": overtakes,
        "lane_changes": lane_changes,
    });

    let summary_path =
        Path::new(&config.video.output_dir).join(format!("{}_summary.json", video_name));
    let json = serde_json::to_string_pretty(&summary)?;
    let mut file = File::create(&summary_path)?;
    file.write_all(json.as_bytes())?;
    info!("💾 Summary saved to: {}", summary_path.display());

    Ok(())
}
