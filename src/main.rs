mod config;
mod inference;
mod lane_detection;
mod overtake_detector;
mod preprocessing;
mod types;
mod video_processor;

use anyhow::Result;
use std::path::Path;
use tracing::{error, info};

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

    // Initialize overtake detector
    let mut overtake_detector = overtake_detector::OvertakeDetector::new(config.clone());

    // Results storage
    let mut overtakes = Vec::new();
    let mut frame_count = 0;

    // Process frames
    while let Some(frame) = reader.read_frame()? {
        frame_count += 1;

        // Show progress every 30 frames
        if frame_count % 30 == 0 {
            info!(
                "Progress: {:.1}% ({}/{})",
                reader.progress(),
                reader.current_frame,
                reader.total_frames
            );
        }

        // Process frame
        match process_frame(&frame, inference_engine, &mut overtake_detector, config).await {
            Ok(result) => {
                // Save overtake event
                if let Some(overtake) = result.overtake {
                    overtakes.push(overtake.clone());
                    info!(
                        "🎯 Overtake #{} detected at {:.2}s",
                        overtakes.len(),
                        overtake.end_timestamp
                    );
                }

                // Draw lanes on frame and write to output video
                if let Some(ref mut w) = writer {
                    if let Ok(annotated) = video_processor::draw_lanes(
                        &frame.data,
                        reader.width,
                        reader.height,
                        &result.lanes,
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

    // Save results to JSON
    save_results(video_path, &overtakes, config)?;

    Ok(ProcessingStats {
        total_frames: frame_count,
        overtakes_detected: overtakes.len(),
        duration_secs: duration.as_secs_f64(),
        avg_fps,
    })
}

struct FrameResult {
    lanes: Vec<types::Lane>,
    overtake: Option<types::OvertakeEvent>,
}

async fn process_frame(
    frame: &types::Frame,
    inference_engine: &mut inference::InferenceEngine,
    overtake_detector: &mut overtake_detector::OvertakeDetector,
    config: &types::Config,
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

    // 4. Find vehicle position and check for overtake
    let overtake = if let Some((lane_idx, lateral_offset)) =
        lane_detection::find_vehicle_lane(&lane_detection.lanes, frame.width as f32)
    {
        overtake_detector.update(lane_idx as i32, lateral_offset, frame.timestamp)
    } else {
        None
    };

    Ok(FrameResult {
        lanes: lane_detection.lanes,
        overtake,
    })
}

fn save_results(
    video_path: &Path,
    overtakes: &[types::OvertakeEvent],
    config: &types::Config,
) -> Result<()> {
    use serde_json;
    use std::fs::File;
    use std::io::Write;

    let video_name = video_path.file_stem().unwrap().to_str().unwrap();
    let output_path =
        Path::new(&config.video.output_dir).join(format!("{}_results.json", video_name));

    let json = serde_json::to_string_pretty(overtakes)?;
    let mut file = File::create(&output_path)?;
    file.write_all(json.as_bytes())?;

    info!("Results saved to: {}", output_path.display());
    Ok(())
}
