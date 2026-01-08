mod config;
mod inference;
mod lane_detection;
mod overtake_detector;
mod preprocessing;
mod types;

use anyhow::Result;
use tracing::{error, info, warn};
use types::Frame;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("overtake_detection=debug,ort=info")
        .init();

    info!("🚗 Overtake Detection System Starting");

    // Load configuration
    let config = types::Config::load("config.yaml")?;
    info!("✓ Configuration loaded");

    // Initialize inference engine
    let inference_engine = inference::InferenceEngine::new(config.clone())?;
    info!("✓ Inference engine ready");

    // Initialize overtake detector
    let mut overtake_detector = overtake_detector::OvertakeDetector::new(config.clone());
    info!("✓ Overtake detector ready");

    info!("🎬 Starting main loop");

    // Demo: Process a test frame
    // Replace this with your actual video pipeline
    let frame = create_test_frame(
        config.video.source_width,
        config.video.source_height,
    );

    match process_frame(&frame, &inference_engine, &mut overtake_detector, &config).await {
        Ok(()) => info!("✓ Frame processed successfully"),
        Err(e) => error!("Failed to process frame: {}", e),
    }

    info!("System ready. In production, connect to video source here.");
    
    // TODO: Add your video pipeline here
    // Example:
    // let (frame_tx, mut frame_rx) = tokio::sync::mpsc::channel(10);
    // tokio::spawn(video_capture_loop(frame_tx));
    // while let Some(frame) = frame_rx.recv().await {
    //     process_frame(&frame, &inference_engine, &mut overtake_detector, &config).await?;
    // }

    Ok(())
}

async fn process_frame(
    frame: &Frame,
    inference_engine: &inference::InferenceEngine,
    overtake_detector: &mut overtake_detector::OvertakeDetector,
    config: &types::Config,
) -> Result<()> {
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

    info!("Detected {} lanes", lane_detection.lanes.len());

    // 4. Find vehicle position
    if let Some((lane_idx, lateral_offset)) =
        lane_detection::find_vehicle_lane(&lane_detection.lanes, frame.width as f32)
    {
        info!(
            "Vehicle in lane {}, offset: {:.2}",
            lane_idx, lateral_offset
        );

        // 5. Check for overtake
        if let Some(overtake) = overtake_detector.update(
            lane_idx as i32,
            lateral_offset,
            frame.timestamp,
        ) {
            info!("🎯 Overtake event: {:?}", overtake);
            
            // TODO: Handle overtake event
            // - Extract video segment
            // - Upload to cloud
            // - Send notification
        }
    } else {
        warn!("Could not determine vehicle position");
    }

    Ok(())
}

fn create_test_frame(width: usize, height: usize) -> Frame {
    // Create a test frame with gradient pattern
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = (x * 255 / width) as u8;
            data[idx + 1] = (y * 255 / height) as u8;
            data[idx + 2] = 128;
        }
    }

    Frame {
        data,
        width,
        height,
        timestamp: 0.0,
    }
}

