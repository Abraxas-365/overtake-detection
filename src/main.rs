// src/main.rs

mod analysis;
mod frame_buffer;
mod inference;
mod lane_detection;
mod overtake_analyzer;
mod overtake_tracker;
mod preprocessing;
mod shadow_overtake; // 🆕
mod types;
mod vehicle_detection;
mod video_processor;

use analysis::LaneChangeAnalyzer;
use anyhow::Result;
use frame_buffer::{
    build_legality_request, print_legality_request, save_legality_request_to_file,
    send_to_legality_api, LaneChangeFrameBuffer,
};
use overtake_analyzer::OvertakeAnalyzer;
use overtake_tracker::{OvertakeResult, OvertakeTracker};
use shadow_overtake::{ShadowOvertakeDetector, ShadowOvertakeEvent}; // 🆕
use std::path::Path;
use tracing::{debug, error, info, warn};
use types::{CurveInfo, DetectedLane, Direction, Frame, Lane, LaneChangeConfig, LaneChangeEvent};
use vehicle_detection::YoloDetector;

/// Configuration for legality analysis
struct LegalityAnalysisConfig {
    num_frames_to_analyze: usize,
    max_buffer_frames: usize,
    save_to_file: bool,
    print_to_console: bool,
    send_to_api: bool,
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
                    100.0 * stats.frames_with_position as f64 / stats.total_frames.max(1) as f64
                );
                info!("  Lane changes detected: {}", stats.lane_changes_detected);
                info!("  ✅ Complete overtakes: {}", stats.complete_overtakes);
                info!("  ⚠️  Incomplete overtakes: {}", stats.incomplete_overtakes);
                info!("  ↔️  Simple lane changes: {}", stats.simple_lane_changes);
                info!("  Events sent to API: {}", stats.events_sent_to_api);
                if stats.curves_detected > 0 {
                    info!("  🌀 Curves detected: {}", stats.curves_detected);
                }

                // Vehicle detection stats
                let yolo_runs = stats.total_frames / 3;
                info!(
                    "  🚙 Vehicle detections: {} (across {} frames)",
                    stats.total_vehicles_detected, yolo_runs
                );
                if yolo_runs > 0 {
                    info!(
                        "  🚙 Average: {:.1} vehicles per frame",
                        stats.total_vehicles_detected as f64 / yolo_runs as f64
                    );
                }
                info!(
                    "  🎯 Vehicles overtaken: {}",
                    stats.total_vehicles_overtaken
                );
                info!(
                    "  🔢 Unique vehicles tracked: {}",
                    stats.unique_vehicles_seen
                );

                // 🆕 Shadow overtake stats
                if stats.shadow_overtakes_detected > 0 {
                    warn!(
                        "  ⚫ SHADOW OVERTAKES: {} detected!",
                        stats.shadow_overtakes_detected
                    );
                } else {
                    info!("  ⚫ Shadow overtakes: 0 (clean overtakes)");
                }

                info!("  Processing Speed: {:.1} FPS", stats.avg_fps);
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
    complete_overtakes: usize,
    incomplete_overtakes: usize,
    simple_lane_changes: usize,
    events_sent_to_api: usize,
    curves_detected: usize,
    total_vehicles_detected: usize,
    total_vehicles_overtaken: usize,
    unique_vehicles_seen: u32,
    shadow_overtakes_detected: usize, // 🆕
    duration_secs: f64,
    avg_fps: f64,
}

// ============================================================================
// HELPER: Extract lane boundaries from detected lanes
// ============================================================================

fn extract_lane_boundaries(
    lanes: &[Lane],
    frame_width: u32,
    frame_height: u32,
    reference_y_ratio: f32,
) -> (Option<f32>, Option<f32>) {
    let reference_y = frame_height as f32 * reference_y_ratio;
    let vehicle_x = frame_width as f32 / 2.0;

    let mut left_x: Option<f32> = None;
    let mut right_x: Option<f32> = None;

    for lane in lanes {
        if let Some(x) = lane.get_x_at_y(reference_y) {
            if x < vehicle_x {
                if left_x.is_none() || x > left_x.unwrap() {
                    left_x = Some(x);
                }
            } else if right_x.is_none() || x < right_x.unwrap() {
                right_x = Some(x);
            }
        }
    }

    (left_x, right_x)
}

// ============================================================================
// HELPER: Attach shadow events to a LaneChangeEvent's metadata
// ============================================================================

fn attach_shadow_metadata(event: &mut LaneChangeEvent, shadow_events: &[ShadowOvertakeEvent]) {
    let detected = !shadow_events.is_empty();

    event.metadata.insert(
        "shadow_overtake_detected".to_string(),
        serde_json::json!(detected),
    );
    event.metadata.insert(
        "shadow_overtake_count".to_string(),
        serde_json::json!(shadow_events.len()),
    );

    if detected {
        let worst_severity = shadow_events
            .iter()
            .map(|e| e.severity)
            .max_by_key(|s| s.rank())
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|| "NONE".to_string());

        let blocking_vehicles: Vec<String> = shadow_events
            .iter()
            .map(|e| {
                format!(
                    "{} (ID #{})",
                    e.blocking_vehicle_class, e.blocking_vehicle_id
                )
            })
            .collect();

        event.metadata.insert(
            "shadow_worst_severity".to_string(),
            serde_json::json!(worst_severity),
        );
        event.metadata.insert(
            "shadow_blocking_vehicles".to_string(),
            serde_json::json!(blocking_vehicles),
        );
        event.metadata.insert(
            "shadow_events".to_string(),
            serde_json::json!(shadow_events),
        );
    }
}

// ============================================================================
// VIDEO PROCESSING
// ============================================================================

async fn process_video(
    video_path: &Path,
    inference_engine: &mut inference::InferenceEngine,
    video_processor: &video_processor::VideoProcessor,
    config: &types::Config,
    legality_config: &LegalityAnalysisConfig,
) -> Result<ProcessingStats> {
    use std::io::Write;
    use std::time::Instant;

    let start_time = Instant::now();

    let mut reader = video_processor.open_video(video_path)?;
    let mut writer =
        video_processor.create_writer(video_path, reader.width, reader.height, reader.fps)?;

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

    let mut yolo_detector = YoloDetector::new("models/yolov8n.onnx")?;
    info!("✓ YOLO vehicle detector ready");

    let mut overtake_analyzer = OvertakeAnalyzer::new(reader.width as f32, reader.height as f32);
    info!("✓ Overtake analyzer ready");

    let mut overtake_tracker = OvertakeTracker::new(30.0, reader.fps);
    info!("✓ Overtake tracker ready (30s timeout)");

    // 🆕 Shadow overtake detector
    let mut shadow_detector =
        ShadowOvertakeDetector::new(reader.width as f32, reader.height as f32);
    info!("✓ Shadow overtake detector ready");

    // Output file
    std::fs::create_dir_all(&config.video.output_dir)?;
    let video_name = video_path.file_stem().unwrap().to_str().unwrap();
    let jsonl_path =
        Path::new(&config.video.output_dir).join(format!("{}_overtakes.jsonl", video_name));
    let mut results_file = std::fs::File::create(&jsonl_path)?;
    info!("💾 Results will be written to: {}", jsonl_path.display());

    // Counters
    let mut lane_changes_count: usize = 0;
    let mut complete_overtakes: usize = 0;
    let mut incomplete_overtakes: usize = 0;
    let mut simple_lane_changes: usize = 0;
    let mut frame_count: u64 = 0;
    let mut frames_with_valid_position: u64 = 0;
    let mut events_sent_to_api: usize = 0;
    let mut curves_detected: usize = 0;
    let mut total_vehicles_detected: usize = 0;
    let mut total_vehicles_overtaken: usize = 0;
    let mut shadow_overtakes_detected: usize = 0; // 🆕

    let mut previous_state = "CENTERED".to_string();
    let mut frame_buffer = LaneChangeFrameBuffer::new(legality_config.max_buffer_frames);
    let lane_confidence_threshold = config.detection.min_lane_confidence;

    // 🆕 Keep latest lane boundaries for shadow detector
    let mut last_left_lane_x: Option<f32> = None;
    let mut last_right_lane_x: Option<f32> = None;

    while let Some(frame) = reader.read_frame()? {
        frame_count += 1;
        let timestamp_ms = frame.timestamp_ms;

        // ── YOLO detection every 3 frames ────────────────────────────────
        if frame_count % 3 == 0 {
            match yolo_detector.detect(&frame.data, frame.width, frame.height, 0.3) {
                Ok(detections) => {
                    total_vehicles_detected += detections.len();
                    overtake_analyzer.update(detections, frame_count);

                    // 🆕 Run shadow detection using latest lane boundaries
                    if shadow_detector.is_monitoring() {
                        if let Some(shadow_event) = shadow_detector.update(
                            overtake_analyzer.get_tracked_vehicles(),
                            last_left_lane_x,
                            last_right_lane_x,
                            frame_count,
                            timestamp_ms,
                        ) {
                            shadow_overtakes_detected += 1;
                            // Save shadow event immediately
                            save_shadow_event(&shadow_event, &mut results_file)?;
                        }
                    }

                    if frame_count % 90 == 0 {
                        let active = overtake_analyzer.get_active_vehicle_count();
                        if active > 0 {
                            info!(
                                "Frame {}: {} active vehicle(s) tracked{}",
                                frame_count,
                                active,
                                if shadow_detector.is_monitoring() {
                                    format!(
                                        " | shadow monitoring: {} candidate(s)",
                                        shadow_detector.active_shadow_count()
                                    )
                                } else {
                                    String::new()
                                }
                            );
                        }
                    }
                }
                Err(e) => debug!("YOLO detection failed on frame {}: {}", frame_count, e),
            }
        }

        // ── Overtake timeout check every 30 frames ───────────────────────
        if frame_count % 30 == 0 {
            if let Some(timeout_result) = overtake_tracker.check_timeout(frame_count) {
                match timeout_result {
                    OvertakeResult::Incomplete {
                        start_event,
                        reason,
                    } => {
                        warn!("⏰ INCOMPLETE OVERTAKE: {}", reason);
                        warn!(
                            "   Started at {:.2}s but never returned to lane",
                            start_event.video_timestamp_ms / 1000.0
                        );
                        incomplete_overtakes += 1;

                        // 🆕 Stop shadow monitoring on timeout
                        let shadow_events = shadow_detector.stop_monitoring();
                        if !shadow_events.is_empty() {
                            warn!(
                                "  ⚫ {} shadow overtake(s) during incomplete maneuver",
                                shadow_events.len()
                            );
                        }

                        let mut incomplete_event = start_event.clone();
                        incomplete_event.metadata.insert(
                            "maneuver_type".to_string(),
                            serde_json::json!("incomplete_overtake"),
                        );
                        incomplete_event
                            .metadata
                            .insert("incomplete_reason".to_string(), serde_json::json!(reason));
                        attach_shadow_metadata(&mut incomplete_event, &shadow_events);

                        save_incomplete_overtake(&start_event, &reason, &mut results_file)?;
                    }
                    _ => {}
                }
            }
        }

        // ── Progress logging ─────────────────────────────────────────────
        if frame_count % 50 == 0 {
            info!(
                "Progress: {:.1}% ({}/{}) | State: {} | Tracking: {} | Shadow: {} | Complete: {} | Incomplete: {}",
                reader.progress(),
                reader.current_frame,
                reader.total_frames,
                analyzer.current_state(),
                if overtake_tracker.is_tracking() { "YES" } else { "NO" },
                if shadow_detector.is_monitoring() {
                    format!("YES({})", shadow_detector.active_shadow_count())
                } else {
                    "NO".to_string()
                },
                complete_overtakes,
                incomplete_overtakes
            );
        }

        // ── Lane detection + analysis ────────────────────────────────────
        match process_frame(&frame, inference_engine, config, lane_confidence_threshold).await {
            Ok(detected_lanes) => {
                let analysis_lanes: Vec<Lane> = detected_lanes
                    .iter()
                    .enumerate()
                    .map(|(i, dl)| Lane::from_detected(i, dl))
                    .collect();

                // 🆕 Update lane boundaries for shadow detection
                let (left_x, right_x) = extract_lane_boundaries(
                    &analysis_lanes,
                    frame.width as u32,
                    frame.height as u32,
                    0.8, // reference_y_ratio
                );
                last_left_lane_x = left_x;
                last_right_lane_x = right_x;

                // Pre-buffer
                if previous_state == "CENTERED" {
                    frame_buffer.add_to_pre_buffer(frame.clone());
                }

                // Lane change analysis
                if let Some(event) = analyzer.analyze(
                    &analysis_lanes,
                    frame.width as u32,
                    frame.height as u32,
                    frame_count,
                    timestamp_ms,
                ) {
                    lane_changes_count += 1;

                    info!(
                        "🚀 LANE CHANGE DETECTED: {} at {:.2}s (frame {})",
                        event.direction_name(),
                        event.video_timestamp_ms / 1000.0,
                        event.end_frame_id
                    );

                    // Process through overtake tracker
                    let tracker_result =
                        overtake_tracker.process_lane_change(event.clone(), frame_count);

                    match tracker_result {
                        None => {
                            // First lane change → overtake just started
                            // 🆕 Start shadow monitoring
                            shadow_detector.start_monitoring(event.direction, frame_count);
                        }

                        Some(OvertakeResult::Complete {
                            start_event,
                            end_event,
                            total_duration_ms,
                            ..
                        }) => {
                            info!(
                                "✅ COMPLETE OVERTAKE: {:.2}s → {:.2}s (duration: {:.1}s)",
                                start_event.video_timestamp_ms / 1000.0,
                                end_event.video_timestamp_ms / 1000.0,
                                total_duration_ms / 1000.0
                            );

                            // 🆕 Stop shadow monitoring and collect events
                            let shadow_events = shadow_detector.stop_monitoring();
                            if !shadow_events.is_empty() {
                                warn!(
                                    "  ⚫ {} SHADOW OVERTAKE(S) during this maneuver!",
                                    shadow_events.len()
                                );
                                shadow_overtakes_detected += shadow_events.len();
                            }

                            // Analyze vehicles overtaken
                            let overtakes = overtake_analyzer.analyze_overtake(
                                start_event.start_frame_id,
                                end_event.end_frame_id,
                                start_event.direction_name(),
                            );

                            if !overtakes.is_empty() {
                                info!(
                                    "🎯 Overtook {} vehicle(s): {}",
                                    overtakes.len(),
                                    overtakes
                                        .iter()
                                        .map(|o| format!("{} (ID #{})", o.class_name, o.vehicle_id))
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                );
                                total_vehicles_overtaken += overtakes.len();
                            } else {
                                info!("ℹ️  No vehicles were overtaken (repositioning maneuver)");
                            }

                            // Create combined event
                            let mut combined_event = create_combined_event(
                                &start_event,
                                &end_event,
                                total_duration_ms,
                                &overtakes,
                            );

                            // 🆕 Attach shadow metadata
                            attach_shadow_metadata(&mut combined_event, &shadow_events);

                            // Curve info
                            let curve_info = analyzer.get_curve_info();
                            if curve_info.is_curve {
                                curves_detected += 1;
                                warn!(
                                    "⚠️  Overtake in CURVE: type={:?}, angle={:.1}°",
                                    curve_info.curve_type, curve_info.angle_degrees
                                );
                            }

                            // Get captured frames
                            let captured_frames = frame_buffer.stop_capture();

                            // Send to API
                            if !captured_frames.is_empty() {
                                if let Err(e) = send_overtake_to_api(
                                    &combined_event,
                                    &captured_frames,
                                    curve_info,
                                    legality_config,
                                    config,
                                    video_processor,
                                )
                                .await
                                {
                                    warn!("Failed to send overtake to API: {}", e);
                                } else {
                                    events_sent_to_api += 1;
                                }
                            }

                            save_complete_overtake(&combined_event, &mut results_file)?;
                            complete_overtakes += 1;

                            // 🆕 Start monitoring again for the return lane change
                            // (the end_event IS a new lane change that could be the start
                            //  of the next overtake sequence — but the tracker already
                            //  went back to Idle, so this is handled naturally next time)
                        }

                        Some(OvertakeResult::Incomplete {
                            start_event,
                            reason,
                        }) => {
                            warn!("⚠️  INCOMPLETE OVERTAKE: {}", reason);
                            incomplete_overtakes += 1;

                            // 🆕 Stop shadow monitoring
                            let shadow_events = shadow_detector.stop_monitoring();
                            if !shadow_events.is_empty() {
                                shadow_overtakes_detected += shadow_events.len();
                            }

                            save_incomplete_overtake(&start_event, &reason, &mut results_file)?;

                            // New overtake started (same direction again) →
                            // start monitoring for the new one
                            shadow_detector.start_monitoring(event.direction, frame_count);
                        }

                        Some(OvertakeResult::SimpleLaneChange { event: _ }) => {
                            info!("↔️  Simple lane change (no return detected yet)");
                            simple_lane_changes += 1;
                        }
                    }
                }

                // ── State transitions for frame buffer ───────────────────
                let current_state = analyzer.current_state().to_string();

                if previous_state == "CENTERED" && current_state == "DRIFTING" {
                    frame_buffer.start_capture(frame_count);
                    debug!(
                        "📸 Started capturing at frame {} (with pre-buffer)",
                        frame_count
                    );
                }

                if frame_buffer.is_capturing() {
                    frame_buffer.add_frame(frame.clone());
                }

                if current_state == "CENTERED" && frame_buffer.is_capturing() {
                    frame_buffer.cancel_capture();
                    debug!("❌ Lane change cancelled");
                }

                previous_state = current_state;

                // Periodic offset logging
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

                // Annotated video output
                if let Some(ref mut w) = writer {
                    if let Ok(annotated) = video_processor::draw_lanes_with_state_enhanced(
                        &frame.data,
                        reader.width,
                        reader.height,
                        &detected_lanes,
                        analyzer.current_state(),
                        analyzer.last_vehicle_state(),
                        overtake_analyzer.get_tracked_vehicles(), // 🆕 Pass tracked vehicles
                        &shadow_detector,                         // 🆕 Pass shadow detector
                        frame_count,                              // 🆕 Pass frame ID
                    ) {
                        use opencv::videoio::VideoWriterTrait;
                        w.write(&annotated)?;
                    }
                }
            }
            Err(e) => error!("Frame {} failed: {}", frame_count, e),
        }
    }

    // ── End of video: clean up any in-progress overtake ──────────────────
    if shadow_detector.is_monitoring() {
        let remaining_shadows = shadow_detector.stop_monitoring();
        if !remaining_shadows.is_empty() {
            shadow_overtakes_detected += remaining_shadows.len();
            warn!(
                "⚫ {} shadow event(s) from unfinished overtake at end of video",
                remaining_shadows.len()
            );
        }
    }

    let duration = start_time.elapsed();
    let avg_fps = frame_count as f64 / duration.as_secs_f64();
    let unique_vehicles = overtake_analyzer.get_total_unique_vehicles();

    info!("\n📊 Final Report:");
    info!("  Total Lane Changes: {}", lane_changes_count);
    info!("  ✅ Complete Overtakes: {}", complete_overtakes);
    info!("  ⚠️  Incomplete Overtakes: {}", incomplete_overtakes);
    info!("  ↔️  Simple Lane Changes: {}", simple_lane_changes);
    info!("  Events Sent to API: {}", events_sent_to_api);
    if curves_detected > 0 {
        info!("  🌀 Curves Detected: {}", curves_detected);
    }
    info!("  🚙 Vehicle Detections: {}", total_vehicles_detected);
    info!("  🎯 Vehicles Overtaken: {}", total_vehicles_overtaken);
    info!("  🔢 Unique Vehicles: {}", unique_vehicles);

    // 🆕 Shadow summary
    if shadow_overtakes_detected > 0 {
        warn!(
            "  ⚫ SHADOW OVERTAKES: {} — DANGEROUS VISIBILITY CONDITIONS",
            shadow_overtakes_detected
        );
    } else {
        info!("  ⚫ Shadow Overtakes: 0");
    }

    info!("  Processing Speed: {:.1} FPS", avg_fps);

    Ok(ProcessingStats {
        total_frames: frame_count,
        frames_with_position: frames_with_valid_position,
        lane_changes_detected: lane_changes_count,
        complete_overtakes,
        incomplete_overtakes,
        simple_lane_changes,
        events_sent_to_api,
        curves_detected,
        total_vehicles_detected,
        total_vehicles_overtaken,
        unique_vehicles_seen: unique_vehicles,
        shadow_overtakes_detected,
        duration_secs: duration.as_secs_f64(),
        avg_fps,
    })
}

// ============================================================================
// HELPERS
// ============================================================================

fn create_combined_event(
    start_event: &LaneChangeEvent,
    end_event: &LaneChangeEvent,
    total_duration_ms: f64,
    overtakes: &[overtake_analyzer::OvertakeEvent],
) -> LaneChangeEvent {
    let mut combined = start_event.clone();

    combined.metadata.insert(
        "maneuver_type".to_string(),
        serde_json::json!("complete_overtake"),
    );
    combined.metadata.insert(
        "total_duration_ms".to_string(),
        serde_json::json!(total_duration_ms),
    );
    combined.metadata.insert(
        "return_frame_id".to_string(),
        serde_json::json!(end_event.end_frame_id),
    );
    combined.metadata.insert(
        "return_timestamp_ms".to_string(),
        serde_json::json!(end_event.video_timestamp_ms),
    );
    combined.metadata.insert(
        "vehicles_overtaken".to_string(),
        serde_json::json!(overtakes.len()),
    );

    if !overtakes.is_empty() {
        combined.metadata.insert(
            "overtaken_vehicle_types".to_string(),
            serde_json::json!(overtakes.iter().map(|o| &o.class_name).collect::<Vec<_>>()),
        );
        combined.metadata.insert(
            "overtaken_vehicle_ids".to_string(),
            serde_json::json!(overtakes.iter().map(|o| o.vehicle_id).collect::<Vec<_>>()),
        );
    }

    combined
}

fn save_complete_overtake(event: &LaneChangeEvent, file: &mut std::fs::File) -> Result<()> {
    use std::io::Write;
    let json_line = serde_json::to_string(&event.to_json())?;
    writeln!(file, "{}", json_line)?;
    file.flush()?;
    info!("💾 Complete overtake saved to JSONL");
    Ok(())
}

fn save_incomplete_overtake(
    start_event: &LaneChangeEvent,
    reason: &str,
    file: &mut std::fs::File,
) -> Result<()> {
    use std::io::Write;
    let mut event = start_event.clone();
    event.metadata.insert(
        "maneuver_type".to_string(),
        serde_json::json!("incomplete_overtake"),
    );
    event
        .metadata
        .insert("incomplete_reason".to_string(), serde_json::json!(reason));

    let json_line = serde_json::to_string(&event.to_json())?;
    writeln!(file, "{}", json_line)?;
    file.flush()?;
    warn!("💾 Incomplete overtake saved to JSONL");
    Ok(())
}

// 🆕 Save shadow overtake event to JSONL
fn save_shadow_event(shadow: &ShadowOvertakeEvent, file: &mut std::fs::File) -> Result<()> {
    use std::io::Write;

    let json_value = serde_json::json!({
        "type": "shadow_overtake",
        "blocking_vehicle_id": shadow.blocking_vehicle_id,
        "blocking_vehicle_class": shadow.blocking_vehicle_class,
        "detected_at_frame": shadow.detected_at_frame,
        "detected_at_timestamp_ms": shadow.detected_at_timestamp_ms,
        "frames_blocked": shadow.frames_blocked,
        "severity": shadow.severity.as_str(),
        "closest_distance_ratio": shadow.closest_distance_ratio,
    });

    let json_line = serde_json::to_string(&json_value)?;
    writeln!(file, "{}", json_line)?;
    file.flush()?;
    warn!("💾 Shadow overtake event saved to JSONL");
    Ok(())
}

async fn send_overtake_to_api(
    event: &LaneChangeEvent,
    captured_frames: &[Frame],
    curve_info: CurveInfo,
    legality_config: &LegalityAnalysisConfig,
    config: &types::Config,
    _video_processor: &video_processor::VideoProcessor,
) -> Result<()> {
    if captured_frames.is_empty() {
        return Ok(());
    }

    let request = build_legality_request(
        event,
        captured_frames,
        legality_config.num_frames_to_analyze,
        curve_info,
    )?;

    if legality_config.print_to_console {
        print_legality_request(&request);
    }

    if legality_config.save_to_file {
        save_legality_request_to_file(&request, &config.video.output_dir)?;
    }

    if legality_config.send_to_api {
        match send_to_legality_api(&request, &legality_config.api_url).await {
            Ok(response) => {
                info!(
                    "✅ Overtake {} sent to API: {} - {}",
                    response.event_id, response.status, response.message
                );
            }
            Err(e) => {
                error!("❌ Failed to send to API: {}", e);
                return Err(e);
            }
        }
    }

    Ok(())
}

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
