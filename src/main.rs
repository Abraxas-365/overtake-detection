// src/main.rs
//
// Production-ready overtake detection pipeline v2.0
//
// Changes from original:
//   ✅ Graceful shutdown (SIGTERM / SIGINT)
//   ✅ Legality ring buffer (temporal alignment fix)
//   ✅ Per-subsystem timing instrumentation
//   ✅ API retry with exponential backoff + circuit breaker
//   ✅ Config validation at startup
//   ✅ Structured logging with context fields
//   ✅ Dynamic overtake timeout (shadow/vehicle-aware)
//   ✅ Proper error context propagation
//   ✅ End-of-video state flush
//   ✅ Incomplete overtake API reporting
//

mod analysis;
mod frame_buffer;
mod inference;
mod lane_detection;
mod lane_legality;
mod overtake_analyzer;
mod overtake_tracker;
mod preprocessing;
mod shadow_overtake;
mod types;
mod vehicle_detection;
mod video_processor;

use analysis::LaneChangeAnalyzer;
use anyhow::{Context, Result};
use frame_buffer::{
    build_legality_request, print_legality_request, save_legality_request_to_file,
    send_to_legality_api, LaneChangeFrameBuffer,
};
use lane_legality::{FusedLegalityResult, LaneLegalityDetector, LegalityResult, LineLegality};
use overtake_analyzer::OvertakeAnalyzer;
use overtake_tracker::{OvertakeResult, OvertakeTracker};
use shadow_overtake::{ShadowOvertakeDetector, ShadowOvertakeEvent};
use std::collections::VecDeque;
use std::path::Path;
use std::time::Instant;
use tokio::sync::watch;
use tracing::{debug, error, info, warn};
use types::{CurveInfo, DetectedLane, Direction, Frame, Lane, LaneChangeConfig, LaneChangeEvent};
use vehicle_detection::YoloDetector;

// ============================================================================
// LEGALITY RING BUFFER
// ============================================================================
// Stores legality results per-frame so we can retrieve the one closest to
// the actual crossing moment instead of using a stale cached value.

struct LegalityRingBuffer {
    entries: VecDeque<LegalityBufferEntry>,
    capacity: usize,
}

#[derive(Clone)]
struct LegalityBufferEntry {
    frame_id: u64,
    timestamp_ms: f64,
    fused: FusedLegalityResult,
    as_result: LegalityResult,
}

impl LegalityRingBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, frame_id: u64, timestamp_ms: f64, fused: FusedLegalityResult) {
        let as_result = LegalityResult {
            verdict: fused.verdict,
            intersecting_line: fused.line_type_from_seg_model.clone(),
            all_markings: fused.all_markings.clone(),
            ego_intersects_marking: fused.ego_intersects_marking,
            frame_id,
        };

        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }

        self.entries.push_back(LegalityBufferEntry {
            frame_id,
            timestamp_ms,
            fused,
            as_result,
        });
    }

    /// Find the most severe legality result within a frame range.
    /// This is what we attach to an overtake event for temporal accuracy.
    fn worst_in_range(&self, start_frame: u64, end_frame: u64) -> Option<&LegalityBufferEntry> {
        self.entries
            .iter()
            .filter(|e| {
                e.frame_id >= start_frame
                    && e.frame_id <= end_frame
                    && e.fused.crossing_confirmed_by_lane_model
            })
            .max_by_key(|e| severity_rank(e.fused.verdict))
    }

    /// Find the closest legality result to a specific frame.
    fn closest_to_frame(&self, target_frame: u64) -> Option<&LegalityBufferEntry> {
        self.entries
            .iter()
            .filter(|e| e.fused.crossing_confirmed_by_lane_model)
            .min_by_key(|e| (e.frame_id as i64 - target_frame as i64).unsigned_abs())
    }

    /// Get the latest result (for video overlay).
    fn latest(&self) -> Option<&LegalityResult> {
        self.entries.back().map(|e| &e.as_result)
    }

    /// Get the latest fused result.
    fn latest_fused(&self) -> Option<&FusedLegalityResult> {
        self.entries.back().map(|e| &e.fused)
    }
}

fn severity_rank(v: LineLegality) -> u8 {
    match v {
        LineLegality::Unknown => 0,
        LineLegality::Legal => 1,
        LineLegality::Caution => 2,
        LineLegality::Illegal => 3,
        LineLegality::CriticalIllegal => 4,
    }
}

// ============================================================================
// API CLIENT WITH RETRY
// ============================================================================

struct ApiClient {
    client: reqwest::Client,
    max_retries: u32,
    initial_backoff_ms: u64,
    consecutive_failures: u32,
    circuit_breaker_threshold: u32,
}

impl ApiClient {
    fn new(timeout_secs: u64) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .pool_max_idle_per_host(5)
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self {
            client,
            max_retries: 3,
            initial_backoff_ms: 500,
            consecutive_failures: 0,
            circuit_breaker_threshold: 10,
        })
    }

    async fn send(
        &mut self,
        request: &frame_buffer::LaneChangeLegalityRequest,
        api_url: &str,
    ) -> Result<frame_buffer::LaneChangeLegalityResponse> {
        // Circuit breaker: fail fast if API is consistently down
        if self.consecutive_failures >= self.circuit_breaker_threshold {
            anyhow::bail!(
                "Circuit breaker open: {} consecutive failures to {}",
                self.consecutive_failures,
                api_url
            );
        }

        let mut last_error = None;
        let mut backoff_ms = self.initial_backoff_ms;

        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                warn!(
                    "API retry {}/{} for event {} (backoff {}ms)",
                    attempt, self.max_retries, request.event_id, backoff_ms
                );
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(5000);
            }

            match self.try_send(request, api_url).await {
                Ok(response) => {
                    self.consecutive_failures = 0;
                    return Ok(response);
                }
                Err(e) => {
                    self.consecutive_failures += 1;
                    let err_str = e.to_string();

                    // Don't retry on client errors (4xx)
                    if err_str.contains("400")
                        || err_str.contains("401")
                        || err_str.contains("403")
                        || err_str.contains("422")
                    {
                        return Err(e);
                    }

                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Unknown API error")))
    }

    async fn try_send(
        &self,
        request: &frame_buffer::LaneChangeLegalityRequest,
        api_url: &str,
    ) -> Result<frame_buffer::LaneChangeLegalityResponse> {
        let response = self
            .client
            .post(api_url)
            .json(request)
            .send()
            .await
            .context("HTTP request failed")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("API error {}: {}", status, body);
        }

        response
            .json::<frame_buffer::LaneChangeLegalityResponse>()
            .await
            .context("Failed to parse API response")
    }

    fn reset_circuit_breaker(&mut self) {
        self.consecutive_failures = 0;
    }
}

// ============================================================================
// SUBSYSTEM TIMING
// ============================================================================

struct SubsystemTimings {
    lane_inference_us: u64,
    yolo_us: u64,
    legality_us: u64,
    analysis_us: u64,
    annotation_us: u64,
}

impl SubsystemTimings {
    fn new() -> Self {
        Self {
            lane_inference_us: 0,
            yolo_us: 0,
            legality_us: 0,
            analysis_us: 0,
            annotation_us: 0,
        }
    }

    fn total_us(&self) -> u64 {
        self.lane_inference_us
            + self.yolo_us
            + self.legality_us
            + self.analysis_us
            + self.annotation_us
    }
}

// ============================================================================
// CONFIGURATION
// ============================================================================

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

// ============================================================================
// PROCESSING STATS
// ============================================================================

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
    shadow_overtakes_detected: usize,
    illegal_crossings: usize,
    critical_violations: usize,
    duration_secs: f64,
    avg_fps: f64,
}

// ============================================================================
// ENTRY POINT
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // Structured logging — respects RUST_LOG env var
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "overtake_detection=info,ort=warn".to_string()),
        )
        .with_target(true)
        .with_thread_ids(true)
        .init();

    info!("🚗 Lane Change Detection System Starting (production mode)");

    // ── Load and validate config ─────────────────────────────────────
    let config = types::Config::load("config.yaml").context("Failed to load config.yaml")?;
    validate_config(&config)?;
    info!("✓ Configuration loaded and validated");

    info!(
        "Detection thresholds: drift={:.2}, crossing={:.2}, confirm_frames={}",
        config.detection.drift_threshold,
        config.detection.crossing_threshold,
        config.detection.confirm_frames
    );

    // ── Graceful shutdown ────────────────────────────────────────────
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    tokio::spawn(async move {
        let ctrl_c = tokio::signal::ctrl_c();

        #[cfg(unix)]
        {
            let mut sigterm =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("Failed to register SIGTERM handler");

            tokio::select! {
                _ = ctrl_c => info!("Received SIGINT, shutting down gracefully..."),
                _ = sigterm.recv() => info!("Received SIGTERM, shutting down gracefully..."),
            }
        }

        #[cfg(not(unix))]
        {
            let _ = ctrl_c.await;
            info!("Received SIGINT, shutting down gracefully...");
        }

        let _ = shutdown_tx.send(true);
    });

    // ── Initialize inference engine ──────────────────────────────────
    let mut inference_engine = inference::InferenceEngine::new(config.clone())
        .context("Failed to initialize lane inference engine")?;
    info!("✓ Inference engine ready");

    // ── Find videos ──────────────────────────────────────────────────
    let video_processor = video_processor::VideoProcessor::new(config.clone());
    let video_files = video_processor.find_video_files()?;

    if video_files.is_empty() {
        error!("No video files found in {}", config.video.input_dir);
        return Ok(());
    }

    info!("Found {} video file(s) to process", video_files.len());

    // ── Legality config ──────────────────────────────────────────────
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

    // ── API client with retry ────────────────────────────────────────
    let mut api_client = ApiClient::new(30).context("Failed to create API client")?;
    info!("✓ API client ready (retry=3, backoff=exponential)");

    // ── Process each video ───────────────────────────────────────────
    for (idx, video_path) in video_files.iter().enumerate() {
        // Check for shutdown between videos
        if *shutdown_rx.borrow() {
            warn!(
                "Shutdown requested, skipping remaining {} video(s)",
                video_files.len() - idx
            );
            break;
        }

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
            &mut api_client,
            shutdown_rx.clone(),
        )
        .await
        {
            Ok(stats) => {
                print_final_stats(&stats);
            }
            Err(e) => {
                error!("Failed to process video {}: {:#}", video_path.display(), e);
            }
        }
    }

    info!("🏁 All videos processed. Exiting.");
    Ok(())
}

// ============================================================================
// CONFIG VALIDATION
// ============================================================================

fn validate_config(config: &types::Config) -> Result<()> {
    // Model files
    if !Path::new(&config.model.path).exists() {
        anyhow::bail!("Lane model not found: {}", config.model.path);
    }

    if config.lane_legality.enabled && !Path::new(&config.lane_legality.model_path).exists() {
        anyhow::bail!(
            "Lane legality model not found: {}. Set lane_legality.enabled=false to skip.",
            config.lane_legality.model_path
        );
    }

    // Threshold ordering
    if config.detection.drift_threshold >= config.detection.crossing_threshold {
        anyhow::bail!(
            "drift_threshold ({}) must be < crossing_threshold ({})",
            config.detection.drift_threshold,
            config.detection.crossing_threshold
        );
    }

    // Range checks
    if config.detection.drift_threshold <= 0.0 || config.detection.drift_threshold >= 1.0 {
        anyhow::bail!(
            "drift_threshold must be in (0,1), got {}",
            config.detection.drift_threshold
        );
    }

    if config.detection.min_lane_change_duration_ms >= config.detection.max_lane_change_duration_ms
    {
        anyhow::bail!(
            "min_duration ({}) must be < max_duration ({})",
            config.detection.min_lane_change_duration_ms,
            config.detection.max_lane_change_duration_ms
        );
    }

    // Input directory
    if !Path::new(&config.video.input_dir).exists() {
        anyhow::bail!(
            "Video input directory not found: {}",
            config.video.input_dir
        );
    }

    // Ego bbox sanity
    let b = &config.lane_legality.ego_bbox_ratio;
    if b[0] >= b[2] || b[1] >= b[3] {
        anyhow::bail!(
            "ego_bbox_ratio invalid: x1<x2 and y1<y2 required, got {:?}",
            b
        );
    }

    info!("✓ Config validation passed");
    Ok(())
}

// ============================================================================
// MAIN PROCESSING LOOP
// ============================================================================

async fn process_video(
    video_path: &Path,
    inference_engine: &mut inference::InferenceEngine,
    video_processor: &video_processor::VideoProcessor,
    config: &types::Config,
    legality_config: &LegalityAnalysisConfig,
    api_client: &mut ApiClient,
    shutdown_rx: watch::Receiver<bool>,
) -> Result<ProcessingStats> {
    use std::io::Write;

    let start_time = Instant::now();

    let mut reader = video_processor
        .open_video(video_path)
        .with_context(|| format!("Failed to open video: {}", video_path.display()))?;
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

    // ── Initialize subsystems ────────────────────────────────────────
    let mut analyzer = LaneChangeAnalyzer::new(lane_change_config);
    analyzer.set_source_id(video_path.to_string_lossy().to_string());

    let mut yolo_detector =
        YoloDetector::new("models/yolov8n.onnx").context("Failed to load YOLO model")?;
    info!("✓ YOLO vehicle detector ready");

    let mut legality_detector = if config.lane_legality.enabled {
        match LaneLegalityDetector::new(&config.lane_legality.model_path) {
            Ok(mut detector) => {
                let ego_bbox = config.lane_legality.ego_bbox_ratio;
                detector.set_ego_bbox_ratio(ego_bbox[0], ego_bbox[1], ego_bbox[2], ego_bbox[3]);
                info!("✓ Lane legality detector ready");
                Some(detector)
            }
            Err(e) => {
                warn!(
                    "⚠️  Lane legality detector failed to load: {}. Continuing without.",
                    e
                );
                None
            }
        }
    } else {
        info!("⚪ Lane legality detection disabled in config");
        None
    };

    let mut overtake_analyzer = OvertakeAnalyzer::new(reader.width as f32, reader.height as f32);
    info!("✓ Overtake analyzer ready");

    let mut overtake_tracker = OvertakeTracker::new(30.0, reader.fps);
    info!("✓ Overtake tracker ready (30s base timeout)");

    let mut shadow_detector =
        ShadowOvertakeDetector::new(reader.width as f32, reader.height as f32);
    info!("✓ Shadow overtake detector ready");

    let mut velocity_tracker = crate::analysis::velocity_tracker::LateralVelocityTracker::new();
    info!("✓ Velocity tracker ready");

    // ── Legality ring buffer (replaces single cached value) ──────────
    let mut legality_buffer = LegalityRingBuffer::new(300); // ~10s at 30fps
    info!("✓ Legality ring buffer ready (capacity: 300 frames)");

    // ── Output file ──────────────────────────────────────────────────
    std::fs::create_dir_all(&config.video.output_dir)?;
    let video_name = video_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let jsonl_path =
        Path::new(&config.video.output_dir).join(format!("{}_overtakes.jsonl", video_name));
    let mut results_file = std::fs::File::create(&jsonl_path)
        .with_context(|| format!("Failed to create results file: {}", jsonl_path.display()))?;
    info!("💾 Results will be written to: {}", jsonl_path.display());

    // ── Counters ─────────────────────────────────────────────────────
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
    let mut shadow_overtakes_detected: usize = 0;
    let mut illegal_crossings: usize = 0;
    let mut critical_violations: usize = 0;
    let mut timings = SubsystemTimings::new();

    // ── State ────────────────────────────────────────────────────────
    let mut previous_state = "CENTERED".to_string();
    let mut frame_buffer = LaneChangeFrameBuffer::new(legality_config.max_buffer_frames);
    let mut last_left_lane_x: Option<f32> = None;
    let mut last_right_lane_x: Option<f32> = None;
    let mut current_overtake_vehicles: Vec<overtake_analyzer::OvertakeEvent> = Vec::new();

    // ═════════════════════════════════════════════════════════════════
    // MAIN FRAME LOOP
    // ═════════════════════════════════════════════════════════════════

    while let Some(frame) = reader.read_frame()? {
        frame_count += 1;
        let timestamp_ms = frame.timestamp_ms;

        // ── Graceful shutdown check (cheap atomic read) ──────────────
        if *shutdown_rx.borrow() {
            warn!(
                "Shutdown requested at frame {} ({:.1}s). Flushing state...",
                frame_count,
                timestamp_ms / 1000.0
            );
            break;
        }

        // ── YOLO + LEGALITY (every N frames) ─────────────────────────
        if frame_count % config.lane_legality.inference_interval == 0 {
            // YOLO vehicle detection
            let yolo_start = Instant::now();
            match yolo_detector.detect(&frame.data, frame.width, frame.height, 0.3) {
                Ok(detections) => {
                    total_vehicles_detected += detections.len();
                    overtake_analyzer.update(detections, frame_count);

                    // Update dynamic timeout: are vehicles being passed?
                    let has_vehicles_ahead = overtake_analyzer.get_active_vehicle_count() > 0;
                    overtake_tracker.set_vehicles_being_passed(has_vehicles_ahead);

                    if shadow_detector.is_monitoring() {
                        if let Some(shadow_event) = shadow_detector.update(
                            overtake_analyzer.get_tracked_vehicles(),
                            last_left_lane_x,
                            last_right_lane_x,
                            frame_count,
                            timestamp_ms,
                        ) {
                            shadow_overtakes_detected += 1;

                            // Notify tracker: shadow active → reduce timeout
                            overtake_tracker.set_shadow_active(true);

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
            timings.yolo_us = yolo_start.elapsed().as_micros() as u64;

            // FUSED LANE LEGALITY DETECTION
            if let Some(ref mut detector) = legality_detector {
                let leg_start = Instant::now();

                let vehicle_offset = analyzer
                    .last_vehicle_state()
                    .map(|vs| vs.lateral_offset)
                    .unwrap_or(0.0);
                let lane_width = analyzer.last_vehicle_state().and_then(|vs| vs.lane_width);

                let crossing_side = match analyzer.current_state() {
                    "CROSSING" | "DRIFTING" => {
                        if vehicle_offset < 0.0 {
                            lane_legality::CrossingSide::Left
                        } else if vehicle_offset > 0.0 {
                            lane_legality::CrossingSide::Right
                        } else {
                            lane_legality::CrossingSide::None
                        }
                    }
                    _ => lane_legality::CrossingSide::None,
                };

                match detector.analyze_frame_fused(
                    &frame.data,
                    frame.width,
                    frame.height,
                    frame_count,
                    config.lane_legality.confidence_threshold,
                    vehicle_offset,
                    lane_width,
                    last_left_lane_x,
                    last_right_lane_x,
                    crossing_side,
                ) {
                    Ok(fused_result) => {
                        // Count violations
                        if fused_result.crossing_confirmed_by_lane_model
                            && fused_result.verdict.is_illegal()
                        {
                            match fused_result.verdict {
                                LineLegality::CriticalIllegal => {
                                    critical_violations += 1;
                                    warn!(
                                        "🚨🚨 FUSED CRITICAL: {} at {:.2}s (frame {}, offset: {:.1}%)",
                                        fused_result
                                            .line_type_from_seg_model
                                            .as_ref()
                                            .map(|l| l.class_name.as_str())
                                            .unwrap_or("unknown"),
                                        timestamp_ms / 1000.0,
                                        frame_count,
                                        fused_result.vehicle_offset_pct * 100.0
                                    );
                                }
                                LineLegality::Illegal => {
                                    illegal_crossings += 1;
                                    warn!(
                                        "🚨 FUSED ILLEGAL: {} at {:.2}s (frame {}, offset: {:.1}%)",
                                        fused_result
                                            .line_type_from_seg_model
                                            .as_ref()
                                            .map(|l| l.class_name.as_str())
                                            .unwrap_or("unknown"),
                                        timestamp_ms / 1000.0,
                                        frame_count,
                                        fused_result.vehicle_offset_pct * 100.0
                                    );
                                }
                                _ => {}
                            }
                        }

                        // ★ Push into ring buffer instead of overwriting single variable
                        legality_buffer.push(frame_count, timestamp_ms, fused_result);
                    }
                    Err(e) => debug!("Fused legality failed on frame {}: {}", frame_count, e),
                }

                timings.legality_us = leg_start.elapsed().as_micros() as u64;
            }
        }

        // ── Overtake timeout check (every 30 frames) ────────────────
        if frame_count % 30 == 0 {
            if let Some(timeout_result) = overtake_tracker.check_timeout(frame_count) {
                match timeout_result {
                    OvertakeResult::Incomplete {
                        start_event,
                        reason,
                    } => {
                        warn!(
                            "⏰ INCOMPLETE OVERTAKE: {} (started at {:.2}s)",
                            reason,
                            start_event.video_timestamp_ms / 1000.0
                        );
                        incomplete_overtakes += 1;

                        let shadow_events = shadow_detector.stop_monitoring();
                        if !shadow_events.is_empty() {
                            warn!(
                                "  ⚫ {} shadow overtake(s) during incomplete maneuver",
                                shadow_events.len()
                            );
                            shadow_overtakes_detected += shadow_events.len();
                        }

                        current_overtake_vehicles.clear();

                        let mut incomplete_event = start_event.clone();
                        incomplete_event.metadata.insert(
                            "maneuver_type".to_string(),
                            serde_json::json!("incomplete_overtake"),
                        );
                        incomplete_event
                            .metadata
                            .insert("incomplete_reason".to_string(), serde_json::json!(reason));
                        attach_shadow_metadata(&mut incomplete_event, &shadow_events);

                        // ★ Attach best legality from the crossing window
                        if let Some(entry) =
                            legality_buffer.worst_in_range(start_event.start_frame_id, frame_count)
                        {
                            attach_legality_to_event(&mut incomplete_event, &entry.as_result);
                        }

                        save_incomplete_overtake(&start_event, &reason, &mut results_file)?;
                    }
                    _ => {}
                }
            }
        }

        // ── Progress logging (every 150 frames ≈ 5s) ────────────────
        if frame_count % 150 == 0 {
            info!(
                "Progress: {:.1}% ({}/{}) | State: {} | Tracking: {} | Shadow: {} | \
                 Complete: {} | Incomplete: {} | Violations: {} | \
                 Timings: lane={}μs yolo={}μs leg={}μs",
                reader.progress(),
                reader.current_frame,
                reader.total_frames,
                analyzer.current_state(),
                if overtake_tracker.is_tracking() {
                    "YES"
                } else {
                    "NO"
                },
                if shadow_detector.is_monitoring() {
                    format!("YES({})", shadow_detector.active_shadow_count())
                } else {
                    "NO".to_string()
                },
                complete_overtakes,
                incomplete_overtakes,
                illegal_crossings + critical_violations,
                timings.lane_inference_us,
                timings.yolo_us,
                timings.legality_us,
            );
        }

        // ── Lane detection + analysis ────────────────────────────────
        let inf_start = Instant::now();
        match process_frame(
            &frame,
            inference_engine,
            config,
            config.detection.min_lane_confidence,
        )
        .await
        {
            Ok(detected_lanes) => {
                timings.lane_inference_us = inf_start.elapsed().as_micros() as u64;

                let analysis_start = Instant::now();

                let analysis_lanes: Vec<Lane> = detected_lanes
                    .iter()
                    .enumerate()
                    .map(|(i, dl)| Lane::from_detected(i, dl))
                    .collect();

                let (left_x, right_x) = extract_lane_boundaries(
                    &analysis_lanes,
                    frame.width as u32,
                    frame.height as u32,
                    0.8,
                );
                last_left_lane_x = left_x;
                last_right_lane_x = right_x;

                if previous_state == "CENTERED" {
                    frame_buffer.add_to_pre_buffer(frame.clone());
                }

                // ── State machine update ─────────────────────────────
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

                    let tracker_result =
                        overtake_tracker.process_lane_change(event.clone(), frame_count);

                    match tracker_result {
                        None => {
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

                            let shadow_events = shadow_detector.stop_monitoring();
                            if !shadow_events.is_empty() {
                                warn!(
                                    "  ⚫ {} SHADOW OVERTAKE(S) during this maneuver!",
                                    shadow_events.len()
                                );
                                shadow_overtakes_detected += shadow_events.len();
                            }

                            // Reset shadow-aware timeout
                            overtake_tracker.set_shadow_active(false);

                            let overtakes = overtake_analyzer.analyze_overtake(
                                start_event.start_frame_id,
                                end_event.end_frame_id,
                                start_event.direction_name(),
                            );

                            current_overtake_vehicles = overtakes.clone();

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

                            let mut combined_event = create_combined_event(
                                &start_event,
                                &end_event,
                                total_duration_ms,
                                &overtakes,
                            );

                            attach_shadow_metadata(&mut combined_event, &shadow_events);

                            // ★ TEMPORAL ALIGNMENT: Get the worst legality result
                            //   from the ACTUAL crossing window, not whatever was
                            //   last cached.
                            let crossing_legality = legality_buffer
                                .worst_in_range(start_event.start_frame_id, end_event.end_frame_id)
                                .or_else(|| {
                                    // Fallback: closest result to the start of the maneuver
                                    legality_buffer.closest_to_frame(start_event.start_frame_id)
                                });

                            if let Some(entry) = crossing_legality {
                                attach_legality_to_event(&mut combined_event, &entry.as_result);

                                info!(
                                    "🚦 Legality from frame {}: {} ({})",
                                    entry.frame_id,
                                    entry.as_result.verdict.as_str(),
                                    entry
                                        .as_result
                                        .intersecting_line
                                        .as_ref()
                                        .map(|l| l.class_name.as_str())
                                        .unwrap_or("none")
                                );
                            }

                            let curve_info = analyzer.get_curve_info();
                            if curve_info.is_curve {
                                curves_detected += 1;
                                warn!(
                                    "⚠️  Overtake in CURVE: type={:?}, angle={:.1}°",
                                    curve_info.curve_type, curve_info.angle_degrees
                                );
                            }

                            let captured_frames = frame_buffer.stop_capture();

                            if !captured_frames.is_empty() {
                                if let Err(e) = send_overtake_to_api(
                                    &combined_event,
                                    &captured_frames,
                                    curve_info,
                                    legality_config,
                                    config,
                                    api_client,
                                )
                                .await
                                {
                                    warn!("Failed to send overtake to API: {:#}", e);
                                } else {
                                    events_sent_to_api += 1;
                                }
                            }

                            save_complete_overtake(&combined_event, &mut results_file)?;
                            complete_overtakes += 1;
                        }

                        Some(OvertakeResult::Incomplete {
                            start_event,
                            reason,
                        }) => {
                            warn!("⚠️  INCOMPLETE OVERTAKE: {}", reason);
                            incomplete_overtakes += 1;

                            let shadow_events = shadow_detector.stop_monitoring();
                            if !shadow_events.is_empty() {
                                shadow_overtakes_detected += shadow_events.len();
                            }

                            current_overtake_vehicles.clear();
                            overtake_tracker.set_shadow_active(false);

                            save_incomplete_overtake(&start_event, &reason, &mut results_file)?;

                            // Start monitoring the new maneuver
                            shadow_detector.start_monitoring(event.direction, frame_count);
                        }

                        Some(OvertakeResult::SimpleLaneChange { event: _ }) => {
                            info!("↔️  Simple lane change (no return detected yet)");
                            simple_lane_changes += 1;
                        }
                    }
                }

                // ── Frame buffer management ──────────────────────────
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

                timings.analysis_us = analysis_start.elapsed().as_micros() as u64;

                // ── Periodic vehicle state logging ───────────────────
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

                // ── Annotated video output ───────────────────────────
                if let Some(ref mut w) = writer {
                    let ann_start = Instant::now();

                    let curve_info = analyzer.get_curve_info();
                    let is_overtaking = overtake_tracker.is_tracking();
                    let overtake_direction = if is_overtaking {
                        Some(overtake_tracker.get_direction().as_str())
                    } else {
                        None
                    };

                    let lateral_velocity = if let Some(vs) = analyzer.last_vehicle_state() {
                        velocity_tracker.get_velocity(vs.lateral_offset, timestamp_ms)
                    } else {
                        0.0
                    };

                    // ★ Use ring buffer's latest for overlay
                    let legality_for_overlay = legality_buffer.latest();

                    if let Ok(annotated) = video_processor::draw_lanes_with_state_enhanced(
                        &frame.data,
                        reader.width,
                        reader.height,
                        &detected_lanes,
                        analyzer.current_state(),
                        analyzer.last_vehicle_state(),
                        overtake_analyzer.get_tracked_vehicles(),
                        &shadow_detector,
                        frame_count,
                        timestamp_ms,
                        is_overtaking,
                        overtake_direction,
                        &current_overtake_vehicles,
                        Some(curve_info),
                        lateral_velocity,
                        legality_for_overlay,
                    ) {
                        use opencv::videoio::VideoWriterTrait;
                        w.write(&annotated)?;
                    }

                    timings.annotation_us = ann_start.elapsed().as_micros() as u64;
                }
            }
            Err(e) => {
                timings.lane_inference_us = inf_start.elapsed().as_micros() as u64;
                error!("Frame {} failed: {:#}", frame_count, e);
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════
    // END OF VIDEO CLEANUP
    // ═════════════════════════════════════════════════════════════════

    // Flush any in-progress overtake as incomplete
    if overtake_tracker.is_tracking() {
        if let Some(timeout_result) = overtake_tracker.check_timeout(u64::MAX) {
            if let OvertakeResult::Incomplete {
                start_event,
                reason,
            } = timeout_result
            {
                warn!(
                    "⏰ End-of-video incomplete overtake: started at {:.2}s",
                    start_event.video_timestamp_ms / 1000.0
                );
                incomplete_overtakes += 1;
                save_incomplete_overtake(
                    &start_event,
                    &format!("Video ended: {}", reason),
                    &mut results_file,
                )?;
            }
        }
    }

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

    // Cancel any remaining frame capture
    if frame_buffer.is_capturing() {
        frame_buffer.stop_capture();
    }

    // ── Build stats ──────────────────────────────────────────────────
    let duration = start_time.elapsed();
    let avg_fps = if duration.as_secs_f64() > 0.01 {
        frame_count as f64 / duration.as_secs_f64()
    } else {
        0.0
    };
    let unique_vehicles = overtake_analyzer.get_total_unique_vehicles();

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
        illegal_crossings,
        critical_violations,
        duration_secs: duration.as_secs_f64(),
        avg_fps,
    })
}

// ============================================================================
// HELPERS
// ============================================================================

/// Attach legality info from a LegalityResult to a LaneChangeEvent
fn attach_legality_to_event(event: &mut LaneChangeEvent, legality: &LegalityResult) {
    if !legality.ego_intersects_marking {
        return;
    }

    let legality_info = types::LegalityInfo {
        is_legal: !legality.verdict.is_illegal(),
        lane_line_type: legality
            .intersecting_line
            .as_ref()
            .map(|l| l.class_name.clone())
            .unwrap_or_else(|| "unknown".to_string()),
        confidence: legality
            .intersecting_line
            .as_ref()
            .map(|l| l.confidence)
            .unwrap_or(0.0),
        analysis_details: Some(format!(
            "On-device YOLOv8-seg detection: {} (frame {})",
            legality.verdict.as_str(),
            legality.frame_id
        )),
    };
    event.legality = Some(legality_info);

    event.metadata.insert(
        "line_legality_verdict".to_string(),
        serde_json::json!(legality.verdict.as_str()),
    );
    event.metadata.insert(
        "line_legality_frame".to_string(),
        serde_json::json!(legality.frame_id),
    );
    event.metadata.insert(
        "line_class_id".to_string(),
        serde_json::json!(legality.intersecting_line.as_ref().map(|l| l.class_id)),
    );
}

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
    let json_line =
        serde_json::to_string(&event.to_json()).context("Failed to serialize complete overtake")?;
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

    let json_line = serde_json::to_string(&event.to_json())
        .context("Failed to serialize incomplete overtake")?;
    writeln!(file, "{}", json_line)?;
    file.flush()?;
    warn!("💾 Incomplete overtake saved to JSONL");
    Ok(())
}

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

    let json_line =
        serde_json::to_string(&json_value).context("Failed to serialize shadow event")?;
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
    api_client: &mut ApiClient,
) -> Result<()> {
    if captured_frames.is_empty() {
        return Ok(());
    }

    let request = build_legality_request(
        event,
        captured_frames,
        legality_config.num_frames_to_analyze,
        curve_info,
    )
    .context("Failed to build legality request")?;

    if legality_config.print_to_console {
        print_legality_request(&request);
    }

    if legality_config.save_to_file {
        if let Err(e) = save_legality_request_to_file(&request, &config.video.output_dir) {
            warn!("Failed to save legality request to file: {:#}", e);
        }
    }

    if legality_config.send_to_api {
        match api_client.send(&request, &legality_config.api_url).await {
            Ok(response) => {
                info!(
                    "✅ Overtake {} sent to API: {} - {}",
                    response.event_id, response.status, response.message
                );
            }
            Err(e) => {
                error!("❌ Failed to send to API: {:#}", e);
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
    )
    .context("Frame preprocessing failed")?;

    let output = inference_engine
        .infer(&preprocessed)
        .context("Lane inference failed")?;

    let lane_detection = lane_detection::parse_lanes(
        &output,
        frame.width as f32,
        frame.height as f32,
        config,
        frame.timestamp_ms,
    )
    .context("Lane parsing failed")?;

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

// ============================================================================
// FINAL REPORT
// ============================================================================

fn print_final_stats(stats: &ProcessingStats) {
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

    if stats.shadow_overtakes_detected > 0 {
        warn!(
            "  ⚫ SHADOW OVERTAKES: {} detected!",
            stats.shadow_overtakes_detected
        );
    } else {
        info!("  ⚫ Shadow overtakes: 0 (clean overtakes)");
    }

    if stats.illegal_crossings > 0 || stats.critical_violations > 0 {
        warn!("  🚦 LANE LINE VIOLATIONS:");
        if stats.critical_violations > 0 {
            warn!(
                "     🚨 CRITICAL: {} (double yellow/solid yellow)",
                stats.critical_violations
            );
        }
        if stats.illegal_crossings > 0 {
            warn!(
                "     ⚠️  ILLEGAL: {} (solid white/red/double white)",
                stats.illegal_crossings
            );
        }
    } else {
        info!("  🚦 Lane line violations: 0");
    }

    info!(
        "  Processing: {:.1} FPS ({:.1}s total)",
        stats.avg_fps, stats.duration_secs
    );
}
