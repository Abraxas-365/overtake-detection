// src/main.rs
//
// Production overtake detection pipeline v4.2 — Maneuver Detection v2 Integrated
//
// Changes from v4.1:
//   ✅ Integrated ManeuverPipeline v2 (signal-fusion architecture)
//   ✅ Parallel A/B validation: old + new pipelines run side-by-side
//   ✅ V2 stats tracked in PipelineState and ProcessingStats
//   ✅ V2 events logged with full source/confidence/legality metadata
//   ✅ Borrow-checker safe: disjoint field access via Rust 2021 edition
//

mod analysis;
mod frame_buffer;
mod inference;
mod lane_detection;
mod lane_legality;
mod overtake_analyzer;
mod overtake_tracker;
mod pipeline;
mod preprocessing;
mod shadow_overtake;
mod types;
mod vehicle_detection;
mod video_processor;

use analysis::fallback_estimator::FallbackPositionEstimator;
use analysis::InferenceScheduler;
use analysis::LaneChangeAnalyzer;

// ── NEW v2 imports ──
use analysis::ego_motion::GrayFrame;
use analysis::lateral_detector::LaneMeasurement;
use analysis::maneuver_pipeline::{ManeuverFrameInput, ManeuverPipeline, ManeuverPipelineConfig};
use analysis::vehicle_tracker::DetectionInput;

use anyhow::{Context, Result};
use frame_buffer::{
    build_legality_request, print_legality_request, save_legality_request_to_file,
    LaneChangeFrameBuffer,
};
use lane_legality::{FusedLegalityResult, LaneLegalityDetector, LegalityResult, LineLegality};
use overtake_analyzer::OvertakeAnalyzer;
use overtake_tracker::{OvertakeResult, OvertakeTracker};
use pipeline::legality_buffer::LegalityRingBuffer;
use shadow_overtake::{ShadowOvertakeDetector, ShadowOvertakeEvent};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::watch;
use tracing::{error, info, warn};
use types::{
    CurveInfo, DetectedLane, Frame, Lane, LaneChangeConfig, LaneChangeEvent, VehicleState,
};
use vehicle_detection::YoloDetector;

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
// API CLIENT WITH RETRY + CIRCUIT BREAKER
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
                    // Don't retry client errors
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
}

// ============================================================================
// PIPELINE STATE — single struct replaces 20+ loose variables
// ============================================================================

struct PipelineState {
    // ── Core analyzers ──
    analyzer: LaneChangeAnalyzer,
    yolo_detector: YoloDetector,
    legality_detector: Option<LaneLegalityDetector>,
    overtake_analyzer: OvertakeAnalyzer,
    overtake_tracker: OvertakeTracker,
    shadow_detector: ShadowOvertakeDetector,
    #[allow(dead_code)] // Reserved for future velocity validation
    velocity_tracker: analysis::velocity_tracker::LateralVelocityTracker,
    fallback_estimator: FallbackPositionEstimator,
    inference_scheduler: InferenceScheduler,

    // ── NEW: Maneuver Detection v2 pipeline ──
    maneuver_pipeline_v2: Option<ManeuverPipeline>,

    // ── Buffers ──
    legality_buffer: LegalityRingBuffer,
    frame_buffer: LaneChangeFrameBuffer,

    // ── Transient state ──
    previous_state: String,
    last_left_lane_x: Option<f32>,
    last_right_lane_x: Option<f32>,
    current_overtake_vehicles: Vec<overtake_analyzer::OvertakeEvent>,
    latest_vehicle_detections: Vec<vehicle_detection::Detection>,

    // ── Counters ──
    frame_count: u64,
    frames_with_valid_position: u64,
    frames_with_fallback: u64,
    lane_changes_count: usize,
    complete_overtakes: usize,
    incomplete_overtakes: usize,
    simple_lane_changes: usize,
    events_sent_to_api: usize,
    total_vehicles_overtaken: usize,
    shadow_overtakes_detected: usize,
    yolo_primary_count: u64,
    ufld_fallback_count: u64,

    // ── NEW: v2 counters ──
    v2_maneuver_events: u64,
    v2_overtakes: u64,
    v2_lane_changes: u64,
    v2_being_overtaken: u64,
}

impl PipelineState {
    fn new(
        config: &types::Config,
        video_path: &Path,
        frame_width: f32,
        frame_height: f32,
        fps: f64,
        max_buffer_frames: usize,
    ) -> Result<Self> {
        let lane_change_config = LaneChangeConfig::from_detection_config(&config.detection);
        let mut analyzer = LaneChangeAnalyzer::new(lane_change_config);
        analyzer.set_source_id(video_path.to_string_lossy().to_string());

        let yolo_detector =
            YoloDetector::new("models/yolov8n.onnx").context("Failed to load YOLO model")?;

        let legality_detector = if config.lane_legality.enabled {
            match LaneLegalityDetector::new(&config.lane_legality.model_path) {
                Ok(mut det) => {
                    let ego = config.lane_legality.ego_bbox_ratio;
                    det.set_ego_bbox_ratio(ego[0], ego[1], ego[2], ego[3]);
                    Some(det)
                }
                Err(e) => {
                    warn!("⚠️ Lane legality detector failed: {}. Continuing.", e);
                    None
                }
            }
        } else {
            None
        };

        // ── Initialize v2 maneuver pipeline ──
        let maneuver_pipeline_v2 = Some(ManeuverPipeline::with_config(
            ManeuverPipelineConfig::mining_route(),
            frame_width,
            frame_height,
        ));
        info!("✓ Maneuver Detection v2 pipeline initialized (mining_route preset)");

        Ok(Self {
            analyzer,
            yolo_detector,
            legality_detector,
            overtake_analyzer: OvertakeAnalyzer::new(frame_width, frame_height),
            overtake_tracker: OvertakeTracker::new(30.0, fps),
            shadow_detector: ShadowOvertakeDetector::new(frame_width, frame_height),
            velocity_tracker: analysis::velocity_tracker::LateralVelocityTracker::new(),
            fallback_estimator: FallbackPositionEstimator::new(frame_width, frame_height),
            inference_scheduler: InferenceScheduler::new(),

            maneuver_pipeline_v2,

            legality_buffer: LegalityRingBuffer::with_capacity(300),
            frame_buffer: LaneChangeFrameBuffer::new(max_buffer_frames),

            previous_state: "CENTERED".to_string(),
            last_left_lane_x: None,
            last_right_lane_x: None,
            current_overtake_vehicles: Vec::new(),
            latest_vehicle_detections: Vec::new(),

            frame_count: 0,
            frames_with_valid_position: 0,
            frames_with_fallback: 0,
            lane_changes_count: 0,
            complete_overtakes: 0,
            incomplete_overtakes: 0,
            simple_lane_changes: 0,
            events_sent_to_api: 0,
            total_vehicles_overtaken: 0,
            shadow_overtakes_detected: 0,
            yolo_primary_count: 0,
            ufld_fallback_count: 0,

            v2_maneuver_events: 0,
            v2_overtakes: 0,
            v2_lane_changes: 0,
            v2_being_overtaken: 0,
        })
    }
}

// ============================================================================
// PROCESSING STATS (final report)
// ============================================================================

struct ProcessingStats {
    total_frames: u64,
    frames_with_position: u64,
    frames_with_fallback: u64,
    lane_changes_detected: usize,
    complete_overtakes: usize,
    incomplete_overtakes: usize,
    simple_lane_changes: usize,
    events_sent_to_api: usize,
    total_vehicles_overtaken: usize,
    #[allow(dead_code)]
    unique_vehicles_seen: u32,
    shadow_overtakes_detected: usize,
    duration_secs: f64,
    avg_fps: f64,
    yolo_primary_count: u64,
    ufld_fallback_count: u64,
    // ── NEW: v2 stats ──
    v2_maneuver_events: u64,
    v2_overtakes: u64,
    v2_lane_changes: u64,
    v2_being_overtaken: u64,
}

impl ProcessingStats {
    fn from_pipeline(ps: &PipelineState, duration: std::time::Duration) -> Self {
        let avg_fps = ps.frame_count as f64 / duration.as_secs_f64();
        Self {
            total_frames: ps.frame_count,
            frames_with_position: ps.frames_with_valid_position,
            frames_with_fallback: ps.frames_with_fallback,
            lane_changes_detected: ps.lane_changes_count,
            complete_overtakes: ps.complete_overtakes,
            incomplete_overtakes: ps.incomplete_overtakes,
            simple_lane_changes: ps.simple_lane_changes,
            events_sent_to_api: ps.events_sent_to_api,
            total_vehicles_overtaken: ps.total_vehicles_overtaken,
            unique_vehicles_seen: ps.overtake_analyzer.get_total_unique_vehicles(),
            shadow_overtakes_detected: ps.shadow_overtakes_detected,
            duration_secs: duration.as_secs_f64(),
            avg_fps,
            yolo_primary_count: ps.yolo_primary_count,
            ufld_fallback_count: ps.ufld_fallback_count,
            v2_maneuver_events: ps.v2_maneuver_events,
            v2_overtakes: ps.v2_overtakes,
            v2_lane_changes: ps.v2_lane_changes,
            v2_being_overtaken: ps.v2_being_overtaken,
        }
    }
}

// ============================================================================
// HELPERS
// ============================================================================

/// Convert FusedLegalityResult → LegalityResult for overlay / event attachment.
fn fused_to_legality_result(fused: &FusedLegalityResult, frame_id: u64) -> LegalityResult {
    LegalityResult {
        verdict: fused.verdict,
        intersecting_line: fused.line_type_from_seg_model.clone(),
        all_markings: fused.all_markings.clone(),
        ego_intersects_marking: fused.ego_intersects_marking,
        frame_id,
    }
}

// ============================================================================
// ENTRY POINT
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "overtake_detection=info,ort=warn".to_string()),
        )
        .with_target(true)
        .with_thread_ids(true)
        .init();

    info!("🚗 Lane Change Detection System Starting (v4.2 — Maneuver v2 Integrated)");

    let config = types::Config::load("config.yaml").context("Failed to load config.yaml")?;
    validate_config(&config)?;
    info!("✓ Configuration loaded and validated");

    // ── Graceful shutdown ──
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

    // ── Inference engine (UFLDv2 fallback) ──
    let mut inference_engine = inference::InferenceEngine::new(config.clone())
        .context("Failed to initialize lane inference engine")?;
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

    let mut api_client = ApiClient::new(30).context("Failed to create API client")?;

    // ── Process each video ──
    for (idx, video_path) in video_files.iter().enumerate() {
        if *shutdown_rx.borrow() {
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
            Ok(stats) => print_final_stats(&stats),
            Err(e) => error!("Failed to process video {}: {:#}", video_path.display(), e),
        }
    }

    info!("🏁 All videos processed. Exiting.");
    Ok(())
}

fn validate_config(config: &types::Config) -> Result<()> {
    if !Path::new(&config.model.path).exists() {
        anyhow::bail!("Lane model not found: {}", config.model.path);
    }
    if config.lane_legality.enabled && !Path::new(&config.lane_legality.model_path).exists() {
        anyhow::bail!(
            "Lane legality model not found: {}.",
            config.lane_legality.model_path
        );
    }
    Ok(())
}

// ============================================================================
// VIDEO PROCESSING — orchestrator
// ============================================================================

async fn process_video(
    video_path: &Path,
    _inference_engine: &mut inference::InferenceEngine,
    video_processor: &video_processor::VideoProcessor,
    config: &types::Config,
    legality_config: &LegalityAnalysisConfig,
    api_client: &mut ApiClient,
    shutdown_rx: watch::Receiver<bool>,
) -> Result<ProcessingStats> {
    let start_time = Instant::now();

    let mut reader = video_processor
        .open_video(video_path)
        .with_context(|| format!("Failed to open video: {}", video_path.display()))?;

    let mut writer =
        video_processor.create_writer(video_path, reader.width, reader.height, reader.fps)?;

    // ── Setup results file ──
    std::fs::create_dir_all(&config.video.output_dir)?;
    let video_name = video_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let jsonl_path =
        Path::new(&config.video.output_dir).join(format!("{}_overtakes.jsonl", video_name));
    let mut results_file = std::fs::File::create(&jsonl_path)?;

    // ── Initialise pipeline state ──
    let mut ps = PipelineState::new(
        config,
        video_path,
        reader.width as f32,
        reader.height as f32,
        reader.fps,
        legality_config.max_buffer_frames,
    )?;

    // ═══════════════════════════════════════════════════════════════════
    // MAIN FRAME LOOP
    // ═══════════════════════════════════════════════════════════════════

    while let Some(frame) = reader.read_frame()? {
        ps.frame_count += 1;
        let timestamp_ms = frame.timestamp_ms;

        if *shutdown_rx.borrow() {
            break;
        }

        // Wrap in Arc for cheap sharing
        let arc_frame = Arc::new(frame);

        // ── STAGE 1: Vehicle Detection ──
        run_vehicle_detection(&mut ps, &arc_frame, timestamp_ms)?;

        // ── STAGE 2: Lane Detection ──
        let (analysis_lanes, detected_lanes_for_draw, detection_source) =
            run_lane_detection(&mut ps, &arc_frame, config, timestamp_ms)?;

        // ── STAGE 3: Position Analysis (old pipeline) ──
        let event_opt = run_position_analysis(&mut ps, &analysis_lanes, timestamp_ms);

        // ── STAGE 3B: Maneuver Detection v2 (parallel) ──
        run_maneuver_pipeline_v2(&mut ps, &arc_frame, &analysis_lanes, timestamp_ms);

        // ── STAGE 4: Handle detected events (old pipeline) ──
        if let Some(event) = event_opt {
            ps.lane_changes_count += 1;
            process_lane_change_event(
                event,
                &mut ps,
                &mut results_file,
                legality_config,
                config,
                api_client,
            )
            .await?;
        }

        // ── STAGE 5: Frame buffer management ──
        manage_frame_buffer(&mut ps, &arc_frame);

        // ── STAGE 6: Overtake timeout check ──
        if ps.frame_count % 30 == 0 {
            check_overtake_timeout(&mut ps, &mut results_file, &analysis_lanes)?;
        }

        // ── STAGE 7: Video annotation ──
        if let Some(ref mut w) = writer {
            run_video_annotation(
                w,
                &ps,
                &arc_frame,
                &detected_lanes_for_draw,
                detection_source,
                reader.width,
                reader.height,
                timestamp_ms,
            )?;
        }
    }

    let duration = start_time.elapsed();
    Ok(ProcessingStats::from_pipeline(&ps, duration))
}

// ============================================================================
// STAGE 1 — Vehicle Detection
// ============================================================================

fn run_vehicle_detection(
    ps: &mut PipelineState,
    frame: &Arc<Frame>,
    timestamp_ms: f64,
) -> Result<()> {
    let frame_count = ps.frame_count;

    let is_maneuvering =
        ps.analyzer.current_state() == "DRIFTING" || ps.analyzer.current_state() == "CROSSING";

    let ufld_confidence = ps
        .analyzer
        .last_vehicle_state()
        .map(|vs| vs.detection_confidence)
        .unwrap_or(0.0);

    let should_run =
        ps.inference_scheduler
            .should_run_yolo(ufld_confidence, ufld_confidence, is_maneuvering, 0);

    if !should_run {
        return Ok(());
    }

    if let Ok(detections) = ps
        .yolo_detector
        .detect(&frame.data, frame.width, frame.height, 0.3)
    {
        ps.latest_vehicle_detections = detections.clone();

        ps.overtake_analyzer.update(detections, frame_count);

        ps.overtake_analyzer
            .set_overtake_active(ps.overtake_tracker.is_tracking());

        ps.overtake_tracker
            .set_vehicles_being_passed(ps.overtake_analyzer.get_active_vehicle_count() > 0);

        if ps.shadow_detector.is_monitoring() {
            if let Some(shadow_event) = ps.shadow_detector.update(
                ps.overtake_analyzer.get_tracked_vehicles(),
                ps.last_left_lane_x,
                ps.last_right_lane_x,
                frame_count,
                timestamp_ms,
            ) {
                ps.shadow_overtakes_detected += 1;
                ps.overtake_tracker.set_shadow_active(true);
                let _ = save_shadow_event(&shadow_event, &mut std::io::sink());
            }
        }
    }

    Ok(())
}

// ============================================================================
// STAGE 2 — Lane Detection
// ============================================================================

fn run_lane_detection(
    ps: &mut PipelineState,
    frame: &Arc<Frame>,
    config: &types::Config,
    timestamp_ms: f64,
) -> Result<(Vec<Lane>, Vec<DetectedLane>, &'static str)> {
    let frame_count = ps.frame_count;
    let mut analysis_lanes: Vec<Lane> = Vec::new();
    let mut detected_lanes_for_draw: Vec<DetectedLane> = Vec::new();
    let mut detection_source: &str = "NONE";
    let center_x = frame.width as f32 / 2.0;

    if let Some(ref mut detector) = ps.legality_detector {
        match detector.estimate_ego_lane_boundaries_stable(
            &frame.data,
            frame.width,
            frame.height,
            center_x,
        ) {
            Ok(Some((left_x, right_x, conf))) => {
                detection_source = "YOLO_SEG";
                ps.yolo_primary_count += 1;

                let y_bottom = frame.height as f32;
                let y_top = frame.height as f32 * 0.45;
                let y_mid = (y_bottom + y_top) / 2.0;

                let left_dl = DetectedLane {
                    points: vec![(left_x, y_bottom), (left_x, y_mid), (left_x, y_top)],
                    confidence: conf,
                };
                let right_dl = DetectedLane {
                    points: vec![(right_x, y_bottom), (right_x, y_mid), (right_x, y_top)],
                    confidence: conf,
                };

                detected_lanes_for_draw = vec![left_dl.clone(), right_dl.clone()];
                analysis_lanes = vec![
                    Lane::from_detected(0, &left_dl),
                    Lane::from_detected(1, &right_dl),
                ];

                if frame_count % 3 == 0 {
                    let vehicle_offset = ps
                        .analyzer
                        .last_vehicle_state()
                        .map(|vs| vs.lateral_offset)
                        .unwrap_or(0.0);
                    let lane_width = ps
                        .analyzer
                        .last_vehicle_state()
                        .and_then(|vs| vs.lane_width);

                    buffer_legality_result(
                        detector,
                        &mut ps.legality_buffer,
                        frame,
                        config,
                        left_x,
                        right_x,
                        frame_count,
                        timestamp_ms,
                        vehicle_offset,
                        lane_width,
                    );
                }
            }
            _ => {
                detection_source = "YOLO_MISS";
                ps.ufld_fallback_count += 1;
            }
        }
    }

    Ok((analysis_lanes, detected_lanes_for_draw, detection_source))
}

/// Helper with disjoint arguments to satisfy borrow checker
#[allow(clippy::too_many_arguments)]
fn buffer_legality_result(
    detector: &mut LaneLegalityDetector,
    buffer: &mut LegalityRingBuffer,
    frame: &Arc<Frame>,
    config: &types::Config,
    left_x: f32,
    right_x: f32,
    frame_count: u64,
    _timestamp_ms: f64,
    vehicle_offset: f32,
    lane_width: Option<f32>,
) {
    let crossing_side = if vehicle_offset < -50.0 {
        lane_legality::CrossingSide::Left
    } else if vehicle_offset > 50.0 {
        lane_legality::CrossingSide::Right
    } else {
        lane_legality::CrossingSide::None
    };

    if let Ok(fused) = detector.analyze_frame_fused(
        &frame.data,
        frame.width,
        frame.height,
        frame_count,
        config.lane_legality.confidence_threshold,
        vehicle_offset,
        lane_width,
        Some(left_x),
        Some(right_x),
        crossing_side,
    ) {
        buffer.push(frame_count, _timestamp_ms, fused);
    }
}

// ============================================================================
// STAGE 3 — Position Analysis (old pipeline)
// ============================================================================

fn run_position_analysis(
    ps: &mut PipelineState,
    analysis_lanes: &[Lane],
    timestamp_ms: f64,
) -> Option<LaneChangeEvent> {
    let frame_count = ps.frame_count;

    let (left_bound, right_bound) = extract_lane_boundaries(analysis_lanes, 1280, 720, 0.8);
    ps.last_left_lane_x = left_bound;
    ps.last_right_lane_x = right_bound;

    let event_opt =
        ps.analyzer
            .analyze_perfect(analysis_lanes, 1280, 720, frame_count, timestamp_ms);

    if ps
        .analyzer
        .last_vehicle_state()
        .map_or(false, |vs| vs.is_valid())
    {
        if let Some(vs) = ps.analyzer.last_vehicle_state() {
            ps.fallback_estimator.sync_from_primary(
                vs.lateral_offset,
                vs.lane_width.unwrap_or(450.0),
                frame_count,
            );
        }
        ps.frames_with_valid_position += 1;
    } else {
        try_fallback_estimation(ps, frame_count, timestamp_ms);
    }

    event_opt
}

fn try_fallback_estimation(ps: &mut PipelineState, frame_count: u64, timestamp_ms: f64) {
    let road_markings = ps
        .legality_buffer
        .latest()
        .map(|r| r.all_markings.clone())
        .unwrap_or_default();

    if let Some(fallback) = ps.fallback_estimator.estimate_fallback(
        &road_markings,
        &ps.latest_vehicle_detections,
        frame_count,
    ) {
        ps.frames_with_fallback += 1;

        let current_state = ps.analyzer.current_state();
        let is_already_maneuvering = current_state == "DRIFTING" || current_state == "CROSSING";

        if is_already_maneuvering
            || (fallback.confidence > 0.75
                && ps.fallback_estimator.fallback_lateral_velocity.abs() > 200.0)
        {
            let synthetic_state = VehicleState {
                lateral_offset: fallback.lateral_offset,
                lane_width: Some(fallback.lane_width),
                heading_offset: 0.0,
                frame_id: frame_count,
                timestamp_ms,
                raw_offset: fallback.lateral_offset,
                detection_confidence: fallback.confidence,
                both_lanes_detected: false,
            };

            if fallback.confidence > 0.25 {
                if let Some(mut fallback_event) =
                    ps.analyzer
                        .analyze_with_state(&synthetic_state, frame_count, timestamp_ms)
                {
                    fallback_event.metadata.insert(
                        "detection_source".to_string(),
                        serde_json::json!(fallback.source.as_str()),
                    );
                    ps.lane_changes_count += 1;
                }
            }
        }
    }
}

// ============================================================================
// STAGE 3B — Maneuver Detection v2 (parallel A/B validation)
// ============================================================================

/// Runs the v2 maneuver detection pipeline in parallel with the old pipeline.
/// Uses disjoint field access to satisfy the borrow checker:
///   - &mut ps.maneuver_pipeline_v2  (mutable, for process_frame)
///   - &ps.legality_buffer           (immutable, passed as reference)
///   - &ps.latest_vehicle_detections (immutable, converted to DetectionInput)
///   - &ps.analyzer                  (immutable, read lane state)
///
/// Rust 2021 edition allows this because the borrows target disjoint struct fields.
fn run_maneuver_pipeline_v2(
    ps: &mut PipelineState,
    frame: &Arc<Frame>,
    analysis_lanes: &[Lane],
    timestamp_ms: f64,
) {
    let pipeline_v2 = match ps.maneuver_pipeline_v2.as_mut() {
        Some(p) => p,
        None => return,
    };

    let frame_count = ps.frame_count;

    // ── Build lane measurement from the old pipeline's analyzer state ──
    let lane_meas: Option<LaneMeasurement> = ps
        .analyzer
        .last_vehicle_state()
        .filter(|vs| vs.is_valid() && vs.detection_confidence > 0.2)
        .map(|vs| LaneMeasurement {
            lateral_offset_px: vs.lateral_offset,
            lane_width_px: vs.lane_width.unwrap_or(450.0),
            confidence: vs.detection_confidence,
            both_lanes: vs.both_lanes_detected,
        });

    // ── Convert existing YOLO detections to v2 DetectionInput ──
    let vehicle_dets: Vec<DetectionInput> = ps
        .latest_vehicle_detections
        .iter()
        .map(|d| DetectionInput {
            bbox: [d.x1, d.y1, d.x2, d.y2],
            class_id: (d.class_id as u32),
            confidence: d.confidence,
        })
        .collect();

    // ── Convert frame to grayscale for ego-motion estimation ──
    // NOTE: OpenCV gives BGR, but the SAD block-matching in ego_motion is
    // channel-agnostic (luminance coefficients differ by <3%) so this is fine.
    let gray = GrayFrame::from_rgb(&frame.data, frame.width as usize, frame.height as usize);

    // ── Extract marking names from the legality buffer's latest entry ──
    let latest_legality = ps.legality_buffer.latest();
    let left_marking: Option<String> = latest_legality.as_ref().and_then(|fused| {
        fused
            .all_markings
            .iter()
            .find(|m| m.side == "left")
            .map(|m| m.class_name.clone())
    });
    let right_marking: Option<String> = latest_legality.as_ref().and_then(|fused| {
        fused
            .all_markings
            .iter()
            .find(|m| m.side == "right")
            .map(|m| m.class_name.clone())
    });

    // ── Process frame through v2 pipeline ──
    let v2_result = pipeline_v2.process_frame(ManeuverFrameInput {
        vehicle_detections: &vehicle_dets,
        lane_measurement: lane_meas,
        gray_frame: Some(&gray),
        left_marking_name: left_marking.as_deref(),
        right_marking_name: right_marking.as_deref(),
        legality_buffer: Some(&ps.legality_buffer),
        timestamp_ms,
        frame_id: frame_count,
    });

    // ── Log v2 events and update counters ──
    for event in &v2_result.maneuver_events {
        ps.v2_maneuver_events += 1;

        match event.maneuver_type {
            analysis::maneuver_classifier::ManeuverType::Overtake => ps.v2_overtakes += 1,
            analysis::maneuver_classifier::ManeuverType::LaneChange => ps.v2_lane_changes += 1,
            analysis::maneuver_classifier::ManeuverType::BeingOvertaken => {
                ps.v2_being_overtaken += 1
            }
        }

        info!(
            "🆕 V2 {} {} | conf={:.2} | sources={} | legality={:?} | frame={}",
            event.maneuver_type.as_str(),
            event.side.as_str(),
            event.confidence,
            event.sources.summary(),
            event.legality,
            frame_count,
        );
    }

    // ── A/B comparison: log when old pipeline detects something this frame ──
    // (The old pipeline event is handled in STAGE 4 — here we just log v2 state
    //  for later comparison in the processing stats)
    if v2_result.pass_in_progress && frame_count % 90 == 0 {
        info!(
            "📊 V2 state: tracks={} | pass=IN_PROGRESS | lateral={} | ego={:.1}px/f",
            v2_result.tracked_vehicle_count,
            v2_result.lateral_state,
            v2_result.ego_lateral_velocity,
        );
    }
}

// ============================================================================
// STAGE 5 — Frame Buffer Management
// ============================================================================

fn manage_frame_buffer(ps: &mut PipelineState, frame: &Arc<Frame>) {
    let current_state = ps.analyzer.current_state().to_string();
    let frame_count = ps.frame_count;

    if ps.previous_state == "CENTERED" {
        ps.frame_buffer.add_to_pre_buffer((**frame).clone());
    }

    if ps.previous_state == "CENTERED" && current_state == "DRIFTING" {
        ps.frame_buffer.start_capture(frame_count);
    }

    if ps.frame_buffer.is_capturing() {
        ps.frame_buffer.add_frame((**frame).clone());
    }

    if current_state == "CENTERED" && ps.frame_buffer.is_capturing() {
        ps.frame_buffer.cancel_capture();
    }

    ps.previous_state = current_state;
}

// ============================================================================
// STAGE 6 — Overtake Timeout Check
// ============================================================================

fn check_overtake_timeout(
    ps: &mut PipelineState,
    results_file: &mut std::fs::File,
    analysis_lanes: &[Lane],
) -> Result<()> {
    let lanes_visible = !analysis_lanes.is_empty();
    let frame_count = ps.frame_count;

    if let Some(timeout_result) = ps
        .overtake_tracker
        .check_timeout(frame_count, lanes_visible)
    {
        if let OvertakeResult::Incomplete {
            start_event,
            reason,
        } = timeout_result
        {
            ps.incomplete_overtakes += 1;
            ps.shadow_detector.stop_monitoring();
            ps.current_overtake_vehicles.clear();
            ps.overtake_tracker.set_shadow_active(false);
            save_incomplete_overtake(&start_event, &reason, results_file)?;
        }
    }
    Ok(())
}

// ============================================================================
// STAGE 7 — Video Annotation
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn run_video_annotation(
    writer: &mut opencv::videoio::VideoWriter,
    ps: &PipelineState,
    frame: &Arc<Frame>,
    detected_lanes: &[DetectedLane],
    detection_source: &str,
    width: i32,
    height: i32,
    timestamp_ms: f64,
) -> Result<()> {
    let frame_count = ps.frame_count;
    let curve_info = ps.analyzer.get_curve_info();
    let is_overtaking = ps.overtake_tracker.is_tracking();
    let overtake_direction = if is_overtaking {
        Some(ps.overtake_tracker.get_direction().as_str())
    } else {
        None
    };

    let lateral_velocity = 0.0_f32;

    let legality_owned: Option<LegalityResult> = ps.legality_buffer.latest_as_legality_result();
    let legality_ref = legality_owned.as_ref();

    if let Ok(annotated) = video_processor::draw_lanes_with_state_enhanced(
        &frame.data,
        width,
        height,
        detected_lanes,
        ps.analyzer.current_state(),
        ps.analyzer.last_vehicle_state(),
        ps.overtake_analyzer.get_tracked_vehicles(),
        &ps.shadow_detector,
        frame_count,
        timestamp_ms,
        is_overtaking,
        overtake_direction,
        &ps.current_overtake_vehicles,
        Some(curve_info),
        lateral_velocity,
        legality_ref,
        detection_source,
    ) {
        use opencv::videoio::VideoWriterTrait;
        writer.write(&annotated)?;
    }

    Ok(())
}

// ============================================================================
// EVENT HANDLING
// ============================================================================

#[allow(clippy::too_many_arguments)]
async fn process_lane_change_event(
    event: LaneChangeEvent,
    ps: &mut PipelineState,
    results_file: &mut std::fs::File,
    legality_config: &LegalityAnalysisConfig,
    config: &types::Config,
    api_client: &mut ApiClient,
) -> Result<()> {
    let frame_count = ps.frame_count;
    let tracker_result = ps
        .overtake_tracker
        .process_lane_change(event.clone(), frame_count);

    match tracker_result {
        None => {
            ps.shadow_detector
                .start_monitoring(event.direction, frame_count);
        }

        Some(OvertakeResult::Complete {
            start_event,
            end_event,
            total_duration_ms,
            ..
        }) => {
            info!(
                "✅ COMPLETE OVERTAKE: duration {:.1}s",
                total_duration_ms / 1000.0
            );

            let shadow_events = ps.shadow_detector.stop_monitoring();
            if !shadow_events.is_empty() {
                ps.shadow_overtakes_detected += shadow_events.len();
            }
            ps.overtake_tracker.set_shadow_active(false);

            let overtakes = ps.overtake_analyzer.analyze_overtake(
                start_event.start_frame_id,
                end_event.end_frame_id,
                start_event.direction_name(),
            );
            ps.current_overtake_vehicles = overtakes.clone();
            ps.total_vehicles_overtaken += overtakes.len();

            let mut combined_event =
                create_combined_event(&start_event, &end_event, total_duration_ms, &overtakes);
            attach_shadow_metadata(&mut combined_event, &shadow_events);

            if let Some(worst_fused) = ps
                .legality_buffer
                .worst_in_range(start_event.start_frame_id, end_event.end_frame_id)
            {
                let lr = fused_to_legality_result(&worst_fused, start_event.start_frame_id);
                attach_legality_to_event(&mut combined_event, &lr);
            } else if let Some(closest_fused) = ps
                .legality_buffer
                .closest_to_frame(start_event.start_frame_id)
            {
                let lr = fused_to_legality_result(&closest_fused, start_event.start_frame_id);
                attach_legality_to_event(&mut combined_event, &lr);
            }

            let curve_info = ps.analyzer.get_curve_info();
            let (captured_frames, buffer_start_id) = ps.frame_buffer.stop_capture();

            if !captured_frames.is_empty() {
                let critical_entry = ps
                    .legality_buffer
                    .worst_in_range(start_event.start_frame_id, end_event.end_frame_id);

                let critical_index = critical_entry.map(|_| captured_frames.len() / 2);

                if let Err(e) = send_overtake_to_api(
                    &combined_event,
                    &captured_frames,
                    curve_info,
                    legality_config,
                    config,
                    api_client,
                    critical_index,
                )
                .await
                {
                    warn!("Failed to send overtake to API: {:#}", e);
                } else {
                    ps.events_sent_to_api += 1;
                }
            }

            save_complete_overtake(&combined_event, results_file)?;
            ps.complete_overtakes += 1;
        }

        Some(OvertakeResult::Incomplete {
            start_event,
            reason,
        }) => {
            warn!("⚠️  INCOMPLETE OVERTAKE: {}", reason);
            ps.incomplete_overtakes += 1;

            let shadow_events = ps.shadow_detector.stop_monitoring();
            if !shadow_events.is_empty() {
                ps.shadow_overtakes_detected += shadow_events.len();
            }
            ps.current_overtake_vehicles.clear();
            ps.overtake_tracker.set_shadow_active(false);

            let mut incomplete_event = start_event.clone();
            incomplete_event.metadata.insert(
                "maneuver_type".to_string(),
                serde_json::json!("incomplete_overtake"),
            );
            incomplete_event
                .metadata
                .insert("incomplete_reason".to_string(), serde_json::json!(reason));
            attach_shadow_metadata(&mut incomplete_event, &shadow_events);

            if let Some(worst_fused) = ps
                .legality_buffer
                .worst_in_range(start_event.start_frame_id, frame_count)
            {
                let lr = fused_to_legality_result(&worst_fused, start_event.start_frame_id);
                attach_legality_to_event(&mut incomplete_event, &lr);
            }

            let (captured_frames, _) = ps.frame_buffer.force_flush();

            if !captured_frames.is_empty() {
                let curve_info = ps.analyzer.get_curve_info();
                if let Err(e) = send_overtake_to_api(
                    &incomplete_event,
                    &captured_frames,
                    curve_info,
                    legality_config,
                    config,
                    api_client,
                    None,
                )
                .await
                {
                    warn!("Failed to send incomplete overtake to API: {:#}", e);
                } else {
                    ps.events_sent_to_api += 1;
                }
            }

            save_incomplete_overtake(&incomplete_event, &reason, results_file)?;

            if ps.analyzer.current_state() == "DRIFTING"
                || ps.analyzer.current_state() == "CROSSING"
            {
                ps.frame_buffer.start_capture(frame_count);
                ps.shadow_detector
                    .start_monitoring(event.direction, frame_count);
            }
        }

        Some(OvertakeResult::SimpleLaneChange { event: _ }) => {
            ps.simple_lane_changes += 1;
        }
    }

    Ok(())
}

// ============================================================================
// EVENT METADATA HELPERS
// ============================================================================

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

fn save_shadow_event(shadow: &ShadowOvertakeEvent, file: &mut impl std::io::Write) -> Result<()> {
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
    Ok(())
}

async fn send_overtake_to_api(
    event: &LaneChangeEvent,
    captured_frames: &[Frame],
    curve_info: CurveInfo,
    legality_config: &LegalityAnalysisConfig,
    config: &types::Config,
    api_client: &mut ApiClient,
    critical_frame_index: Option<usize>,
) -> Result<()> {
    if captured_frames.is_empty() {
        return Ok(());
    }

    let request = build_legality_request(
        event,
        captured_frames,
        legality_config.num_frames_to_analyze,
        curve_info,
        critical_frame_index,
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

fn print_final_stats(stats: &ProcessingStats) {
    info!("\n✓ Video processed successfully!");
    info!("  Total frames: {}", stats.total_frames);
    info!(
        "  Valid position frames: {} ({:.1}%)",
        stats.frames_with_position,
        100.0 * stats.frames_with_position as f64 / stats.total_frames.max(1) as f64
    );
    info!(
        "  🔶 Fallback estimation: {} frames ({:.1}%)",
        stats.frames_with_fallback,
        100.0 * stats.frames_with_fallback as f64 / stats.total_frames.max(1) as f64
    );

    info!("\n╔════════════════════════════════════════════════════════╗");
    info!("║         LANE DETECTION SOURCE STATISTICS              ║");
    info!("╠════════════════════════════════════════════════════════╣");
    info!(
        "║ YOLOv8-seg Primary: {:>5} frames ({:.1}%)             ║",
        stats.yolo_primary_count,
        stats.yolo_primary_count as f64 / stats.total_frames.max(1) as f64 * 100.0
    );
    info!(
        "║ UFLDv2 Fallback:    {:>5} frames ({:.1}%)             ║",
        stats.ufld_fallback_count,
        stats.ufld_fallback_count as f64 / stats.total_frames.max(1) as f64 * 100.0
    );
    info!("╚════════════════════════════════════════════════════════╝");

    info!("\n  Lane changes detected: {}", stats.lane_changes_detected);
    info!("  ✅ Complete overtakes: {}", stats.complete_overtakes);
    info!("  ⚠️  Incomplete overtakes: {}", stats.incomplete_overtakes);
    info!("  ↔️  Simple lane changes: {}", stats.simple_lane_changes);
    info!("  Events sent to API: {}", stats.events_sent_to_api);
    info!(
        "  Shadow overtakes detected: {}",
        stats.shadow_overtakes_detected
    );

    // ── NEW: v2 pipeline stats ──
    info!("\n╔════════════════════════════════════════════════════════╗");
    info!("║       MANEUVER DETECTION v2 (A/B VALIDATION)         ║");
    info!("╠════════════════════════════════════════════════════════╣");
    info!(
        "║ Total v2 maneuver events:   {:>5}                     ║",
        stats.v2_maneuver_events
    );
    info!(
        "║   🚗 Overtakes:             {:>5}                     ║",
        stats.v2_overtakes
    );
    info!(
        "║   🔀 Lane changes:          {:>5}                     ║",
        stats.v2_lane_changes
    );
    info!(
        "║   ⚠️  Being overtaken:       {:>5}                     ║",
        stats.v2_being_overtaken
    );
    info!("╚════════════════════════════════════════════════════════╝");

    // ── A/B comparison summary ──
    if stats.complete_overtakes > 0 || stats.v2_overtakes > 0 {
        let old_total = stats.complete_overtakes + stats.incomplete_overtakes;
        info!(
            "\n  📊 A/B: old_pipeline={} overtakes | v2_pipeline={} overtakes",
            old_total, stats.v2_overtakes
        );
    }

    info!(
        "\n  Processing: {:.1} FPS ({:.1}s total)",
        stats.avg_fps, stats.duration_secs
    );
}
