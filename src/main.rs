// src/main.rs
//
// Production overtake detection pipeline v6.0
//
// v4.11: boundary_coherence → LaneMeasurement for curve-aware detection
// v4.13: polynomial curvature from YOLO-seg masks
// v5.2:  DetectionCache + LineCrossingDetector (wired below)
// v6.0:  RoadClassifier + CrossingFlash + zone-based visualization
//        All previously dead-code modules are now active in the pipeline.

mod analysis;
mod color_analysis;
mod lane_crossing;
mod lane_crossing_integration;
mod lane_detection;
mod lane_legality;
mod lane_legality_patches;
mod pipeline;
mod road_classification;
mod road_overlay;
mod types;
mod vehicle_detection;
mod video_processor;

use analysis::ego_motion::GrayFrame;
use analysis::lateral_detector::LaneMeasurement;
use analysis::maneuver_pipeline::{
    ManeuverFrameInput, ManeuverFrameOutput, ManeuverPipeline, ManeuverPipelineConfig,
};
use analysis::vehicle_tracker::DetectionInput;

use anyhow::{Context, Result};
use lane_crossing::CacheState;
use lane_crossing_integration::LaneCrossingState;
use lane_legality::{FusedLegalityResult, LaneLegalityDetector, LegalityResult};
use pipeline::legality_buffer::LegalityRingBuffer;
use road_classification::{MixedLineSide, PassingLegality, RoadClassification, RoadClassifier};
use road_overlay::CrossingFlashState;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::watch;
use tracing::{debug, error, info, warn};
use types::{DetectedLane, Frame, VehicleState};
use vehicle_detection::YoloDetector;

// ============================================================================
// LAST MANEUVER INFO (for persistent display)
// ============================================================================

#[derive(Clone)]
pub struct LastManeuverInfo {
    pub maneuver_type: String,
    pub side: String,
    pub confidence: f32,
    pub legality: String,
    pub duration_ms: f64,
    pub sources: String,
    pub frame_detected: u64,
    pub timestamp_detected: f64,
    pub vehicles_in_this_maneuver: usize,
}

// ============================================================================
// PIPELINE STATE
// ============================================================================

struct PipelineState {
    // ── Core components ──
    yolo_detector: YoloDetector,
    legality_detector: Option<LaneLegalityDetector>,
    maneuver_pipeline_v2: ManeuverPipeline,
    legality_buffer: LegalityRingBuffer,

    // ── v5.2/v6.0: New subsystems ──
    lane_crossing_state: LaneCrossingState,
    road_classifier: RoadClassifier,
    crossing_flash: Option<CrossingFlashState>,
    last_road_classification: RoadClassification,

    // ── Transient state ──
    latest_vehicle_detections: Vec<vehicle_detection::Detection>,
    last_vehicle_state: Option<VehicleState>,
    last_maneuver: Option<LastManeuverInfo>,
    /// Cached MarkingInfo vec from the most recent legality buffer entry.
    /// Updated every time buffer_legality_result runs (every 3rd frame).
    latest_marking_infos: Vec<road_classification::MarkingInfo>,

    // ── Counters ──
    frame_count: u64,
    yolo_primary_count: u64,
    v2_maneuver_events: u64,
    v2_overtakes: u64,
    v2_lane_changes: u64,
    v2_being_overtaken: u64,
    v2_vehicles_overtaken: u64,
    crossing_events_total: u64,
    crossing_events_illegal: u64,
}

impl PipelineState {
    fn new(
        config: &types::Config,
        video_path: &Path,
        frame_width: f32,
        frame_height: f32,
    ) -> Result<Self> {
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

        let maneuver_pipeline_v2 = ManeuverPipeline::with_config(
            ManeuverPipelineConfig::mining_route(),
            frame_width,
            frame_height,
        );

        info!(
            "✓ Pipeline v6.0 initialized for video: {} ({}×{})",
            video_path.display(),
            frame_width,
            frame_height,
        );

        Ok(Self {
            yolo_detector,
            legality_detector,
            maneuver_pipeline_v2,
            legality_buffer: LegalityRingBuffer::with_capacity(300),

            // v5.2/v6.0: New subsystems
            lane_crossing_state: LaneCrossingState::new(frame_width, frame_height),
            road_classifier: RoadClassifier::new(frame_width),
            crossing_flash: None,
            last_road_classification: RoadClassification::default(),

            latest_vehicle_detections: Vec::new(),
            last_vehicle_state: None,
            last_maneuver: None,
            latest_marking_infos: Vec::new(),

            frame_count: 0,
            yolo_primary_count: 0,
            v2_maneuver_events: 0,
            v2_overtakes: 0,
            v2_lane_changes: 0,
            v2_being_overtaken: 0,
            v2_vehicles_overtaken: 0,
            crossing_events_total: 0,
            crossing_events_illegal: 0,
        })
    }
}

// ============================================================================
// PROCESSING STATS
// ============================================================================

struct ProcessingStats {
    total_frames: u64,
    yolo_primary_count: u64,
    v2_maneuver_events: u64,
    v2_overtakes: u64,
    v2_lane_changes: u64,
    v2_being_overtaken: u64,
    v2_vehicles_overtaken: u64,
    crossing_events_total: u64,
    crossing_events_illegal: u64,
    cache_recoveries: u64,
    duration_secs: f64,
    avg_fps: f64,
}

impl ProcessingStats {
    fn from_pipeline(ps: &PipelineState, duration: std::time::Duration) -> Self {
        let avg_fps = ps.frame_count as f64 / duration.as_secs_f64();
        Self {
            total_frames: ps.frame_count,
            yolo_primary_count: ps.yolo_primary_count,
            v2_maneuver_events: ps.v2_maneuver_events,
            v2_overtakes: ps.v2_overtakes,
            v2_lane_changes: ps.v2_lane_changes,
            v2_being_overtaken: ps.v2_being_overtaken,
            v2_vehicles_overtaken: ps.v2_vehicles_overtaken,
            crossing_events_total: ps.crossing_events_total,
            crossing_events_illegal: ps.crossing_events_illegal,
            cache_recoveries: ps.lane_crossing_state.detection_cache.cache_recoveries,
            duration_secs: duration.as_secs_f64(),
            avg_fps,
        }
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

    info!("🚗 Maneuver Detection System v6.0 Starting");

    let config = types::Config::load("config.yaml").context("Failed to load config.yaml")?;
    validate_config(&config)?;
    info!("✓ Config loaded: {}", serde_json::to_string(&config)?);

    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let _shutdown_handle = tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for Ctrl+C");
        info!("\n🛑 Shutdown requested");
        let _ = shutdown_tx.send(true);
    });

    let video_processor = video_processor::VideoProcessor::new(config.clone());
    let video_files = video_processor.find_video_files()?;

    for video_path in &video_files {
        let stats = process_video(&config, video_path, &shutdown_rx)?;
        print_final_stats(&stats);
    }

    info!("✓ All videos processed");
    Ok(())
}

fn validate_config(config: &types::Config) -> Result<()> {
    if config.video.input_dir.is_empty() {
        anyhow::bail!("video.input_dir is empty");
    }
    Ok(())
}

// ============================================================================
// VIDEO PROCESSING LOOP
// ============================================================================

fn process_video(
    config: &types::Config,
    video_path: &Path,
    shutdown_rx: &watch::Receiver<bool>,
) -> Result<ProcessingStats> {
    let start_time = Instant::now();

    let video_processor = video_processor::VideoProcessor::new(config.clone());
    let mut reader = video_processor
        .open_video(video_path)
        .context(format!("Failed to open video: {}", video_path.display()))?;

    let mut writer =
        video_processor.create_writer(video_path, reader.width, reader.height, reader.fps)?;

    // ── Setup results file ──
    std::fs::create_dir_all(&config.video.output_dir)?;
    let video_name = video_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let jsonl_path =
        Path::new(&config.video.output_dir).join(format!("{}_maneuvers_v2.jsonl", video_name));
    let mut results_file = std::fs::File::create(&jsonl_path)?;

    // ── Initialize pipeline state ──
    let mut ps = PipelineState::new(
        config,
        video_path,
        reader.width as f32,
        reader.height as f32,
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

        let arc_frame = Arc::new(frame);

        // ── STAGE 1: Vehicle Detection ──
        run_vehicle_detection(&mut ps, &arc_frame)?;

        // ── STAGE 2: Lane Detection + Cache + Road Classification ──
        let (detected_lanes, lane_measurement) =
            run_lane_detection(&mut ps, &arc_frame, config, timestamp_ms)?;

        // ── STAGE 3: Maneuver Detection v2 + Crossing Detection ──
        let v2_output =
            run_maneuver_pipeline_v2(&mut ps, &arc_frame, lane_measurement, timestamp_ms);

        // ── STAGE 4: Save events ──
        for event in &v2_output.maneuver_events {
            save_maneuver_event(event, &mut results_file)?;
        }

        // ── STAGE 5: Video annotation (v6.0 — full AnnotationInput) ──
        if let Some(ref mut w) = writer {
            let tracked_vehicles = ps.maneuver_pipeline_v2.tracked_vehicles();

            run_video_annotation(
                w,
                &ps,
                &arc_frame,
                &detected_lanes,
                &v2_output,
                &tracked_vehicles,
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

fn run_vehicle_detection(ps: &mut PipelineState, frame: &Arc<Frame>) -> Result<()> {
    if let Ok(detections) = ps
        .yolo_detector
        .detect(&frame.data, frame.width, frame.height, 0.25)
    {
        ps.latest_vehicle_detections = detections;
    }
    Ok(())
}

// ============================================================================
// STAGE 2 — Lane Detection + Cache + Road Classification
// ============================================================================

fn run_lane_detection(
    ps: &mut PipelineState,
    frame: &Arc<Frame>,
    config: &types::Config,
    timestamp_ms: f64,
) -> Result<(Vec<DetectedLane>, Option<LaneMeasurement>)> {
    let frame_count = ps.frame_count;
    let mut detected_lanes: Vec<DetectedLane> = Vec::new();
    let mut lane_measurement: Option<LaneMeasurement> = None;
    let center_x = frame.width as f32 / 2.0;

    // ── 2a: Run YOLO-seg boundary estimation ──
    let yolo_result: Option<(f32, f32, f32)> = if let Some(ref mut detector) = ps.legality_detector
    {
        match detector.estimate_ego_lane_boundaries_stable(
            &frame.data,
            frame.width,
            frame.height,
            center_x,
        ) {
            Ok(Some((left_x, right_x, conf))) => {
                ps.yolo_primary_count += 1;
                Some((left_x, right_x, conf))
            }
            Ok(None) => None,
            Err(_) => None,
        }
    } else {
        None
    };

    // ── 2b: Get ego lateral velocity for cache ego-compensation ──
    let ego_lat_vel = ps.maneuver_pipeline_v2.last_ego_velocity().unwrap_or(0.0);

    // ── 2c: Feed detection cache (fresh or miss) ──
    // This is the key v5.2 integration: when YOLO misses, the cache
    // provides ego-compensated boundaries from the last good detection.
    let cache_result = lane_crossing_integration::update_lane_cache(
        &mut ps.lane_crossing_state,
        yolo_result,
        &ps.latest_marking_infos, // from last legality buffer entry
        ego_lat_vel,
        frame_count,
        timestamp_ms,
    );

    // ── 2d: Use cache-aware boundaries (may be cached when YOLO missed) ──
    let effective_boundaries: Option<(f32, f32, f32)> = cache_result.map(|(l, r, c, _)| (l, r, c));
    let cache_state = cache_result
        .map(|(_, _, _, s)| s)
        .unwrap_or(CacheState::Empty);

    if let Some((left_x, right_x, conf)) = effective_boundaries {
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
        detected_lanes = vec![left_dl, right_dl];

        let lane_width_px = (right_x - left_x).abs();
        let lateral_offset_px = center_x - (left_x + lane_width_px / 2.0);

        // v4.11/v4.13: boundary coherence + curvature (only from fresh YOLO)
        let (boundary_coherence, curvature) = if let Some(ref detector) = ps.legality_detector {
            (
                detector.boundary_coherence(),
                detector.curvature_estimate().cloned(),
            )
        } else {
            (-1.0, None)
        };

        lane_measurement = Some(LaneMeasurement {
            lateral_offset_px,
            lane_width_px,
            confidence: conf,
            both_lanes: true,
            boundary_coherence,
            curvature,
        });

        ps.last_vehicle_state = Some(VehicleState {
            lateral_offset: lateral_offset_px,
            lane_width: Some(lane_width_px),
            heading_offset: 0.0,
            frame_id: frame_count,
            timestamp_ms,
            raw_offset: lateral_offset_px,
            detection_confidence: conf,
            both_lanes_detected: true,
        });

        // ── 2e: Buffer legality (every 3 frames, only when YOLO is fresh) ──
        if yolo_result.is_some() && frame_count % 3 == 0 {
            buffer_legality_result(
                ps.legality_detector.as_mut().unwrap(),
                &mut ps.legality_buffer,
                frame,
                config,
                left_x,
                right_x,
                frame_count,
                timestamp_ms,
                lateral_offset_px,
                Some(lane_width_px),
            );

            // ── 2f: Convert fresh markings → MarkingInfo for RoadClassifier ──
            if let Some(latest_fused) = ps.legality_buffer.latest() {
                let marking_infos = lane_legality_patches::detections_to_marking_infos(
                    &latest_fused.all_markings,
                    &frame.data,
                    frame.width,
                    frame.height,
                );

                // ── 2g: Update RoadClassifier ──
                let road_class = ps.road_classifier.update(&marking_infos);
                if road_class.confidence > 0.3 {
                    debug!(
                        "🛣️  Road: {} | Passing: {} | Mixed: {:?} | Lanes: {}",
                        road_class.road_type.as_display_str(),
                        road_class.passing_legality.as_str(),
                        road_class.mixed_line_side,
                        road_class.estimated_lanes,
                    );
                    ps.last_road_classification = *road_class;
                }

                // Cache marking infos for use in non-legality frames
                ps.latest_marking_infos = marking_infos;
            }
        }
    }

    Ok((detected_lanes, lane_measurement))
}

#[allow(clippy::too_many_arguments)]
fn buffer_legality_result(
    detector: &mut LaneLegalityDetector,
    buffer: &mut LegalityRingBuffer,
    frame: &Arc<Frame>,
    config: &types::Config,
    left_x: f32,
    right_x: f32,
    frame_count: u64,
    timestamp_ms: f64,
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
        buffer.push(frame_count, timestamp_ms, fused);
    }
}

// ============================================================================
// STAGE 3 — Maneuver Detection v2 + Crossing Detection
// ============================================================================

fn run_maneuver_pipeline_v2(
    ps: &mut PipelineState,
    frame: &Arc<Frame>,
    lane_measurement: Option<LaneMeasurement>,
    timestamp_ms: f64,
) -> ManeuverFrameOutput {
    let frame_count = ps.frame_count;

    // ── Convert YOLO detections to v2 format ──
    let vehicle_dets: Vec<DetectionInput> = ps
        .latest_vehicle_detections
        .iter()
        .map(|d| DetectionInput {
            bbox: d.bbox,
            class_id: d.class_id as u32,
            confidence: d.confidence,
        })
        .collect();

    // ── Convert frame to grayscale for ego-motion ──
    let gray = GrayFrame::from_rgb(&frame.data, frame.width as usize, frame.height as usize);

    // ── Extract marking names from legality buffer ──
    let latest_legality = ps.legality_buffer.latest();
    let frame_center_x = frame.width as f32 / 2.0;

    let left_marking: Option<String> = latest_legality.as_ref().and_then(|fused| {
        fused
            .all_markings
            .iter()
            .filter(|m| {
                let bbox_center_x = (m.bbox[0] + m.bbox[2]) / 2.0;
                bbox_center_x < frame_center_x
            })
            .max_by(|a, b| {
                let a_x = (a.bbox[0] + a.bbox[2]) / 2.0;
                let b_x = (b.bbox[0] + b.bbox[2]) / 2.0;
                a_x.partial_cmp(&b_x).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|m| m.class_name.clone())
    });

    let right_marking: Option<String> = latest_legality.as_ref().and_then(|fused| {
        fused
            .all_markings
            .iter()
            .filter(|m| {
                let bbox_center_x = (m.bbox[0] + m.bbox[2]) / 2.0;
                bbox_center_x >= frame_center_x
            })
            .min_by(|a, b| {
                let a_x = (a.bbox[0] + a.bbox[2]) / 2.0;
                let b_x = (b.bbox[0] + b.bbox[2]) / 2.0;
                a_x.partial_cmp(&b_x).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|m| m.class_name.clone())
    });

    // ── Process frame through v2 pipeline ──
    let v2_result = ps.maneuver_pipeline_v2.process_frame(ManeuverFrameInput {
        vehicle_detections: &vehicle_dets,
        lane_measurement,
        gray_frame: Some(&gray),
        left_marking_name: left_marking.as_deref(),
        right_marking_name: right_marking.as_deref(),
        legality_buffer: Some(&ps.legality_buffer),
        timestamp_ms,
        frame_id: frame_count,
    });

    // ── v5.2: Run line crossing detection ──
    let ego_left = ps.last_vehicle_state.as_ref().and_then(|vs| {
        vs.lane_width
            .map(|w| frame_center_x - w / 2.0 + vs.lateral_offset)
    });
    let ego_right = ps.last_vehicle_state.as_ref().and_then(|vs| {
        vs.lane_width
            .map(|w| frame_center_x + w / 2.0 + vs.lateral_offset)
    });

    let crossing_event = lane_crossing_integration::run_crossing_detection(
        &mut ps.lane_crossing_state,
        ego_left,
        ego_right,
        frame_count,
        timestamp_ms,
    );

    if let Some(ref event) = crossing_event {
        ps.crossing_events_total += 1;
        if matches!(
            event.passing_legality,
            PassingLegality::Prohibited | PassingLegality::MixedProhibited
        ) {
            ps.crossing_events_illegal += 1;
        }

        info!(
            "🚧 LINE CROSSING: {} {} | legality={} | pen={:.2} | frame={}",
            event.line_role.as_str(),
            event.crossing_direction.as_str(),
            event.passing_legality.as_str(),
            event.penetration_ratio,
            frame_count,
        );

        // v6.0: Trigger crossing flash animation
        // Find the marking bbox that was crossed
        if let Some(fused) = ps.legality_buffer.latest() {
            let marking_bbox = fused
                .all_markings
                .iter()
                .find(|m| m.class_name == event.marking_class)
                .map(|m| m.bbox);

            if let Some(bbox) = marking_bbox {
                let was_illegal = matches!(
                    event.passing_legality,
                    PassingLegality::Prohibited | PassingLegality::MixedProhibited
                );
                ps.crossing_flash = Some(CrossingFlashState::new(*event.line_role, bbox));
            }
        }
    }

    // Expire old crossing flash
    if let Some(ref flash) = ps.crossing_flash {
        if !flash.is_active(frame_count) {
            ps.crossing_flash = None;
        }
    }

    // ── Update counters and last maneuver ──
    for event in &v2_result.maneuver_events {
        ps.v2_maneuver_events += 1;

        let vehicles_count = if event.passed_vehicle_id.is_some() {
            1
        } else {
            0
        };

        match event.maneuver_type {
            analysis::maneuver_classifier::ManeuverType::Overtake => {
                ps.v2_overtakes += 1;
                if event.passed_vehicle_id.is_some() {
                    ps.v2_vehicles_overtaken += 1;
                }
            }
            analysis::maneuver_classifier::ManeuverType::LaneChange => ps.v2_lane_changes += 1,
            analysis::maneuver_classifier::ManeuverType::BeingOvertaken => {
                ps.v2_being_overtaken += 1
            }
        }

        ps.last_maneuver = Some(LastManeuverInfo {
            maneuver_type: event.maneuver_type.as_str().to_string(),
            side: event.side.as_str().to_string(),
            confidence: event.confidence,
            legality: format!("{:?}", event.legality),
            duration_ms: event.duration_ms,
            sources: event.sources.summary(),
            frame_detected: frame_count,
            timestamp_detected: timestamp_ms,
            vehicles_in_this_maneuver: vehicles_count,
        });

        info!(
            "🆕 {} {} | conf={:.2} | sources={} | legality={:?} | frame={}",
            event.maneuver_type.as_str(),
            event.side.as_str(),
            event.confidence,
            event.sources.summary(),
            event.legality,
            frame_count,
        );
    }

    v2_result
}

// ============================================================================
// STAGE 4 — Save Events
// ============================================================================

fn save_maneuver_event(
    event: &analysis::maneuver_classifier::ManeuverEvent,
    file: &mut std::fs::File,
) -> Result<()> {
    use std::io::Write;
    let json_line = serde_json::to_string(&event).context("Failed to serialize maneuver event")?;
    writeln!(file, "{}", json_line)?;
    file.flush()?;
    Ok(())
}

// ============================================================================
// STAGE 5 — Video Annotation (v6.0: full AnnotationInput)
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn run_video_annotation(
    writer: &mut opencv::videoio::VideoWriter,
    ps: &PipelineState,
    frame: &Arc<Frame>,
    detected_lanes: &[DetectedLane],
    v2_output: &ManeuverFrameOutput,
    tracked_vehicles: &[&analysis::vehicle_tracker::Track],
    width: i32,
    height: i32,
    timestamp_ms: f64,
) -> Result<()> {
    let legality_owned: Option<LegalityResult> = ps.legality_buffer.latest_as_legality_result();
    let legality_ref = legality_owned.as_ref();

    // v6.0: Derive mixed-line side info from road classification
    let mixed_dashed_is_right = match ps.last_road_classification.mixed_line_side {
        Some(MixedLineSide::DashedRight) => Some(true),
        Some(MixedLineSide::SolidRight) => Some(false),
        _ => None,
    };

    let cache_state = ps.lane_crossing_state.detection_cache.state();
    let cache_stale_frames = ps.lane_crossing_state.detection_cache.stale_frames();

    // Build full AnnotationInput with ALL v6.0 data
    let input = video_processor::AnnotationInput {
        frame_rgb: &frame.data,
        width,
        height,
        frame_id: ps.frame_count,
        timestamp_ms,

        lanes: detected_lanes,
        vehicle_state: ps.last_vehicle_state.as_ref(),
        legality_result: legality_ref,

        // v6.0: Real data, not defaults
        passing_legality: ps.last_road_classification.passing_legality,
        cache_state,
        cache_stale_frames,
        crossing_flash: ps.crossing_flash.as_ref(),
        mixed_dashed_is_right,

        maneuver_events: &v2_output.maneuver_events,
        ego_lateral_velocity: v2_output.ego_lateral_velocity,
        lateral_state: &v2_output.lateral_state,
        total_overtakes: ps.v2_overtakes,
        total_lane_changes: ps.v2_lane_changes,
        total_vehicles_overtaken: ps.v2_vehicles_overtaken,
        last_maneuver: ps.last_maneuver.as_ref(),

        tracked_vehicles,
        vehicle_detections: &ps.latest_vehicle_detections,
    };

    if let Ok(annotated) = video_processor::draw_annotated_frame(&input) {
        use opencv::videoio::VideoWriterTrait;
        writer.write(&annotated)?;
    }

    Ok(())
}

// ============================================================================
// FINAL STATS
// ============================================================================

fn print_final_stats(stats: &ProcessingStats) {
    info!("\n✓ Video processed successfully!");
    info!("  Total frames: {}", stats.total_frames);
    info!(
        "  YOLOv8-seg detections: {} frames ({:.1}%)",
        stats.yolo_primary_count,
        stats.yolo_primary_count as f64 / stats.total_frames.max(1) as f64 * 100.0
    );
    if stats.cache_recoveries > 0 {
        info!(
            "  Cache recoveries (trocha→detection): {}",
            stats.cache_recoveries
        );
    }

    info!("╔════════════════════════════════════════════════════════════╗");
    info!("║       MANEUVER DETECTION v6.0 (FULL PIPELINE)            ║");
    info!("╠════════════════════════════════════════════════════════════╣");
    info!(
        "║ Total v2 maneuver events:   {:>5}                         ║",
        stats.v2_maneuver_events
    );
    info!(
        "║   🚗 Overtakes:             {:>5} ({} vehicles)            ║",
        stats.v2_overtakes, stats.v2_vehicles_overtaken
    );
    info!(
        "║   🔀 Lane changes:          {:>5}                         ║",
        stats.v2_lane_changes
    );
    info!(
        "║   ⚠️  Being overtaken:       {:>5}                         ║",
        stats.v2_being_overtaken
    );
    info!(
        "║   🚧 Line crossings:        {:>5} ({} illegal)            ║",
        stats.crossing_events_total, stats.crossing_events_illegal
    );
    info!("╚════════════════════════════════════════════════════════════╝");

    info!(
        "\n  Processing: {:.1} FPS ({:.1}s total)",
        stats.avg_fps, stats.duration_secs
    );
}

