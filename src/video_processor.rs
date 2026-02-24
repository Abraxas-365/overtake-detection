// src/video_processor.rs
//
// v6.0: Zone-Based Visualization Pipeline
//
// Rewrite of v5.0 with:
//   • Decomposed rendering layers instead of one monolithic function
//   • Zone-based road surface visualization (mask projection + legality strips)
//   • Batched alpha blending (single overlay per layer, not per element)
//   • Proper input struct instead of 17 positional parameters
//   • Crossing flash animation support
//   • Cache state badge (LIVE / CACHED / TROCHA)
//   • Mixed line PERMITIDO/PROHIBIDO indicators
//   • Cleaner BEV radar with legality context
//
// RENDERING ORDER (back → front):
//   Layer 0: Original camera frame
//   Layer 1: Road surface zones (opposing lane tint, ego lane fill)
//   Layer 2: YOLO-seg mask overlays (colored by legality)
//   Layer 3: Legality zone strips (alongside markings)
//   Layer 4: Lane boundary polylines
//   Layer 5: Crossing flash animation
//   Layer 6: Vehicle bounding boxes + labels
//   Layer 7: Ego position indicator + velocity arrow
//   Layer 8: HUD panels (info, BEV radar, event log)
//   Layer 9: Banners (legality, violation, being-overtaken)
//   Layer 10: Badges (cache state, marking labels)
//   Layer 11: Bottom status bar
//   Layer 12: Maneuver border pulse

use crate::lane_crossing::CacheState;
use crate::lane_legality::{DetectedRoadMarking, LegalityResult, LineLegality};
use crate::road_classification::PassingLegality;
use crate::road_overlay::{self, CrossingFlashState, RoadZoneInput};
use crate::types::{Config, DetectedLane, VehicleState};
use anyhow::Result;
use opencv::{
    core::{self, Mat, Vector},
    imgcodecs, imgproc,
    prelude::*,
    videoio::{self, VideoCapture, VideoCaptureTraitConst, VideoWriter},
};
use std::path::{Path, PathBuf};
use tracing::info;
use walkdir::WalkDir;

// ============================================================================
// VIDEO PROCESSOR (file discovery, reader, writer — unchanged)
// ============================================================================

pub struct VideoProcessor {
    config: Config,
}

impl VideoProcessor {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    pub fn find_video_files(&self) -> Result<Vec<PathBuf>> {
        let mut videos = Vec::new();
        let video_extensions = ["mp4", "avi", "mov", "mkv", "MP4", "AVI", "MOV", "MKV"];

        for entry in WalkDir::new(&self.config.video.input_dir)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if video_extensions.contains(&ext.to_str().unwrap_or("")) {
                    videos.push(path.to_path_buf());
                }
            }
        }
        info!("Found {} video files", videos.len());
        Ok(videos)
    }

    pub fn open_video(&self, path: &Path) -> Result<VideoReader> {
        info!("Opening video: {}", path.display());
        let cap = VideoCapture::from_file(path.to_str().unwrap(), videoio::CAP_ANY)?;

        if !cap.is_opened()? {
            anyhow::bail!("Failed to open video file");
        }

        let fps = VideoCaptureTraitConst::get(&cap, videoio::CAP_PROP_FPS)?;
        let total_frames = VideoCaptureTraitConst::get(&cap, videoio::CAP_PROP_FRAME_COUNT)? as i32;
        let width = VideoCaptureTraitConst::get(&cap, videoio::CAP_PROP_FRAME_WIDTH)? as i32;
        let height = VideoCaptureTraitConst::get(&cap, videoio::CAP_PROP_FRAME_HEIGHT)? as i32;

        info!(
            "Video properties: {}x{} @ {:.1} FPS, {} frames",
            width, height, fps, total_frames
        );

        Ok(VideoReader {
            cap,
            fps,
            total_frames,
            current_frame: 0,
            width,
            height,
        })
    }

    pub fn create_writer(
        &self,
        input_path: &Path,
        width: i32,
        height: i32,
        fps: f64,
    ) -> Result<Option<VideoWriter>> {
        if !self.config.video.save_annotated {
            return Ok(None);
        }

        std::fs::create_dir_all(&self.config.video.output_dir)?;

        let input_name = input_path.file_stem().unwrap().to_str().unwrap();
        let output_path = PathBuf::from(&self.config.video.output_dir)
            .join(format!("{}_annotated.mp4", input_name));

        info!("Output video: {}", output_path.display());

        let fourcc = VideoWriter::fourcc('m', 'p', '4', 'v')?;
        let writer = VideoWriter::new(
            output_path.to_str().unwrap(),
            fourcc,
            fps,
            core::Size::new(width, height),
            true,
        )?;

        Ok(Some(writer))
    }

    pub fn save_debug_frame(&self, frame: &crate::types::Frame, label: &str) -> Result<PathBuf> {
        let output_dir = PathBuf::from(&self.config.video.output_dir).join("debug");
        std::fs::create_dir_all(&output_dir)?;
        let filename = format!("{}_{}.png", label, chrono::Utc::now().format("%H%M%S"));
        let file_path = output_dir.join(filename);

        let mat = Mat::from_slice(&frame.data)?;
        let mat = mat.reshape(3, frame.height as i32)?;
        let mut bgr_mat = Mat::default();
        imgproc::cvt_color(&mat, &mut bgr_mat, imgproc::COLOR_RGB2BGR, 0)?;

        let params = Vector::new();
        imgcodecs::imwrite(file_path.to_str().unwrap(), &bgr_mat, &params)?;
        Ok(file_path)
    }
}

pub struct VideoReader {
    pub cap: VideoCapture,
    pub fps: f64,
    pub total_frames: i32,
    pub current_frame: i32,
    pub width: i32,
    pub height: i32,
}

impl VideoReader {
    pub fn read_frame(&mut self) -> Result<Option<crate::types::Frame>> {
        use opencv::videoio::VideoCaptureTrait;
        let mut mat = Mat::default();
        if !VideoCaptureTrait::read(&mut self.cap, &mut mat)? || mat.empty() {
            return Ok(None);
        }
        self.current_frame += 1;
        let timestamp_ms = (self.current_frame as f64 / self.fps) * 1000.0;

        let mut rgb_mat = Mat::default();
        imgproc::cvt_color(&mat, &mut rgb_mat, imgproc::COLOR_BGR2RGB, 0)?;
        let data = rgb_mat.data_bytes()?.to_vec();

        Ok(Some(crate::types::Frame {
            data,
            width: self.width as usize,
            height: self.height as usize,
            timestamp_ms,
        }))
    }

    pub fn progress(&self) -> f32 {
        if self.total_frames == 0 {
            return 0.0;
        }
        (self.current_frame as f32 / self.total_frames as f32) * 100.0
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ANNOTATION INPUT — replaces 17 positional parameters
// ════════════════════════════════════════════════════════════════════════════

/// Everything the annotation pipeline needs for one frame.
///
/// Constructed in main.rs `run_video_annotation()` from pipeline state.
/// This replaces the 17-parameter `draw_lanes_v2` signature.
pub struct AnnotationInput<'a> {
    // Frame data
    pub frame_rgb: &'a [u8],
    pub width: i32,
    pub height: i32,
    pub frame_id: u64,
    pub timestamp_ms: f64,

    // Lane detection
    pub lanes: &'a [DetectedLane],
    pub vehicle_state: Option<&'a VehicleState>,
    pub legality_result: Option<&'a LegalityResult>,

    // v6.0: Road zone context
    pub passing_legality: PassingLegality,
    pub cache_state: CacheState,
    pub cache_stale_frames: u32,
    pub crossing_flash: Option<&'a CrossingFlashState>,
    pub mixed_dashed_is_right: Option<bool>,

    // Maneuver detection
    pub maneuver_events: &'a [crate::analysis::maneuver_classifier::ManeuverEvent],
    pub ego_lateral_velocity: f32,
    pub lateral_state: &'a str,
    pub total_overtakes: u64,
    pub total_shadow_overtakes: u64,
    pub total_lane_changes: u64,
    pub total_vehicles_overtaken: u64,
    pub last_maneuver: Option<&'a crate::LastManeuverInfo>,

    // Vehicle tracking
    pub tracked_vehicles: &'a [&'a crate::analysis::vehicle_tracker::Track],
    pub vehicle_detections: &'a [crate::vehicle_detection::Detection],
}

// ════════════════════════════════════════════════════════════════════════════
// MAIN ENTRY POINT: draw_annotated_frame (v6.0)
// ════════════════════════════════════════════════════════════════════════════

/// Primary annotation entry point (v6.0).
///
/// Renders all visualization layers in correct z-order onto the camera frame.
pub fn draw_annotated_frame(input: &AnnotationInput) -> Result<Mat> {
    let width = input.width;
    let height = input.height;

    // Convert RGB → BGR for OpenCV
    let mat = Mat::from_slice(input.frame_rgb)?;
    let mat = mat.reshape(3, height)?;
    let mut bgr_mat = Mat::default();
    imgproc::cvt_color(&mat, &mut bgr_mat, imgproc::COLOR_RGB2BGR, 0)?;
    let mut output = bgr_mat.try_clone()?;

    let has_new_event = !input.maneuver_events.is_empty();
    let is_shadow_overtake = input.maneuver_events.iter().any(|e| {
        e.maneuver_type == crate::analysis::maneuver_classifier::ManeuverType::ShadowOvertake
    });

    // ──────────────────────────────────────────────────────────────────
    // LAYER 0: Maneuver border pulse (renders as frame border)
    // ──────────────────────────────────────────────────────────────────
    if has_new_event {
        render_maneuver_border_pulse(&mut output, input.maneuver_events, width, height)?;
    }

    // ──────────────────────────────────────────────────────────────────
    // LAYERS 1-3: Road surface zones (road_overlay module)
    //   1. Opposing lane tint
    //   2. Mask overlays
    //   3. Legality zone strips
    //   4. Crossing flash
    //   5. Marking labels
    //   6. Mixed line indicators
    //   7. Legality banner
    //   8. Cache badge
    // ──────────────────────────────────────────────────────────────────
    if let Some(legality) = input.legality_result {
        let zone_input = RoadZoneInput {
            markings: &legality.all_markings,
            ego_left_x: ego_left_x_from_lanes(input.lanes, width),
            ego_right_x: ego_right_x_from_lanes(input.lanes, width),
            passing_legality: input.passing_legality,
            cache_state: input.cache_state,
            cache_stale_frames: input.cache_stale_frames,
            crossing_flash: input.crossing_flash,
            mixed_dashed_is_right: input.mixed_dashed_is_right,
            frame_id: input.frame_id,
        };
        road_overlay::render_road_zones(&mut output, &zone_input, width, height)?;
    } else {
        // No legality data — still show cache badge
        road_overlay::render_cache_badge(
            &mut output,
            input.cache_state,
            input.cache_stale_frames,
            width - 160,
            12,
        )?;
    }

    // ──────────────────────────────────────────────────────────────────
    // LAYER 1b: Ego lane polygon fill (between L/R boundary polylines)
    // ──────────────────────────────────────────────────────────────────
    render_ego_lane_fill(
        &mut output,
        input.lanes,
        input.legality_result,
        width,
        height,
    )?;

    // ──────────────────────────────────────────────────────────────────
    // LAYER 4: Lane boundary polylines
    // ──────────────────────────────────────────────────────────────────
    render_lane_polylines(&mut output, input.lanes)?;

    // ──────────────────────────────────────────────────────────────────
    // LAYER 6: Tracked vehicles
    // ──────────────────────────────────────────────────────────────────
    render_tracked_vehicles(
        &mut output,
        input.tracked_vehicles,
        input.vehicle_detections,
        width,
        height,
    )?;

    // ──────────────────────────────────────────────────────────────────
    // LAYER 7: Ego position indicator + velocity arrow
    // ──────────────────────────────────────────────────────────────────
    render_ego_indicator(
        &mut output,
        input.vehicle_state,
        input.ego_lateral_velocity,
        width,
        height,
    )?;

    // ──────────────────────────────────────────────────────────────────
    // LAYER 5: Shadow overtake warning banner
    // ──────────────────────────────────────────────────────────────────
    if is_shadow_overtake {
        render_shadow_overtake_banner(&mut output, width)?;
    }

    // ──────────────────────────────────────────────────────────────────
    // LAYER 8: HUD panels
    // ──────────────────────────────────────────────────────────────────
    render_left_info_panel(
        &mut output,
        input.lanes,
        input.vehicle_state,
        input.legality_result,
        input.tracked_vehicles,
        input.vehicle_detections,
        input.lateral_state,
        input.ego_lateral_velocity,
        width,
        height,
    )?;

    render_right_event_panel(
        &mut output,
        input.maneuver_events,
        input.total_overtakes,
        input.total_shadow_overtakes,
        input.total_lane_changes,
        input.total_vehicles_overtaken,
        input.last_maneuver,
        input.timestamp_ms,
        width,
        height,
    )?;

    render_bev_radar(
        &mut output,
        input.tracked_vehicles,
        input.vehicle_state,
        input.legality_result,
        width,
        height,
    )?;

    // ──────────────────────────────────────────────────────────────────
    // LAYER 11: Bottom status bar
    // ──────────────────────────────────────────────────────────────────
    render_bottom_status_bar(
        &mut output,
        input.frame_id,
        input.timestamp_ms,
        input.total_overtakes,
        input.total_shadow_overtakes,
        input.total_lane_changes,
        width,
        height,
    )?;

    // ──────────────────────────────────────────────────────────────────
    // LAYER 10: Legend
    // ──────────────────────────────────────────────────────────────────
    render_legend(&mut output, width, height)?;

    Ok(output)
}

// ════════════════════════════════════════════════════════════════════════════
// BACKWARD-COMPAT WRAPPER: draw_lanes_v2 (same signature as v5.0)
// ════════════════════════════════════════════════════════════════════════════

/// Backward-compatible wrapper that constructs `AnnotationInput` from the
/// existing positional parameters used by `run_video_annotation()` in main.rs.
///
/// New code should use `draw_annotated_frame()` directly with `AnnotationInput`.
#[allow(clippy::too_many_arguments)]
pub fn draw_lanes_v2(
    frame: &[u8],
    width: i32,
    height: i32,
    lanes: &[DetectedLane],
    vehicle_state: Option<&VehicleState>,
    maneuver_events: &[crate::analysis::maneuver_classifier::ManeuverEvent],
    tracked_vehicles: &[&crate::analysis::vehicle_tracker::Track],
    ego_lateral_velocity: f32,
    lateral_state: &str,
    frame_id: u64,
    timestamp_ms: f64,
    legality_result: Option<&LegalityResult>,
    vehicle_detections: &[crate::vehicle_detection::Detection],
    total_overtakes: u64,
    total_shadow_overtakes: u64,
    total_lane_changes: u64,
    total_vehicles_overtaken: u64,
    last_maneuver: Option<&crate::LastManeuverInfo>,
) -> Result<Mat> {
    let input = AnnotationInput {
        frame_rgb: frame,
        width,
        height,
        frame_id,
        timestamp_ms,
        lanes,
        vehicle_state,
        legality_result,
        // v6.0 fields — defaults when called via legacy path
        passing_legality: legality_result
            .map(|l| passing_legality_from_markings(&l.all_markings))
            .unwrap_or(PassingLegality::Unknown),
        cache_state: CacheState::Fresh, // Legacy callers don't have cache
        cache_stale_frames: 0,
        crossing_flash: None,
        mixed_dashed_is_right: None,
        maneuver_events,
        ego_lateral_velocity,
        lateral_state,
        total_overtakes,
        total_shadow_overtakes,
        total_lane_changes,
        total_vehicles_overtaken,
        last_maneuver,
        tracked_vehicles,
        vehicle_detections,
    };

    draw_annotated_frame(&input)
}

/// Legacy simple annotation for early pipeline stages / unit tests.
pub fn draw_lanes_with_state(
    frame: &[u8],
    width: i32,
    height: i32,
    lanes: &[DetectedLane],
    state: &str,
    vehicle_state: Option<&VehicleState>,
) -> Result<Mat> {
    let mat = Mat::from_slice(frame)?;
    let mat = mat.reshape(3, height)?;
    let mut bgr_mat = Mat::default();
    imgproc::cvt_color(&mat, &mut bgr_mat, imgproc::COLOR_RGB2BGR, 0)?;
    let mut output = bgr_mat.try_clone()?;

    let colors = [
        core::Scalar::new(0.0, 0.0, 255.0, 0.0),
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        core::Scalar::new(255.0, 0.0, 0.0, 0.0),
        core::Scalar::new(0.0, 255.0, 255.0, 0.0),
    ];

    for (i, lane) in lanes.iter().enumerate() {
        let color = colors[i % colors.len()];
        for point in &lane.points {
            let pt = core::Point::new(point.0 as i32, point.1 as i32);
            imgproc::circle(&mut output, pt, 3, color, -1, imgproc::LINE_8, 0)?;
        }
    }

    let vehicle_x = width / 2;
    let vehicle_y = (height as f32 * 0.85) as i32;
    imgproc::circle(
        &mut output,
        core::Point::new(vehicle_x, vehicle_y),
        10,
        core::Scalar::new(0.0, 255.0, 255.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;

    draw_text_with_shadow(
        &mut output,
        &format!("State: {}", state),
        15,
        32,
        0.8,
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
    )?;

    if let Some(vs) = vehicle_state {
        if vs.is_valid() {
            let normalized = vs.normalized_offset().unwrap_or(0.0);
            let info = format!(
                "Offset: {:.1}px ({:.0}%) | Width: {:.0}px",
                vs.lateral_offset,
                normalized * 100.0,
                vs.lane_width.unwrap_or(0.0),
            );
            draw_text_with_shadow(
                &mut output,
                &info,
                15,
                64,
                0.55,
                core::Scalar::new(200.0, 200.0, 200.0, 0.0),
                1,
            )?;
        }
    }

    Ok(output)
}

// ════════════════════════════════════════════════════════════════════════════
// RENDERING LAYER FUNCTIONS
// ════════════════════════════════════════════════════════════════════════════

// ──────────────────────────────────────────────────────────────────────────
// Ego lane polygon fill
// ──────────────────────────────────────────────────────────────────────────

fn render_ego_lane_fill(
    output: &mut Mat,
    lanes: &[DetectedLane],
    legality_result: Option<&LegalityResult>,
    _width: i32,
    _height: i32,
) -> Result<()> {
    if lanes.len() < 2 {
        return Ok(());
    }

    let left_lane = &lanes[0];
    let right_lane = &lanes[1];

    if left_lane.points.len() < 2 || right_lane.points.len() < 2 {
        return Ok(());
    }

    // Build polygon: left points bottom→top, then right points top→bottom
    let mut poly_pts: Vec<core::Point> = Vec::new();

    let mut left_sorted = left_lane.points.clone();
    left_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for p in &left_sorted {
        poly_pts.push(core::Point::new(p.0 as i32, p.1 as i32));
    }

    let mut right_sorted = right_lane.points.clone();
    right_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for p in &right_sorted {
        poly_pts.push(core::Point::new(p.0 as i32, p.1 as i32));
    }

    if poly_pts.len() < 3 {
        return Ok(());
    }

    let fill_color = if let Some(legality) = legality_result {
        if legality.ego_intersects_marking && legality.verdict.is_illegal() {
            core::Scalar::new(0.0, 0.0, 180.0, 0.0) // Red tint during violation
        } else {
            core::Scalar::new(160.0, 110.0, 0.0, 0.0) // Teal tint normal
        }
    } else {
        core::Scalar::new(160.0, 110.0, 0.0, 0.0)
    };

    let mut overlay = output.try_clone()?;
    let mut pts_vec = Vector::<Vector<core::Point>>::new();
    pts_vec.push(Vector::from_iter(poly_pts.into_iter()));
    imgproc::fill_poly(
        &mut overlay,
        &pts_vec,
        fill_color,
        imgproc::LINE_AA,
        0,
        core::Point::new(0, 0),
    )?;

    let mut blended = Mat::default();
    core::add_weighted(&overlay, 0.14, output, 0.86, 0.0, &mut blended, -1)?;
    blended.copy_to(output)?;

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// Lane boundary polylines
// ──────────────────────────────────────────────────────────────────────────

fn render_lane_polylines(output: &mut Mat, lanes: &[DetectedLane]) -> Result<()> {
    let lane_colors = [
        core::Scalar::new(0.0, 0.0, 255.0, 0.0),   // Red — left
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),   // Green — right
        core::Scalar::new(255.0, 0.0, 0.0, 0.0),   // Blue — extra
        core::Scalar::new(0.0, 255.0, 255.0, 0.0), // Yellow — extra
    ];

    for (i, lane) in lanes.iter().enumerate() {
        let color = lane_colors[i % lane_colors.len()];
        let thickness = if i < 2 { 4 } else { 2 };

        // Dots at each detected point
        for point in &lane.points {
            let pt = core::Point::new(point.0 as i32, point.1 as i32);
            imgproc::circle(output, pt, thickness + 1, color, -1, imgproc::LINE_8, 0)?;
        }

        // Connected polyline
        if lane.points.len() >= 2 {
            for j in 0..lane.points.len() - 1 {
                let pt1 = core::Point::new(lane.points[j].0 as i32, lane.points[j].1 as i32);
                let pt2 =
                    core::Point::new(lane.points[j + 1].0 as i32, lane.points[j + 1].1 as i32);
                imgproc::line(output, pt1, pt2, color, 3, imgproc::LINE_AA, 0)?;
            }

            // Label at the top of each lane line
            if let Some(first_point) = lane.points.first() {
                let label = match i {
                    0 => "LEFT BOUNDARY",
                    1 => "RIGHT BOUNDARY",
                    _ => "LANE",
                };
                draw_text_with_shadow(
                    output,
                    label,
                    first_point.0 as i32 + 10,
                    first_point.1 as i32,
                    0.45,
                    color,
                    1,
                )?;
            }
        }
    }

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// Tracked vehicles
// ──────────────────────────────────────────────────────────────────────────

fn render_tracked_vehicles(
    output: &mut Mat,
    tracked_vehicles: &[&crate::analysis::vehicle_tracker::Track],
    vehicle_detections: &[crate::vehicle_detection::Detection],
    width: i32,
    height: i32,
) -> Result<()> {
    for track in tracked_vehicles {
        if !track.is_confirmed() {
            continue;
        }

        let bbox = &track.bbox;
        let bbox_w = bbox[2] - bbox[0];
        let bbox_h = bbox[3] - bbox[1];

        // Look up class name from nearest raw detection
        let class_name = vehicle_detections
            .iter()
            .find(|d| {
                let d_cx = (d.bbox[0] + d.bbox[2]) / 2.0;
                let t_cx = (bbox[0] + bbox[2]) / 2.0;
                (d_cx - t_cx).abs() < 50.0
            })
            .map(|d| d.class_name.as_str())
            .unwrap_or("vehicle");

        // Color by zone
        let (box_color, zone_str) = zone_color_and_label(track.zone);

        let pt1 = core::Point::new(bbox[0] as i32, bbox[1] as i32);
        let pt2 = core::Point::new(bbox[2] as i32, bbox[3] as i32);

        // Thin full rectangle
        imgproc::rectangle(
            output,
            core::Rect::from_points(pt1, pt2),
            box_color,
            2,
            imgproc::LINE_8,
            0,
        )?;

        // Corner accent brackets (modern detection look)
        let corner_len = (bbox_w.min(bbox_h) * 0.2).max(8.0) as i32;
        draw_corner_accents(output, pt1.x, pt1.y, pt2.x, pt2.y, corner_len, box_color, 4)?;

        // Vehicle label: "car T12 AHEAD"
        let label_y = (bbox[1] as i32 - 8).max(14);
        let label = format!("{} T{} {}", class_name, track.id, zone_str);
        draw_text_with_shadow(output, &label, bbox[0] as i32, label_y, 0.42, box_color, 1)?;

        // Distance estimation (rough: based on bbox height relative to frame)
        let bbox_area_ratio = (bbox_w * bbox_h) / (width as f32 * height as f32);
        if bbox_area_ratio > 0.002 {
            let est_dist = estimate_distance(bbox_h, height as f32);
            let dist_label = format!("~{:.0}m", est_dist);
            draw_text_with_shadow(
                output,
                &dist_label,
                bbox[0] as i32,
                bbox[3] as i32 + 16,
                0.38,
                box_color,
                1,
            )?;
        }
    }

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// Ego position indicator + velocity arrow
// ──────────────────────────────────────────────────────────────────────────

fn render_ego_indicator(
    output: &mut Mat,
    vehicle_state: Option<&VehicleState>,
    ego_lateral_velocity: f32,
    width: i32,
    height: i32,
) -> Result<()> {
    let ego_x = width / 2;
    let ego_y = (height as f32 * 0.82) as i32;

    // Ego diamond marker
    let diamond_pts = vec![
        core::Point::new(ego_x, ego_y - 12),
        core::Point::new(ego_x + 8, ego_y),
        core::Point::new(ego_x, ego_y + 12),
        core::Point::new(ego_x - 8, ego_y),
    ];
    let mut pts_vec = Vector::<Vector<core::Point>>::new();
    pts_vec.push(Vector::from_iter(diamond_pts.into_iter()));
    imgproc::fill_poly(
        output,
        &pts_vec,
        core::Scalar::new(0.0, 255.0, 255.0, 0.0),
        imgproc::LINE_AA,
        0,
        core::Point::new(0, 0),
    )?;

    draw_text_with_shadow(
        output,
        "EGO",
        ego_x - 14,
        ego_y + 28,
        0.4,
        core::Scalar::new(0.0, 255.0, 255.0, 0.0),
        1,
    )?;

    // Lateral velocity arrow on road surface
    if ego_lateral_velocity.abs() > 0.5 {
        let arrow_len = (ego_lateral_velocity * 12.0).clamp(-80.0, 80.0) as i32;
        let arrow_color = if ego_lateral_velocity.abs() > 3.0 {
            core::Scalar::new(0.0, 0.0, 255.0, 0.0)
        } else if ego_lateral_velocity.abs() > 1.5 {
            core::Scalar::new(0.0, 165.0, 255.0, 0.0)
        } else {
            core::Scalar::new(0.0, 230.0, 230.0, 0.0)
        };

        imgproc::arrowed_line(
            output,
            core::Point::new(ego_x, ego_y - 30),
            core::Point::new(ego_x + arrow_len, ego_y - 30),
            arrow_color,
            3,
            imgproc::LINE_AA,
            0,
            0.3,
        )?;
    }

    // Lateral position gauge (compact bar below ego marker)
    if let Some(vs) = vehicle_state {
        let norm = vs.normalized_offset().unwrap_or(0.0);
        let gauge_x = ego_x - 60;
        let gauge_y = ego_y + 40;
        draw_lateral_gauge(output, gauge_x, gauge_y, 120, 10, norm)?;
    }

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// Shadow overtake warning banner
// ──────────────────────────────────────────────────────────────────────────

fn render_shadow_overtake_banner(output: &mut Mat, width: i32) -> Result<()> {
    let banner_h = 40;
    let mut overlay = output.try_clone()?;
    imgproc::rectangle(
        &mut overlay,
        core::Rect::new(0, 0, width, banner_h),
        core::Scalar::new(0.0, 0.0, 200.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;
    let mut blended = Mat::default();
    core::add_weighted(&overlay, 0.8, output, 0.2, 0.0, &mut blended, -1)?;
    blended.copy_to(output)?;

    let text = "!! SHADOW OVERTAKE !!";
    let mut baseline = 0;
    let text_size =
        imgproc::get_text_size(text, imgproc::FONT_HERSHEY_SIMPLEX, 0.75, 2, &mut baseline)?;
    let text_x = (width - text_size.width) / 2;
    imgproc::put_text(
        output,
        text,
        core::Point::new(text_x, banner_h - 10),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.75,
        core::Scalar::new(255.0, 255.0, 255.0, 0.0),
        2,
        imgproc::LINE_AA,
        false,
    )?;

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// Maneuver border pulse
// ──────────────────────────────────────────────────────────────────────────

fn render_maneuver_border_pulse(
    output: &mut Mat,
    events: &[crate::analysis::maneuver_classifier::ManeuverEvent],
    width: i32,
    height: i32,
) -> Result<()> {
    use crate::analysis::maneuver_classifier::ManeuverType;

    let border_color = if events
        .iter()
        .any(|e| e.maneuver_type == ManeuverType::ShadowOvertake)
    {
        core::Scalar::new(0.0, 0.0, 255.0, 0.0) // Red for shadow overtake
    } else if events
        .iter()
        .any(|e| e.maneuver_type == ManeuverType::Overtake)
    {
        core::Scalar::new(0.0, 255.0, 0.0, 0.0) // Green for overtake
    } else {
        core::Scalar::new(0.0, 165.0, 255.0, 0.0) // Orange
    };

    let t = 6; // pulse thickness
               // Top
    imgproc::rectangle(
        output,
        core::Rect::new(0, 0, width, t),
        border_color,
        -1,
        imgproc::LINE_8,
        0,
    )?;
    // Bottom
    imgproc::rectangle(
        output,
        core::Rect::new(0, height - t, width, t),
        border_color,
        -1,
        imgproc::LINE_8,
        0,
    )?;
    // Left
    imgproc::rectangle(
        output,
        core::Rect::new(0, 0, t, height),
        border_color,
        -1,
        imgproc::LINE_8,
        0,
    )?;
    // Right
    imgproc::rectangle(
        output,
        core::Rect::new(width - t, 0, t, height),
        border_color,
        -1,
        imgproc::LINE_8,
        0,
    )?;

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// Left info panel (lane status, confidence, markings, vehicle summary)
// ──────────────────────────────────────────────────────────────────────────

fn render_left_info_panel(
    output: &mut Mat,
    lanes: &[DetectedLane],
    vehicle_state: Option<&VehicleState>,
    legality_result: Option<&LegalityResult>,
    tracked_vehicles: &[&crate::analysis::vehicle_tracker::Track],
    vehicle_detections: &[crate::vehicle_detection::Detection],
    lateral_state: &str,
    ego_lateral_velocity: f32,
    width: i32,
    height: i32,
) -> Result<()> {
    let panel_x = 10;
    let mut panel_y = 60;
    let line_height = 22;

    // Determine panel height based on content
    let panel_h = 260;
    draw_panel_background(output, panel_x - 5, panel_y - 18, 290, panel_h)?;

    // ── Lane detection status ──
    let (lane_status, lane_color, lane_conf) = if lanes.len() >= 2 {
        (
            format!("LANE: {} boundaries", lanes.len()),
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            vehicle_state.map(|v| v.detection_confidence).unwrap_or(0.0),
        )
    } else if !lanes.is_empty() {
        (
            "LANE: partial".to_string(),
            core::Scalar::new(0.0, 180.0, 255.0, 0.0),
            vehicle_state.map(|v| v.detection_confidence).unwrap_or(0.0) * 0.65,
        )
    } else {
        (
            "LANE: no detection".to_string(),
            core::Scalar::new(0.0, 80.0, 200.0, 0.0),
            0.0,
        )
    };

    draw_text_with_shadow(
        output,
        &lane_status,
        panel_x + 18,
        panel_y,
        0.48,
        lane_color,
        1,
    )?;
    panel_y += line_height;

    // Lane confidence bar
    draw_confidence_bar(output, panel_x + 10, panel_y - 4, 180, 10, lane_conf)?;
    draw_text_with_shadow(
        output,
        &format!("{:.0}%", lane_conf * 100.0),
        panel_x + 198,
        panel_y + 4,
        0.38,
        core::Scalar::new(180.0, 180.0, 180.0, 0.0),
        1,
    )?;
    panel_y += line_height - 4;

    // ── Road marking types (left/right) ──
    if let Some(legality) = legality_result {
        if !legality.all_markings.is_empty() {
            let frame_cx = width as f32 / 2.0;

            let left_names: Vec<&str> = legality
                .all_markings
                .iter()
                .filter(|m| (m.bbox[0] + m.bbox[2]) / 2.0 < frame_cx)
                .map(|m| m.class_name.as_str())
                .collect();
            let right_names: Vec<&str> = legality
                .all_markings
                .iter()
                .filter(|m| (m.bbox[0] + m.bbox[2]) / 2.0 >= frame_cx)
                .map(|m| m.class_name.as_str())
                .collect();

            if !left_names.is_empty() {
                draw_text_with_shadow(
                    output,
                    &format!("  L: {}", left_names.join(", ")),
                    panel_x + 10,
                    panel_y,
                    0.42,
                    core::Scalar::new(180.0, 180.0, 180.0, 0.0),
                    1,
                )?;
                panel_y += 18;
            }
            if !right_names.is_empty() {
                draw_text_with_shadow(
                    output,
                    &format!("  R: {}", right_names.join(", ")),
                    panel_x + 10,
                    panel_y,
                    0.42,
                    core::Scalar::new(180.0, 180.0, 180.0, 0.0),
                    1,
                )?;
                panel_y += 18;
            }
        }
    }

    // ── Vehicle tracking summary ──
    let confirmed_count = tracked_vehicles.iter().filter(|t| t.is_confirmed()).count();
    let mut vehicle_classes: std::collections::HashMap<&str, u32> =
        std::collections::HashMap::new();
    for track in tracked_vehicles {
        if track.is_confirmed() {
            let class_name = vehicle_detections
                .iter()
                .find(|d| {
                    let d_cx = (d.bbox[0] + d.bbox[2]) / 2.0;
                    let t_cx = (track.bbox[0] + track.bbox[2]) / 2.0;
                    (d_cx - t_cx).abs() < 50.0
                })
                .map(|d| d.class_name.as_str())
                .unwrap_or("vehicle");
            *vehicle_classes.entry(class_name).or_insert(0) += 1;
        }
    }

    panel_y += 4;
    draw_text_with_shadow(
        output,
        &format!("VEH: {} tracked", confirmed_count),
        panel_x + 18,
        panel_y,
        0.48,
        core::Scalar::new(255.0, 255.0, 0.0, 0.0),
        1,
    )?;
    panel_y += line_height;

    if !vehicle_classes.is_empty() {
        let summary: Vec<String> = vehicle_classes
            .iter()
            .map(|(k, v)| format!("{}×{}", v, k))
            .collect();
        draw_text_with_shadow(
            output,
            &format!("  {}", summary.join(", ")),
            panel_x + 10,
            panel_y,
            0.38,
            core::Scalar::new(160.0, 160.0, 160.0, 0.0),
            1,
        )?;
        panel_y += 18;
    }

    // ── Lateral state ──
    panel_y += 4;
    let state_color = match lateral_state {
        s if s.contains("SHIFT") => core::Scalar::new(0.0, 165.0, 255.0, 0.0),
        s if s.contains("STABLE") => core::Scalar::new(0.0, 200.0, 0.0, 0.0),
        _ => core::Scalar::new(180.0, 180.0, 180.0, 0.0),
    };
    draw_text_with_shadow(
        output,
        &format!("LAT: {} ({:.1}px/f)", lateral_state, ego_lateral_velocity),
        panel_x + 18,
        panel_y,
        0.42,
        state_color,
        1,
    )?;

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// Right event panel (maneuver log + stats)
// ──────────────────────────────────────────────────────────────────────────

fn render_right_event_panel(
    output: &mut Mat,
    maneuver_events: &[crate::analysis::maneuver_classifier::ManeuverEvent],
    total_overtakes: u64,
    total_shadow_overtakes: u64,
    total_lane_changes: u64,
    total_vehicles_overtaken: u64,
    last_maneuver: Option<&crate::LastManeuverInfo>,
    timestamp_ms: f64,
    width: i32,
    _height: i32,
) -> Result<()> {
    let right_panel_x = width - 310;
    let mut right_panel_y = 60;
    let line_height = 22;

    draw_panel_background(output, right_panel_x - 5, right_panel_y - 18, 300, 200)?;

    // Title
    draw_text_with_shadow(
        output,
        "MANEUVER LOG",
        right_panel_x,
        right_panel_y,
        0.48,
        core::Scalar::new(200.0, 200.0, 200.0, 0.0),
        1,
    )?;
    right_panel_y += line_height;

    // Counters
    draw_text_with_shadow(
        output,
        &format!(
            "OVT: {} ({} veh) | LC: {} | SHDW: {}",
            total_overtakes, total_vehicles_overtaken, total_lane_changes, total_shadow_overtakes,
        ),
        right_panel_x + 8,
        right_panel_y,
        0.42,
        core::Scalar::new(0.0, 200.0, 0.0, 0.0),
        1,
    )?;
    right_panel_y += line_height;

    // Last maneuver
    if let Some(maneuver) = last_maneuver {
        let seconds_ago = (timestamp_ms - maneuver.timestamp_detected) / 1000.0;
        let is_recent = seconds_ago < 5.0;

        let ago_color = if is_recent {
            core::Scalar::new(0.0, 255.0, 0.0, 0.0)
        } else {
            core::Scalar::new(150.0, 150.0, 150.0, 0.0)
        };

        draw_text_with_shadow(
            output,
            &format!(
                "{} {} {:.1}s ago",
                maneuver.maneuver_type, maneuver.side, seconds_ago,
            ),
            right_panel_x + 8,
            right_panel_y,
            0.42,
            ago_color,
            1,
        )?;
        right_panel_y += line_height;

        // v6.1b: Tiered legality badge
        //   Tier 1: crossed_line_class is Some → confirmed crossing, full-color badge
        //   Tier 2: crossed_line_class is None but buffer has legality → dimmed "inferred" badge
        //   Tier 3: no data at all → grey N/A
        let has_confirmed_crossing = maneuver.crossed_line_class.is_some();

        let (leg_text, leg_color) = if maneuver.legality.contains("CriticalIllegal") {
            if has_confirmed_crossing {
                ("CRITICAL ILLEGAL", core::Scalar::new(0.0, 0.0, 255.0, 0.0))
            } else {
                ("~CRIT.ILLEGAL", core::Scalar::new(0.0, 0.0, 160.0, 0.0))
            }
        } else if maneuver.legality.contains("Illegal") {
            if has_confirmed_crossing {
                ("ILLEGAL", core::Scalar::new(0.0, 100.0, 255.0, 0.0))
            } else {
                ("~ILLEGAL", core::Scalar::new(0.0, 70.0, 160.0, 0.0))
            }
        } else if maneuver.legality.contains("Legal") {
            if has_confirmed_crossing {
                ("LEGAL", core::Scalar::new(0.0, 255.0, 0.0, 0.0))
            } else {
                ("~LEGAL", core::Scalar::new(0.0, 160.0, 0.0, 0.0))
            }
        } else {
            ("N/A", core::Scalar::new(140.0, 140.0, 140.0, 0.0))
        };

        // Show crossed line class if confirmed, otherwise note it's inferred
        let leg_detail = if let Some(ref cls) = maneuver.crossed_line_class {
            format!(
                "  {} [{}] | conf={:.0}% | {:.1}s",
                leg_text,
                cls,
                maneuver.confidence * 100.0,
                maneuver.duration_ms / 1000.0,
            )
        } else {
            format!(
                "  {} | conf={:.0}% | {:.1}s",
                leg_text,
                maneuver.confidence * 100.0,
                maneuver.duration_ms / 1000.0,
            )
        };

        draw_text_with_shadow(
            output,
            &leg_detail,
            right_panel_x + 8,
            right_panel_y,
            0.38,
            leg_color,
            1,
        )?;
        right_panel_y += line_height;
    }

    // Current-frame events (live alerts)
    for event in maneuver_events.iter().take(3) {
        let ev_color = match event.maneuver_type {
            crate::analysis::maneuver_classifier::ManeuverType::Overtake => {
                core::Scalar::new(0.0, 255.0, 0.0, 0.0)
            }
            crate::analysis::maneuver_classifier::ManeuverType::ShadowOvertake => {
                core::Scalar::new(0.0, 0.0, 255.0, 0.0)
            }
            crate::analysis::maneuver_classifier::ManeuverType::LaneChange => {
                core::Scalar::new(255.0, 200.0, 0.0, 0.0) // Cyan/teal for lane changes
            }
        };
        draw_text_with_shadow(
            output,
            &format!(
                ">> {} {} conf={:.0}%",
                event.maneuver_type.as_str(),
                event.side.as_str(),
                event.confidence * 100.0,
            ),
            right_panel_x + 8,
            right_panel_y,
            0.42,
            ev_color,
            1,
        )?;
        right_panel_y += 20;
    }

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// BEV radar
// ──────────────────────────────────────────────────────────────────────────

fn render_bev_radar(
    output: &mut Mat,
    tracked_vehicles: &[&crate::analysis::vehicle_tracker::Track],
    vehicle_state: Option<&VehicleState>,
    legality_result: Option<&LegalityResult>,
    width: i32,
    height: i32,
) -> Result<()> {
    let bar_h = 36; // bottom status bar height
    let bev_w = 160;
    let bev_h = 220;
    let bev_x = width - bev_w - 20;
    let bev_y = height - bev_h - bar_h - 50;

    draw_panel_background(output, bev_x - 5, bev_y - 20, bev_w + 10, bev_h + 25)?;

    draw_text_with_shadow(
        output,
        "RADAR VIEW",
        bev_x + 30,
        bev_y - 5,
        0.42,
        core::Scalar::new(200.0, 200.0, 200.0, 0.0),
        1,
    )?;

    let bev_cx = bev_x + bev_w / 2;
    let lane_half_w = 35;

    // Road surface
    draw_filled_rect_alpha(
        output,
        bev_cx - lane_half_w - 5,
        bev_y,
        (lane_half_w + 5) * 2,
        bev_h,
        0.3,
    )?;

    // v6.0: Color BEV lane lines by legality
    let left_line_color = if let Some(legality) = legality_result {
        marking_color_for_bev(&legality.all_markings, width as f32, true)
    } else {
        core::Scalar::new(100.0, 100.0, 100.0, 0.0)
    };
    let right_line_color = if let Some(legality) = legality_result {
        marking_color_for_bev(&legality.all_markings, width as f32, false)
    } else {
        core::Scalar::new(100.0, 100.0, 100.0, 0.0)
    };

    // Dashed lane boundary lines
    for dy in (0..bev_h).step_by(16) {
        if dy % 32 < 16 {
            imgproc::line(
                output,
                core::Point::new(bev_cx - lane_half_w, bev_y + dy),
                core::Point::new(bev_cx - lane_half_w, bev_y + dy + 10),
                left_line_color,
                1,
                imgproc::LINE_AA,
                0,
            )?;
            imgproc::line(
                output,
                core::Point::new(bev_cx + lane_half_w, bev_y + dy),
                core::Point::new(bev_cx + lane_half_w, bev_y + dy + 10),
                right_line_color,
                1,
                imgproc::LINE_AA,
                0,
            )?;
        }
    }

    // Ego vehicle in BEV
    let ego_bev_y = bev_y + bev_h - 30;
    let ego_offset_bev = if let Some(vs) = vehicle_state {
        let norm = vs.normalized_offset().unwrap_or(0.0);
        (norm * lane_half_w as f32) as i32
    } else {
        0
    };
    imgproc::rectangle(
        output,
        core::Rect::new(bev_cx + ego_offset_bev - 8, ego_bev_y - 12, 16, 24),
        core::Scalar::new(0.0, 255.0, 255.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;
    imgproc::rectangle(
        output,
        core::Rect::new(bev_cx + ego_offset_bev - 8, ego_bev_y - 12, 16, 24),
        core::Scalar::new(255.0, 255.0, 255.0, 0.0),
        1,
        imgproc::LINE_8,
        0,
    )?;

    // Tracked vehicles in BEV
    for track in tracked_vehicles {
        if !track.is_confirmed() {
            continue;
        }

        let bbox = &track.bbox;
        let t_cx = (bbox[0] + bbox[2]) / 2.0;
        let t_cy = (bbox[1] + bbox[3]) / 2.0;

        let rel_x = (t_cx - width as f32 / 2.0) / (width as f32 / 2.0);
        let bev_vx = bev_cx + (rel_x * lane_half_w as f32 * 1.5) as i32;
        let rel_y = t_cy / height as f32;
        let bev_vy = bev_y + (rel_y * bev_h as f32 * 0.85) as i32;

        let (dot_color, _) = zone_color_and_label(track.zone);
        let bev_vx_c = bev_vx.clamp(bev_x + 5, bev_x + bev_w - 5);
        let bev_vy_c = bev_vy.clamp(bev_y + 5, bev_y + bev_h - 5);

        imgproc::rectangle(
            output,
            core::Rect::new(bev_vx_c - 5, bev_vy_c - 4, 10, 8),
            dot_color,
            -1,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::put_text(
            output,
            &format!("{}", track.id),
            core::Point::new(bev_vx_c + 7, bev_vy_c + 3),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.28,
            dot_color,
            1,
            imgproc::LINE_AA,
            false,
        )?;
    }

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// Bottom status bar
// ──────────────────────────────────────────────────────────────────────────

fn render_bottom_status_bar(
    output: &mut Mat,
    frame_id: u64,
    timestamp_ms: f64,
    total_overtakes: u64,
    total_shadow_overtakes: u64,
    total_lane_changes: u64,
    width: i32,
    height: i32,
) -> Result<()> {
    let bar_h = 36;
    let bar_y = height - bar_h;

    let mut overlay = output.try_clone()?;
    imgproc::rectangle(
        &mut overlay,
        core::Rect::new(0, bar_y, width, bar_h),
        core::Scalar::new(20.0, 20.0, 20.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;
    let mut blended = Mat::default();
    core::add_weighted(&overlay, 0.85, output, 0.15, 0.0, &mut blended, -1)?;
    blended.copy_to(output)?;

    // Accent line
    imgproc::line(
        output,
        core::Point::new(0, bar_y),
        core::Point::new(width, bar_y),
        core::Scalar::new(80.0, 80.0, 80.0, 0.0),
        1,
        imgproc::LINE_8,
        0,
    )?;

    // Left: branding
    imgproc::put_text(
        output,
        "OVERTAKE ANALYTICS v7.0",
        core::Point::new(12, bar_y + 24),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.45,
        core::Scalar::new(0.0, 200.0, 200.0, 0.0),
        1,
        imgproc::LINE_AA,
        false,
    )?;

    // Center: frame / time
    let time_text = format!("F{} | {:.1}s", frame_id, timestamp_ms / 1000.0,);
    let mut baseline = 0;
    let text_size = imgproc::get_text_size(
        &time_text,
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.42,
        1,
        &mut baseline,
    )?;
    let center_x = (width - text_size.width) / 2;
    imgproc::put_text(
        output,
        &time_text,
        core::Point::new(center_x, bar_y + 24),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.42,
        core::Scalar::new(180.0, 180.0, 180.0, 0.0),
        1,
        imgproc::LINE_AA,
        false,
    )?;

    // Right: stats
    let stats_text = format!("OVT:{} LC:{} SHDW:{}", total_overtakes, total_lane_changes, total_shadow_overtakes);
    let stats_size = imgproc::get_text_size(
        &stats_text,
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.42,
        1,
        &mut baseline,
    )?;
    imgproc::put_text(
        output,
        &stats_text,
        core::Point::new(width - stats_size.width - 12, bar_y + 24),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.42,
        core::Scalar::new(0.0, 200.0, 0.0, 0.0),
        1,
        imgproc::LINE_AA,
        false,
    )?;

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// Legend
// ──────────────────────────────────────────────────────────────────────────

fn render_legend(output: &mut Mat, _width: i32, height: i32) -> Result<()> {
    let bar_h = 36;
    let legend_x = 10;
    let legend_y = height - bar_h - 120;

    draw_panel_background(output, legend_x - 5, legend_y - 10, 220, 110)?;

    draw_text_with_shadow(
        output,
        "LEGEND",
        legend_x,
        legend_y,
        0.42,
        core::Scalar::new(180.0, 180.0, 180.0, 0.0),
        1,
    )?;

    let legend_items: [(&str, core::Scalar); 5] = [
        ("AHEAD", core::Scalar::new(255.0, 255.0, 0.0, 0.0)),
        ("BESIDE-L", core::Scalar::new(0.0, 165.0, 255.0, 0.0)),
        ("BESIDE-R", core::Scalar::new(255.0, 0.0, 255.0, 0.0)),
        ("BEHIND", core::Scalar::new(0.0, 255.0, 0.0, 0.0)),
        ("EGO", core::Scalar::new(0.0, 255.0, 255.0, 0.0)),
    ];

    for (i, (label, color)) in legend_items.iter().enumerate() {
        let ly = legend_y + 18 + (i as i32 * 18);
        imgproc::rectangle(
            output,
            core::Rect::new(legend_x + 2, ly - 10, 14, 14),
            *color,
            -1,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::put_text(
            output,
            label,
            core::Point::new(legend_x + 22, ly),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.38,
            *color,
            1,
            imgproc::LINE_AA,
            false,
        )?;
    }

    Ok(())
}

// ════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ════════════════════════════════════════════════════════════════════════════

fn draw_panel_background(img: &mut Mat, x: i32, y: i32, w: i32, h: i32) -> Result<()> {
    let mut overlay = img.clone();
    imgproc::rectangle(
        &mut overlay,
        core::Rect::new(x, y, w, h),
        core::Scalar::new(0.0, 0.0, 0.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;
    let mut result = Mat::default();
    core::add_weighted(&overlay, 0.7, img, 0.3, 0.0, &mut result, -1)?;
    result.copy_to(img)?;

    imgproc::rectangle(
        img,
        core::Rect::new(x, y, w, h),
        core::Scalar::new(60.0, 60.0, 60.0, 0.0),
        1,
        imgproc::LINE_AA,
        0,
    )?;

    Ok(())
}

fn draw_filled_rect_alpha(img: &mut Mat, x: i32, y: i32, w: i32, h: i32, alpha: f64) -> Result<()> {
    let mut overlay = img.clone();
    imgproc::rectangle(
        &mut overlay,
        core::Rect::new(x, y, w, h),
        core::Scalar::new(0.0, 0.0, 0.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;
    let mut result = Mat::default();
    core::add_weighted(&overlay, alpha, img, 1.0 - alpha, 0.0, &mut result, -1)?;
    result.copy_to(img)?;
    Ok(())
}

fn draw_text_with_shadow(
    img: &mut Mat,
    text: &str,
    x: i32,
    y: i32,
    scale: f64,
    color: core::Scalar,
    thickness: i32,
) -> Result<()> {
    // Shadow
    imgproc::put_text(
        img,
        text,
        core::Point::new(x + 1, y + 1),
        imgproc::FONT_HERSHEY_SIMPLEX,
        scale,
        core::Scalar::new(0.0, 0.0, 0.0, 0.0),
        thickness + 1,
        imgproc::LINE_AA,
        false,
    )?;
    // Main
    imgproc::put_text(
        img,
        text,
        core::Point::new(x, y),
        imgproc::FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        imgproc::LINE_AA,
        false,
    )?;
    Ok(())
}

fn draw_corner_accents(
    img: &mut Mat,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    len: i32,
    color: core::Scalar,
    thickness: i32,
) -> Result<()> {
    // Top-left
    imgproc::line(
        img,
        core::Point::new(x1, y1),
        core::Point::new(x1 + len, y1),
        color,
        thickness,
        imgproc::LINE_AA,
        0,
    )?;
    imgproc::line(
        img,
        core::Point::new(x1, y1),
        core::Point::new(x1, y1 + len),
        color,
        thickness,
        imgproc::LINE_AA,
        0,
    )?;
    // Top-right
    imgproc::line(
        img,
        core::Point::new(x2, y1),
        core::Point::new(x2 - len, y1),
        color,
        thickness,
        imgproc::LINE_AA,
        0,
    )?;
    imgproc::line(
        img,
        core::Point::new(x2, y1),
        core::Point::new(x2, y1 + len),
        color,
        thickness,
        imgproc::LINE_AA,
        0,
    )?;
    // Bottom-left
    imgproc::line(
        img,
        core::Point::new(x1, y2),
        core::Point::new(x1 + len, y2),
        color,
        thickness,
        imgproc::LINE_AA,
        0,
    )?;
    imgproc::line(
        img,
        core::Point::new(x1, y2),
        core::Point::new(x1, y2 - len),
        color,
        thickness,
        imgproc::LINE_AA,
        0,
    )?;
    // Bottom-right
    imgproc::line(
        img,
        core::Point::new(x2, y2),
        core::Point::new(x2 - len, y2),
        color,
        thickness,
        imgproc::LINE_AA,
        0,
    )?;
    imgproc::line(
        img,
        core::Point::new(x2, y2),
        core::Point::new(x2, y2 - len),
        color,
        thickness,
        imgproc::LINE_AA,
        0,
    )?;
    Ok(())
}

fn draw_confidence_bar(img: &mut Mat, x: i32, y: i32, w: i32, h: i32, value: f32) -> Result<()> {
    let value = value.clamp(0.0, 1.0);

    // Background
    imgproc::rectangle(
        img,
        core::Rect::new(x, y, w, h),
        core::Scalar::new(40.0, 40.0, 40.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;

    let fill_w = (w as f32 * value) as i32;
    if fill_w > 0 {
        let fill_color = if value > 0.7 {
            core::Scalar::new(0.0, 200.0, 0.0, 0.0)
        } else if value > 0.4 {
            core::Scalar::new(0.0, 180.0, 220.0, 0.0)
        } else {
            core::Scalar::new(0.0, 80.0, 220.0, 0.0)
        };
        imgproc::rectangle(
            img,
            core::Rect::new(x, y, fill_w, h),
            fill_color,
            -1,
            imgproc::LINE_8,
            0,
        )?;
    }

    imgproc::rectangle(
        img,
        core::Rect::new(x, y, w, h),
        core::Scalar::new(100.0, 100.0, 100.0, 0.0),
        1,
        imgproc::LINE_8,
        0,
    )?;

    Ok(())
}

fn draw_lateral_gauge(
    img: &mut Mat,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    normalized: f32,
) -> Result<()> {
    let normalized = normalized.clamp(-1.0, 1.0);

    // Background
    imgproc::rectangle(
        img,
        core::Rect::new(x, y, w, h),
        core::Scalar::new(30.0, 30.0, 30.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;

    // Danger zones at edges
    let danger_w = (w as f32 * 0.15) as i32;
    imgproc::rectangle(
        img,
        core::Rect::new(x, y, danger_w, h),
        core::Scalar::new(0.0, 0.0, 80.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;
    imgproc::rectangle(
        img,
        core::Rect::new(x + w - danger_w, y, danger_w, h),
        core::Scalar::new(0.0, 0.0, 80.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;

    // Center marker
    imgproc::line(
        img,
        core::Point::new(x + w / 2, y),
        core::Point::new(x + w / 2, y + h),
        core::Scalar::new(80.0, 80.0, 80.0, 0.0),
        1,
        imgproc::LINE_8,
        0,
    )?;

    // Needle
    let needle_x = x + (w as f32 * (0.5 + normalized * 0.5)) as i32;
    let needle_color = if normalized.abs() > 0.7 {
        core::Scalar::new(0.0, 0.0, 255.0, 0.0)
    } else if normalized.abs() > 0.3 {
        core::Scalar::new(0.0, 200.0, 255.0, 0.0)
    } else {
        core::Scalar::new(0.0, 255.0, 0.0, 0.0)
    };

    imgproc::rectangle(
        img,
        core::Rect::new(needle_x - 2, y - 2, 4, h + 4),
        needle_color,
        -1,
        imgproc::LINE_8,
        0,
    )?;

    // Border
    imgproc::rectangle(
        img,
        core::Rect::new(x, y, w, h),
        core::Scalar::new(100.0, 100.0, 100.0, 0.0),
        1,
        imgproc::LINE_8,
        0,
    )?;

    Ok(())
}

// ════════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ════════════════════════════════════════════════════════════════════════════

fn zone_color_and_label(
    zone: crate::analysis::vehicle_tracker::VehicleZone,
) -> (core::Scalar, &'static str) {
    use crate::analysis::vehicle_tracker::VehicleZone;
    match zone {
        VehicleZone::Ahead => (core::Scalar::new(255.0, 255.0, 0.0, 0.0), "AHEAD"),
        VehicleZone::BesideLeft => (core::Scalar::new(0.0, 165.0, 255.0, 0.0), "BESIDE-L"),
        VehicleZone::BesideRight => (core::Scalar::new(255.0, 0.0, 255.0, 0.0), "BESIDE-R"),
        VehicleZone::Behind => (core::Scalar::new(0.0, 255.0, 0.0, 0.0), "BEHIND"),
        VehicleZone::Unknown => (core::Scalar::new(128.0, 128.0, 128.0, 0.0), "?"),
    }
}

/// Rough distance estimation from bounding box height.
/// Based on pinhole camera model: d = (real_h × f) / pixel_h
/// Uses typical vehicle height ~1.5m and a dashcam focal length approximation.
fn estimate_distance(bbox_height_px: f32, frame_height_px: f32) -> f32 {
    let typical_vehicle_height_m = 1.5;
    let focal_ratio = frame_height_px * 0.9; // Rough focal length in pixels
    if bbox_height_px < 5.0 {
        return 999.0;
    }
    (typical_vehicle_height_m * focal_ratio) / bbox_height_px
}

/// Extract left boundary X from lane detection for road_overlay integration.
fn ego_left_x_from_lanes(lanes: &[DetectedLane], _width: i32) -> Option<f32> {
    if lanes.is_empty() {
        return None;
    }
    // Left boundary = lane[0], use bottom-most point's X
    lanes[0]
        .points
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|p| p.0)
}

/// Extract right boundary X from lane detection for road_overlay integration.
fn ego_right_x_from_lanes(lanes: &[DetectedLane], _width: i32) -> Option<f32> {
    if lanes.len() < 2 {
        return None;
    }
    lanes[1]
        .points
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|p| p.0)
}

/// Derive PassingLegality from the detected markings when not explicitly provided.
fn passing_legality_from_markings(markings: &[DetectedRoadMarking]) -> PassingLegality {
    // If any marking is critical illegal (double solid), overall is Prohibited
    if markings
        .iter()
        .any(|m| m.legality == LineLegality::CriticalIllegal)
    {
        return PassingLegality::Prohibited;
    }
    if markings.iter().any(|m| m.legality == LineLegality::Illegal) {
        return PassingLegality::Prohibited;
    }
    if markings.iter().any(|m| m.legality == LineLegality::Legal) {
        return PassingLegality::Allowed;
    }
    PassingLegality::Unknown
}

/// Get BEV lane line color based on marking legality.
fn marking_color_for_bev(
    markings: &[DetectedRoadMarking],
    frame_width: f32,
    is_left: bool,
) -> core::Scalar {
    let center = frame_width / 2.0;
    let relevant = markings.iter().find(|m| {
        let mx = (m.bbox[0] + m.bbox[2]) / 2.0;
        if is_left {
            mx < center
        } else {
            mx >= center
        }
    });

    match relevant.map(|m| &m.legality) {
        Some(LineLegality::Legal) => core::Scalar::new(0.0, 200.0, 0.0, 0.0),
        Some(LineLegality::Illegal | LineLegality::CriticalIllegal) => {
            core::Scalar::new(0.0, 0.0, 200.0, 0.0)
        }
        _ => core::Scalar::new(100.0, 100.0, 100.0, 0.0),
    }
}
