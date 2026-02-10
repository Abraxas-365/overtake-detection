// src/video_processor.rs
use crate::lane_legality::LegalityResult;
use crate::overtake_analyzer::{OvertakeEvent, TrackedVehicle};
use crate::shadow_overtake::{ShadowOvertakeDetector, ShadowSeverity};
use crate::types::CurveInfo;
use crate::types::{Config, DetectedLane, VehicleState};
use anyhow::Result;
use opencv::{
    core::{self, Mat, Vector},
    imgcodecs, imgproc,
    prelude::*,
    videoio::{self, VideoCapture, VideoCaptureTraitConst, VideoWriter},
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::info;
use walkdir::WalkDir;

pub struct VideoProcessor {
    config: Config,
}

impl VideoProcessor {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    pub fn find_video_files(&self) -> Result<Vec<PathBuf>> {
        let mut videos = Vec::new();
        let video_extensions = vec!["mp4", "avi", "mov", "mkv", "MP4", "AVI", "MOV", "MKV"];

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

    /// Save a specific frame as an image file
    pub fn save_frame_to_disk(
        &self,
        frame: &crate::types::Frame,
        filename: &str,
    ) -> Result<PathBuf> {
        let output_dir = Path::new(&self.config.video.output_dir).join("evidence");
        std::fs::create_dir_all(&output_dir)?;

        let file_path = output_dir.join(filename);

        // Frame data is RGB, OpenCV needs BGR
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

// ══════════════════════════════════════════════════════════════════════════
// YOLO-PRIMARY VISUALIZATION - Full telemetry overlay
// ══════════════════════════════════════════════════════════════════════════

/// YOLO-seg primary visualization with full telemetry overlay + LEGALITY
pub fn draw_lanes_with_state_enhanced(
    frame: &[u8],
    width: i32,
    height: i32,
    lanes: &[DetectedLane],
    state: &str,
    vehicle_state: Option<&VehicleState>,
    tracked_vehicles: &HashMap<u32, TrackedVehicle>,
    shadow_detector: &ShadowOvertakeDetector,
    frame_id: u64,
    timestamp_ms: f64,
    is_overtaking: bool,
    overtake_direction: Option<&str>,
    vehicles_overtaken_this_maneuver: &[OvertakeEvent],
    curve_info: Option<CurveInfo>,
    lateral_velocity: f32,
    legality_result: Option<&LegalityResult>,
    detection_source: &str,
) -> Result<Mat> {
    let mat = Mat::from_slice(frame)?;
    let mat = mat.reshape(3, height)?;
    let mut bgr_mat = Mat::default();
    imgproc::cvt_color(&mat, &mut bgr_mat, imgproc::COLOR_RGB2BGR, 0)?;
    let mut output = bgr_mat.try_clone()?;

    // ══════════════════════════════════════════════════════════════════════
    // 1. DRAW LANE MARKINGS (YOLO-seg detected boundaries)
    // ══════════════════════════════════════════════════════════════════════
    let lane_colors = vec![
        core::Scalar::new(0.0, 0.0, 255.0, 0.0), // Red - left boundary
        core::Scalar::new(0.0, 255.0, 0.0, 0.0), // Green - right boundary
        core::Scalar::new(255.0, 0.0, 0.0, 0.0), // Blue
        core::Scalar::new(0.0, 255.0, 255.0, 0.0), // Yellow
    ];

    for (i, lane) in lanes.iter().enumerate() {
        let color = lane_colors[i % lane_colors.len()];
        let thickness = if i < 2 { 4 } else { 2 }; // Thicker for primary ego boundaries

        // Draw lane points
        for point in &lane.points {
            let pt = core::Point::new(point.0 as i32, point.1 as i32);
            imgproc::circle(&mut output, pt, thickness, color, -1, imgproc::LINE_8, 0)?;
        }

        // Connect lane points with lines for better visibility
        if lane.points.len() >= 2 {
            for j in 0..lane.points.len() - 1 {
                let pt1 = core::Point::new(lane.points[j].0 as i32, lane.points[j].1 as i32);
                let pt2 =
                    core::Point::new(lane.points[j + 1].0 as i32, lane.points[j + 1].1 as i32);
                imgproc::line(&mut output, pt1, pt2, color, 2, imgproc::LINE_AA, 0)?;
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 2. DRAW TRACKED VEHICLES (YOLO) WITH STATUS INDICATORS
    // ══════════════════════════════════════════════════════════════════════

    let shadow_vehicle_ids: std::collections::HashSet<u32> = if shadow_detector.is_monitoring() {
        shadow_detector
            .get_current_shadows()
            .iter()
            .map(|s| s.blocking_vehicle_id)
            .collect()
    } else {
        std::collections::HashSet::new()
    };

    let overtaken_vehicle_ids: std::collections::HashSet<u32> = vehicles_overtaken_this_maneuver
        .iter()
        .map(|o| o.vehicle_id)
        .collect();

    for (vehicle_id, vehicle) in tracked_vehicles {
        if let Some(latest_pos) = vehicle.position_history.last() {
            if frame_id.saturating_sub(latest_pos.frame_id) < 10 {
                let bbox = &vehicle.bbox;

                let is_shadow_blocker = shadow_vehicle_ids.contains(vehicle_id);
                let was_overtaken = overtaken_vehicle_ids.contains(vehicle_id);

                let box_color = if is_shadow_blocker {
                    core::Scalar::new(0.0, 0.0, 255.0, 0.0) // RED
                } else if was_overtaken {
                    core::Scalar::new(255.0, 255.0, 0.0, 0.0) // CYAN
                } else {
                    core::Scalar::new(0.0, 255.0, 0.0, 0.0) // GREEN
                };

                let thickness = if is_shadow_blocker { 4 } else { 2 };

                // Draw bounding box
                let pt1 = core::Point::new(bbox[0] as i32, bbox[1] as i32);
                let pt2 = core::Point::new(bbox[2] as i32, bbox[3] as i32);
                imgproc::rectangle(
                    &mut output,
                    core::Rect::from_points(pt1, pt2),
                    box_color,
                    thickness,
                    imgproc::LINE_8,
                    0,
                )?;

                // Label with YOLO class + confidence
                let label = if is_shadow_blocker {
                    format!("! BLOCKING {} #{}", vehicle.class_name, vehicle_id)
                } else if was_overtaken {
                    format!("v {} #{}", vehicle.class_name, vehicle_id)
                } else {
                    format!("{} #{}", vehicle.class_name, vehicle_id)
                };

                let label_pos = core::Point::new(bbox[0] as i32, (bbox[1] as i32) - 5);

                let label_size =
                    imgproc::get_text_size(&label, imgproc::FONT_HERSHEY_SIMPLEX, 0.5, 1, &mut 0)?;

                let label_bg_pt1 =
                    core::Point::new(label_pos.x - 2, label_pos.y - label_size.height - 2);
                let label_bg_pt2 =
                    core::Point::new(label_pos.x + label_size.width + 2, label_pos.y + 2);

                imgproc::rectangle(
                    &mut output,
                    core::Rect::from_points(label_bg_pt1, label_bg_pt2),
                    core::Scalar::new(0.0, 0.0, 0.0, 0.0),
                    -1,
                    imgproc::LINE_8,
                    0,
                )?;

                imgproc::put_text(
                    &mut output,
                    &label,
                    label_pos,
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    core::Scalar::new(255.0, 255.0, 255.0, 0.0),
                    1,
                    imgproc::LINE_8,
                    false,
                )?;
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 3. DRAW EGO VEHICLE MARKER
    // ══════════════════════════════════════════════════════════════════════
    let vehicle_x = width / 2;
    let vehicle_y = (height as f32 * 0.85) as i32;
    imgproc::circle(
        &mut output,
        core::Point::new(vehicle_x, vehicle_y),
        12,
        core::Scalar::new(0.0, 255.0, 255.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;

    // Draw direction arrow if overtaking
    if is_overtaking {
        if let Some(direction) = overtake_direction {
            let arrow_start = core::Point::new(vehicle_x, vehicle_y);
            let arrow_end = if direction == "LEFT" {
                core::Point::new(vehicle_x - 40, vehicle_y)
            } else {
                core::Point::new(vehicle_x + 40, vehicle_y)
            };
            imgproc::arrowed_line(
                &mut output,
                arrow_start,
                arrow_end,
                core::Scalar::new(0.0, 255.0, 255.0, 0.0),
                3,
                imgproc::LINE_8,
                0,
                0.3,
            )?;
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 4. TOP BANNER: CRITICAL WARNINGS
    // ══════════════════════════════════════════════════════════════════════
    let mut banner_active = false;
    let mut banner_text = String::new();
    let mut banner_color = core::Scalar::new(0.0, 0.0, 200.0, 0.0);

    if shadow_detector.is_monitoring() && shadow_detector.active_shadow_count() > 0 {
        banner_active = true;
        banner_text = format!(
            "! SHADOW OVERTAKE: {} vehicle(s) blocking visibility! DANGER!",
            shadow_detector.active_shadow_count()
        );
        banner_color = core::Scalar::new(0.0, 0.0, 255.0, 0.0);
    } else if let Some(curve) = curve_info {
        if curve.is_curve && is_overtaking {
            banner_active = true;
            banner_text = format!(
                "! OVERTAKING IN {} CURVE ({:.1}) - ILLEGAL!",
                match curve.curve_type {
                    crate::types::CurveType::Sharp => "SHARP",
                    crate::types::CurveType::Moderate => "MODERATE",
                    _ => "CURVE",
                },
                curve.angle_degrees
            );
            banner_color = core::Scalar::new(0.0, 100.0, 255.0, 0.0);
        }
    }

    if banner_active {
        let banner_height = 60;
        imgproc::rectangle(
            &mut output,
            core::Rect::new(0, 0, width, banner_height),
            banner_color,
            -1,
            imgproc::LINE_8,
            0,
        )?;

        imgproc::put_text(
            &mut output,
            &banner_text,
            core::Point::new(20, 40),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.9,
            core::Scalar::new(255.0, 255.0, 255.0, 0.0),
            2,
            imgproc::LINE_8,
            false,
        )?;
    }

    // ══════════════════════════════════════════════════════════════════════
    // 4.5 LEGALITY BANNER (Overrides normal banner if illegal)
    // ══════════════════════════════════════════════════════════════════════
    if let Some(legality) = legality_result {
        if legality.ego_intersects_marking && legality.verdict.is_illegal() {
            let legality_banner_y = if banner_active { 60 } else { 0 };
            let banner_h = 50;

            let color = match legality.verdict {
                crate::lane_legality::LineLegality::CriticalIllegal => {
                    core::Scalar::new(0.0, 0.0, 255.0, 0.0)
                }
                crate::lane_legality::LineLegality::Illegal => {
                    core::Scalar::new(0.0, 50.0, 200.0, 0.0)
                }
                _ => core::Scalar::new(0.0, 200.0, 255.0, 0.0),
            };

            imgproc::rectangle(
                &mut output,
                core::Rect::new(0, legality_banner_y, width, banner_h),
                color,
                -1,
                imgproc::LINE_8,
                0,
            )?;

            let line_name = legality
                .intersecting_line
                .as_ref()
                .map(|l| l.class_name.as_str())
                .unwrap_or("SOLID LINE");

            let text = format!(
                "! ILLEGAL CROSSING: {} - VIOLATION DS 016-2009-MTC",
                line_name.to_uppercase()
            );

            draw_text_with_shadow(
                &mut output,
                &text,
                20,
                legality_banner_y + 35,
                0.8,
                core::Scalar::new(255.0, 255.0, 255.0, 0.0),
                2,
            )?;

            banner_active = true;
        }

        // Draw detected road markings with color-coded bboxes
        for marking in &legality.all_markings {
            use crate::lane_legality::LineLegality;

            let box_color = match marking.legality {
                LineLegality::CriticalIllegal => core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                LineLegality::Illegal => core::Scalar::new(0.0, 100.0, 255.0, 0.0),
                LineLegality::Legal => core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                _ => core::Scalar::new(200.0, 200.0, 200.0, 0.0),
            };

            let pt1 = core::Point::new(marking.bbox[0] as i32, marking.bbox[1] as i32);
            let pt2 = core::Point::new(marking.bbox[2] as i32, marking.bbox[3] as i32);
            imgproc::rectangle(
                &mut output,
                core::Rect::from_points(pt1, pt2),
                box_color,
                2,
                imgproc::LINE_8,
                0,
            )?;

            let label = format!(
                "{} ({:.0}%)",
                marking.class_name,
                marking.confidence * 100.0
            );
            draw_text_with_shadow(
                &mut output,
                &label,
                marking.bbox[0] as i32,
                marking.bbox[1] as i32 - 5,
                0.4,
                box_color,
                1,
            )?;
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 5. LEFT PANEL: YOLO-SEG DETECTION & VEHICLE STATE
    // ══════════════════════════════════════════════════════════════════════
    let panel_x = 15;
    let mut panel_y = if banner_active { 80 } else { 30 };
    let line_height = 26;

    draw_panel_background(&mut output, 5, panel_y - 10, 460, 340)?;

    // Title — YOLO-primary branding
    draw_text_with_shadow(
        &mut output,
        "YOLO-SEG LANE DETECTION",
        panel_x,
        panel_y,
        0.7,
        core::Scalar::new(0.0, 255.0, 255.0, 0.0), // Cyan title
        2,
    )?;
    panel_y += line_height;

    // Frame info
    draw_text_with_shadow(
        &mut output,
        &format!("Frame: {} | Time: {:.2}s", frame_id, timestamp_ms / 1000.0),
        panel_x,
        panel_y,
        0.5,
        core::Scalar::new(200.0, 200.0, 200.0, 0.0),
        1,
    )?;
    panel_y += line_height;

    // Detection source — color-coded by detector
    let (source_label, source_color) = match detection_source {
        s if s.contains("YOLO_SEG") => (
            "YOLO-seg (Primary)",
            core::Scalar::new(0.0, 255.0, 0.0, 0.0), // Green — primary hit
        ),
        s if s.contains("YOLO_MISS") => (
            "YOLO-seg MISS",
            core::Scalar::new(0.0, 0.0, 255.0, 0.0), // Red — no detection
        ),
        s if s.contains("UFLD") || s.contains("FALLBACK") => (
            "Fallback (UFLDv2)",
            core::Scalar::new(0.0, 165.0, 255.0, 0.0), // Orange — fallback
        ),
        _ => (
            detection_source,
            core::Scalar::new(200.0, 200.0, 200.0, 0.0),
        ),
    };

    draw_text_with_shadow(
        &mut output,
        &format!("Detector: {}", source_label),
        panel_x,
        panel_y,
        0.6,
        source_color,
        2,
    )?;
    panel_y += line_height;

    // Boundary count indicator
    let boundary_count = lanes.len();
    let (boundary_label, boundary_color) = match boundary_count {
        0 => ("Boundaries: NONE", core::Scalar::new(0.0, 0.0, 255.0, 0.0)),
        1 => (
            "Boundaries: 1 (estimated)",
            core::Scalar::new(0.0, 200.0, 255.0, 0.0),
        ),
        2 => (
            "Boundaries: 2 (L+R)",
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        ),
        n => (
            "Boundaries: MULTI",
            core::Scalar::new(0.0, 255.0, 200.0, 0.0),
        ),
    };

    // We need to format dynamically for the multi case
    let boundary_text = if boundary_count > 2 {
        format!("Boundaries: {} detected", boundary_count)
    } else {
        boundary_label.to_string()
    };

    draw_text_with_shadow(
        &mut output,
        &boundary_text,
        panel_x,
        panel_y,
        0.55,
        boundary_color,
        1,
    )?;
    panel_y += line_height;

    // Lane change state with color coding
    let state_color = match state {
        "CENTERED" => core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        "DRIFTING" => core::Scalar::new(0.0, 165.0, 255.0, 0.0),
        "CROSSING" => core::Scalar::new(0.0, 0.0, 255.0, 0.0),
        "COMPLETED" => core::Scalar::new(255.0, 255.0, 0.0, 0.0),
        _ => core::Scalar::new(255.0, 255.0, 255.0, 0.0),
    };

    draw_text_with_shadow(
        &mut output,
        &format!("State: {}", state),
        panel_x,
        panel_y,
        0.65,
        state_color,
        2,
    )?;
    panel_y += line_height;

    // Vehicle offset and velocity
    if let Some(vs) = vehicle_state {
        if vs.is_valid() {
            let normalized = vs.normalized_offset().unwrap_or(0.0);

            draw_text_with_shadow(
                &mut output,
                &format!(
                    "Lateral Offset: {:.1}px ({:+.1}%)",
                    vs.lateral_offset,
                    normalized * 100.0
                ),
                panel_x,
                panel_y,
                0.55,
                core::Scalar::new(255.0, 255.0, 255.0, 0.0),
                1,
            )?;
            panel_y += line_height;

            // Velocity with color ramp
            let vel_color = if lateral_velocity.abs() > 200.0 {
                core::Scalar::new(0.0, 0.0, 255.0, 0.0) // Red — spike
            } else if lateral_velocity.abs() > 100.0 {
                core::Scalar::new(0.0, 165.0, 255.0, 0.0) // Orange — high
            } else if lateral_velocity.abs() > 60.0 {
                core::Scalar::new(0.0, 255.0, 255.0, 0.0) // Yellow — medium
            } else {
                core::Scalar::new(255.0, 255.0, 255.0, 0.0) // White — normal
            };

            draw_text_with_shadow(
                &mut output,
                &format!("Lateral Velocity: {:.1} px/s", lateral_velocity),
                panel_x,
                panel_y,
                0.55,
                vel_color,
                1,
            )?;
            panel_y += line_height;

            if let Some(lw) = vs.lane_width {
                draw_text_with_shadow(
                    &mut output,
                    &format!("Lane Width: {:.0}px", lw),
                    panel_x,
                    panel_y,
                    0.55,
                    core::Scalar::new(200.0, 200.0, 200.0, 0.0),
                    1,
                )?;
                panel_y += line_height;
            }
        } else {
            // Invalid state — show occlusion indicator
            draw_text_with_shadow(
                &mut output,
                "Position: NO VALID LANES",
                panel_x,
                panel_y,
                0.55,
                core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                1,
            )?;
            panel_y += line_height;
        }
    }

    // Tracked vehicles
    let active_vehicles = tracked_vehicles
        .values()
        .filter(|v| {
            v.position_history
                .last()
                .map(|p| frame_id.saturating_sub(p.frame_id) < 10)
                .unwrap_or(false)
        })
        .count();

    draw_text_with_shadow(
        &mut output,
        &format!("YOLO Vehicles: {}", active_vehicles),
        panel_x,
        panel_y,
        0.55,
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        1,
    )?;

    // ══════════════════════════════════════════════════════════════════════
    // 6. RIGHT PANEL: OVERTAKE STATUS
    // ══════════════════════════════════════════════════════════════════════
    let right_panel_x = width - 480;
    let mut right_panel_y = if banner_active { 80 } else { 30 };

    draw_panel_background(
        &mut output,
        right_panel_x - 10,
        right_panel_y - 10,
        470,
        350,
    )?;

    // Title
    draw_text_with_shadow(
        &mut output,
        "OVERTAKE STATUS",
        right_panel_x,
        right_panel_y,
        0.7,
        core::Scalar::new(100.0, 200.0, 255.0, 0.0),
        2,
    )?;
    right_panel_y += line_height;

    if is_overtaking {
        if let Some(direction) = overtake_direction {
            draw_text_with_shadow(
                &mut output,
                &format!(">> OVERTAKING {} >>", direction),
                right_panel_x,
                right_panel_y,
                0.7,
                core::Scalar::new(0.0, 165.0, 255.0, 0.0),
                2,
            )?;
            right_panel_y += line_height;
        }
    } else {
        draw_text_with_shadow(
            &mut output,
            "Status: Normal driving",
            right_panel_x,
            right_panel_y,
            0.6,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            1,
        )?;
        right_panel_y += line_height;
    }

    // Vehicles overtaken this maneuver
    if !vehicles_overtaken_this_maneuver.is_empty() {
        draw_text_with_shadow(
            &mut output,
            &format!(
                "Overtaken: {} vehicle(s)",
                vehicles_overtaken_this_maneuver.len()
            ),
            right_panel_x,
            right_panel_y,
            0.6,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            1,
        )?;
        right_panel_y += line_height;

        for overtaken in vehicles_overtaken_this_maneuver.iter().take(5) {
            draw_text_with_shadow(
                &mut output,
                &format!(
                    "  * {} (ID #{})",
                    overtaken.class_name, overtaken.vehicle_id
                ),
                right_panel_x + 10,
                right_panel_y,
                0.5,
                core::Scalar::new(200.0, 255.0, 200.0, 0.0),
                1,
            )?;
            right_panel_y += 22;
        }

        if vehicles_overtaken_this_maneuver.len() > 5 {
            draw_text_with_shadow(
                &mut output,
                &format!(
                    "  ... and {} more",
                    vehicles_overtaken_this_maneuver.len() - 5
                ),
                right_panel_x + 10,
                right_panel_y,
                0.5,
                core::Scalar::new(200.0, 200.0, 200.0, 0.0),
                1,
            )?;
            right_panel_y += 22;
        }
    } else if is_overtaking {
        draw_text_with_shadow(
            &mut output,
            "Overtaken: 0 (no vehicles passed yet)",
            right_panel_x,
            right_panel_y,
            0.55,
            core::Scalar::new(200.0, 200.0, 200.0, 0.0),
            1,
        )?;
        right_panel_y += line_height;
    }

    right_panel_y += 10;

    // Shadow overtake status
    if shadow_detector.is_monitoring() {
        let shadow_count = shadow_detector.active_shadow_count();

        if shadow_count > 0 {
            draw_text_with_shadow(
                &mut output,
                &format!("! SHADOW: {} vehicle(s)", shadow_count),
                right_panel_x,
                right_panel_y,
                0.65,
                core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                2,
            )?;
            right_panel_y += line_height;

            if let Some(severity) = shadow_detector.worst_active_severity() {
                let severity_color = match severity {
                    ShadowSeverity::Critical => core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                    ShadowSeverity::Dangerous => core::Scalar::new(0.0, 100.0, 255.0, 0.0),
                    ShadowSeverity::Warning => core::Scalar::new(0.0, 255.0, 255.0, 0.0),
                };

                draw_text_with_shadow(
                    &mut output,
                    &format!("Severity: {}", severity.as_str()),
                    right_panel_x + 10,
                    right_panel_y,
                    0.6,
                    severity_color,
                    2,
                )?;
                right_panel_y += line_height;
            }

            for shadow in shadow_detector.get_current_shadows().iter().take(3) {
                draw_text_with_shadow(
                    &mut output,
                    &format!(
                        "  * {} ID #{} blocking",
                        shadow.blocking_vehicle_class, shadow.blocking_vehicle_id
                    ),
                    right_panel_x + 10,
                    right_panel_y,
                    0.5,
                    core::Scalar::new(255.0, 150.0, 150.0, 0.0),
                    1,
                )?;
                right_panel_y += 22;
            }
        } else {
            draw_text_with_shadow(
                &mut output,
                "Shadow: Monitoring (clear)",
                right_panel_x,
                right_panel_y,
                0.55,
                core::Scalar::new(0.0, 255.0, 255.0, 0.0),
                1,
            )?;
            right_panel_y += line_height;
        }
    }

    // Curve detection
    if let Some(curve) = curve_info {
        if curve.is_curve {
            draw_text_with_shadow(
                &mut output,
                &format!(
                    "Curve: {} ({:.1} deg)",
                    match curve.curve_type {
                        crate::types::CurveType::Sharp => "SHARP",
                        crate::types::CurveType::Moderate => "MODERATE",
                        _ => "DETECTED",
                    },
                    curve.angle_degrees
                ),
                right_panel_x,
                right_panel_y,
                0.6,
                core::Scalar::new(0.0, 200.0, 255.0, 0.0),
                1,
            )?;
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 7. LEGEND (Bottom Right)
    // ══════════════════════════════════════════════════════════════════════
    let legend_y = height - 160;
    draw_panel_background(&mut output, width - 280, legend_y - 10, 270, 155)?;

    draw_text_with_shadow(
        &mut output,
        "LEGEND",
        width - 270,
        legend_y,
        0.5,
        core::Scalar::new(200.0, 200.0, 200.0, 0.0),
        1,
    )?;

    let legend_items: Vec<(&str, core::Scalar)> = vec![
        (
            "[GREEN] YOLO-seg primary hit",
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        ),
        (
            "[ORANGE] UFLDv2 fallback",
            core::Scalar::new(0.0, 165.0, 255.0, 0.0),
        ),
        (
            "[RED] No lane detection",
            core::Scalar::new(0.0, 0.0, 255.0, 0.0),
        ),
        (
            "[GREEN box] Tracked vehicle",
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        ),
        (
            "[CYAN box] Overtaken vehicle",
            core::Scalar::new(255.0, 255.0, 0.0, 0.0),
        ),
        (
            "[RED box] Shadow blocker!",
            core::Scalar::new(0.0, 0.0, 255.0, 0.0),
        ),
        (
            "[YELLOW] Ego vehicle",
            core::Scalar::new(0.0, 255.0, 255.0, 0.0),
        ),
    ];

    for (i, (label, color)) in legend_items.iter().enumerate() {
        draw_text_with_shadow(
            &mut output,
            label,
            width - 265,
            legend_y + 20 + (i as i32 * 18),
            0.38,
            *color,
            1,
        )?;
    }

    Ok(output)
}

// ══════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS FOR DRAWING
// ══════════════════════════════════════════════════════════════════════════

fn draw_panel_background(img: &mut Mat, x: i32, y: i32, width: i32, height: i32) -> Result<()> {
    let mut overlay = img.clone();
    imgproc::rectangle(
        &mut overlay,
        core::Rect::new(x, y, width, height),
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
        core::Rect::new(x, y, width, height),
        core::Scalar::new(80.0, 80.0, 80.0, 0.0),
        2,
        imgproc::LINE_8,
        0,
    )?;

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
        core::Point::new(x + 2, y + 2),
        imgproc::FONT_HERSHEY_SIMPLEX,
        scale,
        core::Scalar::new(0.0, 0.0, 0.0, 0.0),
        thickness,
        imgproc::LINE_8,
        false,
    )?;

    // Main text
    imgproc::put_text(
        img,
        text,
        core::Point::new(x, y),
        imgproc::FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        imgproc::LINE_8,
        false,
    )?;

    Ok(())
}

/// Simple visualization fallback (minimal overlay)
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

    let colors = vec![
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

        // Connect points
        if lane.points.len() >= 2 {
            for j in 0..lane.points.len() - 1 {
                let pt1 = core::Point::new(lane.points[j].0 as i32, lane.points[j].1 as i32);
                let pt2 =
                    core::Point::new(lane.points[j + 1].0 as i32, lane.points[j + 1].1 as i32);
                imgproc::line(&mut output, pt1, pt2, color, 2, imgproc::LINE_AA, 0)?;
            }
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

    imgproc::put_text(
        &mut output,
        &format!("State: {}", state),
        core::Point::new(15, 32),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.8,
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        false,
    )?;

    if let Some(vs) = vehicle_state {
        if vs.is_valid() {
            let normalized = vs.normalized_offset().unwrap_or(0.0);
            let info = format!(
                "Offset: {:.1}px ({:+.1}%)",
                vs.lateral_offset,
                normalized * 100.0
            );
            imgproc::put_text(
                &mut output,
                &info,
                core::Point::new(200, 32),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                core::Scalar::new(255.0, 255.0, 255.0, 0.0),
                1,
                imgproc::LINE_8,
                false,
            )?;
        }
    }

    Ok(output)
}

// ══════════════════════════════════════════════════════════════════════════
// V2 PIPELINE VISUALIZATION - Maneuver Detection v2
// ══════════════════════════════════════════════════════════════════════════

/// Visualization for v2 maneuver detection pipeline
/// Visualization for v2 maneuver detection pipeline with vehicle tracking
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
) -> Result<Mat> {
    let mat = Mat::from_slice(frame)?;
    let mat = mat.reshape(3, height)?;
    let mut bgr_mat = Mat::default();
    imgproc::cvt_color(&mat, &mut bgr_mat, imgproc::COLOR_RGB2BGR, 0)?;
    let mut output = bgr_mat.try_clone()?;

    // ══════════════════════════════════════════════════════════════════════
    // 1. DRAW LANE MARKINGS
    // ══════════════════════════════════════════════════════════════════════
    let lane_colors = vec![
        core::Scalar::new(0.0, 0.0, 255.0, 0.0),   // Red - left
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),   // Green - right
        core::Scalar::new(255.0, 0.0, 0.0, 0.0),   // Blue
        core::Scalar::new(0.0, 255.0, 255.0, 0.0), // Yellow
    ];

    for (i, lane) in lanes.iter().enumerate() {
        let color = lane_colors[i % lane_colors.len()];
        let thickness = if i < 2 { 4 } else { 2 };

        for point in &lane.points {
            let pt = core::Point::new(point.0 as i32, point.1 as i32);
            imgproc::circle(&mut output, pt, thickness, color, -1, imgproc::LINE_8, 0)?;
        }

        if lane.points.len() >= 2 {
            for j in 0..lane.points.len() - 1 {
                let pt1 = core::Point::new(lane.points[j].0 as i32, lane.points[j].1 as i32);
                let pt2 =
                    core::Point::new(lane.points[j + 1].0 as i32, lane.points[j + 1].1 as i32);
                imgproc::line(&mut output, pt1, pt2, color, 2, imgproc::LINE_AA, 0)?;
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 2. DRAW TRACKED VEHICLES WITH IDs
    // ══════════════════════════════════════════════════════════════════════
    for track in tracked_vehicles {
        if !track.is_confirmed() {
            continue;
        }

        let bbox = &track.bbox;

        // Color based on zone
        let box_color = match track.zone {
            crate::analysis::vehicle_tracker::VehicleZone::Ahead => {
                core::Scalar::new(255.0, 255.0, 0.0, 0.0) // Cyan - ahead
            }
            crate::analysis::vehicle_tracker::VehicleZone::BesideLeft => {
                core::Scalar::new(0.0, 165.0, 255.0, 0.0) // Orange - beside left
            }
            crate::analysis::vehicle_tracker::VehicleZone::BesideRight => {
                core::Scalar::new(255.0, 0.0, 255.0, 0.0) // Magenta - beside right
            }
            crate::analysis::vehicle_tracker::VehicleZone::Behind => {
                core::Scalar::new(0.0, 255.0, 0.0, 0.0) // Green - behind (passed)
            }
            crate::analysis::vehicle_tracker::VehicleZone::Unknown => {
                core::Scalar::new(128.0, 128.0, 128.0, 0.0) // Gray - unknown
            }
        };

        // Draw bounding box
        let pt1 = core::Point::new(bbox[0] as i32, bbox[1] as i32);
        let pt2 = core::Point::new(bbox[2] as i32, bbox[3] as i32);
        imgproc::rectangle(
            &mut output,
            core::Rect::from_points(pt1, pt2),
            box_color,
            3,
            imgproc::LINE_8,
            0,
        )?;

        // Label with ID, zone, and confidence
        let zone_str = match track.zone {
            crate::analysis::vehicle_tracker::VehicleZone::Ahead => "AHEAD",
            crate::analysis::vehicle_tracker::VehicleZone::BesideLeft => "BESIDE-L",
            crate::analysis::vehicle_tracker::VehicleZone::BesideRight => "BESIDE-R",
            crate::analysis::vehicle_tracker::VehicleZone::Behind => "BEHIND",
            crate::analysis::vehicle_tracker::VehicleZone::Unknown => "UNKNOWN",
        };

        let label = format!(
            "ID:{} {} ({:.0}%)",
            track.id,
            zone_str,
            track.last_confidence * 100.0
        );

        let label_pos = core::Point::new(bbox[0] as i32, (bbox[1] as i32) - 8);

        // Background for label
        let label_size =
            imgproc::get_text_size(&label, imgproc::FONT_HERSHEY_SIMPLEX, 0.6, 2, &mut 0)?;

        let label_bg_pt1 = core::Point::new(label_pos.x - 2, label_pos.y - label_size.height - 4);
        let label_bg_pt2 = core::Point::new(label_pos.x + label_size.width + 4, label_pos.y + 4);

        imgproc::rectangle(
            &mut output,
            core::Rect::from_points(label_bg_pt1, label_bg_pt2),
            core::Scalar::new(0.0, 0.0, 0.0, 0.0),
            -1,
            imgproc::LINE_8,
            0,
        )?;

        // Label text
        imgproc::put_text(
            &mut output,
            &label,
            label_pos,
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.6,
            core::Scalar::new(255.0, 255.0, 255.0, 0.0),
            2,
            imgproc::LINE_8,
            false,
        )?;

        // Draw track age indicator
        let age_text = format!("Age:{}", track.age);
        let age_pos = core::Point::new(bbox[0] as i32, (bbox[2] as i32) + 20);

        draw_text_with_shadow(
            &mut output,
            &age_text,
            age_pos.x,
            age_pos.y,
            0.4,
            core::Scalar::new(200.0, 200.0, 200.0, 0.0),
            1,
        )?;
    }

    // ══════════════════════════════════════════════════════════════════════
    // 3. DRAW EGO VEHICLE MARKER
    // ══════════════════════════════════════════════════════════════════════
    let vehicle_x = width / 2;
    let vehicle_y = (height as f32 * 0.85) as i32;
    imgproc::circle(
        &mut output,
        core::Point::new(vehicle_x, vehicle_y),
        12,
        core::Scalar::new(0.0, 255.0, 255.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;

    // ══════════════════════════════════════════════════════════════════════
    // 4. LEGALITY BANNER (if illegal crossing detected)
    // ══════════════════════════════════════════════════════════════════════
    let mut banner_active = false;

    if let Some(legality) = legality_result {
        if legality.ego_intersects_marking && legality.verdict.is_illegal() {
            let banner_h = 50;
            let color = match legality.verdict {
                crate::lane_legality::LineLegality::CriticalIllegal => {
                    core::Scalar::new(0.0, 0.0, 255.0, 0.0)
                }
                crate::lane_legality::LineLegality::Illegal => {
                    core::Scalar::new(0.0, 50.0, 200.0, 0.0)
                }
                _ => core::Scalar::new(0.0, 200.0, 255.0, 0.0),
            };

            imgproc::rectangle(
                &mut output,
                core::Rect::new(0, 0, width, banner_h),
                color,
                -1,
                imgproc::LINE_8,
                0,
            )?;

            let line_name = legality
                .intersecting_line
                .as_ref()
                .map(|l| l.class_name.as_str())
                .unwrap_or("SOLID LINE");

            let text = format!(
                "! ILLEGAL CROSSING: {} - VIOLATION",
                line_name.to_uppercase()
            );

            draw_text_with_shadow(
                &mut output,
                &text,
                20,
                35,
                0.8,
                core::Scalar::new(255.0, 255.0, 255.0, 0.0),
                2,
            )?;

            banner_active = true;
        }

        // Draw detected road markings
        for marking in &legality.all_markings {
            use crate::lane_legality::LineLegality;

            let box_color = match marking.legality {
                LineLegality::CriticalIllegal => core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                LineLegality::Illegal => core::Scalar::new(0.0, 100.0, 255.0, 0.0),
                LineLegality::Legal => core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                _ => core::Scalar::new(200.0, 200.0, 200.0, 0.0),
            };

            let pt1 = core::Point::new(marking.bbox[0] as i32, marking.bbox[1] as i32);
            let pt2 = core::Point::new(marking.bbox[2] as i32, marking.bbox[3] as i32);
            imgproc::rectangle(
                &mut output,
                core::Rect::from_points(pt1, pt2),
                box_color,
                2,
                imgproc::LINE_8,
                0,
            )?;
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 5. LEFT PANEL: V2 PIPELINE STATUS
    // ══════════════════════════════════════════════════════════════════════
    let panel_x = 15;
    let mut panel_y = if banner_active { 70 } else { 30 };
    let line_height = 26;

    draw_panel_background(&mut output, 5, panel_y - 10, 480, 300)?;

    // Title
    draw_text_with_shadow(
        &mut output,
        "MANEUVER DETECTION v2",
        panel_x,
        panel_y,
        0.7,
        core::Scalar::new(100.0, 200.0, 255.0, 0.0),
        2,
    )?;
    panel_y += line_height;

    // Frame info
    draw_text_with_shadow(
        &mut output,
        &format!("Frame: {} | Time: {:.2}s", frame_id, timestamp_ms / 1000.0),
        panel_x,
        panel_y,
        0.5,
        core::Scalar::new(200.0, 200.0, 200.0, 0.0),
        1,
    )?;
    panel_y += line_height;

    // Tracked vehicles with zone breakdown
    let mut zone_counts = std::collections::HashMap::new();
    for track in tracked_vehicles {
        if track.is_confirmed() {
            *zone_counts.entry(format!("{:?}", track.zone)).or_insert(0) += 1;
        }
    }

    draw_text_with_shadow(
        &mut output,
        &format!("Tracked Vehicles: {}", tracked_vehicles.len()),
        panel_x,
        panel_y,
        0.55,
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        1,
    )?;
    panel_y += line_height;
    // Show zone breakdown if vehicles present
    if !tracked_vehicles.is_empty() {
        for (zone, count) in &zone_counts {
            let zone_display = zone.replace("VehicleZone::", "");
            draw_text_with_shadow(
                &mut output,
                &format!("  • {}: {}", zone_display, count),
                panel_x + 10,
                panel_y,
                0.45,
                core::Scalar::new(180.0, 180.0, 180.0, 0.0),
                1,
            )?;
            panel_y += 20;
        }
    } else {
        draw_text_with_shadow(
            &mut output,
            "  (no vehicles tracked)",
            panel_x + 10,
            panel_y,
            0.45,
            core::Scalar::new(150.0, 150.0, 150.0, 0.0),
            1,
        )?;
        panel_y += 20;
    }

    // Lateral state
    let state_color = match lateral_state {
        s if s.contains("CENTERED") || s.contains("STABLE") => {
            core::Scalar::new(0.0, 255.0, 0.0, 0.0)
        }
        s if s.contains("SHIFT") => core::Scalar::new(0.0, 165.0, 255.0, 0.0),
        s if s.contains("RECOVERING") => core::Scalar::new(0.0, 255.0, 255.0, 0.0),
        _ => core::Scalar::new(255.0, 255.0, 255.0, 0.0),
    };

    draw_text_with_shadow(
        &mut output,
        &format!("Lateral State: {}", lateral_state),
        panel_x,
        panel_y,
        0.6,
        state_color,
        1,
    )?;
    panel_y += line_height;

    // Ego lateral velocity
    let vel_color = if ego_lateral_velocity.abs() > 3.0 {
        core::Scalar::new(0.0, 0.0, 255.0, 0.0) // Red - high
    } else if ego_lateral_velocity.abs() > 1.5 {
        core::Scalar::new(0.0, 165.0, 255.0, 0.0) // Orange - medium
    } else {
        core::Scalar::new(255.0, 255.0, 255.0, 0.0) // White - low
    };

    draw_text_with_shadow(
        &mut output,
        &format!("Ego Lateral: {:.2} px/frame", ego_lateral_velocity),
        panel_x,
        panel_y,
        0.55,
        vel_color,
        1,
    )?;
    panel_y += line_height;

    // Vehicle position
    if let Some(vs) = vehicle_state {
        if vs.is_valid() {
            let normalized = vs.normalized_offset().unwrap_or(0.0);
            draw_text_with_shadow(
                &mut output,
                &format!(
                    "Lateral Offset: {:.1}px ({:+.1}%)",
                    vs.lateral_offset,
                    normalized * 100.0
                ),
                panel_x,
                panel_y,
                0.55,
                core::Scalar::new(255.0, 255.0, 255.0, 0.0),
                1,
            )?;
            panel_y += line_height;

            if let Some(lw) = vs.lane_width {
                draw_text_with_shadow(
                    &mut output,
                    &format!("Lane Width: {:.0}px", lw),
                    panel_x,
                    panel_y,
                    0.55,
                    core::Scalar::new(200.0, 200.0, 200.0, 0.0),
                    1,
                )?;
                panel_y += line_height;
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 6. RIGHT PANEL: DETECTED MANEUVERS
    // ══════════════════════════════════════════════════════════════════════
    let right_panel_x = width - 480;
    let mut right_panel_y = if banner_active { 70 } else { 30 };

    draw_panel_background(
        &mut output,
        right_panel_x - 10,
        right_panel_y - 10,
        470,
        280,
    )?;

    // Title
    draw_text_with_shadow(
        &mut output,
        "DETECTED MANEUVERS",
        right_panel_x,
        right_panel_y,
        0.7,
        core::Scalar::new(100.0, 255.0, 100.0, 0.0),
        2,
    )?;
    right_panel_y += line_height;

    if maneuver_events.is_empty() {
        draw_text_with_shadow(
            &mut output,
            "No maneuvers detected this frame",
            right_panel_x,
            right_panel_y,
            0.55,
            core::Scalar::new(200.0, 200.0, 200.0, 0.0),
            1,
        )?;
    } else {
        for event in maneuver_events.iter().take(5) {
            let maneuver_color = match event.maneuver_type.as_str() {
                "OVERTAKE" => core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                "LANE_CHANGE" => core::Scalar::new(0.0, 165.0, 255.0, 0.0),
                "BEING_OVERTAKEN" => core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                _ => core::Scalar::new(255.0, 255.0, 255.0, 0.0),
            };

            draw_text_with_shadow(
                &mut output,
                &format!(
                    ">> {} {} | conf={:.2}",
                    event.maneuver_type.as_str(),
                    event.side.as_str(),
                    event.confidence
                ),
                right_panel_x,
                right_panel_y,
                0.6,
                maneuver_color,
                2,
            )?;
            right_panel_y += line_height;

            // Sources
            draw_text_with_shadow(
                &mut output,
                &format!("  Sources: {}", event.sources.summary()),
                right_panel_x + 10,
                right_panel_y,
                0.5,
                core::Scalar::new(200.0, 200.0, 200.0, 0.0),
                1,
            )?;
            right_panel_y += 22;

            // Legality
            let legality_text = format!("  Legality: {:?}", event.legality);
            let legality_color = match event.legality {
                crate::lane_legality::LineLegality::Legal => {
                    core::Scalar::new(0.0, 255.0, 0.0, 0.0)
                }
                crate::lane_legality::LineLegality::Illegal => {
                    core::Scalar::new(0.0, 100.0, 255.0, 0.0)
                }
                crate::lane_legality::LineLegality::CriticalIllegal => {
                    core::Scalar::new(0.0, 0.0, 255.0, 0.0)
                }
                _ => core::Scalar::new(200.0, 200.0, 200.0, 0.0),
            };

            draw_text_with_shadow(
                &mut output,
                &legality_text,
                right_panel_x + 10,
                right_panel_y,
                0.5,
                legality_color,
                1,
            )?;
            right_panel_y += 22;

            // Duration
            draw_text_with_shadow(
                &mut output,
                &format!("  Duration: {:.1}s", event.duration_ms / 1000.0),
                right_panel_x + 10,
                right_panel_y,
                0.5,
                core::Scalar::new(200.0, 200.0, 200.0, 0.0),
                1,
            )?;
            right_panel_y += 24;
        }

        if maneuver_events.len() > 5 {
            draw_text_with_shadow(
                &mut output,
                &format!("... and {} more", maneuver_events.len() - 5),
                right_panel_x,
                right_panel_y,
                0.5,
                core::Scalar::new(200.0, 200.0, 200.0, 0.0),
                1,
            )?;
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 7. LEGEND (Bottom Right)
    // ══════════════════════════════════════════════════════════════════════
    let legend_y = height - 190;
    draw_panel_background(&mut output, width - 340, legend_y - 10, 330, 185)?;

    draw_text_with_shadow(
        &mut output,
        "VEHICLE TRACKING LEGEND",
        width - 330,
        legend_y,
        0.5,
        core::Scalar::new(200.0, 200.0, 200.0, 0.0),
        1,
    )?;

    let legend_items: Vec<(&str, core::Scalar)> = vec![
        (
            "[CYAN] Vehicle AHEAD",
            core::Scalar::new(255.0, 255.0, 0.0, 0.0),
        ),
        (
            "[ORANGE] Vehicle BESIDE-LEFT",
            core::Scalar::new(0.0, 165.0, 255.0, 0.0),
        ),
        (
            "[MAGENTA] Vehicle BESIDE-RIGHT",
            core::Scalar::new(255.0, 0.0, 255.0, 0.0),
        ),
        (
            "[GREEN] Vehicle BEHIND (passed)",
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        ),
        (
            "[GRAY] Vehicle UNKNOWN zone",
            core::Scalar::new(128.0, 128.0, 128.0, 0.0),
        ),
        (
            "[YELLOW] Ego vehicle marker",
            core::Scalar::new(0.0, 255.0, 255.0, 0.0),
        ),
        (
            "[RED/GREEN] Lane boundaries",
            core::Scalar::new(0.0, 200.0, 200.0, 0.0),
        ),
    ];

    for (i, (label, color)) in legend_items.iter().enumerate() {
        draw_text_with_shadow(
            &mut output,
            label,
            width - 325,
            legend_y + 20 + (i as i32 * 22),
            0.42,
            *color,
            1,
        )?;
    }

    Ok(output)
}
