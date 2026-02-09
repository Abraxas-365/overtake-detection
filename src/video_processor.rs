// src/video_processor.rs
use crate::lane_legality::LegalityResult; // 🆕 NEW IMPORT
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

        // Use imgcodecs::imwrite
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

/// Ultra-rich visualization with complete telemetry overlay + LEGALITY
// src/video_processor.rs

// ... existing imports ...

/// Ultra-rich visualization with complete telemetry overlay + LEGALITY + SOURCE
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
    detection_source: &str, // 🆕 ADDED: Source parameter
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
        core::Scalar::new(0.0, 0.0, 255.0, 0.0),   // Red
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),   // Green
        core::Scalar::new(255.0, 0.0, 0.0, 0.0),   // Blue
        core::Scalar::new(0.0, 255.0, 255.0, 0.0), // Yellow
    ];

    for (i, lane) in lanes.iter().enumerate() {
        let color = lane_colors[i % lane_colors.len()];
        for point in &lane.points {
            let pt = core::Point::new(point.0 as i32, point.1 as i32);
            imgproc::circle(&mut output, pt, 3, color, -1, imgproc::LINE_8, 0)?;
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

                // Color coding:
                // RED = shadow blocker (danger!)
                // CYAN = overtaken vehicle
                // GREEN = normal tracked vehicle
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

                // Label with status icons
                let label = if is_shadow_blocker {
                    format!("⚠ BLOCKING {} #{}", vehicle.class_name, vehicle_id)
                } else if was_overtaken {
                    format!("✓ {} #{}", vehicle.class_name, vehicle_id)
                } else {
                    format!("{} #{}", vehicle.class_name, vehicle_id)
                };

                let label_pos = core::Point::new(bbox[0] as i32, (bbox[1] as i32) - 5);

                // Background for label
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
    let mut banner_color = core::Scalar::new(0.0, 0.0, 200.0, 0.0); // Default red

    if shadow_detector.is_monitoring() && shadow_detector.active_shadow_count() > 0 {
        banner_active = true;
        banner_text = format!(
            "⚠ SHADOW OVERTAKE: {} vehicle(s) blocking visibility! DANGER!",
            shadow_detector.active_shadow_count()
        );
        banner_color = core::Scalar::new(0.0, 0.0, 255.0, 0.0); // Bright red
    } else if let Some(curve) = curve_info {
        if curve.is_curve && is_overtaking {
            banner_active = true;
            banner_text = format!(
                "⚠ OVERTAKING IN {} CURVE ({:.1}°) - ILLEGAL!",
                match curve.curve_type {
                    crate::types::CurveType::Sharp => "SHARP",
                    crate::types::CurveType::Moderate => "MODERATE",
                    _ => "CURVE",
                },
                curve.angle_degrees
            );
            banner_color = core::Scalar::new(0.0, 100.0, 255.0, 0.0); // Orange-red
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
                } // Bright red
                crate::lane_legality::LineLegality::Illegal => {
                    core::Scalar::new(0.0, 50.0, 200.0, 0.0)
                } // Dark red
                _ => core::Scalar::new(0.0, 200.0, 255.0, 0.0), // Yellow
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
                "⚠ ILLEGAL CROSSING: {} — VIOLATION DS 016-2009-MTC",
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

            banner_active = true; // Mark banner as active for panel positioning
        }

        // Draw detected road markings with color-coded bboxes
        for marking in &legality.all_markings {
            use crate::lane_legality::LineLegality;

            let box_color = match marking.legality {
                LineLegality::CriticalIllegal => core::Scalar::new(0.0, 0.0, 255.0, 0.0), // RED
                LineLegality::Illegal => core::Scalar::new(0.0, 100.0, 255.0, 0.0), // ORANGE-RED
                LineLegality::Legal => core::Scalar::new(0.0, 255.0, 0.0, 0.0),     // GREEN
                _ => core::Scalar::new(200.0, 200.0, 200.0, 0.0),                   // GRAY
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
    // 5. LEFT PANEL: VEHICLE STATE & POSITIONING
    // ══════════════════════════════════════════════════════════════════════
    let panel_x = 15;
    let mut panel_y = if banner_active { 80 } else { 30 };
    let line_height = 26;

    // Semi-transparent background
    draw_panel_background(&mut output, 5, panel_y - 10, 450, 310)?;

    // Title
    draw_text_with_shadow(
        &mut output,
        "VEHICLE STATUS",
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

    // 🆕 DETECTION SOURCE (YOLO vs UFLD)
    let source_color = if detection_source.contains("YOLO") {
        core::Scalar::new(0.0, 255.0, 255.0, 0.0) // Yellow for YOLO
    } else {
        core::Scalar::new(200.0, 200.0, 200.0, 0.0) // Gray for UFLD/Fallback
    };

    draw_text_with_shadow(
        &mut output,
        &format!("Source: {}", detection_source),
        panel_x,
        panel_y,
        0.6,
        source_color,
        2,
    )?;
    panel_y += line_height;

    // Lane change state with color coding
    let state_color = match state {
        "CENTERED" => core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        "DRIFTING" => core::Scalar::new(0.0, 165.0, 255.0, 0.0),
        "CROSSING" => core::Scalar::new(0.0, 0.0, 255.0, 0.0),
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

            draw_text_with_shadow(
                &mut output,
                &format!("Lateral Velocity: {:.1} px/s", lateral_velocity),
                panel_x,
                panel_y,
                0.55,
                if lateral_velocity.abs() > 100.0 {
                    core::Scalar::new(0.0, 165.0, 255.0, 0.0) // Orange for high velocity
                } else {
                    core::Scalar::new(255.0, 255.0, 255.0, 0.0)
                },
                1,
            )?;
            panel_y += line_height;

            if let Some(width) = vs.lane_width {
                draw_text_with_shadow(
                    &mut output,
                    &format!("Lane Width: {:.0}px", width),
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

    // Lane detection quality
    draw_text_with_shadow(
        &mut output,
        &format!("Lanes Detected: {}", lanes.len()),
        panel_x,
        panel_y,
        0.55,
        core::Scalar::new(200.0, 200.0, 200.0, 0.0),
        1,
    )?;
    panel_y += line_height;

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
        &format!("Vehicles Tracked: {}", active_vehicles),
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
                &format!("🚀 OVERTAKING {} ➜", direction),
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
                "✓ Overtaken: {} vehicle(s)",
                vehicles_overtaken_this_maneuver.len()
            ),
            right_panel_x,
            right_panel_y,
            0.6,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            1,
        )?;
        right_panel_y += line_height;

        // List overtaken vehicles
        for (i, overtaken) in vehicles_overtaken_this_maneuver.iter().take(5).enumerate() {
            draw_text_with_shadow(
                &mut output,
                &format!(
                    "  • {} (ID #{})",
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

    right_panel_y += 10; // Spacing

    // Shadow overtake status
    if shadow_detector.is_monitoring() {
        let shadow_count = shadow_detector.active_shadow_count();

        if shadow_count > 0 {
            draw_text_with_shadow(
                &mut output,
                &format!("⚠ SHADOW: {} vehicle(s)", shadow_count),
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

            // List shadow blockers
            for shadow in shadow_detector.get_current_shadows().iter().take(3) {
                draw_text_with_shadow(
                    &mut output,
                    &format!(
                        "  • {} ID #{} blocking",
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
                    "🌀 Curve: {} ({:.1}°)",
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
    let legend_y = height - 140; // Extended to fit legality colors
    draw_panel_background(&mut output, width - 250, legend_y - 10, 240, 130)?;

    draw_text_with_shadow(
        &mut output,
        "LEGEND",
        width - 240,
        legend_y,
        0.5,
        core::Scalar::new(200.0, 200.0, 200.0, 0.0),
        1,
    )?;

    let legend_items = vec![
        (
            "🟢 Green box:",
            "Tracked vehicle",
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        ),
        (
            "🔵 Cyan box:",
            "Overtaken vehicle",
            core::Scalar::new(255.0, 255.0, 0.0, 0.0),
        ),
        (
            "🔴 Red box:",
            "Shadow blocker!",
            core::Scalar::new(0.0, 0.0, 255.0, 0.0),
        ),
        (
            "🟡 Yellow:",
            "Your vehicle",
            core::Scalar::new(0.0, 255.0, 255.0, 0.0),
        ),
        (
            "🟢 Green line:",
            "LEGAL crossing",
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        ),
        (
            "🔴 Red line:",
            "ILLEGAL crossing",
            core::Scalar::new(0.0, 0.0, 255.0, 0.0),
        ),
    ];

    for (i, (label, desc, color)) in legend_items.iter().enumerate() {
        draw_text_with_shadow(
            &mut output,
            &format!("{} {}", label, desc),
            width - 235,
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
    // Create overlay on a clone to avoid borrow conflict
    let mut overlay = img.clone();
    imgproc::rectangle(
        &mut overlay,
        core::Rect::new(x, y, width, height),
        core::Scalar::new(0.0, 0.0, 0.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;

    // Blend with original (70% overlay, 30% original)
    let mut result = Mat::default();
    core::add_weighted(&overlay, 0.7, img, 0.3, 0.0, &mut result, -1)?;

    // Copy result back to img
    result.copy_to(img)?;

    // Border
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
