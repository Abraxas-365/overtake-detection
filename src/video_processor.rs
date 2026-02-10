// src/video_processor.rs
use crate::lane_legality::LegalityResult;
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
// VIDEO PROCESSOR
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

// ============================================================================
// VIDEO READER
// ============================================================================

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
// MANEUVER DETECTION V2 VISUALIZATION
// ══════════════════════════════════════════════════════════════════════════

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
    total_lane_changes: u64,
    total_vehicles_overtaken: u64,
    last_maneuver: Option<&crate::LastManeuverInfo>,
) -> Result<Mat> {
    let mat = Mat::from_slice(frame)?;
    let mat = mat.reshape(3, height)?;
    let mut bgr_mat = Mat::default();
    imgproc::cvt_color(&mat, &mut bgr_mat, imgproc::COLOR_RGB2BGR, 0)?;
    let mut output = bgr_mat.try_clone()?;

    // ══════════════════════════════════════════════════════════════════════
    // 1. DRAW LANE MARKINGS WITH TYPE LABELS
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

            // Label lanes
            if let Some(first_point) = lane.points.first() {
                let lane_label = if i == 0 {
                    "LEFT LANE"
                } else if i == 1 {
                    "RIGHT LANE"
                } else {
                    &format!("LANE {}", i + 1)
                };

                draw_text_with_shadow(
                    &mut output,
                    lane_label,
                    first_point.0 as i32 + 10,
                    first_point.1 as i32,
                    0.4,
                    color,
                    1,
                )?;
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 2. DRAW TRACKED VEHICLES WITH CLASS NAMES
    // ══════════════════════════════════════════════════════════════════════
    for track in tracked_vehicles {
        if !track.is_confirmed() {
            continue;
        }

        let bbox = &track.bbox;

        // Get vehicle class name from detections
        let class_name = vehicle_detections
            .iter()
            .find(|d| {
                let d_center_x = (d.bbox[0] + d.bbox[2]) / 2.0;
                let t_center_x = (bbox[0] + bbox[2]) / 2.0;
                (d_center_x - t_center_x).abs() < 50.0
            })
            .map(|d| d.class_name.as_str())
            .unwrap_or("vehicle");

        // Color based on zone
        let box_color = match track.zone {
            crate::analysis::vehicle_tracker::VehicleZone::Ahead => {
                core::Scalar::new(255.0, 255.0, 0.0, 0.0) // Cyan
            }
            crate::analysis::vehicle_tracker::VehicleZone::BesideLeft => {
                core::Scalar::new(0.0, 165.0, 255.0, 0.0) // Orange
            }
            crate::analysis::vehicle_tracker::VehicleZone::BesideRight => {
                core::Scalar::new(255.0, 0.0, 255.0, 0.0) // Magenta
            }
            crate::analysis::vehicle_tracker::VehicleZone::Behind => {
                core::Scalar::new(0.0, 255.0, 0.0, 0.0) // Green
            }
            crate::analysis::vehicle_tracker::VehicleZone::Unknown => {
                core::Scalar::new(128.0, 128.0, 128.0, 0.0) // Gray
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

        // Zone string
        let zone_str = match track.zone {
            crate::analysis::vehicle_tracker::VehicleZone::Ahead => "AHEAD",
            crate::analysis::vehicle_tracker::VehicleZone::BesideLeft => "BESIDE-L",
            crate::analysis::vehicle_tracker::VehicleZone::BesideRight => "BESIDE-R",
            crate::analysis::vehicle_tracker::VehicleZone::Behind => "BEHIND",
            crate::analysis::vehicle_tracker::VehicleZone::Unknown => "UNKNOWN",
        };

        // Label with class name + zone
        let label = format!(
            "{} ID:{} {} ({:.0}%)",
            class_name.to_uppercase(),
            track.id,
            zone_str,
            track.last_confidence * 100.0
        );

        let label_pos = core::Point::new(bbox[0] as i32, (bbox[1] as i32) - 8);

        // Background for label
        let label_size =
            imgproc::get_text_size(&label, imgproc::FONT_HERSHEY_SIMPLEX, 0.5, 2, &mut 0)?;

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
            0.5,
            core::Scalar::new(255.0, 255.0, 255.0, 0.0),
            2,
            imgproc::LINE_8,
            false,
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
    // 4. LEGALITY BANNER
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

        // Draw detected road markings with names
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

            // Label the marking type
            draw_text_with_shadow(
                &mut output,
                &marking.class_name,
                marking.bbox[0] as i32,
                marking.bbox[1] as i32 - 5,
                0.4,
                box_color,
                1,
            )?;
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 5. LEFT PANEL: DETECTION DETAILS
    // ══════════════════════════════════════════════════════════════════════
    let panel_x = 15;
    let mut panel_y = if banner_active { 70 } else { 30 };
    let line_height = 26;

    draw_panel_background(&mut output, 5, panel_y - 10, 500, 360)?;

    // Title
    draw_text_with_shadow(
        &mut output,
        "AI DETECTION STATUS",
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

    // Lane detection status
    let lane_status = if lanes.len() >= 2 {
        format!("✓ {} lanes detected (L+R)", lanes.len())
    } else if lanes.len() == 1 {
        "⚠ 1 lane detected (estimated)".to_string()
    } else {
        "✗ No lanes detected".to_string()
    };

    let lane_color = if lanes.len() >= 2 {
        core::Scalar::new(0.0, 255.0, 0.0, 0.0)
    } else {
        core::Scalar::new(0.0, 165.0, 255.0, 0.0)
    };

    draw_text_with_shadow(
        &mut output,
        &lane_status,
        panel_x,
        panel_y,
        0.55,
        lane_color,
        1,
    )?;
    panel_y += line_height;

    // Road marking types if detected
    if let Some(legality) = legality_result {
        if !legality.all_markings.is_empty() {
            let frame_center_x = width as f32 / 2.0;

            let left_markings: Vec<String> = legality
                .all_markings
                .iter()
                .filter(|m| (m.bbox[0] + m.bbox[2]) / 2.0 < frame_center_x)
                .map(|m| m.class_name.clone())
                .collect();

            let right_markings: Vec<String> = legality
                .all_markings
                .iter()
                .filter(|m| (m.bbox[0] + m.bbox[2]) / 2.0 >= frame_center_x)
                .map(|m| m.class_name.clone())
                .collect();

            if !left_markings.is_empty() {
                draw_text_with_shadow(
                    &mut output,
                    &format!("  L: {}", left_markings.join(", ")),
                    panel_x + 10,
                    panel_y,
                    0.45,
                    core::Scalar::new(180.0, 180.0, 180.0, 0.0),
                    1,
                )?;
                panel_y += 20;
            }

            if !right_markings.is_empty() {
                draw_text_with_shadow(
                    &mut output,
                    &format!("  R: {}", right_markings.join(", ")),
                    panel_x + 10,
                    panel_y,
                    0.45,
                    core::Scalar::new(180.0, 180.0, 180.0, 0.0),
                    1,
                )?;
                panel_y += 20;
            }
        }
    }

    // Vehicle tracking with classes
    let mut vehicle_classes = std::collections::HashMap::new();
    for track in tracked_vehicles {
        if track.is_confirmed() {
            let class_name = vehicle_detections
                .iter()
                .find(|d| {
                    let d_center_x = (d.bbox[0] + d.bbox[2]) / 2.0;
                    let t_center_x = (track.bbox[0] + track.bbox[2]) / 2.0;
                    (d_center_x - t_center_x).abs() < 50.0
                })
                .map(|d| d.class_name.as_str())
                .unwrap_or("vehicle");

            *vehicle_classes.entry(class_name).or_insert(0) += 1;
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

    // Show vehicle type breakdown
    if !vehicle_classes.is_empty() {
        for (class_name, count) in &vehicle_classes {
            draw_text_with_shadow(
                &mut output,
                &format!("  • {}: {}", class_name, count),
                panel_x + 10,
                panel_y,
                0.45,
                core::Scalar::new(180.0, 180.0, 180.0, 0.0),
                1,
            )?;
            panel_y += 20;
        }
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
        core::Scalar::new(0.0, 0.0, 255.0, 0.0)
    } else if ego_lateral_velocity.abs() > 1.5 {
        core::Scalar::new(0.0, 165.0, 255.0, 0.0)
    } else {
        core::Scalar::new(255.0, 255.0, 255.0, 0.0)
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
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 6. RIGHT PANEL: PERSISTENT MANEUVER + TOTALS
    // ══════════════════════════════════════════════════════════════════════
    let right_panel_x = width - 520;
    let mut right_panel_y = if banner_active { 70 } else { 30 };

    draw_panel_background(
        &mut output,
        right_panel_x - 10,
        right_panel_y - 10,
        510,
        400,
    )?;

    // Title
    draw_text_with_shadow(
        &mut output,
        "MANEUVER DETECTION",
        right_panel_x,
        right_panel_y,
        0.7,
        core::Scalar::new(100.0, 255.0, 100.0, 0.0),
        2,
    )?;
    right_panel_y += line_height;

    // SESSION TOTALS
    draw_text_with_shadow(
        &mut output,
        "═══ SESSION TOTALS ═══",
        right_panel_x,
        right_panel_y,
        0.6,
        core::Scalar::new(255.0, 200.0, 100.0, 0.0),
        1,
    )?;
    right_panel_y += line_height;

    draw_text_with_shadow(
        &mut output,
        &format!(
            "🚗 Overtakes: {} ({} vehicles)",
            total_overtakes, total_vehicles_overtaken
        ),
        right_panel_x,
        right_panel_y,
        0.6,
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
    )?;
    right_panel_y += line_height;

    draw_text_with_shadow(
        &mut output,
        &format!("🔀 Lane Changes: {}", total_lane_changes),
        right_panel_x,
        right_panel_y,
        0.6,
        core::Scalar::new(0.0, 165.0, 255.0, 0.0),
        1,
    )?;
    right_panel_y += line_height + 10;

    // LAST MANEUVER (PERSISTENT)
    if let Some(maneuver) = last_maneuver {
        let seconds_ago = (timestamp_ms - maneuver.timestamp_detected) / 1000.0;

        draw_text_with_shadow(
            &mut output,
            "═══ LAST MANEUVER ═══",
            right_panel_x,
            right_panel_y,
            0.6,
            core::Scalar::new(255.0, 200.0, 100.0, 0.0),
            1,
        )?;
        right_panel_y += line_height;

        let maneuver_color = match maneuver.maneuver_type.as_str() {
            "OVERTAKE" => core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            "LANE_CHANGE" => core::Scalar::new(0.0, 165.0, 255.0, 0.0),
            "BEING_OVERTAKEN" => core::Scalar::new(0.0, 0.0, 255.0, 0.0),
            _ => core::Scalar::new(255.0, 255.0, 255.0, 0.0),
        };

        draw_text_with_shadow(
            &mut output,
            &format!(
                ">> {} {} | conf={:.2}",
                maneuver.maneuver_type, maneuver.side, maneuver.confidence
            ),
            right_panel_x,
            right_panel_y,
            0.65,
            maneuver_color,
            2,
        )?;
        right_panel_y += line_height;

        draw_text_with_shadow(
            &mut output,
            &format!("  Detected: {:.1}s ago", seconds_ago),
            right_panel_x + 10,
            right_panel_y,
            0.5,
            core::Scalar::new(180.0, 180.0, 180.0, 0.0),
            1,
        )?;
        right_panel_y += 22;

        draw_text_with_shadow(
            &mut output,
            &format!("  Sources: {}", maneuver.sources),
            right_panel_x + 10,
            right_panel_y,
            0.5,
            core::Scalar::new(200.0, 200.0, 200.0, 0.0),
            1,
        )?;
        right_panel_y += 22;

        let legality_color = if maneuver.legality.contains("CriticalIllegal") {
            core::Scalar::new(0.0, 0.0, 255.0, 0.0)
        } else if maneuver.legality.contains("Illegal") {
            core::Scalar::new(0.0, 100.0, 255.0, 0.0)
        } else if maneuver.legality.contains("Legal") {
            core::Scalar::new(0.0, 255.0, 0.0, 0.0)
        } else {
            core::Scalar::new(200.0, 200.0, 200.0, 0.0)
        };

        draw_text_with_shadow(
            &mut output,
            &format!("  Legality: {}", maneuver.legality),
            right_panel_x + 10,
            right_panel_y,
            0.5,
            legality_color,
            1,
        )?;
        right_panel_y += 22;

        draw_text_with_shadow(
            &mut output,
            &format!("  Duration: {:.1}s", maneuver.duration_ms / 1000.0),
            right_panel_x + 10,
            right_panel_y,
            0.5,
            core::Scalar::new(200.0, 200.0, 200.0, 0.0),
            1,
        )?;
        right_panel_y += 22;

        // Show vehicle count for this specific maneuver
        if maneuver.maneuver_type == "OVERTAKE" {
            draw_text_with_shadow(
                &mut output,
                &format!(
                    "  🚗 Vehicles in maneuver: {}",
                    maneuver.vehicles_in_this_maneuver
                ),
                right_panel_x + 10,
                right_panel_y,
                0.55,
                core::Scalar::new(0.0, 255.0, 100.0, 0.0),
                2,
            )?;
            right_panel_y += 22;
        }
    } else {
        draw_text_with_shadow(
            &mut output,
            "No maneuvers detected yet",
            right_panel_x,
            right_panel_y,
            0.55,
            core::Scalar::new(150.0, 150.0, 150.0, 0.0),
            1,
        )?;
        right_panel_y += line_height;
    }

    // NEW THIS FRAME indicator
    if !maneuver_events.is_empty() {
        right_panel_y += 15;

        draw_text_with_shadow(
            &mut output,
            "🆕 NEW THIS FRAME!",
            right_panel_x,
            right_panel_y,
            0.65,
            core::Scalar::new(255.0, 255.0, 0.0, 0.0),
            2,
        )?;
        right_panel_y += 22;

        for event in maneuver_events.iter().take(2) {
            draw_text_with_shadow(
                &mut output,
                &format!(
                    "  • {} {}",
                    event.maneuver_type.as_str(),
                    event.side.as_str()
                ),
                right_panel_x + 10,
                right_panel_y,
                0.5,
                core::Scalar::new(200.0, 255.0, 200.0, 0.0),
                1,
            )?;
            right_panel_y += 20;
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 7. LEGEND
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

// ══════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
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
