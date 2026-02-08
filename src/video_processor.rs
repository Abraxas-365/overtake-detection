// src/video_processor.rs
use crate::overtake_analyzer::TrackedVehicle;
use crate::shadow_overtake::ShadowOvertakeDetector;
use std::collections::HashMap;

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
    // 2. DRAW TRACKED VEHICLES (YOLO)
    // ══════════════════════════════════════════════════════════════════════

    // Get shadow overtake vehicle IDs for highlighting
    let shadow_vehicle_ids: std::collections::HashSet<u32> = if shadow_detector.is_monitoring() {
        shadow_detector
            .get_current_shadows()
            .iter()
            .map(|s| s.blocking_vehicle_id)
            .collect()
    } else {
        std::collections::HashSet::new()
    };

    for (vehicle_id, vehicle) in tracked_vehicles {
        // Get latest position
        if let Some(latest_pos) = vehicle.position_history.last() {
            // Only draw if this position is recent (within last 10 frames)
            if frame_id.saturating_sub(latest_pos.frame_id) < 10 {
                let bbox = &vehicle.bbox;

                // Check if this vehicle is a shadow blocker
                let is_shadow_blocker = shadow_vehicle_ids.contains(vehicle_id);

                // Color: RED for shadow blockers, GREEN for normal vehicles
                let box_color = if is_shadow_blocker {
                    core::Scalar::new(0.0, 0.0, 255.0, 0.0) // RED - DANGER!
                } else {
                    core::Scalar::new(0.0, 255.0, 0.0, 0.0) // GREEN - normal
                };

                let thickness = if is_shadow_blocker { 3 } else { 2 };

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

                // Draw label with ID and class
                let label = if is_shadow_blocker {
                    format!("⚠ {} #{}", vehicle.class_name, vehicle_id)
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
                    core::Scalar::new(0.0, 0.0, 0.0, 0.0), // Black background
                    -1,
                    imgproc::LINE_8,
                    0,
                )?;

                // Draw text
                imgproc::put_text(
                    &mut output,
                    &label,
                    label_pos,
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    core::Scalar::new(255.0, 255.0, 255.0, 0.0), // White text
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
        10,
        core::Scalar::new(0.0, 255.0, 255.0, 0.0), // Yellow
        -1,
        imgproc::LINE_8,
        0,
    )?;

    // ══════════════════════════════════════════════════════════════════════
    // 4. STATUS OVERLAY (TOP LEFT)
    // ══════════════════════════════════════════════════════════════════════
    let mut y_offset = 32;
    let line_height = 28;

    // Lane change state
    let state_color = match state {
        "CENTERED" => core::Scalar::new(0.0, 255.0, 0.0, 0.0), // Green
        "DRIFTING" => core::Scalar::new(0.0, 165.0, 255.0, 0.0), // Orange
        "CROSSING" => core::Scalar::new(0.0, 0.0, 255.0, 0.0), // Red
        _ => core::Scalar::new(255.0, 255.0, 255.0, 0.0),      // White
    };

    imgproc::put_text(
        &mut output,
        &format!("State: {}", state),
        core::Point::new(15, y_offset),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.7,
        state_color,
        2,
        imgproc::LINE_8,
        false,
    )?;
    y_offset += line_height;

    // Vehicle offset info
    if let Some(vs) = vehicle_state {
        if vs.is_valid() {
            let normalized = vs.normalized_offset().unwrap_or(0.0);
            let offset_info = format!(
                "Offset: {:.1}px ({:+.1}%)",
                vs.lateral_offset,
                normalized * 100.0
            );
            imgproc::put_text(
                &mut output,
                &offset_info,
                core::Point::new(15, y_offset),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                core::Scalar::new(255.0, 255.0, 255.0, 0.0),
                1,
                imgproc::LINE_8,
                false,
            )?;
            y_offset += line_height;
        }
    }

    // Vehicle tracking info
    let active_vehicles = tracked_vehicles
        .values()
        .filter(|v| {
            v.position_history
                .last()
                .map(|p| frame_id.saturating_sub(p.frame_id) < 10)
                .unwrap_or(false)
        })
        .count();

    imgproc::put_text(
        &mut output,
        &format!("Tracked: {} vehicles", active_vehicles),
        core::Point::new(15, y_offset),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        core::Scalar::new(200.0, 200.0, 200.0, 0.0),
        1,
        imgproc::LINE_8,
        false,
    )?;
    y_offset += line_height;

    // ══════════════════════════════════════════════════════════════════════
    // 5. SHADOW OVERTAKE WARNING (if active)
    // ══════════════════════════════════════════════════════════════════════
    if shadow_detector.is_monitoring() && shadow_detector.active_shadow_count() > 0 {
        let warning_text = format!(
            "⚠ SHADOW OVERTAKE: {} vehicle(s) blocking!",
            shadow_detector.active_shadow_count()
        );

        // Draw warning banner at the top
        let banner_height = 50;
        imgproc::rectangle(
            &mut output,
            core::Rect::new(0, 0, width, banner_height),
            core::Scalar::new(0.0, 0.0, 200.0, 0.0), // Red background
            -1,
            imgproc::LINE_8,
            0,
        )?;

        imgproc::put_text(
            &mut output,
            &warning_text,
            core::Point::new(width / 2 - 250, 32),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.8,
            core::Scalar::new(255.0, 255.0, 255.0, 0.0), // White text
            2,
            imgproc::LINE_8,
            false,
        )?;

        // Severity indicator
        if let Some(severity) = shadow_detector.worst_active_severity() {
            let severity_text = format!("Severity: {}", severity.as_str());
            imgproc::put_text(
                &mut output,
                &severity_text,
                core::Point::new(15, y_offset),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.7,
                core::Scalar::new(0.0, 0.0, 255.0, 0.0), // Red
                2,
                imgproc::LINE_8,
                false,
            )?;
        }
    } else if shadow_detector.is_monitoring() {
        // Show that monitoring is active but no shadows detected
        imgproc::put_text(
            &mut output,
            "Shadow monitoring: ACTIVE",
            core::Point::new(15, y_offset),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.6,
            core::Scalar::new(0.0, 255.0, 255.0, 0.0), // Cyan
            1,
            imgproc::LINE_8,
            false,
        )?;
    }

    Ok(output)
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
