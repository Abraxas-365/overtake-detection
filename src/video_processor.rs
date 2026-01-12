// src/video_processor.rs

use crate::types::{Config, Lane, VehicleState};
use anyhow::Result;
use opencv::{
    core::{self, Mat},
    imgproc,
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
        let timestamp = (self.current_frame as f64) / self.fps;

        let mut rgb_mat = Mat::default();
        imgproc::cvt_color(&mat, &mut rgb_mat, imgproc::COLOR_BGR2RGB, 0)?;

        let data = rgb_mat.data_bytes()?.to_vec();

        Ok(Some(crate::types::Frame {
            data,
            width: self.width as usize,
            height: self.height as usize,
            timestamp,
        }))
    }

    pub fn progress(&self) -> f32 {
        if self.total_frames == 0 {
            return 0.0;
        }
        (self.current_frame as f32 / self.total_frames as f32) * 100.0
    }
}

/// Draw lanes on frame (simple version)
pub fn draw_lanes(frame: &[u8], _width: i32, height: i32, lanes: &[Lane]) -> Result<Mat> {
    let mat = Mat::from_slice(frame)?;
    let mat = mat.reshape(3, height)?;

    let mut bgr_mat = Mat::default();
    imgproc::cvt_color(&mat, &mut bgr_mat, imgproc::COLOR_RGB2BGR, 0)?;

    let mut output = bgr_mat.try_clone()?;

    let colors = vec![
        core::Scalar::new(0.0, 0.0, 255.0, 0.0),   // Red
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),   // Green
        core::Scalar::new(255.0, 0.0, 0.0, 0.0),   // Blue
        core::Scalar::new(0.0, 255.0, 255.0, 0.0), // Yellow
    ];

    for (i, lane) in lanes.iter().enumerate() {
        let color = colors[i % colors.len()];

        for point in &lane.points {
            let pt = core::Point::new(point.0 as i32, point.1 as i32);
            imgproc::circle(&mut output, pt, 3, color, -1, imgproc::LINE_8, 0)?;
        }

        for window in lane.points.windows(2) {
            let pt1 = core::Point::new(window[0].0 as i32, window[0].1 as i32);
            let pt2 = core::Point::new(window[1].0 as i32, window[1].1 as i32);
            imgproc::line(&mut output, pt1, pt2, color, 2, imgproc::LINE_AA, 0)?;
        }
    }

    Ok(output)
}

/// Draw lanes with Python-style state machine info overlay
pub fn draw_lanes_with_state(
    frame: &[u8],
    width: i32,
    height: i32,
    lanes: &[Lane],
    state: &str,
    vehicle_state: Option<&VehicleState>,
) -> Result<Mat> {
    let mat = Mat::from_slice(frame)?;
    let mat = mat.reshape(3, height)?;

    let mut bgr_mat = Mat::default();
    imgproc::cvt_color(&mat, &mut bgr_mat, imgproc::COLOR_RGB2BGR, 0)?;
    let mut output = bgr_mat.try_clone()?;

    // Lane colors
    let colors = vec![
        core::Scalar::new(0.0, 0.0, 255.0, 0.0),   // Red
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),   // Green
        core::Scalar::new(255.0, 0.0, 0.0, 0.0),   // Blue
        core::Scalar::new(0.0, 255.0, 255.0, 0.0), // Yellow
    ];

    // Draw lanes
    for (i, lane) in lanes.iter().enumerate() {
        let color = colors[i % colors.len()];

        // Draw lane points
        for point in &lane.points {
            let pt = core::Point::new(point.0 as i32, point.1 as i32);
            imgproc::circle(&mut output, pt, 3, color, -1, imgproc::LINE_8, 0)?;
        }

        // Draw lane lines
        for window in lane.points.windows(2) {
            let pt1 = core::Point::new(window[0].0 as i32, window[0].1 as i32);
            let pt2 = core::Point::new(window[1].0 as i32, window[1].1 as i32);
            imgproc::line(&mut output, pt1, pt2, color, 2, imgproc::LINE_AA, 0)?;
        }
    }

    // Draw vehicle position marker (center of frame at 85% height)
    let vehicle_x = width / 2;
    let vehicle_y = (height as f32 * 0.85) as i32;

    // Draw vertical center line (reference)
    imgproc::line(
        &mut output,
        core::Point::new(vehicle_x, height - 50),
        core::Point::new(vehicle_x, (height as f32 * 0.6) as i32),
        core::Scalar::new(128.0, 128.0, 128.0, 0.0), // Gray
        1,
        imgproc::LINE_AA,
        0,
    )?;

    // Draw vehicle marker circle
    imgproc::circle(
        &mut output,
        core::Point::new(vehicle_x, vehicle_y),
        10,
        core::Scalar::new(0.0, 255.0, 255.0, 0.0), // Cyan
        -1,
        imgproc::LINE_8,
        0,
    )?;

    // Draw vehicle marker outline
    imgproc::circle(
        &mut output,
        core::Point::new(vehicle_x, vehicle_y),
        10,
        core::Scalar::new(0.0, 0.0, 0.0, 0.0), // Black outline
        2,
        imgproc::LINE_8,
        0,
    )?;

    // State color based on state machine state
    let state_color = match state {
        "CENTERED" => core::Scalar::new(0.0, 255.0, 0.0, 0.0), // Green
        "DRIFTING" => core::Scalar::new(0.0, 255.0, 255.0, 0.0), // Yellow
        "CROSSING" => core::Scalar::new(0.0, 165.0, 255.0, 0.0), // Orange
        "COMPLETED" => core::Scalar::new(0.0, 0.0, 255.0, 0.0), // Red
        _ => core::Scalar::new(255.0, 255.0, 255.0, 0.0),      // White
    };

    // Draw info overlay background
    imgproc::rectangle(
        &mut output,
        core::Rect::new(5, 5, 550, 70),
        core::Scalar::new(0.0, 0.0, 0.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;

    // Add semi-transparent overlay effect by drawing a darker rectangle
    imgproc::rectangle(
        &mut output,
        core::Rect::new(5, 5, 550, 70),
        core::Scalar::new(40.0, 40.0, 40.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;

    // Draw state indicator box
    let state_box_color = match state {
        "CENTERED" => core::Scalar::new(0.0, 100.0, 0.0, 0.0), // Dark green
        "DRIFTING" => core::Scalar::new(0.0, 100.0, 100.0, 0.0), // Dark yellow
        "CROSSING" => core::Scalar::new(0.0, 80.0, 150.0, 0.0), // Dark orange
        "COMPLETED" => core::Scalar::new(0.0, 0.0, 150.0, 0.0), // Dark red
        _ => core::Scalar::new(80.0, 80.0, 80.0, 0.0),
    };

    imgproc::rectangle(
        &mut output,
        core::Rect::new(10, 10, 150, 30),
        state_box_color,
        -1,
        imgproc::LINE_8,
        0,
    )?;

    // Draw state text
    imgproc::put_text(
        &mut output,
        state,
        core::Point::new(15, 32),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.7,
        state_color,
        2,
        imgproc::LINE_8,
        false,
    )?;

    // Draw vehicle state info if available
    if let Some(vs) = vehicle_state {
        if vs.is_valid() {
            let normalized = vs.normalized_offset().unwrap_or(0.0);

            // Line 1: Offset info
            let offset_info = format!(
                "Offset: {:.1}px ({:+.1}%)",
                vs.lateral_offset,
                normalized * 100.0
            );
            imgproc::put_text(
                &mut output,
                &offset_info,
                core::Point::new(170, 32),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                core::Scalar::new(255.0, 255.0, 255.0, 0.0),
                1,
                imgproc::LINE_8,
                false,
            )?;

            // Line 2: Lane width info
            let width_info = format!("Lane Width: {:.0}px", vs.lane_width.unwrap_or(0.0));
            imgproc::put_text(
                &mut output,
                &width_info,
                core::Point::new(15, 60),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                core::Scalar::new(200.0, 200.0, 200.0, 0.0),
                1,
                imgproc::LINE_8,
                false,
            )?;

            // Draw offset indicator bar
            draw_offset_bar(&mut output, normalized, 380, 45)?;
        } else {
            // Invalid state
            imgproc::put_text(
                &mut output,
                "No valid lane data",
                core::Point::new(170, 32),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                core::Scalar::new(150.0, 150.0, 150.0, 0.0),
                1,
                imgproc::LINE_8,
                false,
            )?;
        }
    } else {
        // No vehicle state
        imgproc::put_text(
            &mut output,
            "Analyzing...",
            core::Point::new(170, 32),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.6,
            core::Scalar::new(150.0, 150.0, 150.0, 0.0),
            1,
            imgproc::LINE_8,
            false,
        )?;
    }

    // Draw lanes count
    let lanes_info = format!("Lanes: {}", lanes.len());
    imgproc::put_text(
        &mut output,
        &lanes_info,
        core::Point::new(200, 60),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        core::Scalar::new(200.0, 200.0, 200.0, 0.0),
        1,
        imgproc::LINE_8,
        false,
    )?;

    Ok(output)
}

/// Draw a visual offset indicator bar
fn draw_offset_bar(output: &mut Mat, normalized_offset: f32, x: i32, y: i32) -> Result<()> {
    let bar_width = 150;
    let bar_height = 20;

    // Background bar
    imgproc::rectangle(
        output,
        core::Rect::new(x, y, bar_width, bar_height),
        core::Scalar::new(60.0, 60.0, 60.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;

    // Center line
    imgproc::line(
        output,
        core::Point::new(x + bar_width / 2, y),
        core::Point::new(x + bar_width / 2, y + bar_height),
        core::Scalar::new(255.0, 255.0, 255.0, 0.0),
        1,
        imgproc::LINE_8,
        0,
    )?;

    // Threshold markers (drift threshold at ±20%)
    let drift_offset = (0.2 * bar_width as f32 / 2.0) as i32;
    imgproc::line(
        output,
        core::Point::new(x + bar_width / 2 - drift_offset, y),
        core::Point::new(x + bar_width / 2 - drift_offset, y + bar_height),
        core::Scalar::new(0.0, 200.0, 200.0, 0.0), // Yellow
        1,
        imgproc::LINE_8,
        0,
    )?;
    imgproc::line(
        output,
        core::Point::new(x + bar_width / 2 + drift_offset, y),
        core::Point::new(x + bar_width / 2 + drift_offset, y + bar_height),
        core::Scalar::new(0.0, 200.0, 200.0, 0.0), // Yellow
        1,
        imgproc::LINE_8,
        0,
    )?;

    // Threshold markers (crossing threshold at ±40%)
    let crossing_offset = (0.4 * bar_width as f32 / 2.0) as i32;
    imgproc::line(
        output,
        core::Point::new(x + bar_width / 2 - crossing_offset, y),
        core::Point::new(x + bar_width / 2 - crossing_offset, y + bar_height),
        core::Scalar::new(0.0, 100.0, 255.0, 0.0), // Orange
        1,
        imgproc::LINE_8,
        0,
    )?;
    imgproc::line(
        output,
        core::Point::new(x + bar_width / 2 + crossing_offset, y),
        core::Point::new(x + bar_width / 2 + crossing_offset, y + bar_height),
        core::Scalar::new(0.0, 100.0, 255.0, 0.0), // Orange
        1,
        imgproc::LINE_8,
        0,
    )?;

    // Current position indicator
    let clamped_offset = normalized_offset.clamp(-1.0, 1.0);
    let indicator_x = x + bar_width / 2 + (clamped_offset * bar_width as f32 / 2.0) as i32;

    // Color based on offset magnitude
    let indicator_color = if normalized_offset.abs() >= 0.4 {
        core::Scalar::new(0.0, 0.0, 255.0, 0.0) // Red - crossing
    } else if normalized_offset.abs() >= 0.2 {
        core::Scalar::new(0.0, 165.0, 255.0, 0.0) // Orange - drifting
    } else {
        core::Scalar::new(0.0, 255.0, 0.0, 0.0) // Green - centered
    };

    imgproc::circle(
        output,
        core::Point::new(indicator_x, y + bar_height / 2),
        6,
        indicator_color,
        -1,
        imgproc::LINE_8,
        0,
    )?;

    imgproc::circle(
        output,
        core::Point::new(indicator_x, y + bar_height / 2),
        6,
        core::Scalar::new(255.0, 255.0, 255.0, 0.0),
        1,
        imgproc::LINE_8,
        0,
    )?;

    Ok(())
}

/// Draw lanes with lane change event highlight
pub fn draw_lanes_with_event(
    frame: &[u8],
    width: i32,
    height: i32,
    lanes: &[Lane],
    state: &str,
    vehicle_state: Option<&VehicleState>,
    event_direction: Option<&str>,
) -> Result<Mat> {
    // Start with the base drawing
    let mut output = draw_lanes_with_state(frame, width, height, lanes, state, vehicle_state)?;

    // If there's a lane change event, draw a highlight
    if let Some(direction) = event_direction {
        // Draw event notification box
        let event_color = if direction == "LEFT" {
            core::Scalar::new(255.0, 100.0, 0.0, 0.0) // Blue-ish for left
        } else {
            core::Scalar::new(0.0, 100.0, 255.0, 0.0) // Orange for right
        };

        imgproc::rectangle(
            &mut output,
            core::Rect::new(width - 250, 10, 240, 50),
            event_color,
            -1,
            imgproc::LINE_8,
            0,
        )?;

        let event_text = format!("LANE CHANGE: {}", direction);
        imgproc::put_text(
            &mut output,
            &event_text,
            core::Point::new(width - 240, 42),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.7,
            core::Scalar::new(255.0, 255.0, 255.0, 0.0),
            2,
            imgproc::LINE_8,
            false,
        )?;

        // Draw arrow indicator
        let arrow_x = width / 2;
        let arrow_y = height / 2;
        let arrow_length = 80;

        let (start_x, end_x) = if direction == "LEFT" {
            (arrow_x + arrow_length / 2, arrow_x - arrow_length / 2)
        } else {
            (arrow_x - arrow_length / 2, arrow_x + arrow_length / 2)
        };

        imgproc::arrowed_line(
            &mut output,
            core::Point::new(start_x, arrow_y),
            core::Point::new(end_x, arrow_y),
            event_color,
            4,
            imgproc::LINE_AA,
            0,
            0.3,
        )?;
    }

    Ok(output)
}
