// src/video_processor.rs

use crate::types::Config;
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

pub fn draw_lanes(
    frame: &[u8],
    _width: i32,
    height: i32,
    lanes: &[crate::types::Lane],
) -> Result<Mat> {
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
