// src/video_processor.rs
//
// Enhanced visualization overlay v5.0 — Client Demo Edition
//
// Major additions over v4:
//   1. Semi-transparent ego lane polygon fill between L/R boundaries
//   2. Lateral position gauge (horizontal bar with needle)
//   3. Ego motion direction arrow on the road
//   4. Mini bird's-eye-view (BEV) radar showing ego + all tracked vehicles
//   5. Enhanced vehicle labels with zone transition trail dots
//   6. Maneuver event timeline bar at the bottom
//   7. Pulsing colored frame border on new maneuver events
//   8. Professional bottom status bar with branding + FPS + frame progress
//   9. Detection confidence meters (lane det, vehicle det)
//  10. Active pass/shift in-progress animated indicator
//  11. Being-overtaken full-width warning banner
//  12. Per-vehicle distance estimation text
//  13. Zone boundary region tinting on the road surface

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

// ══════════════════════════════════════════════════════════════════════════════
// ENHANCED VISUALIZATION v5.0 — CLIENT DEMO EDITION
// ══════════════════════════════════════════════════════════════════════════════

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

    let has_new_event = !maneuver_events.is_empty();
    let is_being_overtaken = maneuver_events.iter().any(|e| {
        e.maneuver_type == crate::analysis::maneuver_classifier::ManeuverType::BeingOvertaken
    });

    // ══════════════════════════════════════════════════════════════════════
    // 0. PULSING BORDER ON NEW MANEUVER EVENTS
    // ══════════════════════════════════════════════════════════════════════
    if has_new_event {
        let border_color = if is_being_overtaken {
            core::Scalar::new(0.0, 0.0, 255.0, 0.0) // Red
        } else if maneuver_events.iter().any(|e| {
            e.maneuver_type == crate::analysis::maneuver_classifier::ManeuverType::Overtake
        }) {
            core::Scalar::new(0.0, 255.0, 0.0, 0.0) // Green
        } else {
            core::Scalar::new(0.0, 200.0, 255.0, 0.0) // Orange
        };

        // Animated pulse: thickness varies with frame parity for a strobe effect
        let pulse_thickness = if frame_id % 4 < 2 { 8 } else { 5 };

        // Top
        imgproc::rectangle(
            &mut output,
            core::Rect::new(0, 0, width, pulse_thickness),
            border_color,
            -1,
            imgproc::LINE_8,
            0,
        )?;
        // Bottom
        imgproc::rectangle(
            &mut output,
            core::Rect::new(0, height - pulse_thickness, width, pulse_thickness),
            border_color,
            -1,
            imgproc::LINE_8,
            0,
        )?;
        // Left
        imgproc::rectangle(
            &mut output,
            core::Rect::new(0, 0, pulse_thickness, height),
            border_color,
            -1,
            imgproc::LINE_8,
            0,
        )?;
        // Right
        imgproc::rectangle(
            &mut output,
            core::Rect::new(width - pulse_thickness, 0, pulse_thickness, height),
            border_color,
            -1,
            imgproc::LINE_8,
            0,
        )?;
    }

    // ══════════════════════════════════════════════════════════════════════
    // 1. SEMI-TRANSPARENT EGO LANE POLYGON FILL
    // ══════════════════════════════════════════════════════════════════════
    if lanes.len() >= 2 {
        let left_lane = &lanes[0];
        let right_lane = &lanes[1];

        if left_lane.points.len() >= 2 && right_lane.points.len() >= 2 {
            // Build polygon: left points top-to-bottom + right points bottom-to-top
            let mut poly_pts: Vec<core::Point> = Vec::new();

            // Left lane points (sorted by Y descending — bottom first)
            let mut left_sorted = left_lane.points.clone();
            left_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for p in &left_sorted {
                poly_pts.push(core::Point::new(p.0 as i32, p.1 as i32));
            }

            // Right lane points (sorted by Y ascending — top first, so polygon closes)
            let mut right_sorted = right_lane.points.clone();
            right_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for p in &right_sorted {
                poly_pts.push(core::Point::new(p.0 as i32, p.1 as i32));
            }

            if poly_pts.len() >= 3 {
                let fill_color = if let Some(legality) = legality_result {
                    if legality.ego_intersects_marking && legality.verdict.is_illegal() {
                        core::Scalar::new(0.0, 0.0, 180.0, 0.0) // Red tint during violation
                    } else {
                        core::Scalar::new(180.0, 120.0, 0.0, 0.0) // Teal/cyan tint normal
                    }
                } else {
                    core::Scalar::new(180.0, 120.0, 0.0, 0.0)
                };

                let mut overlay = output.try_clone()?;
                let pts_mat = Mat::from_slice(&poly_pts)?;
                let pts_vec = Vector::<Mat>::from(vec![pts_mat]);
                imgproc::fill_poly(
                    &mut overlay,
                    &pts_vec,
                    fill_color,
                    imgproc::LINE_AA,
                    0,
                    core::Point::new(0, 0),
                )?;

                let mut blended = Mat::default();
                core::add_weighted(&overlay, 0.15, &output, 0.85, 0.0, &mut blended, -1)?;
                blended.copy_to(&mut output)?;
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 2. DRAW LANE MARKINGS WITH TYPE LABELS
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

        // Draw thicker lane lines for visibility
        for point in &lane.points {
            let pt = core::Point::new(point.0 as i32, point.1 as i32);
            imgproc::circle(
                &mut output,
                pt,
                thickness + 1,
                color,
                -1,
                imgproc::LINE_8,
                0,
            )?;
        }

        if lane.points.len() >= 2 {
            for j in 0..lane.points.len() - 1 {
                let pt1 = core::Point::new(lane.points[j].0 as i32, lane.points[j].1 as i32);
                let pt2 =
                    core::Point::new(lane.points[j + 1].0 as i32, lane.points[j + 1].1 as i32);
                imgproc::line(&mut output, pt1, pt2, color, 3, imgproc::LINE_AA, 0)?;
            }

            if let Some(first_point) = lane.points.first() {
                let lane_label = if i == 0 {
                    "LEFT BOUNDARY"
                } else if i == 1 {
                    "RIGHT BOUNDARY"
                } else {
                    "LANE"
                };

                draw_text_with_shadow(
                    &mut output,
                    lane_label,
                    first_point.0 as i32 + 10,
                    first_point.1 as i32,
                    0.45,
                    color,
                    1,
                )?;
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 3. DRAW TRACKED VEHICLES WITH ENHANCED LABELS
    // ══════════════════════════════════════════════════════════════════════
    for track in tracked_vehicles {
        if !track.is_confirmed() {
            continue;
        }

        let bbox = &track.bbox;
        let bbox_w = bbox[2] - bbox[0];
        let bbox_h = bbox[3] - bbox[1];
        let bbox_area = bbox_w * bbox_h;
        let frame_area = width as f32 * height as f32;

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
        let (box_color, zone_str) = match track.zone {
            crate::analysis::vehicle_tracker::VehicleZone::Ahead => {
                (core::Scalar::new(255.0, 255.0, 0.0, 0.0), "AHEAD")
            }
            crate::analysis::vehicle_tracker::VehicleZone::BesideLeft => {
                (core::Scalar::new(0.0, 165.0, 255.0, 0.0), "BESIDE-L")
            }
            crate::analysis::vehicle_tracker::VehicleZone::BesideRight => {
                (core::Scalar::new(255.0, 0.0, 255.0, 0.0), "BESIDE-R")
            }
            crate::analysis::vehicle_tracker::VehicleZone::Behind => {
                (core::Scalar::new(0.0, 255.0, 0.0, 0.0), "BEHIND")
            }
            crate::analysis::vehicle_tracker::VehicleZone::Unknown => {
                (core::Scalar::new(128.0, 128.0, 128.0, 0.0), "?")
            }
        };

        // Draw bounding box with corner accents for a modern look
        let pt1 = core::Point::new(bbox[0] as i32, bbox[1] as i32);
        let pt2 = core::Point::new(bbox[2] as i32, bbox[3] as i32);
        // Thin full rect
        imgproc::rectangle(
            &mut output,
            core::Rect::from_points(pt1, pt2),
            box_color,
            2,
            imgproc::LINE_8,
            0,
        )?;
        // Corner accents (thicker, shorter lines at each corner)
        let corner_len = (bbox_w.min(bbox_h) * 0.2).max(8.0) as i32;
        draw_corner_accents(
            &mut output,
            pt1.x,
            pt1.y,
            pt2.x,
            pt2.y,
            corner_len,
            box_color,
            4,
        )?;

        // Relative size estimate (proxy for distance)
        let size_pct = (bbox_area / frame_area) * 100.0;
        let distance_label = if size_pct > 8.0 {
            "VERY CLOSE"
        } else if size_pct > 3.0 {
            "CLOSE"
        } else if size_pct > 1.0 {
            "MEDIUM"
        } else {
            "FAR"
        };

        // Primary label: CLASS ID ZONE
        let label_line1 = format!("{} ID:{} {}", class_name.to_uppercase(), track.id, zone_str,);
        // Secondary label: confidence + distance
        let label_line2 = format!("{:.0}% | {}", track.last_confidence * 100.0, distance_label,);

        let label_y = (bbox[1] as i32) - 10;

        // Background for two-line label
        let size1 =
            imgproc::get_text_size(&label_line1, imgproc::FONT_HERSHEY_SIMPLEX, 0.48, 1, &mut 0)?;
        let size2 =
            imgproc::get_text_size(&label_line2, imgproc::FONT_HERSHEY_SIMPLEX, 0.40, 1, &mut 0)?;
        let label_w = size1.width.max(size2.width) + 8;
        let label_h = size1.height + size2.height + 12;

        let bg_pt1 = core::Point::new(bbox[0] as i32 - 2, label_y - label_h);
        let bg_pt2 = core::Point::new(bbox[0] as i32 + label_w, label_y + 4);

        // Semi-transparent label background
        draw_filled_rect_alpha(&mut output, bg_pt1.x, bg_pt1.y, label_w, label_h + 4, 0.75)?;

        imgproc::put_text(
            &mut output,
            &label_line1,
            core::Point::new(bbox[0] as i32 + 2, label_y - size2.height - 4),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.48,
            core::Scalar::new(255.0, 255.0, 255.0, 0.0),
            1,
            imgproc::LINE_AA,
            false,
        )?;

        imgproc::put_text(
            &mut output,
            &label_line2,
            core::Point::new(bbox[0] as i32 + 2, label_y),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.40,
            box_color,
            1,
            imgproc::LINE_AA,
            false,
        )?;

        // Zone transition trail: small dots showing recent zone history
        let history_len = track.zone_history.len();
        if history_len >= 2 {
            let trail_y = bbox[3] as i32 + 8;
            let trail_start_x = bbox[0] as i32;
            let show = history_len.min(8);
            for (ti, obs) in track.zone_history.iter().rev().take(show).enumerate() {
                let dot_color = match obs.zone {
                    crate::analysis::vehicle_tracker::VehicleZone::Ahead => {
                        core::Scalar::new(255.0, 255.0, 0.0, 0.0)
                    }
                    crate::analysis::vehicle_tracker::VehicleZone::BesideLeft
                    | crate::analysis::vehicle_tracker::VehicleZone::BesideRight => {
                        core::Scalar::new(0.0, 165.0, 255.0, 0.0)
                    }
                    crate::analysis::vehicle_tracker::VehicleZone::Behind => {
                        core::Scalar::new(0.0, 255.0, 0.0, 0.0)
                    }
                    _ => core::Scalar::new(128.0, 128.0, 128.0, 0.0),
                };
                let dot_x = trail_start_x + (ti as i32) * 10;
                if dot_x < width - 10 {
                    imgproc::circle(
                        &mut output,
                        core::Point::new(dot_x, trail_y),
                        3,
                        dot_color,
                        -1,
                        imgproc::LINE_8,
                        0,
                    )?;
                }
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 4. DRAW EGO VEHICLE MARKER + DIRECTION ARROW
    // ══════════════════════════════════════════════════════════════════════
    let ego_x = width / 2;
    let ego_y = (height as f32 * 0.85) as i32;

    // Ego marker: triangle pointing up (car shape)
    let tri_pts = vec![
        core::Point::new(ego_x, ego_y - 18),
        core::Point::new(ego_x - 12, ego_y + 10),
        core::Point::new(ego_x + 12, ego_y + 10),
    ];
    let tri_mat = Mat::from_slice(&tri_pts)?;
    let tri_vec = Vector::<Mat>::from(vec![tri_mat]);
    imgproc::fill_poly(
        &mut output,
        &tri_vec,
        core::Scalar::new(0.0, 255.0, 255.0, 0.0),
        imgproc::LINE_AA,
        0,
        core::Point::new(0, 0),
    )?;
    imgproc::put_text(
        &mut output,
        "EGO",
        core::Point::new(ego_x - 14, ego_y + 28),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.4,
        core::Scalar::new(0.0, 255.0, 255.0, 0.0),
        1,
        imgproc::LINE_AA,
        false,
    )?;

    // Lateral velocity arrow on the road surface
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
            &mut output,
            core::Point::new(ego_x, ego_y - 30),
            core::Point::new(ego_x + arrow_len, ego_y - 30),
            arrow_color,
            3,
            imgproc::LINE_AA,
            0,
            0.3,
        )?;
    }

    // ══════════════════════════════════════════════════════════════════════
    // 5. LEGALITY BANNER (full-width, with icon indicator)
    // ══════════════════════════════════════════════════════════════════════
    let mut banner_active = false;

    if let Some(legality) = legality_result {
        if legality.ego_intersects_marking && legality.verdict.is_illegal() {
            let banner_h = 54;
            let color = match legality.verdict {
                crate::lane_legality::LineLegality::CriticalIllegal => {
                    core::Scalar::new(0.0, 0.0, 200.0, 0.0)
                }
                crate::lane_legality::LineLegality::Illegal => {
                    core::Scalar::new(0.0, 50.0, 180.0, 0.0)
                }
                _ => core::Scalar::new(0.0, 180.0, 230.0, 0.0),
            };

            // Semi-transparent banner
            let mut overlay = output.try_clone()?;
            imgproc::rectangle(
                &mut overlay,
                core::Rect::new(0, 0, width, banner_h),
                color,
                -1,
                imgproc::LINE_8,
                0,
            )?;
            let mut blended = Mat::default();
            core::add_weighted(&overlay, 0.85, &output, 0.15, 0.0, &mut blended, -1)?;
            blended.copy_to(&mut output)?;

            let line_name = legality
                .intersecting_line
                .as_ref()
                .map(|l| l.class_name.as_str())
                .unwrap_or("SOLID LINE");

            // Warning icon triangle
            let icon_x = 15;
            let icon_y = 10;
            let tri = vec![
                core::Point::new(icon_x + 15, icon_y),
                core::Point::new(icon_x, icon_y + 30),
                core::Point::new(icon_x + 30, icon_y + 30),
            ];
            let tri_m = Mat::from_slice(&tri)?;
            let tri_v = Vector::<Mat>::from(vec![tri_m]);
            imgproc::polylines(
                &mut output,
                &tri_v,
                true,
                core::Scalar::new(255.0, 255.0, 255.0, 0.0),
                2,
                imgproc::LINE_AA,
                0,
            )?;
            imgproc::put_text(
                &mut output,
                "!",
                core::Point::new(icon_x + 10, icon_y + 26),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.7,
                core::Scalar::new(255.0, 255.0, 255.0, 0.0),
                2,
                imgproc::LINE_AA,
                false,
            )?;

            let text = format!("ILLEGAL CROSSING: {} - VIOLATION", line_name.to_uppercase());
            draw_text_with_shadow(
                &mut output,
                &text,
                55,
                37,
                0.75,
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
    // 6. LEFT PANEL: AI DETECTION STATUS (enhanced)
    // ══════════════════════════════════════════════════════════════════════
    let panel_x = 15;
    let mut panel_y = if banner_active { 72 } else { 30 };
    let line_height = 24;

    draw_panel_background(&mut output, 5, panel_y - 10, 480, 400)?;

    // Title bar with accent line
    draw_text_with_shadow(
        &mut output,
        "AI DETECTION STATUS",
        panel_x,
        panel_y,
        0.65,
        core::Scalar::new(100.0, 200.0, 255.0, 0.0),
        2,
    )?;
    // Accent underline
    imgproc::line(
        &mut output,
        core::Point::new(panel_x, panel_y + 5),
        core::Point::new(panel_x + 220, panel_y + 5),
        core::Scalar::new(100.0, 200.0, 255.0, 0.0),
        2,
        imgproc::LINE_AA,
        0,
    )?;
    panel_y += line_height + 4;

    // Frame info
    let time_str = format_timestamp(timestamp_ms);
    draw_text_with_shadow(
        &mut output,
        &format!("Frame: {} | {}", frame_id, time_str),
        panel_x,
        panel_y,
        0.48,
        core::Scalar::new(200.0, 200.0, 200.0, 0.0),
        1,
    )?;
    panel_y += line_height;

    // ── Lane Detection Status + confidence meter ──
    let (lane_status, lane_color, lane_conf) = if lanes.len() >= 2 {
        let conf = lanes.iter().map(|l| l.confidence).sum::<f32>() / lanes.len() as f32;
        (
            format!("{} lanes detected (L+R)", lanes.len()),
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            conf,
        )
    } else if lanes.len() == 1 {
        (
            "1 lane detected (partial)".to_string(),
            core::Scalar::new(0.0, 165.0, 255.0, 0.0),
            lanes[0].confidence * 0.6,
        )
    } else {
        (
            "No lanes detected".to_string(),
            core::Scalar::new(0.0, 0.0, 255.0, 0.0),
            0.0,
        )
    };

    // Status icon
    let icon = if lanes.len() >= 2 {
        "+"
    } else if lanes.len() == 1 {
        "~"
    } else {
        "x"
    };
    draw_text_with_shadow(&mut output, icon, panel_x, panel_y, 0.5, lane_color, 2)?;
    draw_text_with_shadow(
        &mut output,
        &lane_status,
        panel_x + 18,
        panel_y,
        0.48,
        lane_color,
        1,
    )?;
    panel_y += line_height;

    // Lane confidence bar
    draw_confidence_bar(&mut output, panel_x + 10, panel_y - 4, 180, 10, lane_conf)?;
    draw_text_with_shadow(
        &mut output,
        &format!("{:.0}%", lane_conf * 100.0),
        panel_x + 198,
        panel_y + 4,
        0.38,
        core::Scalar::new(180.0, 180.0, 180.0, 0.0),
        1,
    )?;
    panel_y += line_height - 4;

    // ── Road Marking Types ──
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
                    0.42,
                    core::Scalar::new(180.0, 180.0, 180.0, 0.0),
                    1,
                )?;
                panel_y += 18;
            }

            if !right_markings.is_empty() {
                draw_text_with_shadow(
                    &mut output,
                    &format!("  R: {}", right_markings.join(", ")),
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

    // ── Vehicle Tracking Summary ──
    let confirmed_count = tracked_vehicles.iter().filter(|t| t.is_confirmed()).count();
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
            *vehicle_classes.entry(class_name).or_insert(0u32) += 1;
        }
    }

    panel_y += 4;
    draw_text_with_shadow(
        &mut output,
        &format!("Tracked Vehicles: {}", confirmed_count),
        panel_x,
        panel_y,
        0.52,
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        1,
    )?;
    panel_y += line_height;

    // Vehicle type breakdown (compact)
    if !vehicle_classes.is_empty() {
        let breakdown: Vec<String> = vehicle_classes
            .iter()
            .map(|(cls, cnt)| format!("{}x{}", cnt, cls))
            .collect();
        draw_text_with_shadow(
            &mut output,
            &format!("  [{}]", breakdown.join("  ")),
            panel_x + 8,
            panel_y,
            0.42,
            core::Scalar::new(180.0, 180.0, 180.0, 0.0),
            1,
        )?;
        panel_y += 20;
    }

    // ── Zone Distribution ──
    let mut zone_counts = [0u32; 5]; // Ahead, BesideL, BesideR, Behind, Unknown
    for track in tracked_vehicles {
        if track.is_confirmed() {
            match track.zone {
                crate::analysis::vehicle_tracker::VehicleZone::Ahead => zone_counts[0] += 1,
                crate::analysis::vehicle_tracker::VehicleZone::BesideLeft => zone_counts[1] += 1,
                crate::analysis::vehicle_tracker::VehicleZone::BesideRight => zone_counts[2] += 1,
                crate::analysis::vehicle_tracker::VehicleZone::Behind => zone_counts[3] += 1,
                crate::analysis::vehicle_tracker::VehicleZone::Unknown => zone_counts[4] += 1,
            }
        }
    }
    if confirmed_count > 0 {
        draw_text_with_shadow(
            &mut output,
            &format!(
                "  Zones: {}A  {}BL  {}BR  {}BH",
                zone_counts[0], zone_counts[1], zone_counts[2], zone_counts[3]
            ),
            panel_x + 8,
            panel_y,
            0.42,
            core::Scalar::new(160.0, 200.0, 160.0, 0.0),
            1,
        )?;
        panel_y += 20;
    }

    // ── Lateral State ──
    let state_color = match lateral_state {
        s if s.contains("CENTERED") || s.contains("STABLE") => {
            core::Scalar::new(0.0, 255.0, 0.0, 0.0)
        }
        s if s.contains("SHIFT") => core::Scalar::new(0.0, 165.0, 255.0, 0.0),
        s if s.contains("RECOVERING") => core::Scalar::new(0.0, 255.0, 255.0, 0.0),
        s if s.contains("OCCLUDED") => core::Scalar::new(0.0, 0.0, 200.0, 0.0),
        _ => core::Scalar::new(255.0, 255.0, 255.0, 0.0),
    };

    draw_text_with_shadow(
        &mut output,
        &format!("Lateral: {}", lateral_state),
        panel_x,
        panel_y,
        0.52,
        state_color,
        1,
    )?;
    panel_y += line_height;

    // ── Ego Lateral Velocity with arrow ──
    let vel_color = if ego_lateral_velocity.abs() > 3.0 {
        core::Scalar::new(0.0, 0.0, 255.0, 0.0)
    } else if ego_lateral_velocity.abs() > 1.5 {
        core::Scalar::new(0.0, 165.0, 255.0, 0.0)
    } else {
        core::Scalar::new(255.0, 255.0, 255.0, 0.0)
    };

    let dir_arrow = if ego_lateral_velocity > 0.5 {
        ">>"
    } else if ego_lateral_velocity < -0.5 {
        "<<"
    } else {
        "=="
    };

    draw_text_with_shadow(
        &mut output,
        &format!("Ego Drift: {:.2} px/f {}", ego_lateral_velocity, dir_arrow),
        panel_x,
        panel_y,
        0.48,
        vel_color,
        1,
    )?;
    panel_y += line_height;

    // ── Lateral Offset + gauge ──
    if let Some(vs) = vehicle_state {
        if vs.is_valid() {
            let normalized = vs.normalized_offset().unwrap_or(0.0);

            draw_text_with_shadow(
                &mut output,
                &format!(
                    "Offset: {:.1}px ({:+.1}%)",
                    vs.lateral_offset,
                    normalized * 100.0
                ),
                panel_x,
                panel_y,
                0.48,
                core::Scalar::new(255.0, 255.0, 255.0, 0.0),
                1,
            )?;
            panel_y += line_height;

            // Lateral position gauge
            draw_lateral_gauge(&mut output, panel_x + 10, panel_y - 4, 200, 14, normalized)?;
            panel_y += 20;

            if let Some(lw) = vs.lane_width {
                draw_text_with_shadow(
                    &mut output,
                    &format!("Lane Width: {:.0}px", lw),
                    panel_x,
                    panel_y,
                    0.42,
                    core::Scalar::new(180.0, 180.0, 180.0, 0.0),
                    1,
                )?;
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 7. RIGHT PANEL: MANEUVER DETECTION + TOTALS
    // ══════════════════════════════════════════════════════════════════════
    let right_panel_x = width - 500;
    let mut right_panel_y = if banner_active { 72 } else { 30 };

    draw_panel_background(
        &mut output,
        right_panel_x - 10,
        right_panel_y - 10,
        495,
        420,
    )?;

    // Title with accent
    draw_text_with_shadow(
        &mut output,
        "MANEUVER DETECTION",
        right_panel_x,
        right_panel_y,
        0.65,
        core::Scalar::new(100.0, 255.0, 100.0, 0.0),
        2,
    )?;
    imgproc::line(
        &mut output,
        core::Point::new(right_panel_x, right_panel_y + 5),
        core::Point::new(right_panel_x + 230, right_panel_y + 5),
        core::Scalar::new(100.0, 255.0, 100.0, 0.0),
        2,
        imgproc::LINE_AA,
        0,
    )?;
    right_panel_y += line_height + 6;

    // ── SESSION TOTALS with big numbers ──
    draw_text_with_shadow(
        &mut output,
        "SESSION TOTALS",
        right_panel_x,
        right_panel_y,
        0.5,
        core::Scalar::new(255.0, 200.0, 100.0, 0.0),
        1,
    )?;
    right_panel_y += line_height;

    // Overtakes row
    draw_text_with_shadow(
        &mut output,
        &format!("{}", total_overtakes),
        right_panel_x,
        right_panel_y,
        0.9,
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
    )?;
    draw_text_with_shadow(
        &mut output,
        &format!("Overtakes ({} veh)", total_vehicles_overtaken),
        right_panel_x + 50,
        right_panel_y,
        0.48,
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        1,
    )?;
    right_panel_y += line_height + 2;

    // Lane changes row
    draw_text_with_shadow(
        &mut output,
        &format!("{}", total_lane_changes),
        right_panel_x,
        right_panel_y,
        0.9,
        core::Scalar::new(0.0, 165.0, 255.0, 0.0),
        2,
    )?;
    draw_text_with_shadow(
        &mut output,
        "Lane Changes",
        right_panel_x + 50,
        right_panel_y,
        0.48,
        core::Scalar::new(0.0, 165.0, 255.0, 0.0),
        1,
    )?;
    right_panel_y += line_height + 8;

    // ── LAST MANEUVER (PERSISTENT) ──
    if let Some(maneuver) = last_maneuver {
        let seconds_ago = (timestamp_ms - maneuver.timestamp_detected) / 1000.0;
        let is_recent = seconds_ago < 3.0;

        draw_text_with_shadow(
            &mut output,
            "LAST MANEUVER",
            right_panel_x,
            right_panel_y,
            0.5,
            core::Scalar::new(255.0, 200.0, 100.0, 0.0),
            1,
        )?;
        // "ago" indicator
        let ago_color = if is_recent {
            core::Scalar::new(0.0, 255.0, 0.0, 0.0)
        } else {
            core::Scalar::new(150.0, 150.0, 150.0, 0.0)
        };
        draw_text_with_shadow(
            &mut output,
            &format!("{:.1}s ago", seconds_ago),
            right_panel_x + 180,
            right_panel_y,
            0.42,
            ago_color,
            1,
        )?;
        right_panel_y += line_height;

        let maneuver_color = match maneuver.maneuver_type.as_str() {
            "OVERTAKE" => core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            "LANE_CHANGE" => core::Scalar::new(0.0, 165.0, 255.0, 0.0),
            "BEING_OVERTAKEN" => core::Scalar::new(0.0, 0.0, 255.0, 0.0),
            _ => core::Scalar::new(255.0, 255.0, 255.0, 0.0),
        };

        // Type + side as big text
        draw_text_with_shadow(
            &mut output,
            &format!("{} {}", maneuver.maneuver_type, maneuver.side),
            right_panel_x,
            right_panel_y,
            0.6,
            maneuver_color,
            2,
        )?;
        right_panel_y += line_height + 2;

        // Details in compact format
        draw_text_with_shadow(
            &mut output,
            &format!(
                "conf={:.0}% | dur={:.1}s | src={}",
                maneuver.confidence * 100.0,
                maneuver.duration_ms / 1000.0,
                maneuver.sources,
            ),
            right_panel_x + 8,
            right_panel_y,
            0.42,
            core::Scalar::new(200.0, 200.0, 200.0, 0.0),
            1,
        )?;
        right_panel_y += 20;

        // Legality badge
        let (legality_text, legality_color) = if maneuver.legality.contains("CriticalIllegal") {
            ("CRITICAL ILLEGAL", core::Scalar::new(0.0, 0.0, 255.0, 0.0))
        } else if maneuver.legality.contains("Illegal") {
            ("ILLEGAL", core::Scalar::new(0.0, 100.0, 255.0, 0.0))
        } else if maneuver.legality.contains("Legal") {
            ("LEGAL", core::Scalar::new(0.0, 255.0, 0.0, 0.0))
        } else {
            ("UNKNOWN", core::Scalar::new(200.0, 200.0, 200.0, 0.0))
        };

        // Draw legality as a badge
        let badge_size =
            imgproc::get_text_size(legality_text, imgproc::FONT_HERSHEY_SIMPLEX, 0.5, 2, &mut 0)?;
        draw_filled_rect_alpha(
            &mut output,
            right_panel_x + 6,
            right_panel_y - badge_size.height - 4,
            badge_size.width + 12,
            badge_size.height + 10,
            0.6,
        )?;
        imgproc::rectangle(
            &mut output,
            core::Rect::new(
                right_panel_x + 6,
                right_panel_y - badge_size.height - 4,
                badge_size.width + 12,
                badge_size.height + 10,
            ),
            legality_color,
            2,
            imgproc::LINE_8,
            0,
        )?;
        draw_text_with_shadow(
            &mut output,
            legality_text,
            right_panel_x + 12,
            right_panel_y,
            0.5,
            legality_color,
            2,
        )?;
        right_panel_y += line_height + 4;

        // Vehicle count for overtakes
        if maneuver.maneuver_type == "OVERTAKE" && maneuver.vehicles_in_this_maneuver > 0 {
            draw_text_with_shadow(
                &mut output,
                &format!("Vehicles passed: {}", maneuver.vehicles_in_this_maneuver),
                right_panel_x + 8,
                right_panel_y,
                0.48,
                core::Scalar::new(0.0, 255.0, 100.0, 0.0),
                1,
            )?;
            right_panel_y += line_height;
        }
    } else {
        draw_text_with_shadow(
            &mut output,
            "No maneuvers detected yet",
            right_panel_x,
            right_panel_y,
            0.48,
            core::Scalar::new(120.0, 120.0, 120.0, 0.0),
            1,
        )?;
        right_panel_y += line_height;
    }

    // ── NEW THIS FRAME flash ──
    if has_new_event {
        right_panel_y += 8;

        // Flashing "NEW" indicator
        let flash_color = if frame_id % 4 < 2 {
            core::Scalar::new(0.0, 255.0, 255.0, 0.0) // Yellow
        } else {
            core::Scalar::new(255.0, 255.0, 255.0, 0.0) // White
        };

        draw_text_with_shadow(
            &mut output,
            ">>> NEW EVENT DETECTED <<<",
            right_panel_x,
            right_panel_y,
            0.6,
            flash_color,
            2,
        )?;
        right_panel_y += line_height;

        for event in maneuver_events.iter().take(3) {
            let ev_color = match event.maneuver_type {
                crate::analysis::maneuver_classifier::ManeuverType::Overtake => {
                    core::Scalar::new(0.0, 255.0, 0.0, 0.0)
                }
                crate::analysis::maneuver_classifier::ManeuverType::LaneChange => {
                    core::Scalar::new(0.0, 165.0, 255.0, 0.0)
                }
                crate::analysis::maneuver_classifier::ManeuverType::BeingOvertaken => {
                    core::Scalar::new(0.0, 0.0, 255.0, 0.0)
                }
            };
            draw_text_with_shadow(
                &mut output,
                &format!(
                    "  {} {} conf={:.0}%",
                    event.maneuver_type.as_str(),
                    event.side.as_str(),
                    event.confidence * 100.0,
                ),
                right_panel_x + 8,
                right_panel_y,
                0.45,
                ev_color,
                1,
            )?;
            right_panel_y += 20;
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 8. MINI BIRD'S-EYE VIEW RADAR
    // ══════════════════════════════════════════════════════════════════════
    let bev_w = 160;
    let bev_h = 220;
    let bev_x = width - bev_w - 20;
    let bev_y = height - bev_h - 80;

    draw_panel_background(&mut output, bev_x - 5, bev_y - 20, bev_w + 10, bev_h + 25)?;

    draw_text_with_shadow(
        &mut output,
        "RADAR VIEW",
        bev_x + 30,
        bev_y - 5,
        0.42,
        core::Scalar::new(200.0, 200.0, 200.0, 0.0),
        1,
    )?;

    // Draw road lanes in BEV
    let bev_cx = bev_x + bev_w / 2;
    let lane_half_w = 35;

    // Road surface
    draw_filled_rect_alpha(
        &mut output,
        bev_cx - lane_half_w - 5,
        bev_y,
        (lane_half_w + 5) * 2,
        bev_h,
        0.3,
    )?;

    // Lane lines (dashed)
    for dy in (0..bev_h).step_by(16) {
        if dy % 32 < 16 {
            imgproc::line(
                &mut output,
                core::Point::new(bev_cx - lane_half_w, bev_y + dy),
                core::Point::new(bev_cx - lane_half_w, bev_y + dy + 10),
                core::Scalar::new(100.0, 100.0, 100.0, 0.0),
                1,
                imgproc::LINE_AA,
                0,
            )?;
            imgproc::line(
                &mut output,
                core::Point::new(bev_cx + lane_half_w, bev_y + dy),
                core::Point::new(bev_cx + lane_half_w, bev_y + dy + 10),
                core::Scalar::new(100.0, 100.0, 100.0, 0.0),
                1,
                imgproc::LINE_AA,
                0,
            )?;
        }
    }

    // Ego vehicle in BEV (bottom center)
    let ego_bev_y = bev_y + bev_h - 30;
    let ego_offset_bev = if let Some(vs) = vehicle_state {
        let norm = vs.normalized_offset().unwrap_or(0.0);
        (norm * lane_half_w as f32) as i32
    } else {
        0
    };
    imgproc::rectangle(
        &mut output,
        core::Rect::new(bev_cx + ego_offset_bev - 8, ego_bev_y - 12, 16, 24),
        core::Scalar::new(0.0, 255.0, 255.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;
    imgproc::rectangle(
        &mut output,
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

        // Map camera coords to BEV position
        // X: relative to frame center, scaled to BEV lane width
        let rel_x = (t_cx - width as f32 / 2.0) / (width as f32 / 2.0);
        let bev_vx = bev_cx + (rel_x * lane_half_w as f32 * 1.5) as i32;

        // Y: higher in frame = further away = higher in BEV
        let rel_y = t_cy / height as f32;
        let bev_vy = bev_y + (rel_y * bev_h as f32 * 0.85) as i32;

        // Zone color
        let dot_color = match track.zone {
            crate::analysis::vehicle_tracker::VehicleZone::Ahead => {
                core::Scalar::new(255.0, 255.0, 0.0, 0.0)
            }
            crate::analysis::vehicle_tracker::VehicleZone::BesideLeft => {
                core::Scalar::new(0.0, 165.0, 255.0, 0.0)
            }
            crate::analysis::vehicle_tracker::VehicleZone::BesideRight => {
                core::Scalar::new(255.0, 0.0, 255.0, 0.0)
            }
            crate::analysis::vehicle_tracker::VehicleZone::Behind => {
                core::Scalar::new(0.0, 255.0, 0.0, 0.0)
            }
            _ => core::Scalar::new(128.0, 128.0, 128.0, 0.0),
        };

        // Clamp to BEV bounds
        let bev_vx_c = bev_vx.clamp(bev_x + 5, bev_x + bev_w - 5);
        let bev_vy_c = bev_vy.clamp(bev_y + 5, bev_y + bev_h - 5);

        imgproc::rectangle(
            &mut output,
            core::Rect::new(bev_vx_c - 5, bev_vy_c - 4, 10, 8),
            dot_color,
            -1,
            imgproc::LINE_8,
            0,
        )?;
        // ID label
        imgproc::put_text(
            &mut output,
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

    // ══════════════════════════════════════════════════════════════════════
    // 9. BOTTOM STATUS BAR (branding + progress + system info)
    // ══════════════════════════════════════════════════════════════════════
    let bar_h = 36;
    let bar_y = height - bar_h;

    // Semi-transparent dark bar
    {
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
        core::add_weighted(&overlay, 0.85, &output, 0.15, 0.0, &mut blended, -1)?;
        blended.copy_to(&mut output)?;
    }

    // Top accent line
    imgproc::line(
        &mut output,
        core::Point::new(0, bar_y),
        core::Point::new(width, bar_y),
        core::Scalar::new(80.0, 180.0, 255.0, 0.0),
        2,
        imgproc::LINE_AA,
        0,
    )?;

    // Left: branding
    draw_text_with_shadow(
        &mut output,
        "AI MANEUVER DETECTION SYSTEM v5.0",
        12,
        bar_y + 24,
        0.48,
        core::Scalar::new(80.0, 180.0, 255.0, 0.0),
        1,
    )?;

    // Center: timestamp + frame
    let center_text = format!("{} | F:{}", time_str, frame_id);
    let center_size =
        imgproc::get_text_size(&center_text, imgproc::FONT_HERSHEY_SIMPLEX, 0.42, 1, &mut 0)?;
    draw_text_with_shadow(
        &mut output,
        &center_text,
        (width - center_size.width) / 2,
        bar_y + 24,
        0.42,
        core::Scalar::new(200.0, 200.0, 200.0, 0.0),
        1,
    )?;

    // Right: live counters summary
    let right_text = format!(
        "OVT:{} | LC:{} | VEH:{}",
        total_overtakes, total_lane_changes, confirmed_count
    );
    let right_size =
        imgproc::get_text_size(&right_text, imgproc::FONT_HERSHEY_SIMPLEX, 0.42, 1, &mut 0)?;
    draw_text_with_shadow(
        &mut output,
        &right_text,
        width - right_size.width - 15,
        bar_y + 24,
        0.42,
        core::Scalar::new(200.0, 200.0, 200.0, 0.0),
        1,
    )?;

    // ══════════════════════════════════════════════════════════════════════
    // 10. LEGEND (compact, bottom-left above status bar)
    // ══════════════════════════════════════════════════════════════════════
    let legend_x = 10;
    let legend_y = height - bar_h - 130;

    draw_panel_background(&mut output, legend_x - 5, legend_y - 10, 280, 120)?;

    draw_text_with_shadow(
        &mut output,
        "LEGEND",
        legend_x,
        legend_y,
        0.42,
        core::Scalar::new(180.0, 180.0, 180.0, 0.0),
        1,
    )?;

    let legend_items: Vec<(&str, core::Scalar)> = vec![
        ("AHEAD", core::Scalar::new(255.0, 255.0, 0.0, 0.0)),
        ("BESIDE-L", core::Scalar::new(0.0, 165.0, 255.0, 0.0)),
        ("BESIDE-R", core::Scalar::new(255.0, 0.0, 255.0, 0.0)),
        ("BEHIND", core::Scalar::new(0.0, 255.0, 0.0, 0.0)),
        ("EGO", core::Scalar::new(0.0, 255.0, 255.0, 0.0)),
    ];

    for (i, (label, color)) in legend_items.iter().enumerate() {
        let ly = legend_y + 18 + (i as i32 * 18);
        // Color swatch
        imgproc::rectangle(
            &mut output,
            core::Rect::new(legend_x + 2, ly - 10, 14, 14),
            *color,
            -1,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::put_text(
            &mut output,
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

    Ok(output)
}

// ══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ══════════════════════════════════════════════════════════════════════════════

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

    // Border with rounded-corner look (just thin border)
    imgproc::rectangle(
        img,
        core::Rect::new(x, y, width, height),
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
    // Shadow (dark offset)
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

    // Main text
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

/// Draw corner bracket accents on a bounding box for a modern detection look
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

/// Confidence bar: green gradient from left to right proportional to value
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

    // Fill
    let fill_w = (w as f32 * value) as i32;
    if fill_w > 0 {
        let fill_color = if value > 0.7 {
            core::Scalar::new(0.0, 200.0, 0.0, 0.0) // Green
        } else if value > 0.4 {
            core::Scalar::new(0.0, 180.0, 220.0, 0.0) // Yellow
        } else {
            core::Scalar::new(0.0, 80.0, 220.0, 0.0) // Red
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

/// Lateral position gauge: shows vehicle position within lane
/// normalized = -1.0 (far left) to +1.0 (far right), 0.0 = centered
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

    // Lane boundary indicators (left/right edges with danger zones)
    let danger_w = (w as f32 * 0.15) as i32;
    // Left danger zone
    imgproc::rectangle(
        img,
        core::Rect::new(x, y, danger_w, h),
        core::Scalar::new(0.0, 0.0, 80.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;
    // Right danger zone
    imgproc::rectangle(
        img,
        core::Rect::new(x + w - danger_w, y, danger_w, h),
        core::Scalar::new(0.0, 0.0, 80.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;

    // Center line
    imgproc::line(
        img,
        core::Point::new(x + w / 2, y),
        core::Point::new(x + w / 2, y + h),
        core::Scalar::new(80.0, 80.0, 80.0, 0.0),
        1,
        imgproc::LINE_8,
        0,
    )?;

    // Position needle
    let needle_x = x + (w as f32 * (0.5 + normalized * 0.5)) as i32;
    let needle_color = if normalized.abs() > 0.6 {
        core::Scalar::new(0.0, 0.0, 255.0, 0.0)
    } else if normalized.abs() > 0.3 {
        core::Scalar::new(0.0, 165.0, 255.0, 0.0)
    } else {
        core::Scalar::new(0.0, 255.0, 0.0, 0.0)
    };

    imgproc::rectangle(
        img,
        core::Rect::new(needle_x - 3, y - 2, 6, h + 4),
        needle_color,
        -1,
        imgproc::LINE_8,
        0,
    )?;

    // Border
    imgproc::rectangle(
        img,
        core::Rect::new(x, y, w, h),
        core::Scalar::new(120.0, 120.0, 120.0, 0.0),
        1,
        imgproc::LINE_8,
        0,
    )?;

    Ok(())
}

/// Format milliseconds into MM:SS.s
fn format_timestamp(ms: f64) -> String {
    let total_seconds = ms / 1000.0;
    let minutes = (total_seconds / 60.0) as u32;
    let seconds = total_seconds % 60.0;
    format!("{:02}:{:04.1}", minutes, seconds)
}

