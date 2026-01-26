// src/analysis/fallback_detector.rs
//
// FALLBACK LANE CHANGE DETECTION
// Used when primary lane detection fails (stuck/invalid)
//

use crate::types::{Direction, LaneChangeEvent};
use opencv::{
    core::{self, Mat},
    imgproc,
    prelude::*,
    video,
};
use std::collections::VecDeque;
use tracing::{info, warn};

// ============================================================================
// FALLBACK DETECTION THRESHOLDS
// ============================================================================
const POSITION_STUCK_VARIANCE_THRESHOLD: f32 = 0.0001; // Very small variance = stuck
const POSITION_STUCK_MIN_FRAMES: u32 = 15; // 0.5 seconds at 30fps

// Optical flow thresholds
const OPTICAL_FLOW_MIN_VELOCITY: f32 = 2.0; // px/frame horizontal flow
const OPTICAL_FLOW_SUSTAINED_FRAMES: usize = 15; // Must sustain for 0.5s
const OPTICAL_FLOW_MIN_DURATION_MS: f64 = 500.0; // Minimum 0.5s movement
const OPTICAL_FLOW_ACCUMULATED_THRESHOLD: f32 = 50.0; // Total px moved

// Position jump thresholds
const POSITION_JUMP_THRESHOLD: f32 = 0.30; // 30% lane width jump

// ============================================================================
// FALLBACK DETECTION METHODS
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FallbackMethod {
    OpticalFlow,
    PositionJump,
}

#[derive(Debug, Clone)]
pub struct FallbackDetection {
    pub direction: Direction,
    pub confidence: f32,
    pub method: FallbackMethod,
    pub accumulated_flow: f32,
}

impl FallbackDetection {
    pub fn to_event(
        &self,
        timestamp_ms: f64,
        start_frame: u64,
        end_frame: u64,
        source_id: String,
    ) -> LaneChangeEvent {
        let mut event = LaneChangeEvent::new(
            timestamp_ms,
            start_frame,
            end_frame,
            self.direction,
            self.confidence,
        );
        event.source_id = source_id;
        event.metadata.insert(
            "fallback_method".to_string(),
            serde_json::json!(format!("{:?}", self.method)),
        );
        event.metadata.insert(
            "accumulated_flow".to_string(),
            serde_json::json!(self.accumulated_flow),
        );
        event
    }
}

// ============================================================================
// FALLBACK LANE CHANGE DETECTOR
// ============================================================================

pub struct FallbackLaneChangeDetector {
    // Optical flow state
    prev_gray: Option<Mat>,
    flow_history: VecDeque<f32>,

    // Position stuck detection
    last_positions: VecDeque<f32>,
    stuck_frames: u32,
    last_valid_position: Option<f32>,

    // Detection state
    is_fallback_active: bool,
    fallback_start_time: Option<f64>,
    fallback_start_frame: Option<u64>,
    accumulated_lateral_flow: f32,
    peak_lateral_flow: f32,

    // Frame dimensions for optical flow ROI
    frame_width: i32,
    frame_height: i32,
}

impl FallbackLaneChangeDetector {
    pub fn new() -> Self {
        Self {
            prev_gray: None,
            flow_history: VecDeque::with_capacity(30),
            last_positions: VecDeque::with_capacity(60),
            stuck_frames: 0,
            last_valid_position: None,
            is_fallback_active: false,
            fallback_start_time: None,
            fallback_start_frame: None,
            accumulated_lateral_flow: 0.0,
            peak_lateral_flow: 0.0,
            frame_width: 0,
            frame_height: 0,
        }
    }

    /// Check if primary lane detection position is stuck
    pub fn is_position_stuck(&mut self, current_position: f32) -> bool {
        self.last_positions.push_back(current_position);
        if self.last_positions.len() > 60 {
            self.last_positions.pop_front();
        }

        if self.last_positions.len() < 30 {
            return false;
        }

        // Calculate variance of last 30 positions
        let recent: Vec<f32> = self.last_positions.iter().rev().take(30).copied().collect();
        let mean: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
        let variance: f32 =
            recent.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / recent.len() as f32;

        // Position is stuck if variance is extremely small
        let is_stuck = variance < POSITION_STUCK_VARIANCE_THRESHOLD;

        if is_stuck {
            self.stuck_frames += 1;
            if self.stuck_frames == POSITION_STUCK_MIN_FRAMES {
                warn!(
                    "⚠️  Position STUCK detected: variance={:.6}, mean={:.2}%",
                    variance,
                    mean * 100.0
                );
                self.last_valid_position = Some(mean);
            }
        } else {
            if self.stuck_frames > 0 {
                self.stuck_frames = 0;
            }
        }

        // Activate fallback if stuck for threshold frames
        self.stuck_frames >= POSITION_STUCK_MIN_FRAMES
    }

    /// Calculate lateral movement using dense optical flow
    pub fn detect_with_optical_flow(
        &mut self,
        frame_rgb: &[u8],
        frame_width: usize,
        frame_height: usize,
        frame_id: u64,
        timestamp_ms: f64,
    ) -> Option<FallbackDetection> {
        self.frame_width = frame_width as i32;
        self.frame_height = frame_height as i32;

        // Convert RGB to grayscale
        let gray_mat = match self.rgb_to_gray(frame_rgb, frame_width, frame_height) {
            Ok(mat) => mat,
            Err(e) => {
                warn!("Failed to convert to grayscale: {}", e);
                return None;
            }
        };

        if let Some(prev_gray) = &self.prev_gray {
            // Calculate dense optical flow (Farneback)
            let mut flow = Mat::default();

            let result = video::calc_optical_flow_farneback(
                prev_gray, &gray_mat, &mut flow, 0.5, // pyr_scale
                3,   // levels
                15,  // winsize
                3,   // iterations
                5,   // poly_n
                1.2, // poly_sigma
                0,   // flags
            );

            if result.is_err() {
                warn!("Optical flow calculation failed");
                self.prev_gray = Some(gray_mat);
                return None;
            }

            // Extract horizontal flow from road region (bottom 40% of frame)
            let road_start_y = (frame_height as f32 * 0.6) as i32;

            let avg_horizontal_flow =
                self.extract_horizontal_flow(&flow, road_start_y, frame_height as i32);

            // Store in history
            self.flow_history.push_back(avg_horizontal_flow);
            if self.flow_history.len() > 30 {
                self.flow_history.pop_front();
            }

            // Manage fallback state
            if !self.is_fallback_active && avg_horizontal_flow.abs() > OPTICAL_FLOW_MIN_VELOCITY {
                // Start tracking potential lane change
                self.is_fallback_active = true;
                self.fallback_start_time = Some(timestamp_ms);
                self.fallback_start_frame = Some(frame_id);
                self.accumulated_lateral_flow = 0.0;
                self.peak_lateral_flow = 0.0;
                info!(
                    "🔄 FALLBACK: Optical flow movement detected ({:.1} px/frame)",
                    avg_horizontal_flow
                );
            }

            if self.is_fallback_active {
                self.accumulated_lateral_flow += avg_horizontal_flow;
                if avg_horizontal_flow.abs() > self.peak_lateral_flow.abs() {
                    self.peak_lateral_flow = avg_horizontal_flow;
                }

                // Check if we have enough evidence
                let duration = timestamp_ms - self.fallback_start_time.unwrap_or(timestamp_ms);

                // Require sustained movement for minimum duration
                if duration >= OPTICAL_FLOW_MIN_DURATION_MS {
                    let abs_accumulated = self.accumulated_lateral_flow.abs();

                    // Threshold: accumulated flow > 50 pixels over the period
                    if abs_accumulated > OPTICAL_FLOW_ACCUMULATED_THRESHOLD {
                        // Flow direction: positive = scene moving right = vehicle moving left
                        let direction = if self.accumulated_lateral_flow > 0.0 {
                            Direction::Left
                        } else {
                            Direction::Right
                        };

                        let confidence = (abs_accumulated / 100.0).min(0.90);

                        // Reset state
                        self.is_fallback_active = false;
                        self.accumulated_lateral_flow = 0.0;

                        info!(
                            "✅ FALLBACK OPTICAL FLOW: {} detected (accumulated: {:.1}px, dur: {:.0}ms)",
                            direction.as_str(),
                            abs_accumulated,
                            duration
                        );

                        return Some(FallbackDetection {
                            direction,
                            confidence,
                            method: FallbackMethod::OpticalFlow,
                            accumulated_flow: abs_accumulated,
                        });
                    }

                    // Cancel if flow reversed or stopped
                    if duration > 2000.0 && abs_accumulated < 30.0 {
                        info!("🔄 FALLBACK: Movement stopped, cancelling");
                        self.is_fallback_active = false;
                        self.accumulated_lateral_flow = 0.0;
                    }
                }
            }
        }

        self.prev_gray = Some(gray_mat);
        None
    }

    /// Convert RGB frame to grayscale Mat
    fn rgb_to_gray(
        &self,
        frame_rgb: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Mat, opencv::Error> {
        // Create Mat from RGB data
        let rgb_mat = unsafe {
            Mat::new_rows_cols_with_data(
                height as i32,
                width as i32,
                core::CV_8UC3,
                frame_rgb.as_ptr() as *mut core::c_void,
                core::Mat_AUTO_STEP,
            )?
        };

        // Convert to grayscale
        let mut gray_mat = Mat::default();
        imgproc::cvt_color(&rgb_mat, &mut gray_mat, imgproc::COLOR_RGB2GRAY, 0)?;

        Ok(gray_mat)
    }

    /// Extract average horizontal flow component from road region
    fn extract_horizontal_flow(&self, flow: &Mat, start_y: i32, end_y: i32) -> f32 {
        let mut total_horizontal_flow = 0.0f32;
        let mut count = 0;

        let rows = flow.rows();
        let cols = flow.cols();

        let actual_start_y = start_y.max(0).min(rows);
        let actual_end_y = end_y.max(0).min(rows);

        for y in actual_start_y..actual_end_y {
            for x in 0..cols {
                // OpenCV optical flow returns 2-channel Mat (horizontal, vertical)
                if let Ok(flow_vec) = flow.at_2d::<core::Vec2f>(y, x) {
                    total_horizontal_flow += flow_vec[0]; // Horizontal component
                    count += 1;
                }
            }
        }

        if count > 0 {
            total_horizontal_flow / count as f32
        } else {
            0.0
        }
    }

    /// Detect sudden position jump after stuck period
    pub fn detect_position_jump(
        &self,
        old_position: f32,
        new_position: f32,
    ) -> Option<FallbackDetection> {
        let jump = new_position - old_position;

        // If position jumps > 30% of lane width after being stuck
        if jump.abs() > POSITION_JUMP_THRESHOLD && self.stuck_frames >= POSITION_STUCK_MIN_FRAMES {
            let direction = if jump > 0.0 {
                Direction::Right
            } else {
                Direction::Left
            };

            info!(
                "✅ FALLBACK POSITION JUMP: {} detected (jump: {:.1}% after {} stuck frames)",
                direction.as_str(),
                jump * 100.0,
                self.stuck_frames
            );

            return Some(FallbackDetection {
                direction,
                confidence: (jump.abs() / 0.5).min(0.85),
                method: FallbackMethod::PositionJump,
                accumulated_flow: jump.abs(),
            });
        }

        None
    }

    pub fn reset(&mut self) {
        self.prev_gray = None;
        self.flow_history.clear();
        self.last_positions.clear();
        self.stuck_frames = 0;
        self.last_valid_position = None;
        self.is_fallback_active = false;
        self.fallback_start_time = None;
        self.fallback_start_frame = None;
        self.accumulated_lateral_flow = 0.0;
        self.peak_lateral_flow = 0.0;
    }

    pub fn is_stuck(&self) -> bool {
        self.stuck_frames >= POSITION_STUCK_MIN_FRAMES
    }

    pub fn get_last_valid_position(&self) -> Option<f32> {
        self.last_valid_position
    }
}
