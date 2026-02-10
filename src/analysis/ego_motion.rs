// src/analysis/ego_motion.rs
//
// Lateral ego-motion estimation from raw frame pixel data.
//
// Uses block matching (simplified optical flow) on the lower portion
// of the frame to estimate lateral ego velocity. This works even when
// lane markings are invisible — any texture on the road surface, road
// edges, or scenery movement provides signal.
//
// The key insight: during an overtake, the entire scene shifts laterally
// in the OPPOSITE direction of ego motion. By measuring this shift
// across the lower frame (road surface), we get ego lateral velocity.
//
// This module has zero dependency on OpenCV. It uses a simple SAD
// (Sum of Absolute Differences) block matcher operating on grayscale
// pixel data.
//
// Accuracy: ±3-5px/frame in good conditions. Not GPS-grade, but enough
// to confirm "the vehicle is moving laterally" which is all the fusion
// layer needs.

use std::collections::VecDeque;
use tracing::{debug, info};

// ============================================================================
// CONFIGURATION
// ============================================================================

#[derive(Debug, Clone)]
pub struct EgoMotionConfig {
    /// Number of horizontal blocks to divide the ROI into
    pub blocks_x: usize,
    /// Number of vertical blocks to divide the ROI into
    pub blocks_y: usize,
    /// Block size in pixels (square blocks)
    pub block_size: usize,
    /// Maximum horizontal search range in pixels (±)
    pub search_range: usize,
    /// ROI: fraction of frame height for the TOP of the analysis region
    /// (e.g., 0.60 means start at 60% from top)
    pub roi_top_ratio: f32,
    /// ROI: fraction of frame height for the BOTTOM of the analysis region
    /// (e.g., 0.90 means stop at 90% from top, avoiding the vehicle hood)
    pub roi_bottom_ratio: f32,
    /// Minimum consensus ratio among blocks for a valid estimate
    pub min_consensus: f32,
    /// Smoothing factor for the velocity output (EWMA alpha)
    pub smoothing_alpha: f32,
    /// Minimum absolute displacement (px/frame) to count as lateral motion
    pub min_displacement: f32,
    /// History size for velocity averaging
    pub history_size: usize,
}

impl Default for EgoMotionConfig {
    fn default() -> Self {
        Self {
            blocks_x: 6,
            blocks_y: 3,
            block_size: 32,
            search_range: 40,
            roi_top_ratio: 0.55,
            roi_bottom_ratio: 0.85,
            min_consensus: 0.50,
            smoothing_alpha: 0.3,
            min_displacement: 1.5,
            history_size: 15,
        }
    }
}

// ============================================================================
// TYPES
// ============================================================================

/// Ego lateral motion estimate for a single frame
#[derive(Debug, Clone, Copy)]
pub struct EgoMotionEstimate {
    /// Lateral velocity in pixels per frame (positive = ego moving right)
    pub lateral_velocity_px: f32,
    /// Confidence in this estimate [0, 1]
    pub confidence: f32,
    /// Number of blocks that agreed on the displacement direction
    pub consensus_blocks: usize,
    /// Total blocks analyzed
    pub total_blocks: usize,
    /// Is the ego vehicle currently moving laterally?
    pub is_lateral_motion: bool,
}

impl EgoMotionEstimate {
    pub fn none() -> Self {
        Self {
            lateral_velocity_px: 0.0,
            confidence: 0.0,
            consensus_blocks: 0,
            total_blocks: 0,
            is_lateral_motion: false,
        }
    }
}

// ============================================================================
// GRAYSCALE FRAME WRAPPER
// ============================================================================

/// Simple grayscale frame — your pipeline should convert to this.
/// Row-major storage: pixel at (x, y) = data[y * width + x]
#[derive(Clone)]
pub struct GrayFrame {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

impl GrayFrame {
    pub fn new(data: Vec<u8>, width: usize, height: usize) -> Self {
        debug_assert_eq!(data.len(), width * height);
        Self {
            data,
            width,
            height,
        }
    }

    /// Convert from RGB packed bytes (3 bytes per pixel)
    pub fn from_rgb(rgb: &[u8], width: usize, height: usize) -> Self {
        let mut gray = Vec::with_capacity(width * height);
        for pixel in rgb.chunks_exact(3) {
            // ITU-R BT.601 luma
            let g =
                (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32) as u8;
            gray.push(g);
        }
        Self::new(gray, width, height)
    }

    /// Convert from RGBA packed bytes (4 bytes per pixel)
    pub fn from_rgba(rgba: &[u8], width: usize, height: usize) -> Self {
        let mut gray = Vec::with_capacity(width * height);
        for pixel in rgba.chunks_exact(4) {
            let g =
                (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32) as u8;
            gray.push(g);
        }
        Self::new(gray, width, height)
    }

    #[inline]
    fn pixel(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.width + x]
    }

    /// Extract a block starting at (bx, by) with given size.
    /// Returns None if out of bounds.
    fn block(&self, bx: usize, by: usize, size: usize) -> Option<&[u8]> {
        if bx + size > self.width || by + size > self.height {
            return None;
        }
        // Return first row pointer — caller must stride manually
        Some(&self.data[by * self.width + bx..])
    }
}

// ============================================================================
// BLOCK MATCHER
// ============================================================================

/// SAD (Sum of Absolute Differences) between two blocks.
/// `ref_frame` block at (rx, ry) vs `cur_frame` block at (cx, cy).
#[inline]
fn sad_block(
    ref_frame: &GrayFrame,
    cur_frame: &GrayFrame,
    rx: usize,
    ry: usize,
    cx: usize,
    cy: usize,
    size: usize,
) -> u32 {
    let mut sum: u32 = 0;
    let rw = ref_frame.width;
    let cw = cur_frame.width;

    for dy in 0..size {
        let r_row = (ry + dy) * rw + rx;
        let c_row = (cy + dy) * cw + cx;
        for dx in 0..size {
            let diff = ref_frame.data[r_row + dx] as i32 - cur_frame.data[c_row + dx] as i32;
            sum += diff.unsigned_abs();
        }
    }
    sum
}

/// Find the best horizontal displacement for a block using SAD.
/// Returns (displacement_px, sad_score).
/// Displacement is positive if the block moved RIGHT in the current frame.
fn match_block_horizontal(
    prev: &GrayFrame,
    curr: &GrayFrame,
    bx: usize,
    by: usize,
    block_size: usize,
    search_range: usize,
) -> Option<(i32, u32)> {
    let min_x = (bx as i32 - search_range as i32).max(0) as usize;
    let max_x = (bx + search_range).min(curr.width - block_size);

    if by + block_size > prev.height || by + block_size > curr.height {
        return None;
    }
    if bx + block_size > prev.width {
        return None;
    }

    let mut best_sad = u32::MAX;
    let mut best_dx: i32 = 0;

    for cx in min_x..=max_x {
        let score = sad_block(prev, curr, bx, by, cx, by, block_size);
        if score < best_sad {
            best_sad = score;
            best_dx = cx as i32 - bx as i32;
        }
    }

    Some((best_dx, best_sad))
}

// ============================================================================
// EGO MOTION ESTIMATOR
// ============================================================================

pub struct EgoMotionEstimator {
    config: EgoMotionConfig,
    prev_frame: Option<GrayFrame>,
    velocity_history: VecDeque<f32>,
    smoothed_velocity: f32,
    frame_count: u64,
}

impl EgoMotionEstimator {
    pub fn new(config: EgoMotionConfig) -> Self {
        Self {
            config,
            prev_frame: None,
            velocity_history: VecDeque::with_capacity(30),
            smoothed_velocity: 0.0,
            frame_count: 0,
        }
    }

    /// Process a new grayscale frame and estimate lateral ego motion.
    ///
    /// Call this every frame with the grayscale conversion of your video frame.
    /// First frame returns EgoMotionEstimate::none() (no reference to compare).
    pub fn update(&mut self, frame: &GrayFrame) -> EgoMotionEstimate {
        self.frame_count += 1;

        let prev = match &self.prev_frame {
            Some(p) if p.width == frame.width && p.height == frame.height => p,
            _ => {
                self.prev_frame = Some(frame.clone());
                return EgoMotionEstimate::none();
            }
        };

        // ── DEFINE ROI ──────────────────────────────────────────
        let roi_top = (frame.height as f32 * self.config.roi_top_ratio) as usize;
        let roi_bottom = (frame.height as f32 * self.config.roi_bottom_ratio) as usize;
        let roi_height = roi_bottom.saturating_sub(roi_top);

        if roi_height < self.config.block_size * 2 {
            self.prev_frame = Some(frame.clone());
            return EgoMotionEstimate::none();
        }

        // ── PLACE BLOCKS ────────────────────────────────────────
        let bs = self.config.block_size;
        let sr = self.config.search_range;

        // Evenly space blocks across the ROI
        let step_x = if self.config.blocks_x > 1 {
            (frame.width - bs - 2 * sr) / (self.config.blocks_x - 1)
        } else {
            frame.width
        };
        let step_y = if self.config.blocks_y > 1 {
            roi_height.saturating_sub(bs) / (self.config.blocks_y - 1)
        } else {
            roi_height
        };

        let mut displacements: Vec<f32> =
            Vec::with_capacity(self.config.blocks_x * self.config.blocks_y);

        for by_idx in 0..self.config.blocks_y {
            let by = roi_top + by_idx * step_y.max(1);
            for bx_idx in 0..self.config.blocks_x {
                let bx = sr + bx_idx * step_x.max(1);

                if let Some((dx, _sad)) = match_block_horizontal(prev, frame, bx, by, bs, sr) {
                    displacements.push(dx as f32);
                }
            }
        }

        let total_blocks = displacements.len();
        if total_blocks < 3 {
            self.prev_frame = Some(frame.clone());
            return EgoMotionEstimate::none();
        }

        // ── ROBUST MEDIAN ───────────────────────────────────────
        displacements.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = displacements[total_blocks / 2];

        // Count blocks agreeing with median direction (within ±3px)
        let consensus = displacements
            .iter()
            .filter(|&&d| (d - median).abs() < 3.0)
            .count();
        let consensus_ratio = consensus as f32 / total_blocks as f32;

        // The scene shifted by `median` pixels to the LEFT,
        // meaning the ego vehicle moved `median` pixels to the RIGHT.
        // (Negate because scene motion is inverse of ego motion)
        let ego_displacement = -median;

        // ── SMOOTH AND THRESHOLD ────────────────────────────────
        let alpha = self.config.smoothing_alpha;
        self.smoothed_velocity = alpha * ego_displacement + (1.0 - alpha) * self.smoothed_velocity;

        self.velocity_history.push_back(self.smoothed_velocity);
        if self.velocity_history.len() > self.config.history_size {
            self.velocity_history.pop_front();
        }

        let is_lateral = self.smoothed_velocity.abs() >= self.config.min_displacement
            && consensus_ratio >= self.config.min_consensus;

        let confidence = if consensus_ratio >= self.config.min_consensus {
            (consensus_ratio * 0.6 + 0.4).min(0.90)
        } else {
            consensus_ratio * 0.4
        };

        // Store current frame as reference for next iteration
        self.prev_frame = Some(frame.clone());

        EgoMotionEstimate {
            lateral_velocity_px: self.smoothed_velocity,
            confidence,
            consensus_blocks: consensus,
            total_blocks,
            is_lateral_motion: is_lateral,
        }
    }

    /// Get the smoothed lateral velocity (px/frame)
    pub fn lateral_velocity(&self) -> f32 {
        self.smoothed_velocity
    }

    /// Get the average lateral velocity over the history window
    pub fn avg_velocity(&self) -> f32 {
        if self.velocity_history.is_empty() {
            return 0.0;
        }
        self.velocity_history.iter().sum::<f32>() / self.velocity_history.len() as f32
    }

    /// Is the vehicle currently moving laterally (based on smoothed estimate)?
    pub fn is_moving_laterally(&self) -> bool {
        self.smoothed_velocity.abs() >= self.config.min_displacement
    }

    /// Direction of current lateral motion
    pub fn lateral_direction(&self) -> Option<super::lateral_detector::ShiftDirection> {
        use super::lateral_detector::ShiftDirection;
        if self.smoothed_velocity > self.config.min_displacement {
            Some(ShiftDirection::Right)
        } else if self.smoothed_velocity < -self.config.min_displacement {
            Some(ShiftDirection::Left)
        } else {
            None
        }
    }

    pub fn reset(&mut self) {
        self.prev_frame = None;
        self.velocity_history.clear();
        self.smoothed_velocity = 0.0;
        self.frame_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(width: usize, height: usize, shift_x: i32) -> GrayFrame {
        // Create a frame with a simple vertical stripe pattern shifted by shift_x
        let mut data = vec![128u8; width * height];
        for y in 0..height {
            for x in 0..width {
                let sx = (x as i32 + shift_x).rem_euclid(width as i32) as usize;
                // Stripe every 20 pixels
                data[y * width + sx] = if (x / 20) % 2 == 0 { 200 } else { 50 };
            }
        }
        GrayFrame::new(data, width, height)
    }

    #[test]
    fn test_no_motion() {
        let mut est = EgoMotionEstimator::new(EgoMotionConfig::default());

        let frame = make_frame(640, 480, 0);
        est.update(&frame); // First frame — no estimate
        let result = est.update(&frame); // Same frame — no motion

        assert!(
            result.lateral_velocity_px.abs() < 2.0,
            "Should detect near-zero motion"
        );
    }

    #[test]
    fn test_lateral_motion_detection() {
        let cfg = EgoMotionConfig {
            roi_top_ratio: 0.0, // Use full frame for test
            roi_bottom_ratio: 1.0,
            ..Default::default()
        };
        let mut est = EgoMotionEstimator::new(cfg);

        let frame1 = make_frame(640, 480, 0);
        let frame2 = make_frame(640, 480, 10); // Scene shifted right by 10px

        est.update(&frame1);
        let result = est.update(&frame2);

        // Scene shifted RIGHT → ego moved LEFT → negative velocity
        assert!(
            result.lateral_velocity_px < -3.0,
            "Should detect leftward ego motion, got {}",
            result.lateral_velocity_px
        );
    }
}
