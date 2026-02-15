// src/road_overlay.rs
//
// v6.0: Zone-Based Road Surface Visualization
//
// ════════════════════════════════════════════════════════════════════════════
// OVERVIEW
// ════════════════════════════════════════════════════════════════════════════
//
// Instead of just drawing line segments and bboxes, this module renders
// the actual road surface with semantically meaningful zones:
//
//   ┌─────────────────────────────────────────────┐
//   │                  SKY / SCENE                │
//   │─────────────────────────────────────────────│
//   │  OPPOSING    ║ ←mask │ ←mask ║   SHOULDER   │
//   │   LANE       ║  RED  │ GREEN ║              │
//   │  (red tint)  ║       │       ║              │
//   │              ║  SOLID│DASHED ║              │
//   │              ║       │       ║              │
//   │    ┌─────────║───────│───────║──────────┐   │
//   │    │         ║       │       ║          │   │
//   │    │   EGO LANE (teal/green tint)       │   │
//   │    │    ◇ vehicle center                │   │
//   │    └─────────║───────│───────║──────────┘   │
//   └─────────────────────────────────────────────┘
//
// The key visual elements:
//
//   1. MASK PROJECTION — Each YOLO-seg mask is projected onto the road as
//      a colored overlay. Solid lines glow red, dashed glow green, mixed
//      lines have a split red/green glow.
//
//   2. LEGALITY ZONE STRIPS — Wide (30-40px) semi-transparent strips along
//      each lane marking, color-coded by crossing legality:
//        • Green: legal to cross (dashed lines)
//        • Red: illegal to cross (solid, double-solid)
//        • Split red/green: mixed line (depends on side)
//        • Grey: unknown legality
//
//   3. OPPOSING LANE TINT — When center markings indicate no-passing, the
//      opposing lane (left of center) gets a faint red tint as a warning.
//
//   4. CROSSING FLASH — When LineCrossingDetector fires, the crossed marking
//      pulses bright with an animated ripple effect for ~15 frames.
//
//   5. CACHE STATE BADGE — Shows whether lane data is fresh from YOLO,
//      cached (with age), or expired (trocha/unmarked road).
//
//   6. MIXED LINE INDICATOR — Arrows showing which side is dashed
//      (→ PERMITIDO) vs solid (→ PROHIBIDO).

use anyhow::Result;
use opencv::{
    core::{self, Mat, Vector},
    imgproc,
    prelude::*,
};

use crate::lane_crossing::{CacheState, LineCrossingEvent, LineRole};
use crate::lane_legality::{DetectedRoadMarking, LineLegality};
use crate::road_classification::PassingLegality;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Colors used for road zone rendering (BGR format for OpenCV).
pub mod colors {
    use opencv::core::Scalar;

    // Legality zone strip colors
    pub const LEGAL_GREEN: Scalar = Scalar::new(0.0, 200.0, 60.0, 0.0);
    pub const ILLEGAL_RED: Scalar = Scalar::new(0.0, 40.0, 220.0, 0.0);
    pub const MIXED_ALLOWED: Scalar = Scalar::new(0.0, 180.0, 80.0, 0.0);
    pub const MIXED_PROHIBITED: Scalar = Scalar::new(0.0, 60.0, 200.0, 0.0);
    pub const UNKNOWN_GREY: Scalar = Scalar::new(120.0, 120.0, 120.0, 0.0);

    // Mask overlay colors
    pub const MASK_SOLID_RED: Scalar = Scalar::new(60.0, 50.0, 200.0, 0.0);
    pub const MASK_DASHED_GREEN: Scalar = Scalar::new(60.0, 200.0, 60.0, 0.0);
    pub const MASK_DOUBLE_RED: Scalar = Scalar::new(40.0, 30.0, 240.0, 0.0);
    pub const MASK_MIXED_YELLOW: Scalar = Scalar::new(0.0, 180.0, 240.0, 0.0);
    pub const MASK_DEFAULT: Scalar = Scalar::new(180.0, 160.0, 100.0, 0.0);

    // Zone tints
    pub const EGO_LANE_NORMAL: Scalar = Scalar::new(160.0, 110.0, 0.0, 0.0);
    pub const EGO_LANE_VIOLATION: Scalar = Scalar::new(0.0, 0.0, 180.0, 0.0);
    pub const OPPOSING_LANE_DANGER: Scalar = Scalar::new(0.0, 20.0, 100.0, 0.0);

    // Crossing flash
    pub const CROSSING_FLASH_LEGAL: Scalar = Scalar::new(100.0, 255.0, 100.0, 0.0);
    pub const CROSSING_FLASH_ILLEGAL: Scalar = Scalar::new(80.0, 80.0, 255.0, 0.0);

    // Cache state badge
    pub const BADGE_FRESH: Scalar = Scalar::new(0.0, 200.0, 0.0, 0.0);
    pub const BADGE_CACHED: Scalar = Scalar::new(0.0, 180.0, 240.0, 0.0);
    pub const BADGE_EXPIRED: Scalar = Scalar::new(0.0, 100.0, 200.0, 0.0);
    pub const BADGE_EMPTY: Scalar = Scalar::new(100.0, 100.0, 100.0, 0.0);
}

/// Alpha (transparency) values for different overlay layers.
pub mod alpha {
    pub const MASK_OVERLAY: f64 = 0.25;
    pub const ZONE_STRIP: f64 = 0.18;
    pub const EGO_LANE_FILL: f64 = 0.12;
    pub const OPPOSING_LANE_TINT: f64 = 0.10;
    pub const CROSSING_FLASH_PEAK: f64 = 0.50;
}

// ============================================================================
// 1. YOLO SEGMENTATION MASK PROJECTION
// ============================================================================

/// Project a YOLO-seg 160x160 mask onto the output frame as a colored overlay.
///
/// Maps each active pixel in the mask to the corresponding region in the
/// original image (using bbox + letterbox transform params) and fills it
/// with a legality-colored tint.
pub fn render_mask_overlay(
    output: &mut Mat,
    marking: &DetectedRoadMarking,
    orig_w: i32,
    orig_h: i32,
) -> Result<()> {
    if marking.mask.is_empty() || marking.mask_width == 0 || marking.mask_height == 0 {
        return Ok(());
    }

    let mask_w = marking.mask_width;
    let mask_h = marking.mask_height;

    // Compute letterbox transform params for mask → image mapping
    let target = 640.0f32;
    let scale = (target / orig_w as f32).min(target / orig_h as f32);
    let pad_x = (target - orig_w as f32 * scale) / 2.0;
    let pad_y = (target - orig_h as f32 * scale) / 2.0;
    let ratio = mask_w as f32 / target; // 160/640 = 0.25

    // Choose color based on marking legality
    let fill_color = mask_color_for_legality(&marking.legality);

    // Create overlay for alpha blending
    let mut overlay = output.try_clone()?;

    // Iterate mask pixels and project to image space
    let bbox = &marking.bbox;
    // Compute mask region corresponding to bbox
    let mask_x1 = ((bbox[0] * scale + pad_x) * ratio).floor().max(0.0) as usize;
    let mask_y1 = ((bbox[1] * scale + pad_y) * ratio).floor().max(0.0) as usize;
    let mask_x2 = ((bbox[2] * scale + pad_x) * ratio)
        .ceil()
        .min(mask_w as f32 - 1.0) as usize;
    let mask_y2 = ((bbox[3] * scale + pad_y) * ratio)
        .ceil()
        .min(mask_h as f32 - 1.0) as usize;

    // For efficiency, build horizontal runs of active pixels per image row
    // and draw them as thin filled rectangles
    for my in mask_y1..=mask_y2 {
        let mut run_start: Option<usize> = None;

        for mx in mask_x1..=mask_x2 + 1 {
            let is_active = if mx <= mask_x2 {
                let idx = my * mask_w + mx;
                idx < marking.mask.len() && marking.mask[idx] == 255
            } else {
                false
            };

            if is_active && run_start.is_none() {
                run_start = Some(mx);
            } else if !is_active && run_start.is_some() {
                let rs = run_start.unwrap();
                // Map run endpoints to image coordinates
                let img_x1 = ((rs as f32 / ratio - pad_x) / scale) as i32;
                let img_x2 = ((mx as f32 / ratio - pad_x) / scale) as i32;
                let img_y = ((my as f32 / ratio - pad_y) / scale) as i32;
                let next_y = (((my + 1) as f32 / ratio - pad_y) / scale) as i32;

                // Clamp to image bounds
                let x1 = img_x1.max(0).min(orig_w - 1);
                let x2 = img_x2.max(0).min(orig_w);
                let y1 = img_y.max(0).min(orig_h - 1);
                let y2 = next_y.max(y1 + 1).min(orig_h);

                if x2 > x1 && y2 > y1 {
                    imgproc::rectangle(
                        &mut overlay,
                        core::Rect::new(x1, y1, x2 - x1, y2 - y1),
                        fill_color,
                        -1,
                        imgproc::LINE_8,
                        0,
                    )?;
                }
                run_start = None;
            }
        }
    }

    // Alpha blend overlay onto output
    let mut blended = Mat::default();
    core::add_weighted(
        &overlay,
        alpha::MASK_OVERLAY,
        output,
        1.0 - alpha::MASK_OVERLAY,
        0.0,
        &mut blended,
        -1,
    )?;
    blended.copy_to(output)?;

    Ok(())
}

/// Render all marking masks in a single pass (batched for efficiency).
pub fn render_all_mask_overlays(
    output: &mut Mat,
    markings: &[DetectedRoadMarking],
    orig_w: i32,
    orig_h: i32,
) -> Result<()> {
    // Build a single overlay with all masks, then blend once (much faster)
    if markings.is_empty() {
        return Ok(());
    }

    let target = 640.0f32;
    let scale = (target / orig_w as f32).min(target / orig_h as f32);
    let pad_x = (target - orig_w as f32 * scale) / 2.0;
    let pad_y = (target - orig_h as f32 * scale) / 2.0;
    let ratio = markings[0].mask_width as f32 / target;

    let mut overlay = output.try_clone()?;
    let mut any_rendered = false;

    for marking in markings {
        if marking.mask.is_empty() || marking.mask_width == 0 {
            continue;
        }

        let mask_w = marking.mask_width;
        let fill_color = mask_color_for_legality(&marking.legality);
        let bbox = &marking.bbox;

        let mask_x1 = ((bbox[0] * scale + pad_x) * ratio).floor().max(0.0) as usize;
        let mask_y1 = ((bbox[1] * scale + pad_y) * ratio).floor().max(0.0) as usize;
        let mask_x2 = ((bbox[2] * scale + pad_x) * ratio)
            .ceil()
            .min(mask_w as f32 - 1.0) as usize;
        let mask_y2 = ((bbox[3] * scale + pad_y) * ratio)
            .ceil()
            .min(marking.mask_height as f32 - 1.0) as usize;

        for my in mask_y1..=mask_y2 {
            let mut run_start: Option<usize> = None;
            for mx in mask_x1..=mask_x2 + 1 {
                let is_active = if mx <= mask_x2 {
                    let idx = my * mask_w + mx;
                    idx < marking.mask.len() && marking.mask[idx] == 255
                } else {
                    false
                };

                if is_active && run_start.is_none() {
                    run_start = Some(mx);
                } else if !is_active && run_start.is_some() {
                    let rs = run_start.unwrap();
                    let img_x1 = ((rs as f32 / ratio - pad_x) / scale) as i32;
                    let img_x2 = ((mx as f32 / ratio - pad_x) / scale) as i32;
                    let img_y = ((my as f32 / ratio - pad_y) / scale) as i32;
                    let next_y = (((my + 1) as f32 / ratio - pad_y) / scale) as i32;

                    let x1 = img_x1.max(0).min(orig_w - 1);
                    let x2 = img_x2.max(0).min(orig_w);
                    let y1 = img_y.max(0).min(orig_h - 1);
                    let y2 = next_y.max(y1 + 1).min(orig_h);

                    if x2 > x1 && y2 > y1 {
                        imgproc::rectangle(
                            &mut overlay,
                            core::Rect::new(x1, y1, x2 - x1, y2 - y1),
                            fill_color,
                            -1,
                            imgproc::LINE_8,
                            0,
                        )?;
                        any_rendered = true;
                    }
                    run_start = None;
                }
            }
        }
    }

    if any_rendered {
        let mut blended = Mat::default();
        core::add_weighted(
            &overlay,
            alpha::MASK_OVERLAY,
            output,
            1.0 - alpha::MASK_OVERLAY,
            0.0,
            &mut blended,
            -1,
        )?;
        blended.copy_to(output)?;
    }

    Ok(())
}

// ============================================================================
// 2. LEGALITY ZONE STRIPS
// ============================================================================

/// Draw a wide semi-transparent strip along a lane marking to indicate
/// whether crossing it is legal or illegal.
///
/// The strip extends horizontally from the marking's bbox edge outward
/// (toward the ego lane), creating a visual "zone" effect on the road.
pub fn render_legality_zone_strip(
    output: &mut Mat,
    marking: &DetectedRoadMarking,
    ego_center_x: f32,
    _orig_w: i32,
    orig_h: i32,
    strip_width_px: i32,
) -> Result<()> {
    let bbox = &marking.bbox;
    let marking_cx = (bbox[0] + bbox[2]) / 2.0;

    // Strip extends from the marking bbox edge toward ego center
    let (strip_x, strip_w) = if marking_cx < ego_center_x {
        // Marking is to the left — strip extends rightward from bbox right edge
        (bbox[2] as i32, strip_width_px)
    } else {
        // Marking is to the right — strip extends leftward from bbox left edge
        ((bbox[0] as i32 - strip_width_px).max(0), strip_width_px)
    };

    let strip_y = bbox[1] as i32;
    let strip_h = ((bbox[3] - bbox[1]) as i32).min(orig_h - strip_y);

    if strip_w <= 0 || strip_h <= 0 {
        return Ok(());
    }

    let strip_color = legality_strip_color(&marking.legality);

    let mut overlay = output.try_clone()?;
    imgproc::rectangle(
        &mut overlay,
        core::Rect::new(strip_x, strip_y, strip_w, strip_h),
        strip_color,
        -1,
        imgproc::LINE_8,
        0,
    )?;

    let mut blended = Mat::default();
    core::add_weighted(
        &overlay,
        alpha::ZONE_STRIP,
        output,
        1.0 - alpha::ZONE_STRIP,
        0.0,
        &mut blended,
        -1,
    )?;
    blended.copy_to(output)?;

    Ok(())
}

/// Render legality zone strips for all markings (batched).
pub fn render_all_legality_strips(
    output: &mut Mat,
    markings: &[DetectedRoadMarking],
    ego_center_x: f32,
    orig_w: i32,
    orig_h: i32,
) -> Result<()> {
    if markings.is_empty() {
        return Ok(());
    }

    let strip_width = (orig_w as f32 * 0.025).max(20.0).min(50.0) as i32; // ~30px at 1280w

    let mut overlay = output.try_clone()?;
    let mut any_drawn = false;

    for marking in markings {
        let bbox = &marking.bbox;
        let marking_cx = (bbox[0] + bbox[2]) / 2.0;

        let (strip_x, strip_w) = if marking_cx < ego_center_x {
            (bbox[2] as i32, strip_width)
        } else {
            ((bbox[0] as i32 - strip_width).max(0), strip_width)
        };

        let strip_y = bbox[1] as i32;
        let strip_h = ((bbox[3] - bbox[1]) as i32).min(orig_h - strip_y);

        if strip_w <= 0 || strip_h <= 0 {
            continue;
        }

        let strip_color = legality_strip_color(&marking.legality);
        imgproc::rectangle(
            &mut overlay,
            core::Rect::new(strip_x, strip_y, strip_w, strip_h),
            strip_color,
            -1,
            imgproc::LINE_8,
            0,
        )?;
        any_drawn = true;
    }

    if any_drawn {
        let mut blended = Mat::default();
        core::add_weighted(
            &overlay,
            alpha::ZONE_STRIP,
            output,
            1.0 - alpha::ZONE_STRIP,
            0.0,
            &mut blended,
            -1,
        )?;
        blended.copy_to(output)?;
    }

    Ok(())
}

// ============================================================================
// 3. OPPOSING LANE DANGER TINT
// ============================================================================

/// Tint the opposing lane (left of the leftmost marking) with a faint red
/// when passing is prohibited.
///
/// This creates a visual "danger zone" effect that makes it immediately
/// obvious that crossing into the opposing lane is illegal.
pub fn render_opposing_lane_tint(
    output: &mut Mat,
    left_boundary_x: f32,
    orig_w: i32,
    orig_h: i32,
    passing_legality: &PassingLegality,
) -> Result<()> {
    // Only tint when passing is clearly prohibited
    if !passing_legality.is_illegal() {
        return Ok(());
    }

    let tint_x = 0;
    let tint_w = (left_boundary_x as i32 - 15).max(0);
    // Start from horizon area (~40% from top) to bottom
    let tint_y = (orig_h as f32 * 0.40) as i32;
    let tint_h = orig_h - tint_y;

    if tint_w <= 0 || tint_h <= 0 {
        return Ok(());
    }

    // Build a trapezoidal tint (wider at bottom, narrower at top) for perspective
    let mut overlay = output.try_clone()?;

    // Use a polygon: narrow at top-left, wider at bottom-left
    let top_right = (tint_w as f32 * 0.7) as i32; // Perspective narrowing at horizon
    let poly_pts = vec![
        core::Point::new(tint_x, tint_y),
        core::Point::new(top_right, tint_y),
        core::Point::new(tint_w, orig_h),
        core::Point::new(tint_x, orig_h),
    ];

    let mut pts_vec = Vector::<Vector<core::Point>>::new();
    pts_vec.push(Vector::from_iter(poly_pts.into_iter()));
    imgproc::fill_poly(
        &mut overlay,
        &pts_vec,
        colors::OPPOSING_LANE_DANGER,
        imgproc::LINE_8,
        0,
        core::Point::new(0, 0),
    )?;

    let mut blended = Mat::default();
    core::add_weighted(
        &overlay,
        alpha::OPPOSING_LANE_TINT,
        output,
        1.0 - alpha::OPPOSING_LANE_TINT,
        0.0,
        &mut blended,
        -1,
    )?;
    blended.copy_to(output)?;

    Ok(())
}

// ============================================================================
// 4. CROSSING FLASH EFFECT
// ============================================================================

/// State for the crossing flash animation.
pub struct CrossingFlashState {
    /// Frame when the crossing was detected
    pub trigger_frame: u64,
    /// Duration of the flash in frames
    pub duration_frames: u32,
    /// The line role that was crossed
    pub line_role: LineRole,
    /// Whether crossing was illegal
    pub was_illegal: bool,
    /// The marking's bbox for positioning
    pub marking_bbox: [f32; 4],
}

impl CrossingFlashState {
    pub fn new(event: &LineCrossingEvent, bbox: [f32; 4]) -> Self {
        Self {
            trigger_frame: event.frame_id,
            duration_frames: 18, // ~600ms at 30fps
            line_role: event.line_role,
            was_illegal: event.passing_legality.is_illegal(),
            marking_bbox: bbox,
        }
    }

    /// Returns the animation progress [0.0, 1.0] where 0 = just started, 1 = done.
    pub fn progress(&self, current_frame: u64) -> f32 {
        let elapsed = current_frame.saturating_sub(self.trigger_frame) as f32;
        (elapsed / self.duration_frames as f32).min(1.0)
    }

    pub fn is_active(&self, current_frame: u64) -> bool {
        self.progress(current_frame) < 1.0
    }
}

/// Render the crossing flash effect.
///
/// Creates a pulsing glow along the crossed marking that fades over time.
/// Red for illegal crossings, green for legal.
pub fn render_crossing_flash(
    output: &mut Mat,
    flash: &CrossingFlashState,
    current_frame: u64,
    orig_w: i32,
    orig_h: i32,
) -> Result<()> {
    if !flash.is_active(current_frame) {
        return Ok(());
    }

    let progress = flash.progress(current_frame);
    // Intensity: peaks at ~0.2 progress, then fades out
    let intensity = if progress < 0.2 {
        progress / 0.2
    } else {
        1.0 - (progress - 0.2) / 0.8
    };

    let flash_alpha = alpha::CROSSING_FLASH_PEAK * intensity as f64;
    if flash_alpha < 0.02 {
        return Ok(());
    }

    let bbox = &flash.marking_bbox;
    let flash_color = if flash.was_illegal {
        colors::CROSSING_FLASH_ILLEGAL
    } else {
        colors::CROSSING_FLASH_LEGAL
    };

    // Expand the bbox outward for the glow effect
    let expand = (30.0 * intensity) as i32;
    let x1 = (bbox[0] as i32 - expand).max(0);
    let y1 = (bbox[1] as i32 - expand / 3).max(0);
    let x2 = (bbox[2] as i32 + expand).min(orig_w);
    let y2 = (bbox[3] as i32 + expand / 3).min(orig_h);

    let mut overlay = output.try_clone()?;
    imgproc::rectangle(
        &mut overlay,
        core::Rect::new(x1, y1, x2 - x1, y2 - y1),
        flash_color,
        -1,
        imgproc::LINE_8,
        0,
    )?;

    let mut blended = Mat::default();
    core::add_weighted(
        &overlay,
        flash_alpha,
        output,
        1.0 - flash_alpha,
        0.0,
        &mut blended,
        -1,
    )?;
    blended.copy_to(output)?;

    // Draw a bright border pulse on top
    let border_thickness = (3.0 * intensity) as i32 + 1;
    imgproc::rectangle(
        output,
        core::Rect::new(x1, y1, x2 - x1, y2 - y1),
        flash_color,
        border_thickness,
        imgproc::LINE_AA,
        0,
    )?;

    Ok(())
}

// ============================================================================
// 5. CACHE STATE BADGE
// ============================================================================

/// Render a compact badge showing the lane detection cache state.
///
/// Positioned in the top-right area, shows:
///   🟢 LIVE       — fresh YOLO detection this frame
///   🟡 CACHED 3f  — using cached data, 3 frames old
///   🔴 TROCHA     — cache expired, no lane markings
///   ⚪ NO DATA    — never detected lanes
pub fn render_cache_badge(
    output: &mut Mat,
    cache_state: CacheState,
    stale_frames: u32,
    badge_x: i32,
    badge_y: i32,
) -> Result<()> {
    let (label, dot_color) = match cache_state {
        CacheState::Fresh => ("LIVE".to_string(), colors::BADGE_FRESH),
        CacheState::Cached => (format!("CACHED {}f", stale_frames), colors::BADGE_CACHED),
        CacheState::Expired => ("TROCHA".to_string(), colors::BADGE_EXPIRED),
        CacheState::Empty => ("NO DATA".to_string(), colors::BADGE_EMPTY),
    };

    // Measure text width for dynamic badge sizing
    let mut baseline = 0;
    let text_size = imgproc::get_text_size(
        &label,
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.42,
        1,
        &mut baseline,
    )?;

    let badge_w = text_size.width + 30;
    let badge_h = 22;

    // Background
    let mut overlay = output.try_clone()?;
    imgproc::rectangle(
        &mut overlay,
        core::Rect::new(badge_x, badge_y, badge_w, badge_h),
        core::Scalar::new(20.0, 20.0, 20.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;
    let mut blended = Mat::default();
    core::add_weighted(&overlay, 0.75, output, 0.25, 0.0, &mut blended, -1)?;
    blended.copy_to(output)?;

    // Border
    imgproc::rectangle(
        output,
        core::Rect::new(badge_x, badge_y, badge_w, badge_h),
        dot_color,
        1,
        imgproc::LINE_AA,
        0,
    )?;

    // Status dot
    imgproc::circle(
        output,
        core::Point::new(badge_x + 10, badge_y + badge_h / 2),
        5,
        dot_color,
        -1,
        imgproc::LINE_AA,
        0,
    )?;

    // Label
    imgproc::put_text(
        output,
        &label,
        core::Point::new(badge_x + 20, badge_y + badge_h - 6),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.42,
        core::Scalar::new(220.0, 220.0, 220.0, 0.0),
        1,
        imgproc::LINE_AA,
        false,
    )?;

    Ok(())
}

// ============================================================================
// 6. MIXED LINE INDICATOR
// ============================================================================

/// Render a visual indicator showing which side of a mixed line is dashed
/// (permitted) vs solid (prohibited).
///
/// Draws two mini arrows on either side of the mixed marking with
/// "PERMITIDO" / "PROHIBIDO" labels.
pub fn render_mixed_line_indicator(
    output: &mut Mat,
    marking: &DetectedRoadMarking,
    dashed_is_right: bool,
) -> Result<()> {
    if !marking.class_name.contains("mixed") && marking.class_id != 99 {
        return Ok(());
    }

    let bbox = &marking.bbox;
    let cx = ((bbox[0] + bbox[2]) / 2.0) as i32;
    let cy = ((bbox[1] + bbox[3]) / 2.0) as i32;

    // Position labels on each side
    let (left_label, left_color, right_label, right_color) = if dashed_is_right {
        (
            "PROHIBIDO",
            colors::ILLEGAL_RED,
            "PERMITIDO",
            colors::LEGAL_GREEN,
        )
    } else {
        (
            "PERMITIDO",
            colors::LEGAL_GREEN,
            "PROHIBIDO",
            colors::ILLEGAL_RED,
        )
    };

    // Left side label
    draw_text_with_bg(output, left_label, cx - 95, cy - 5, 0.35, left_color)?;

    // Right side label
    draw_text_with_bg(output, right_label, cx + 10, cy - 5, 0.35, right_color)?;

    // Direction arrows
    // Left arrow: ← (from center pointing left)
    let arrow_y = cy + 12;
    imgproc::arrowed_line(
        output,
        core::Point::new(cx - 5, arrow_y),
        core::Point::new(cx - 45, arrow_y),
        left_color,
        2,
        imgproc::LINE_AA,
        0,
        0.3,
    )?;

    // Right arrow: → (from center pointing right)
    imgproc::arrowed_line(
        output,
        core::Point::new(cx + 5, arrow_y),
        core::Point::new(cx + 45, arrow_y),
        right_color,
        2,
        imgproc::LINE_AA,
        0,
        0.3,
    )?;

    Ok(())
}

// ============================================================================
// 7. MARKING TYPE LABEL (improved)
// ============================================================================

/// Render a compact label on each marking showing its detected type
/// with a colored background pill.
pub fn render_marking_label(
    output: &mut Mat,
    marking: &DetectedRoadMarking,
    _orig_h: i32,
) -> Result<()> {
    let bbox = &marking.bbox;
    let cx = ((bbox[0] + bbox[2]) / 2.0) as i32;
    // Position label near the top of the marking
    let label_y = (bbox[1] as i32 - 8).max(14);

    let short_name = short_class_name(&marking.class_name);
    let color = legality_strip_color(&marking.legality);

    draw_text_with_bg(output, &short_name, cx - 30, label_y, 0.38, color)?;

    Ok(())
}

/// Render labels for all markings.
pub fn render_all_marking_labels(
    output: &mut Mat,
    markings: &[DetectedRoadMarking],
    orig_h: i32,
) -> Result<()> {
    for marking in markings {
        render_marking_label(output, marking, orig_h)?;
    }
    Ok(())
}

// ============================================================================
// 8. PASSING LEGALITY BANNER
// ============================================================================

/// Render a top-of-screen banner showing the current overall passing legality.
///
/// Uses the Peru MTC vocabulary: "ADELANTAR PERMITIDO" / "ADELANTAR PROHIBIDO" /
/// "MIXTA: PERMITIDO (lado segmentado)" etc.
pub fn render_legality_banner(
    output: &mut Mat,
    legality: &PassingLegality,
    orig_w: i32,
) -> Result<()> {
    let text = legality.as_str();
    let (bg_color, text_color) = match legality {
        PassingLegality::Allowed => (
            core::Scalar::new(0.0, 80.0, 0.0, 0.0),
            core::Scalar::new(180.0, 255.0, 180.0, 0.0),
        ),
        PassingLegality::Prohibited => (
            core::Scalar::new(0.0, 0.0, 100.0, 0.0),
            core::Scalar::new(200.0, 200.0, 255.0, 0.0),
        ),
        PassingLegality::MixedAllowed => (
            core::Scalar::new(0.0, 80.0, 40.0, 0.0),
            core::Scalar::new(180.0, 255.0, 200.0, 0.0),
        ),
        PassingLegality::MixedProhibited => (
            core::Scalar::new(0.0, 20.0, 100.0, 0.0),
            core::Scalar::new(200.0, 200.0, 255.0, 0.0),
        ),
        PassingLegality::Unknown => return Ok(()), // Don't show banner for unknown
    };

    // Measure text to center it
    let mut baseline = 0;
    let text_size =
        imgproc::get_text_size(text, imgproc::FONT_HERSHEY_SIMPLEX, 0.55, 1, &mut baseline)?;

    let banner_h = 30;
    let banner_w = text_size.width + 40;
    let banner_x = (orig_w - banner_w) / 2;
    let banner_y = 8;

    // Semi-transparent background
    let mut overlay = output.try_clone()?;
    imgproc::rectangle(
        &mut overlay,
        core::Rect::new(banner_x, banner_y, banner_w, banner_h),
        bg_color,
        -1,
        imgproc::LINE_8,
        0,
    )?;
    let mut blended = Mat::default();
    core::add_weighted(&overlay, 0.80, output, 0.20, 0.0, &mut blended, -1)?;
    blended.copy_to(output)?;

    // Border
    imgproc::rectangle(
        output,
        core::Rect::new(banner_x, banner_y, banner_w, banner_h),
        text_color,
        1,
        imgproc::LINE_AA,
        0,
    )?;

    // Centered text
    let text_x = banner_x + (banner_w - text_size.width) / 2;
    let text_y = banner_y + banner_h - 8;
    imgproc::put_text(
        output,
        text,
        core::Point::new(text_x, text_y),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.55,
        text_color,
        1,
        imgproc::LINE_AA,
        false,
    )?;

    Ok(())
}

// ============================================================================
// 9. FULL ROAD ZONE RENDER PASS
// ============================================================================

/// Input data for the full road zone rendering pass.
pub struct RoadZoneInput<'a> {
    pub markings: &'a [DetectedRoadMarking],
    pub ego_left_x: Option<f32>,
    pub ego_right_x: Option<f32>,
    pub passing_legality: PassingLegality,
    pub cache_state: CacheState,
    pub cache_stale_frames: u32,
    pub crossing_flash: Option<&'a CrossingFlashState>,
    pub mixed_dashed_is_right: Option<bool>,
    pub frame_id: u64,
}

/// Execute the full road zone rendering pipeline.
///
/// Call this BEFORE the existing draw_lanes_v2 vehicle/HUD rendering
/// so zone overlays appear behind vehicles and UI elements.
///
/// Rendering order (back-to-front):
///   1. Opposing lane danger tint (deepest layer)
///   2. YOLO mask overlays (on road surface)
///   3. Legality zone strips (alongside markings)
///   4. Crossing flash effect (animated)
///   5. Marking type labels (on top of markings)
///   6. Mixed line indicators (arrows + labels)
///   7. Passing legality banner (top center)
///   8. Cache state badge (top right)
pub fn render_road_zones(
    output: &mut Mat,
    input: &RoadZoneInput,
    orig_w: i32,
    orig_h: i32,
) -> Result<()> {
    let ego_center_x = orig_w as f32 / 2.0;

    // 1. Opposing lane danger tint
    if let Some(left_x) = input.ego_left_x {
        render_opposing_lane_tint(output, left_x, orig_w, orig_h, &input.passing_legality)?;
    }

    // 2. YOLO mask overlays
    render_all_mask_overlays(output, input.markings, orig_w, orig_h)?;

    // 3. Legality zone strips
    render_all_legality_strips(output, input.markings, ego_center_x, orig_w, orig_h)?;

    // 4. Crossing flash effect
    if let Some(flash) = input.crossing_flash {
        render_crossing_flash(output, flash, input.frame_id, orig_w, orig_h)?;
    }

    // 5. Marking type labels
    render_all_marking_labels(output, input.markings, orig_h)?;

    // 6. Mixed line indicators
    if let Some(dashed_right) = input.mixed_dashed_is_right {
        for marking in input.markings {
            if marking.class_id == 99 || marking.class_name.contains("mixed") {
                render_mixed_line_indicator(output, marking, dashed_right)?;
            }
        }
    }

    // 7. Passing legality banner
    render_legality_banner(output, &input.passing_legality, orig_w)?;

    // 8. Cache state badge (top-right area)
    let badge_x = orig_w - 160;
    let badge_y = 12;
    render_cache_badge(
        output,
        input.cache_state,
        input.cache_stale_frames,
        badge_x,
        badge_y,
    )?;

    Ok(())
}

// ============================================================================
// INTERNAL HELPERS
// ============================================================================

fn mask_color_for_legality(legality: &LineLegality) -> core::Scalar {
    match legality {
        LineLegality::Legal => colors::MASK_DASHED_GREEN,
        LineLegality::Illegal => colors::MASK_SOLID_RED,
        LineLegality::CriticalIllegal => colors::MASK_DOUBLE_RED,
        _ => colors::MASK_DEFAULT,
    }
}

fn legality_strip_color(legality: &LineLegality) -> core::Scalar {
    match legality {
        LineLegality::Legal => colors::LEGAL_GREEN,
        LineLegality::Illegal => colors::ILLEGAL_RED,
        LineLegality::CriticalIllegal => colors::ILLEGAL_RED,
        _ => colors::UNKNOWN_GREY,
    }
}

fn short_class_name(name: &str) -> String {
    // Convert "solid_single_yellow" → "SLD-Y"
    // Convert "dashed_single_white" → "DSH-W"
    // Convert "double_solid_yellow" → "DBL-Y"
    // Convert "mixed_double_yellow" → "MIX-Y"
    let is_yellow = name.contains("yellow");
    let color_suffix = if is_yellow { "Y" } else { "W" };

    if name.contains("mixed") {
        format!("MIX-{}", color_suffix)
    } else if name.contains("double") {
        format!("DBL-{}", color_suffix)
    } else if name.contains("dashed") {
        format!("DSH-{}", color_suffix)
    } else if name.contains("solid") {
        format!("SLD-{}", color_suffix)
    } else {
        name.chars().take(6).collect::<String>().to_uppercase()
    }
}

/// Draw text with a filled background rectangle (pill/badge style).
fn draw_text_with_bg(
    img: &mut Mat,
    text: &str,
    x: i32,
    y: i32,
    scale: f64,
    color: core::Scalar,
) -> Result<()> {
    let mut baseline = 0;
    let text_size =
        imgproc::get_text_size(text, imgproc::FONT_HERSHEY_SIMPLEX, scale, 1, &mut baseline)?;

    let pad = 4;
    let bg_x = x - pad;
    let bg_y = y - text_size.height - pad;
    let bg_w = text_size.width + pad * 2;
    let bg_h = text_size.height + pad * 2 + baseline;

    // Dark background
    imgproc::rectangle(
        img,
        core::Rect::new(bg_x, bg_y, bg_w, bg_h),
        core::Scalar::new(15.0, 15.0, 15.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;

    // Colored left accent bar
    imgproc::rectangle(
        img,
        core::Rect::new(bg_x, bg_y, 3, bg_h),
        color,
        -1,
        imgproc::LINE_8,
        0,
    )?;

    // Text
    imgproc::put_text(
        img,
        text,
        core::Point::new(x, y),
        imgproc::FONT_HERSHEY_SIMPLEX,
        scale,
        core::Scalar::new(230.0, 230.0, 230.0, 0.0),
        1,
        imgproc::LINE_AA,
        false,
    )?;

    Ok(())
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_class_name() {
        assert_eq!(short_class_name("solid_single_yellow"), "SLD-Y");
        assert_eq!(short_class_name("dashed_single_white"), "DSH-W");
        assert_eq!(short_class_name("double_solid_yellow"), "DBL-Y");
        assert_eq!(short_class_name("mixed_double_yellow"), "MIX-Y");
        assert_eq!(short_class_name("unknown_thing"), "UNKNOW");
    }

    #[test]
    fn test_crossing_flash_progress() {
        let event = LineCrossingEvent {
            line_role: LineRole::LeftBoundary,
            marking_class: "solid_single_yellow".to_string(),
            marking_class_id: 5,
            passing_legality: PassingLegality::Prohibited,
            crossing_direction: crate::lane_crossing::CrossingDirection::Leftward,
            confidence: 0.85,
            penetration_ratio: 0.5,
            frame_id: 100,
            timestamp_ms: 3333.0,
        };

        let flash = CrossingFlashState::new(&event, [300.0, 400.0, 320.0, 700.0]);

        assert!(flash.is_active(100));
        assert!(flash.is_active(110));
        assert!(!flash.is_active(120)); // 18 frames duration
        assert!((flash.progress(109) - 0.5).abs() < 0.1);
    }
}
