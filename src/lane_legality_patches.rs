// src/lane_legality_patches.rs
//
// v5.0: Drop-in replacement functions for lane_legality.rs
//
// This file contains corrected implementations that should replace the
// corresponding functions in lane_legality.rs. Each function documents
// which original function it replaces and what was fixed.
//
// INTEGRATION INSTRUCTIONS:
//   1. Add `mod color_analysis;` and `mod road_classification;` to main.rs
//   2. Replace `verify_line_color()` calls in lane_legality.rs with
//      `verify_line_color_hsv()` from this module
//   3. Replace hardcoded pixel thresholds with proportional versions
//   4. Wire `RoadClassifier` into the pipeline state

use crate::color_analysis::{self, classify_marking_color, verify_and_correct_line_color};
use crate::road_classification::{MarkingInfo, RoadClassification, RoadClassifier};
use tracing::debug;

// ============================================================================
// 1. REPLACEMENT: verify_line_color → verify_line_color_hsv
// ============================================================================
//
// In lane_legality.rs, find the call:
//
//   for marking in &mut detections {
//       verify_line_color(frame_rgb, orig_w, orig_h, marking);
//   }
//
// Replace with:
//
//   for marking in &mut detections {
//       verify_line_color_hsv(frame_rgb, orig_w, orig_h, marking);
//   }

/// HSV-based line color verification. Replaces the old `verify_line_color`.
///
/// Key improvements over the original:
///   - Works in HSV space (robust to lighting changes)
///   - Bidirectional correction (white↔yellow, not just white→yellow)
///   - Uses segmentation mask when available (not just bbox center)
///   - Handles red markings (Peru MTC edge lines)
pub fn verify_line_color_hsv(
    frame_rgb: &[u8],
    frame_width: usize,
    frame_height: usize,
    marking_class_id: &mut usize,
    marking_class_name: &mut String,
    marking_bbox: &[f32; 4],
    marking_legality: &mut crate::lane_legality::LineLegality,
    mask: Option<&[u8]>,
    mask_width: usize,
    mask_height: usize,
) {
    let changed = verify_and_correct_line_color(
        frame_rgb,
        frame_width,
        frame_height,
        marking_class_id,
        marking_class_name,
        marking_bbox,
        mask,
        mask_width,
        mask_height,
    );

    // If the class changed, update legality too
    if changed {
        *marking_legality = class_id_to_legality(*marking_class_id);
    }
}

/// Mirrors the legality logic from lane_legality.rs
fn class_id_to_legality(class_id: usize) -> crate::lane_legality::LineLegality {
    use crate::lane_legality::LineLegality;
    const CRITICAL: [usize; 2] = [5, 8];
    const ILLEGAL: [usize; 5] = [4, 5, 6, 7, 8];
    const LEGAL: [usize; 2] = [9, 10];

    if CRITICAL.contains(&class_id) {
        LineLegality::CriticalIllegal
    } else if ILLEGAL.contains(&class_id) {
        LineLegality::Illegal
    } else if LEGAL.contains(&class_id) {
        LineLegality::Legal
    } else {
        LineLegality::Caution
    }
}

// ============================================================================
// 2. FIX: Proportional merge threshold (replaces hardcoded 100.0px)
// ============================================================================
//
// In lane_legality.rs, find in `merge_composite_lines`:
//
//   if iou > 0.1 && dist_x < 100.0 {
//
// Replace with:
//
//   if iou > 0.1 && dist_x < merge_distance_threshold {
//
// Where merge_distance_threshold is passed in or computed as:

/// Compute the merge distance threshold proportional to frame width.
///
/// Original code used a hardcoded 100.0px which is:
///   - ~7.8% of 1280px → reasonable
///   - ~15.6% of 640px → way too generous (merges separate lines)
///   - ~5.2% of 1920px → too tight (misses valid pairs)
///
/// This function returns ~7% of frame width as the threshold.
pub fn proportional_merge_distance(frame_width: f32) -> f32 {
    (frame_width * 0.07).max(30.0).min(150.0)
}

// ============================================================================
// 3. HELPER: Convert YOLO detections to MarkingInfo for RoadClassifier
// ============================================================================

/// Convert raw YOLO detections into MarkingInfo structs for the RoadClassifier.
///
/// Call this after `postprocess()` in lane_legality.rs and feed the result
/// to `RoadClassifier::update()`.
///
/// **v5.1**: Now passes segmentation mask data through to enable mixed line
/// side analysis (determining which side is dashed vs solid).
pub fn detections_to_marking_infos(
    detections: &[crate::lane_legality::DetectedRoadMarking],
    frame_rgb: &[u8],
    frame_width: usize,
    frame_height: usize,
) -> Vec<MarkingInfo> {
    detections
        .iter()
        .map(|d| {
            let cx = (d.bbox[0] + d.bbox[2]) / 2.0;

            // Run HSV color analysis for each marking
            let color_result = classify_marking_color(
                frame_rgb,
                frame_width,
                frame_height,
                &d.bbox,
                if d.mask.is_empty() {
                    None
                } else {
                    Some(d.mask.as_slice())
                },
                d.mask_width,
                d.mask_height,
            );

            MarkingInfo {
                class_id: d.class_id,
                class_name: d.class_name.clone(),
                center_x: cx,
                bbox: d.bbox,
                confidence: d.confidence,
                detected_color: if color_result.samples >= 10 {
                    Some(color_result.detected_color)
                } else {
                    None
                },
                // Pass mask through for mixed line side analysis
                mask: d.mask.clone(),
                mask_width: d.mask_width,
                mask_height: d.mask_height,
            }
        })
        .collect()
}

// ============================================================================
// 4. FIX: Dynamic ROW_ANCHOR scaling for lane_detection.rs
// ============================================================================
//
// In lane_detection.rs, the constants are hardcoded for 720p:
//
//   const ROW_ANCHOR_START: f32 = 160.0;
//   const ROW_ANCHOR_END: f32 = 710.0;
//   const ORIGINAL_HEIGHT: f32 = 720.0;
//
// These are correct for the CULane dataset (720p). The parse_lanes function
// already scales them by (y_norm / ORIGINAL_HEIGHT) * frame_height, which
// is correct IF the model was trained on CULane.
//
// However, if you're using a model trained on a different resolution,
// these need to match the training data. The fix is to make them configurable:

/// Row anchor parameters that should come from config, not hardcoded.
#[derive(Debug, Clone)]
pub struct RowAnchorConfig {
    pub start: f32,
    pub end: f32,
    pub original_height: f32,
}

impl Default for RowAnchorConfig {
    fn default() -> Self {
        // CULane / TuSimple defaults (720p training)
        Self {
            start: 160.0,
            end: 710.0,
            original_height: 720.0,
        }
    }
}

impl RowAnchorConfig {
    /// For CULane trained models (most UFLDv2 checkpoints)
    pub fn culane() -> Self {
        Self::default()
    }

    /// For TuSimple trained models
    pub fn tusimple() -> Self {
        Self {
            start: 160.0,
            end: 710.0,
            original_height: 720.0,
        }
    }
}

// ============================================================================
// 5. INTEGRATION GUIDE
// ============================================================================
//
// To integrate all fixes into main.rs / pipeline:
//
// A) Add modules to main.rs:
//    ```
//    mod color_analysis;
//    mod road_classification;
//    mod lane_legality_patches;
//    ```
//
// B) Add RoadClassifier to PipelineState:
//    ```
//    struct PipelineState {
//        ...
//        road_classifier: RoadClassifier,
//    }
//    ```
//
// C) Initialize in PipelineState::new():
//    ```
//    road_classifier: RoadClassifier::new(frame_width),
//    ```
//
// D) In run_lane_detection(), after getting markings from legality detector:
//    ```
//    // Convert detections to marking infos
//    let marking_infos = lane_legality_patches::detections_to_marking_infos(
//        &markings, &frame.data, frame.width, frame.height,
//    );
//
//    // Classify road type
//    let road_class = ps.road_classifier.update(&marking_infos);
//    debug!("Road: {} | Passing: {} | Lanes: {}",
//        road_class.road_type.as_display_str(),
//        road_class.passing_legality.as_str(),
//        road_class.estimated_lanes,
//    );
//    ```
//
// E) In video_processor.rs visualization, add road type display:
//    ```
//    let road_text = format!("VÍA: {} | {}",
//        road_class.road_type.as_display_str(),
//        road_class.passing_legality.as_str(),
//    );
//    draw_text_with_shadow(&mut output, &road_text, 15, 65, 0.55, white_color, 1)?;
//    ```
//
// F) In lane_legality.rs postprocess(), replace:
//    ```
//    // OLD:
//    for marking in &mut detections {
//        verify_line_color(frame_rgb, orig_w, orig_h, marking);
//    }
//
//    // NEW:
//    for marking in &mut detections {
//        lane_legality_patches::verify_line_color_hsv(
//            frame_rgb, orig_w, orig_h,
//            &mut marking.class_id,
//            &mut marking.class_name,
//            &marking.bbox,
//            &mut marking.legality,
//            if marking.mask.is_empty() { None } else { Some(&marking.mask) },
//            marking.mask_width,
//            marking.mask_height,
//        );
//    }
//    ```
//
// G) In lane_legality.rs merge_composite_lines(), replace:
//    ```
//    // OLD:
//    if iou > 0.1 && dist_x < 100.0 {
//
//    // NEW:
//    let merge_dist = lane_legality_patches::proportional_merge_distance(frame_width);
//    if iou > 0.1 && dist_x < merge_dist {
//    ```
//    (You'll need to pass frame_width into merge_composite_lines)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proportional_merge_distance() {
        // 1280px → ~89.6px
        let d1280 = proportional_merge_distance(1280.0);
        assert!(d1280 > 80.0 && d1280 < 100.0);

        // 640px → ~44.8px (not 100px!)
        let d640 = proportional_merge_distance(640.0);
        assert!(d640 > 40.0 && d640 < 50.0);

        // 1920px → ~134.4px
        let d1920 = proportional_merge_distance(1920.0);
        assert!(d1920 > 130.0 && d1920 < 140.0);
    }

    #[test]
    fn test_row_anchor_defaults() {
        let config = RowAnchorConfig::default();
        assert_eq!(config.original_height, 720.0);
        assert!(config.start < config.end);
    }
}
