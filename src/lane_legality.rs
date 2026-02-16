// src/lane_legality.rs
//
// v4.13: Added polynomial curvature estimation from YOLO-seg masks.
//        Stores raw lane markings per frame and computes per-boundary
//        polynomial fits for direct geometric curve detection.
// v6.1g: Mask-based mixed line detection — analyzes class 8 segmentation
//        masks to distinguish true double-solid from mixed (solid+dashed)
//        center lines. Reclassifies to class 99 when one stripe is dashed.

use crate::analysis::curvature_estimator::{
    self, CurvatureEstimate, MaskInput, MaskTransformParams,
};
use anyhow::Result;
use ort::{
    execution_providers::CUDAExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::collections::VecDeque;
use tracing::{debug, info};

const SEG_INPUT_SIZE: usize = 640;
const NUM_CLASSES: usize = 25;
const MASK_SIZE: usize = 160;

/// Class IDs that make crossing ILLEGAL
const ILLEGAL_CLASS_IDS: [usize; 5] = [4, 5, 6, 7, 8];
/// Class IDs that make crossing LEGAL
const LEGAL_CLASS_IDS: [usize; 2] = [9, 10];
/// Class IDs considered CRITICAL violations
const CRITICAL_CLASS_IDS: [usize; 2] = [5, 8];

// Lane line class IDs (excludes arrows, crosswalks, text, etc.)
const LANE_LINE_CLASS_IDS: [usize; 8] = [4, 5, 6, 7, 8, 9, 10, 99];

// ---------------------------------------------------------------------------
// Ego lane boundary estimation constants (v2)
// ---------------------------------------------------------------------------
const MERGE_THRESHOLD_RATIO: f32 = 0.06;
const MIN_LANE_WIDTH: f32 = 200.0;
const MAX_LANE_WIDTH: f32 = 900.0;
const IDEAL_LANE_WIDTH: f32 = 450.0;
const DEFAULT_SINGLE_MARKING_WIDTH: f32 = 450.0;
const WIDTH_HISTORY_SIZE: usize = 20;
const SINGLE_BOUNDARY_CONFIDENCE_PENALTY: f32 = 0.65;

/// v4.11 (Fix 1): Minimum boundary movement (pixels/frame) to compute coherence.
/// Below this, boundary deltas are noise — coherence is meaningless.
const MIN_BOUNDARY_DELTA: f32 = 0.5;

// ===========================================================================
// CROSSING SIDE
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossingSide {
    None,
    Left,
    Right,
}

// ===========================================================================
// TYPES
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineLegality {
    Legal,
    Illegal,
    CriticalIllegal,
    Caution,
    Unknown,
}

impl LineLegality {
    pub fn as_str(&self) -> &'static str {
        match self {
            LineLegality::Legal => "LEGAL",
            LineLegality::Illegal => "ILLEGAL",
            LineLegality::CriticalIllegal => "CRITICAL_ILLEGAL",
            LineLegality::Caution => "CAUTION",
            LineLegality::Unknown => "UNKNOWN",
        }
    }

    pub fn is_illegal(&self) -> bool {
        matches!(self, LineLegality::Illegal | LineLegality::CriticalIllegal)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedRoadMarking {
    pub class_id: usize,
    pub class_name: String,
    pub confidence: f32,
    pub bbox: [f32; 4],
    pub legality: LineLegality,
    pub mask: Vec<u8>,
    pub mask_width: usize,
    pub mask_height: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalityResult {
    pub verdict: LineLegality,
    pub intersecting_line: Option<DetectedRoadMarking>,
    pub all_markings: Vec<DetectedRoadMarking>,
    pub ego_intersects_marking: bool,
    pub frame_id: u64,
}

impl LegalityResult {
    pub fn no_detection(frame_id: u64) -> Self {
        Self {
            verdict: LineLegality::Unknown,
            intersecting_line: None,
            all_markings: Vec::new(),
            ego_intersects_marking: false,
            frame_id,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FusedLegalityResult {
    pub verdict: LineLegality,
    pub crossing_confirmed_by_lane_model: bool,
    pub line_type_from_seg_model: Option<DetectedRoadMarking>,
    pub vehicle_offset_pct: f32,
    pub all_markings: Vec<DetectedRoadMarking>,
    pub ego_intersects_marking: bool,
}

// ===========================================================================
// TEMPORAL FILTER
// ===========================================================================

struct TemporalViolationFilter {
    recent_verdicts: Vec<(u64, LineLegality)>,
    min_consecutive: usize,
    max_frame_gap: u64,
}

impl TemporalViolationFilter {
    fn new(min_consecutive: usize, max_frame_gap: u64) -> Self {
        Self {
            recent_verdicts: Vec::with_capacity(20),
            min_consecutive,
            max_frame_gap,
        }
    }

    fn update(&mut self, frame_id: u64, verdict: LineLegality) -> bool {
        self.recent_verdicts.push((frame_id, verdict));
        if self.recent_verdicts.len() > 20 {
            self.recent_verdicts.remove(0);
        }

        if !verdict.is_illegal() {
            return false;
        }

        let mut consecutive = 0;
        let mut last_frame: Option<u64> = None;

        for &(fid, v) in self.recent_verdicts.iter().rev() {
            if !v.is_illegal() {
                break;
            }
            if let Some(prev) = last_frame {
                if prev - fid > self.max_frame_gap {
                    break;
                }
            }
            consecutive += 1;
            last_frame = Some(fid);
        }

        consecutive >= self.min_consecutive
    }
}

// ===========================================================================
// HELPERS
// ===========================================================================

fn class_id_to_name(class_id: usize) -> &'static str {
    match class_id {
        4 => "solid_single_white",
        5 => "solid_single_yellow",
        6 => "solid_single_red",
        7 => "solid_double_white",
        8 => "solid_double_yellow",
        9 => "dashed_single_white",
        10 => "dashed_single_yellow",
        99 => "mixed_double_yellow",
        _ => "other_marking",
    }
}

fn class_id_to_legality(class_id: usize) -> LineLegality {
    if CRITICAL_CLASS_IDS.contains(&class_id) {
        LineLegality::CriticalIllegal
    } else if ILLEGAL_CLASS_IDS.contains(&class_id) {
        LineLegality::Illegal
    } else if LEGAL_CLASS_IDS.contains(&class_id) {
        LineLegality::Legal
    } else {
        LineLegality::Caution
    }
}

fn is_lane_line_class(class_id: usize) -> bool {
    LANE_LINE_CLASS_IDS.contains(&class_id)
}

// ===========================================================================
// COLOR VERIFICATION
// ===========================================================================

fn verify_line_color(
    frame_rgb: &[u8],
    frame_width: usize,
    frame_height: usize,
    marking: &mut DetectedRoadMarking,
) {
    let is_white_class = matches!(marking.class_id, 4 | 7 | 9);
    if !is_white_class {
        return;
    }

    let cx = ((marking.bbox[0] + marking.bbox[2]) / 2.0) as usize;
    let cy = ((marking.bbox[1] + marking.bbox[3]) / 2.0) as usize;
    let half_w = ((marking.bbox[2] - marking.bbox[0]) / 4.0).max(5.0) as usize;
    let half_h = ((marking.bbox[3] - marking.bbox[1]) / 4.0).max(5.0) as usize;

    let mut total_r: u32 = 0;
    let mut total_g: u32 = 0;
    let mut total_b: u32 = 0;
    let mut count: u32 = 0;

    let y_start = cy.saturating_sub(half_h).min(frame_height - 1);
    let y_end = (cy + half_h).min(frame_height - 1);
    let x_start = cx.saturating_sub(half_w).min(frame_width - 1);
    let x_end = (cx + half_w).min(frame_width - 1);

    for y in y_start..=y_end {
        for x in x_start..=x_end {
            let idx = (y * frame_width + x) * 3;
            if idx + 2 < frame_rgb.len() {
                let r = frame_rgb[idx] as u32;
                let g = frame_rgb[idx + 1] as u32;
                let b = frame_rgb[idx + 2] as u32;

                let brightness = (r + g + b) / 3;
                if brightness > 60 {
                    total_r += r;
                    total_g += g;
                    total_b += b;
                    count += 1;
                }
            }
        }
    }

    if count < 5 {
        return;
    }

    let avg_r = (total_r / count) as f32;
    let avg_g = (total_g / count) as f32;
    let avg_b = (total_b / count) as f32;

    let r_b_ratio = if avg_b > 1.0 { avg_r / avg_b } else { 2.0 };
    let g_b_ratio = if avg_b > 1.0 { avg_g / avg_b } else { 2.0 };

    let is_yellow = avg_b > 1.0
        && r_b_ratio > 1.08
        && g_b_ratio > 1.05
        && avg_r > 90.0
        && avg_g > 90.0
        && (avg_r - avg_b) > 15.0;

    if is_yellow {
        let new_class = match marking.class_id {
            4 => 5,
            7 => 8,
            9 => 10,
            _ => marking.class_id,
        };

        if new_class != marking.class_id {
            debug!(
                "🎨 Color correction: {} → {} (R={:.0}, G={:.0}, B={:.0}, R/B={:.2})",
                marking.class_name,
                class_id_to_name(new_class),
                avg_r,
                avg_g,
                avg_b,
                r_b_ratio
            );
            marking.class_id = new_class;
            marking.class_name = class_id_to_name(new_class).to_string();
            marking.legality = class_id_to_legality(new_class);
        }
    }
}

// ===========================================================================
// v6.1g: MASK-BASED MIXED LINE DETECTION
// ===========================================================================
//
// The YOLO model detects both stripes of a mixed double yellow line as a
// single solid_double_yellow (class 8) bounding box. This function splits
// the 160×160 segmentation mask into left/right halves and measures
// row-by-row coverage to distinguish solid from dashed stripes.
//
// Pipeline order in postprocess():
//   1. verify_line_color()             — fix color misclassification
//   2. analyze_double_line_for_mixed() — class 8 → 99 when mixed  ← THIS
//   3. merge_composite_lines()         — merge separate solid+dashed pairs
//   4. nms_markings()                  — suppress overlapping detections
//
// Two paths create class 99 mixed lines:
//   Path A (this function): Single class 8 detection with one dashed stripe
//   Path B (merge_composite_lines): Separate class 5 + class 10 detections

/// v6.1g: Analyze a solid_double_yellow (class 8) segmentation mask to detect
/// if one stripe is actually dashed (making it a mixed double yellow line).
///
/// Splits the mask into left/right halves at the column-sum valley between
/// the two stripe peaks, then measures per-half coverage and gap transitions.
///
/// If mixed, reclassifies the detection:
///   - dashed on right → class 99, mixed_double_yellow_dashed_right (Legal)
///   - dashed on left  → class 99, mixed_double_yellow_solid_right (CriticalIllegal)
fn analyze_double_line_for_mixed(
    marking: &mut DetectedRoadMarking,
    scale: f32,
    pad_x: f32,
    pad_y: f32,
) {
    // Only analyze solid_double_yellow (class 8)
    if marking.class_id != 8 {
        return;
    }

    let mask = &marking.mask;
    let mw = marking.mask_width;
    let mh = marking.mask_height;

    if mask.is_empty() || mw == 0 || mh == 0 {
        return;
    }

    // Map bbox from original image → 640-space → 160-space (mask coordinates)
    let ratio = MASK_SIZE as f32 / SEG_INPUT_SIZE as f32; // 160/640 = 0.25

    let mx1 = ((marking.bbox[0] * scale + pad_x) * ratio).floor().max(0.0) as usize;
    let my1 = ((marking.bbox[1] * scale + pad_y) * ratio).floor().max(0.0) as usize;
    let mx2 = ((marking.bbox[2] * scale + pad_x) * ratio)
        .ceil()
        .min(mw as f32 - 1.0) as usize;
    let my2 = ((marking.bbox[3] * scale + pad_y) * ratio)
        .ceil()
        .min(mh as f32 - 1.0) as usize;

    let col_width = mx2.saturating_sub(mx1) + 1;
    let row_height = my2.saturating_sub(my1) + 1;

    // Need enough resolution to distinguish two stripes
    if col_width < 4 || row_height < 8 {
        return;
    }

    // ── Step 1: Column sums to find the stripe split point ──────────
    // A double line produces two vertical peaks with a valley between them.
    let mut col_sums = vec![0u32; col_width];

    for my in my1..=my2 {
        for mx in mx1..=mx2 {
            let idx = my * mw + mx;
            if idx < mask.len() && mask[idx] == 255 {
                col_sums[mx - mx1] += 1;
            }
        }
    }

    // Find the valley (minimum) in the central 50% of columns
    let search_start = col_width / 4;
    let search_end = (col_width * 3) / 4;

    if search_end <= search_start {
        return;
    }

    let split = (search_start..=search_end)
        .min_by_key(|&c| col_sums[c])
        .unwrap_or(col_width / 2);

    // Verify there actually are two stripes (valley significantly lower than peaks)
    let left_peak = col_sums[..split].iter().copied().max().unwrap_or(0);
    let right_peak = col_sums[split..].iter().copied().max().unwrap_or(0);
    let valley_val = col_sums[split];

    let peak_min = left_peak.min(right_peak);
    if peak_min == 0 || valley_val as f32 > peak_min as f32 * 0.6 {
        return; // Not a clear two-stripe pattern — keep as solid_double_yellow
    }

    // ── Step 2: Row-by-row coverage for each half ───────────────────
    let mut left_present_rows = 0u32;
    let mut right_present_rows = 0u32;
    let mut left_gap_transitions = 0u32;
    let mut right_gap_transitions = 0u32;
    let mut prev_left = false;
    let mut prev_right = false;
    let mut total_rows = 0u32;

    for my in my1..=my2 {
        let mut left_has_pixel = false;
        let mut right_has_pixel = false;

        for mx in mx1..=mx2 {
            let idx = my * mw + mx;
            if idx < mask.len() && mask[idx] == 255 {
                if mx - mx1 < split {
                    left_has_pixel = true;
                } else {
                    right_has_pixel = true;
                }
            }
        }

        total_rows += 1;
        if left_has_pixel {
            left_present_rows += 1;
        }
        if right_has_pixel {
            right_present_rows += 1;
        }

        // Count transitions (present→absent or absent→present) = gap pattern
        if total_rows > 1 {
            if left_has_pixel != prev_left {
                left_gap_transitions += 1;
            }
            if right_has_pixel != prev_right {
                right_gap_transitions += 1;
            }
        }
        prev_left = left_has_pixel;
        prev_right = right_has_pixel;
    }

    if total_rows < 8 {
        return;
    }

    let left_coverage = left_present_rows as f32 / total_rows as f32;
    let right_coverage = right_present_rows as f32 / total_rows as f32;

    // ── Step 3: Classify each half ──────────────────────────────────
    // Solid: high coverage + few transitions
    // Dashed: low coverage AND many transitions (both conditions required)
    //
    // v6.1g-fix: The original OR logic caused false positives on true double
    // solid lines — mask quantization noise at certain perspectives produces
    // ≥4 gap transitions even on solid stripes. Requiring BOTH low coverage
    // AND frequent gaps eliminates these false reclassifications.
    const SOLID_MIN_COVERAGE: f32 = 0.65;
    const DASHED_MAX_COVERAGE: f32 = 0.55;
    const DASHED_MIN_TRANSITIONS: u32 = 6;

    let left_is_solid =
        left_coverage >= SOLID_MIN_COVERAGE && left_gap_transitions < DASHED_MIN_TRANSITIONS;
    let right_is_solid =
        right_coverage >= SOLID_MIN_COVERAGE && right_gap_transitions < DASHED_MIN_TRANSITIONS;
    // v6.1g-fix: AND instead of OR — a stripe is only dashed when it has
    // BOTH low coverage AND frequent gaps. This prevents mask noise
    // (high transitions on a solid stripe) from triggering false mixed.
    let left_is_dashed =
        left_coverage < DASHED_MAX_COVERAGE && left_gap_transitions >= DASHED_MIN_TRANSITIONS;
    let right_is_dashed =
        right_coverage < DASHED_MAX_COVERAGE && right_gap_transitions >= DASHED_MIN_TRANSITIONS;

    // v6.1g-fix: Safety guard — if both halves have high coverage (>= 60%),
    // this is almost certainly a true double solid. Don't reclassify even
    // if one half has noisy transitions.
    if left_coverage >= 0.60 && right_coverage >= 0.60 {
        debug!(
            "🔍 v6.1g: both halves high coverage ({:.0}%/{:.0}%) — keeping as solid_double_yellow",
            left_coverage * 100.0,
            right_coverage * 100.0,
        );
        return;
    }

    // ── Step 4: Reclassify if mixed ─────────────────────────────────
    if left_is_solid && right_is_dashed && !right_is_solid {
        // Solid on left (opposing), dashed on right (ego side) → ego CAN pass
        marking.class_id = 99;
        marking.class_name = "mixed_double_yellow_dashed_right".to_string();
        marking.legality = LineLegality::Legal;
        info!(
            "🔍 v6.1g mask→mixed: solid_double_yellow → dashed_right (Legal) \
             | L: {:.0}% cov, {} trans | R: {:.0}% cov, {} trans | {} rows",
            left_coverage * 100.0,
            left_gap_transitions,
            right_coverage * 100.0,
            right_gap_transitions,
            total_rows,
        );
    } else if right_is_solid && left_is_dashed && !left_is_solid {
        // Dashed on left (opposing), solid on right (ego side) → ego CANNOT pass
        marking.class_id = 99;
        marking.class_name = "mixed_double_yellow_solid_right".to_string();
        marking.legality = LineLegality::CriticalIllegal;
        info!(
            "🔍 v6.1g mask→mixed: solid_double_yellow → solid_right (CriticalIllegal) \
             | L: {:.0}% cov, {} trans | R: {:.0}% cov, {} trans | {} rows",
            left_coverage * 100.0,
            left_gap_transitions,
            right_coverage * 100.0,
            right_gap_transitions,
            total_rows,
        );
    } else {
        // Both solid or ambiguous → keep as solid_double_yellow (class 8, no change)
        debug!(
            "🔍 v6.1g mask: solid_double_yellow confirmed double-solid \
             | L: {:.0}% cov, {} trans | R: {:.0}% cov, {} trans | {} rows",
            left_coverage * 100.0,
            left_gap_transitions,
            right_coverage * 100.0,
            right_gap_transitions,
            total_rows,
        );
    }
}

// ===========================================================================
// MERGE COMPOSITE LINES (FIX FOR MIXED LINES)
// ===========================================================================

fn merge_composite_lines(dets: Vec<DetectedRoadMarking>) -> Vec<DetectedRoadMarking> {
    let mut merged = Vec::new();
    let mut used_indices = HashSet::new();

    for i in 0..dets.len() {
        if used_indices.contains(&i) {
            continue;
        }
        let a = &dets[i];
        let mut found_partner = false;

        for j in 0..dets.len() {
            if i == j || used_indices.contains(&j) {
                continue;
            }
            let b = &dets[j];

            let is_pair_solid_dashed = (matches!(a.class_id, 4 | 5)
                && matches!(b.class_id, 9 | 10))
                || (matches!(a.class_id, 9 | 10) && matches!(b.class_id, 4 | 5));

            if is_pair_solid_dashed {
                let iou = calculate_iou_arr(&a.bbox, &b.bbox);
                let cx_a = (a.bbox[0] + a.bbox[2]) / 2.0;
                let cx_b = (b.bbox[0] + b.bbox[2]) / 2.0;
                let dist_x = (cx_a - cx_b).abs();

                if iou > 0.1 && dist_x < 100.0 {
                    let (solid_part, dashed_part) = if matches!(a.class_id, 9 | 10) {
                        (b, a)
                    } else {
                        (a, b)
                    };
                    let solid_cx = (solid_part.bbox[0] + solid_part.bbox[2]) / 2.0;
                    let dashed_cx = (dashed_part.bbox[0] + dashed_part.bbox[2]) / 2.0;

                    let mut new_marking = solid_part.clone();
                    new_marking.class_id = 99;
                    new_marking.confidence = (a.confidence + b.confidence) / 2.0;
                    new_marking.bbox = [
                        a.bbox[0].min(b.bbox[0]),
                        a.bbox[1].min(b.bbox[1]),
                        a.bbox[2].max(b.bbox[2]),
                        a.bbox[3].max(b.bbox[3]),
                    ];

                    if dashed_cx > solid_cx {
                        new_marking.class_name = "mixed_double_yellow_dashed_right".to_string();
                        new_marking.legality = LineLegality::Legal;
                        debug!("🧩 Merged Mixed Line: Dashed RIGHT (Legal)");
                    } else {
                        new_marking.class_name = "mixed_double_yellow_solid_right".to_string();
                        new_marking.legality = LineLegality::CriticalIllegal;
                        debug!("🧩 Merged Mixed Line: Solid RIGHT (Illegal)");
                    }

                    merged.push(new_marking);
                    used_indices.insert(i);
                    used_indices.insert(j);
                    found_partner = true;
                    break;
                }
            }
        }

        if !found_partner {
            merged.push(a.clone());
        }
    }
    merged
}

// ===========================================================================
// MERGED MARKING POSITION (for boundary estimation v2)
// ===========================================================================

#[derive(Debug, Clone)]
struct MergedMarkingPosition {
    center_x: f32,
    confidence: f32,
    class_name: String,
    source_count: u32,
}

fn merge_nearby_markings(
    markings: &[&DetectedRoadMarking],
    merge_threshold: f32,
) -> Vec<MergedMarkingPosition> {
    let mut sorted: Vec<(f32, f32, &str)> = markings
        .iter()
        .map(|m| {
            let cx = (m.bbox[0] + m.bbox[2]) / 2.0;
            (cx, m.confidence, m.class_name.as_str())
        })
        .collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut merged: Vec<MergedMarkingPosition> = Vec::new();

    for (cx, conf, name) in &sorted {
        if let Some(last) = merged.last_mut() {
            if (*cx - last.center_x).abs() < merge_threshold {
                let w_old = last.confidence * last.source_count as f32;
                let w_new = *conf;
                last.center_x = (last.center_x * w_old + cx * w_new) / (w_old + w_new);
                last.confidence = (w_old + w_new) / (last.source_count as f32 + 1.0);
                last.source_count += 1;
                if *conf > last.confidence {
                    last.class_name = name.to_string();
                }
                continue;
            }
        }
        merged.push(MergedMarkingPosition {
            center_x: *cx,
            confidence: *conf,
            class_name: name.to_string(),
            source_count: 1,
        });
    }

    merged
}

// ===========================================================================
// MAIN DETECTOR IMPL
// ===========================================================================

pub struct LaneLegalityDetector {
    session: Session,
    ego_bbox_ratio: [f32; 4],
    temporal_filter: TemporalViolationFilter,
    lane_width_history: VecDeque<f32>,
    last_valid_width: Option<f32>,

    last_left_lane_x: Option<f32>,
    last_right_lane_x: Option<f32>,

    /// v4.11 (Fix 1): Rolling coherence of left/right boundary movement.
    /// +1 = boundaries co-moving (curve), -1 = diverging (lane change), <-0.5 = no data.
    last_boundary_coherence: f32,

    /// v4.13: Raw lane-line markings from the last boundary estimation call.
    /// Stored so the curvature estimator can access their masks without re-running inference.
    pub last_lane_markings: Vec<DetectedRoadMarking>,
    /// v4.13: Original image dimensions from the last call (for mask coordinate conversion).
    last_image_dims: (usize, usize),
    /// v4.13: Polynomial curvature estimate from mask geometry.
    last_curvature: Option<CurvatureEstimate>,
}

impl LaneLegalityDetector {
    pub fn new(model_path: &str) -> Result<Self> {
        info!("Loading lane legality model: {}", model_path);
        let session = Session::builder()?
            .with_execution_providers([CUDAExecutionProvider::default().with_device_id(0).build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        info!("✓ Lane legality detector initialized");
        Ok(Self {
            session,
            ego_bbox_ratio: [0.30, 0.75, 0.70, 0.98],
            temporal_filter: TemporalViolationFilter::new(2, 6),
            lane_width_history: VecDeque::with_capacity(WIDTH_HISTORY_SIZE),
            last_valid_width: None,
            last_left_lane_x: None,
            last_right_lane_x: None,
            last_boundary_coherence: -1.0, // v4.11: no data yet
            last_lane_markings: Vec::new(),
            last_image_dims: (0, 0),
            last_curvature: None,
        })
    }

    // -----------------------------------------------------------------------
    // WIDTH MEMORY
    // -----------------------------------------------------------------------

    fn record_lane_width(&mut self, width: f32) {
        if width >= MIN_LANE_WIDTH && width <= MAX_LANE_WIDTH {
            self.lane_width_history.push_back(width);
            if self.lane_width_history.len() > WIDTH_HISTORY_SIZE {
                self.lane_width_history.pop_front();
            }
            self.last_valid_width = Some(width);
        }
    }

    fn estimated_lane_width(&self) -> f32 {
        if self.lane_width_history.len() >= 3 {
            let mut sorted: Vec<f32> = self.lane_width_history.iter().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            return sorted[sorted.len() / 2];
        }
        self.last_valid_width
            .unwrap_or(DEFAULT_SINGLE_MARKING_WIDTH)
    }

    pub fn estimate_ego_lane_boundaries_stable(
        &mut self,
        frame: &[u8],
        width: usize,
        height: usize,
        vehicle_center_x: f32,
    ) -> Result<Option<(f32, f32, f32)>> {
        let raw_estimate =
            self.estimate_ego_lane_boundaries(frame, width, height, vehicle_center_x)?;

        if let Some((l, r, c)) = raw_estimate {
            // PRODUCTION FIX: Temporal dampening of the YOLO box jump
            let smooth_l = match self.last_left_lane_x {
                Some(prev) => prev * 0.7 + l * 0.3,
                None => l,
            };
            let smooth_r = match self.last_right_lane_x {
                Some(prev) => prev * 0.7 + r * 0.3,
                None => r,
            };

            // ═══════════════════════════════════════════════════════
            // v4.11 (Fix 1): Compute boundary coherence from per-frame deltas
            // ═══════════════════════════════════════════════════════
            //
            // On a curve: both boundaries shift in the SAME direction
            //   dl and dr have the same sign → coherence ≈ +1.0
            //
            // On a lane change: one boundary approaches, the other recedes
            //   dl and dr have OPPOSITE signs → coherence ≈ -1.0
            //
            self.last_boundary_coherence = if let (Some(prev_l), Some(prev_r)) =
                (self.last_left_lane_x, self.last_right_lane_x)
            {
                let dl = smooth_l - prev_l;
                let dr = smooth_r - prev_r;
                let abs_dl = dl.abs();
                let abs_dr = dr.abs();

                if abs_dl < MIN_BOUNDARY_DELTA && abs_dr < MIN_BOUNDARY_DELTA {
                    // Both static — neutral
                    0.0
                } else if abs_dl < MIN_BOUNDARY_DELTA || abs_dr < MIN_BOUNDARY_DELTA {
                    // Only one moved — inconclusive
                    0.0
                } else {
                    // Both have meaningful movement
                    let direction_sign = (dl * dr).signum();
                    let mag_ratio = abs_dl.min(abs_dr) / abs_dl.max(abs_dr);
                    let raw_coherence = direction_sign * mag_ratio;

                    // EWMA smoothing
                    let prev_coherence = self.last_boundary_coherence;
                    if prev_coherence < -0.5 {
                        raw_coherence // no previous data
                    } else {
                        0.6 * raw_coherence + 0.4 * prev_coherence
                    }
                }
            } else {
                -1.0 // No previous boundaries
            };

            self.last_left_lane_x = Some(smooth_l);
            self.last_right_lane_x = Some(smooth_r);

            // ═══════════════════════════════════════════════════════
            // v4.13: Polynomial curvature from YOLO-seg mask geometry
            // ═══════════════════════════════════════════════════════
            //
            // Extract mask spines for the markings closest to our selected
            // L/R boundaries, fit polynomials, and compare curvatures.
            // This is a DIRECT geometric measurement — not a temporal proxy
            // like boundary coherence — so it works on a single frame.
            //
            self.last_curvature = self.compute_mask_curvature(l, r);

            return Ok(Some((smooth_l, smooth_r, c)));
        }

        // No boundaries this frame — decay coherence
        if self.last_boundary_coherence > -0.5 {
            self.last_boundary_coherence *= 0.8;
        }

        Ok(None)
    }

    /// v4.11 (Fix 1): Get the current boundary coherence metric.
    ///
    /// Returns a value in [-1, +1]:
    ///   +1.0 = boundaries co-moving (curve)
    ///   -1.0 = boundaries diverging (lane change)
    ///    0.0 = inconclusive / static
    ///   < -0.5 = no data available
    ///
    /// Call after `estimate_ego_lane_boundaries_stable()` and pass into
    /// `LaneMeasurement.boundary_coherence` for the lateral detector.
    pub fn boundary_coherence(&self) -> f32 {
        self.last_boundary_coherence
    }

    /// v4.13: Get the polynomial curvature estimate from the last frame.
    ///
    /// Returns None if no valid polynomial fit was possible (insufficient
    /// mask points, poor fit quality, or no lane markings detected).
    pub fn curvature_estimate(&self) -> Option<&CurvatureEstimate> {
        self.last_curvature.as_ref()
    }

    /// v6.1d: Access the raw lane-line markings from the last inference run.
    ///
    /// These are available even when `estimate_ego_lane_boundaries` returns None,
    /// because the YOLO-seg model detected individual markings that just didn't
    /// form a valid boundary pair. Use as fallback for crossing detection when
    /// the detection cache has expired.
    pub fn last_lane_markings(&self) -> &[DetectedRoadMarking] {
        &self.last_lane_markings
    }

    /// v4.13: Compute polynomial curvature from stored mask data.
    ///
    /// Finds the markings whose bbox centers are closest to the selected
    /// left/right boundary positions, extracts their mask spines, fits
    /// quadratic polynomials, and compares curvatures.
    fn compute_mask_curvature(
        &self,
        raw_left_x: f32,
        raw_right_x: f32,
    ) -> Option<CurvatureEstimate> {
        if self.last_lane_markings.is_empty() || self.last_image_dims == (0, 0) {
            return None;
        }

        let (orig_w, orig_h) = self.last_image_dims;
        let params = MaskTransformParams::from_image_dims(orig_w, orig_h);

        // Find the marking closest to each selected boundary position.
        // A marking's center_x is (bbox[0] + bbox[2]) / 2.0.
        let left_marking = self.find_closest_marking(raw_left_x);
        let right_marking = self.find_closest_marking(raw_right_x);

        // Don't use the same marking for both boundaries
        let (left_input, right_input) = match (left_marking, right_marking) {
            (Some(lm), Some(rm)) => {
                let same = std::ptr::eq(lm, rm);
                if same {
                    // Only one marking — assign to whichever side is closer
                    let lcx = (lm.bbox[0] + lm.bbox[2]) / 2.0;
                    if (lcx - raw_left_x).abs() < (lcx - raw_right_x).abs() {
                        (Some(lm), None)
                    } else {
                        (None, Some(rm))
                    }
                } else {
                    (Some(lm), Some(rm))
                }
            }
            pair => pair,
        };

        let left_mi = left_input.map(|m| MaskInput {
            mask: &m.mask,
            mask_w: m.mask_width,
            mask_h: m.mask_height,
            bbox: m.bbox,
            center_x: (m.bbox[0] + m.bbox[2]) / 2.0,
        });
        let right_mi = right_input.map(|m| MaskInput {
            mask: &m.mask,
            mask_w: m.mask_width,
            mask_h: m.mask_height,
            bbox: m.bbox,
            center_x: (m.bbox[0] + m.bbox[2]) / 2.0,
        });

        let est = curvature_estimator::estimate_curvature_from_masks(
            left_mi.as_ref(),
            right_mi.as_ref(),
            &params,
        );

        // Only return if we got at least one valid polynomial
        if est.left_poly.is_some() || est.right_poly.is_some() {
            Some(est)
        } else {
            None
        }
    }

    /// Find the marking whose bbox center-x is closest to `target_x`.
    /// Returns None if no markings are stored or closest is > 100px away.
    fn find_closest_marking(&self, target_x: f32) -> Option<&DetectedRoadMarking> {
        const MAX_MATCH_DISTANCE: f32 = 100.0;

        self.last_lane_markings
            .iter()
            .map(|m| {
                let cx = (m.bbox[0] + m.bbox[2]) / 2.0;
                let dist = (cx - target_x).abs();
                (m, dist)
            })
            .filter(|(_, dist)| *dist < MAX_MATCH_DISTANCE)
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(m, _)| m)
    }

    // -----------------------------------------------------------------------
    // EGO LANE BOUNDARY ESTIMATION  v2
    // -----------------------------------------------------------------------

    pub fn estimate_ego_lane_boundaries(
        &mut self,
        frame: &[u8],
        width: usize,
        height: usize,
        vehicle_center_x: f32,
    ) -> Result<Option<(f32, f32, f32)>> {
        let markings = self.get_markings_only(frame, width, height, 0.20)?;

        if markings.is_empty() {
            return Ok(None);
        }

        let lane_lines: Vec<&DetectedRoadMarking> = markings
            .iter()
            .filter(|m| is_lane_line_class(m.class_id))
            .collect();

        if lane_lines.is_empty() {
            self.last_lane_markings.clear(); // v4.13
            return Ok(None);
        }

        // v4.13: Store raw markings for curvature estimation (needs mask data).
        self.last_lane_markings = lane_lines.iter().map(|m| (*m).clone()).collect();
        self.last_image_dims = (width, height);

        let merge_threshold = width as f32 * MERGE_THRESHOLD_RATIO;
        let merged = merge_nearby_markings(&lane_lines, merge_threshold);

        debug!(
            "🔍 YOLO boundary: {} raw → {} merged [{}]",
            lane_lines.len(),
            merged.len(),
            merged
                .iter()
                .map(|m| format!(
                    "{}@{:.0}({:.0}%)",
                    m.class_name,
                    m.center_x,
                    m.confidence * 100.0
                ))
                .collect::<Vec<_>>()
                .join(", ")
        );

        if merged.len() >= 2 {
            if let Some(result) = self.find_best_lane_pair(&merged, vehicle_center_x, width) {
                return Ok(Some(result));
            }
        }

        if let Some(result) = self.single_marking_fallback(&merged, vehicle_center_x, width) {
            return Ok(Some(result));
        }

        debug!(
            "⚠️ YOLO boundary: no valid config from {} merged positions",
            merged.len()
        );
        Ok(None)
    }

    fn find_best_lane_pair(
        &mut self,
        merged: &[MergedMarkingPosition],
        vehicle_center_x: f32,
        frame_width: usize,
    ) -> Option<(f32, f32, f32)> {
        let mut best: Option<(f32, f32, f32, f32)> = None;

        for i in 0..merged.len() {
            for j in (i + 1)..merged.len() {
                let left_x = merged[i].center_x;
                let right_x = merged[j].center_x;
                let lane_width = right_x - left_x;

                if lane_width < MIN_LANE_WIDTH || lane_width > MAX_LANE_WIDTH {
                    continue;
                }

                let lane_center = (left_x + right_x) / 2.0;
                let center_offset = (lane_center - vehicle_center_x).abs();

                let width_score =
                    1.0 - ((lane_width - IDEAL_LANE_WIDTH).abs() / IDEAL_LANE_WIDTH).min(1.0);
                let center_score = 1.0 - (center_offset / (frame_width as f32 * 0.35)).min(1.0);
                let conf = (merged[i].confidence + merged[j].confidence) / 2.0;
                let score = width_score * 0.25 + center_score * 0.40 + conf * 0.35;

                if best.is_none() || score > best.unwrap().3 {
                    best = Some((left_x, right_x, conf, score));
                }
            }
        }

        if let Some((left_x, right_x, conf, score)) = best {
            let lane_width = right_x - left_x;
            self.record_lane_width(lane_width);
            debug!(
                "✅ YOLO boundary (pair): L={:.0}, R={:.0}, W={:.0}, conf={:.2}, score={:.2}",
                left_x, right_x, lane_width, conf, score
            );
            Some((left_x, right_x, conf))
        } else {
            None
        }
    }

    fn single_marking_fallback(
        &self,
        merged: &[MergedMarkingPosition],
        vehicle_center_x: f32,
        frame_width: usize,
    ) -> Option<(f32, f32, f32)> {
        let best = merged.iter().min_by(|a, b| {
            let da = (a.center_x - vehicle_center_x).abs();
            let db = (b.center_x - vehicle_center_x).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })?;

        let est_width = self.estimated_lane_width();
        let conf = best.confidence * SINGLE_BOUNDARY_CONFIDENCE_PENALTY;

        if best.center_x < vehicle_center_x {
            let right_x = best.center_x + est_width;
            let right_x = if right_x > frame_width as f32 {
                debug!(
                    "⚠️ YOLO boundary (left-only): est_R={:.0} > frame_w={}, clamping",
                    right_x, frame_width
                );
                (frame_width as f32 - 10.0).max(best.center_x + MIN_LANE_WIDTH)
            } else {
                right_x
            };
            let effective_conf = if right_x >= frame_width as f32 - 10.0 {
                conf * 0.8
            } else {
                conf
            };
            debug!(
                "⚠️ YOLO boundary (left-only): L={:.0} ({}), est_R={:.0}, est_W={:.0}, conf={:.2}",
                best.center_x, best.class_name, right_x, est_width, effective_conf
            );
            Some((best.center_x, right_x, effective_conf))
        } else {
            let left_x = best.center_x - est_width;
            let left_x = if left_x < 0.0 {
                debug!(
                    "⚠️ YOLO boundary (right-only): est_L={:.0} < 0, clamping",
                    left_x
                );
                (10.0_f32).min(best.center_x - MIN_LANE_WIDTH).max(0.0)
            } else {
                left_x
            };
            let effective_conf = if left_x <= 10.0 { conf * 0.8 } else { conf };
            debug!(
                "⚠️ YOLO boundary (right-only): est_L={:.0}, R={:.0} ({}), est_W={:.0}, conf={:.2}",
                left_x, best.center_x, best.class_name, est_width, effective_conf
            );
            Some((left_x, best.center_x, effective_conf))
        }
    }

    pub fn get_markings_only(
        &mut self,
        frame: &[u8],
        width: usize,
        height: usize,
        confidence_threshold: f32,
    ) -> Result<Vec<DetectedRoadMarking>> {
        let (input, scale, pad_x, pad_y) = self.preprocess(frame, width, height)?;
        let (box_output, mask_proto, _) = self.infer(&input)?;
        self.postprocess(
            &box_output,
            &mask_proto,
            &[],
            scale,
            pad_x,
            pad_y,
            width,
            height,
            confidence_threshold,
            frame,
        )
    }

    pub fn set_ego_bbox_ratio(&mut self, x1: f32, y1: f32, x2: f32, y2: f32) {
        self.ego_bbox_ratio = [x1, y1, x2, y2];
    }

    pub fn analyze_frame_fused(
        &mut self,
        frame: &[u8],
        width: usize,
        height: usize,
        frame_id: u64,
        confidence_threshold: f32,
        vehicle_offset_px: f32,
        lane_width: Option<f32>,
        left_lane_x: Option<f32>,
        right_lane_x: Option<f32>,
        crossing_side: CrossingSide,
    ) -> Result<FusedLegalityResult> {
        let (input, scale, pad_x, pad_y) = self.preprocess(frame, width, height)?;
        let (box_output, mask_proto, _) = self.infer(&input)?;
        let all_markings = self.postprocess(
            &box_output,
            &mask_proto,
            &[],
            scale,
            pad_x,
            pad_y,
            width,
            height,
            confidence_threshold,
            frame,
        )?;

        let offset_pct = match lane_width {
            Some(w) if w > 50.0 => (vehicle_offset_px / w).abs(),
            _ => 0.0,
        };

        let lane_model_confirms_crossing = crossing_side != CrossingSide::None && offset_pct > 0.25;

        if !lane_model_confirms_crossing {
            return Ok(FusedLegalityResult {
                verdict: LineLegality::Unknown,
                crossing_confirmed_by_lane_model: false,
                line_type_from_seg_model: None,
                vehicle_offset_pct: offset_pct,
                all_markings,
                ego_intersects_marking: false,
            });
        }

        let crossing_line = self.find_crossing_line(
            &all_markings,
            crossing_side,
            left_lane_x,
            right_lane_x,
            width,
            height,
        );
        let ego_intersects_marking = crossing_line.is_some();
        let (verdict, intersecting_line) = match crossing_line {
            Some(marking) => {
                let confirmed = self.temporal_filter.update(frame_id, marking.legality);
                if confirmed {
                    (marking.legality, Some(marking))
                } else {
                    (LineLegality::Unknown, Some(marking))
                }
            }
            None => {
                self.temporal_filter.update(frame_id, LineLegality::Unknown);
                (LineLegality::Unknown, None)
            }
        };

        Ok(FusedLegalityResult {
            verdict,
            crossing_confirmed_by_lane_model: true,
            line_type_from_seg_model: intersecting_line,
            vehicle_offset_pct: offset_pct,
            all_markings,
            ego_intersects_marking,
        })
    }

    fn find_crossing_line(
        &self,
        markings: &[DetectedRoadMarking],
        crossing_side: CrossingSide,
        left_lane_x: Option<f32>,
        right_lane_x: Option<f32>,
        frame_width: usize,
        _frame_height: usize,
    ) -> Option<DetectedRoadMarking> {
        let boundary_x = match crossing_side {
            CrossingSide::Left => left_lane_x,
            CrossingSide::Right => right_lane_x,
            CrossingSide::None => return None,
        }?;

        let tolerance = frame_width as f32 * 0.12;
        let mut best_match: Option<(f32, &DetectedRoadMarking)> = None;

        for marking in markings {
            if !ILLEGAL_CLASS_IDS.contains(&marking.class_id)
                && !LEGAL_CLASS_IDS.contains(&marking.class_id)
                && marking.class_id != 99
            {
                continue;
            }
            let marking_center_x = (marking.bbox[0] + marking.bbox[2]) / 2.0;
            let distance = (marking_center_x - boundary_x).abs();
            if distance < tolerance {
                if best_match.is_none() || distance < best_match.unwrap().0 {
                    best_match = Some((distance, marking));
                }
            }
        }

        best_match.map(|(_, marking)| marking.clone())
    }

    fn preprocess(
        &self,
        src: &[u8],
        src_w: usize,
        src_h: usize,
    ) -> Result<(Vec<f32>, f32, f32, f32)> {
        let target = SEG_INPUT_SIZE;
        let scale = (target as f32 / src_w as f32).min(target as f32 / src_h as f32);
        let scaled_w = (src_w as f32 * scale) as usize;
        let scaled_h = (src_h as f32 * scale) as usize;
        let pad_x = (target - scaled_w) as f32 / 2.0;
        let pad_y = (target - scaled_h) as f32 / 2.0;

        let resized = resize_bilinear(src, src_w, src_h, scaled_w, scaled_h);
        let mut canvas = vec![114u8; target * target * 3];
        for y in 0..scaled_h {
            for x in 0..scaled_w {
                let src_idx = (y * scaled_w + x) * 3;
                let dst_x = x + pad_x as usize;
                let dst_y = y + pad_y as usize;
                if dst_x < target && dst_y < target {
                    let dst_idx = (dst_y * target + dst_x) * 3;
                    canvas[dst_idx..dst_idx + 3].copy_from_slice(&resized[src_idx..src_idx + 3]);
                }
            }
        }

        let mut input = vec![0.0f32; 3 * target * target];
        for c in 0..3 {
            for h in 0..target {
                for w in 0..target {
                    let hwc_idx = (h * target + w) * 3 + c;
                    let chw_idx = c * target * target + h * target + w;
                    input[chw_idx] = canvas[hwc_idx] as f32 / 255.0;
                }
            }
        }
        Ok((input, scale, pad_x, pad_y))
    }

    fn infer(&mut self, input: &[f32]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let shape = [1, 3, SEG_INPUT_SIZE, SEG_INPUT_SIZE];
        let input_value =
            ort::value::Value::from_array((shape.as_slice(), input.to_vec().into_boxed_slice()))?;
        let outputs = self.session.run(ort::inputs!["images" => input_value])?;
        let output0 = &outputs[0];
        let (_, data0) = output0.try_extract_tensor::<f32>()?;
        let output1 = &outputs[1];
        let (_, data1) = output1.try_extract_tensor::<f32>()?;
        Ok((data0.to_vec(), data1.to_vec(), Vec::new()))
    }

    fn postprocess(
        &self,
        output: &[f32],
        mask_proto: &[f32],
        _mask_coeffs: &[f32],
        scale: f32,
        pad_x: f32,
        pad_y: f32,
        orig_w: usize,
        orig_h: usize,
        conf_thresh: f32,
        frame_rgb: &[u8],
    ) -> Result<Vec<DetectedRoadMarking>> {
        let mut detections = Vec::new();
        let num_detections = 8400;

        for i in 0..num_detections {
            let cx = output[i];
            let cy = output[num_detections + i];
            let w = output[num_detections * 2 + i];
            let h = output[num_detections * 3 + i];

            let mut max_conf = 0.0f32;
            let mut best_class = 0;
            for c in 0..NUM_CLASSES {
                let conf = output[num_detections * (4 + c) + i];
                if conf > max_conf {
                    max_conf = conf;
                    best_class = c;
                }
            }
            if max_conf < conf_thresh {
                continue;
            }

            let mut mask_coeffs_det = [0.0f32; 32];
            for mc in 0..32 {
                mask_coeffs_det[mc] = output[num_detections * (4 + NUM_CLASSES + mc) + i];
            }

            let x1 = ((cx - w / 2.0) - pad_x) / scale;
            let y1 = ((cy - h / 2.0) - pad_y) / scale;
            let x2 = ((cx + w / 2.0) - pad_x) / scale;
            let y2 = ((cy + h / 2.0) - pad_y) / scale;

            let x1 = x1.max(0.0).min(orig_w as f32);
            let y1 = y1.max(0.0).min(orig_h as f32);
            let x2 = x2.max(0.0).min(orig_w as f32);
            let y2 = y2.max(0.0).min(orig_h as f32);

            let mask = self.generate_mask(mask_proto, &mask_coeffs_det);

            detections.push(DetectedRoadMarking {
                class_id: best_class,
                class_name: class_id_to_name(best_class).to_string(),
                confidence: max_conf,
                bbox: [x1, y1, x2, y2],
                legality: class_id_to_legality(best_class),
                mask,
                mask_width: MASK_SIZE,
                mask_height: MASK_SIZE,
            });
        }

        // Step 1: Color verification (existing)
        for marking in &mut detections {
            verify_line_color(frame_rgb, orig_w, orig_h, marking);
        }

        // Step 2: v6.1g — Mask-based mixed line detection
        // Check if any solid_double_yellow (class 8) is actually a mixed line
        // by analyzing the segmentation mask for dashed stripe patterns.
        // Must run BEFORE merge_composite_lines so that reclassified mixed
        // lines (now class 99) don't get suppressed by double-solid guards.
        for marking in &mut detections {
            analyze_double_line_for_mixed(marking, scale, pad_x, pad_y);
        }

        // Step 3: Merge separate solid+dashed detection pairs into mixed (existing)
        let detections = merge_composite_lines(detections);
        // Step 4: NMS (existing)
        let detections = nms_markings(detections, 0.45);
        Ok(detections)
    }

    fn generate_mask(&self, mask_proto: &[f32], mask_coeffs: &[f32; 32]) -> Vec<u8> {
        let mh = MASK_SIZE;
        let mw = MASK_SIZE;
        let mut mask = vec![0u8; mh * mw];
        for y in 0..mh {
            for x in 0..mw {
                let mut sum = 0.0f32;
                for c in 0..32 {
                    let proto_idx = c * mh * mw + y * mw + x;
                    if proto_idx < mask_proto.len() {
                        sum += mask_coeffs[c] * mask_proto[proto_idx];
                    }
                }
                let prob = 1.0 / (1.0 + (-sum).exp());
                if prob > 0.5 {
                    mask[y * mw + x] = 255;
                }
            }
        }
        mask
    }
}

// ===========================================================================
// UTILS
// ===========================================================================

fn resize_bilinear(src: &[u8], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<u8> {
    let mut dst = vec![0u8; dst_h * dst_w * 3];
    let x_ratio = src_w as f32 / dst_w as f32;
    let y_ratio = src_h as f32 / dst_h as f32;
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = dx as f32 * x_ratio;
            let sy = dy as f32 * y_ratio;
            let sx0 = sx.floor() as usize;
            let sy0 = sy.floor() as usize;
            let sx1 = (sx0 + 1).min(src_w - 1);
            let sy1 = (sy0 + 1).min(src_h - 1);
            let fx = sx - sx0 as f32;
            let fy = sy - sy0 as f32;
            for c in 0..3 {
                let p00 = src[(sy0 * src_w + sx0) * 3 + c] as f32;
                let p10 = src[(sy0 * src_w + sx1) * 3 + c] as f32;
                let p01 = src[(sy1 * src_w + sx0) * 3 + c] as f32;
                let p11 = src[(sy1 * src_w + sx1) * 3 + c] as f32;
                let val = p00 * (1.0 - fx) * (1.0 - fy)
                    + p10 * fx * (1.0 - fy)
                    + p01 * (1.0 - fx) * fy
                    + p11 * fx * fy;
                dst[(dy * dst_w + dx) * 3 + c] = val.round() as u8;
            }
        }
    }
    dst
}

fn nms_markings(mut dets: Vec<DetectedRoadMarking>, iou_thresh: f32) -> Vec<DetectedRoadMarking> {
    if dets.is_empty() {
        return dets;
    }
    dets.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    let mut keep = Vec::new();
    let mut suppressed_indices = HashSet::new();
    for i in 0..dets.len() {
        if suppressed_indices.contains(&i) {
            continue;
        }
        let current = &dets[i];
        let mut should_keep_current = true;
        for j in 0..dets.len() {
            if i == j || suppressed_indices.contains(&j) {
                continue;
            }
            let other = &dets[j];
            let iou = calculate_iou_arr(&current.bbox, &other.bbox);
            if iou > iou_thresh {
                let is_curr_mixed = current.class_id == 99;
                let is_other_mixed = other.class_id == 99;
                if is_other_mixed && !is_curr_mixed {
                    should_keep_current = false;
                    break;
                } else {
                    suppressed_indices.insert(j);
                }
            }
        }
        if should_keep_current {
            keep.push(current.clone());
        }
    }
    keep
}

fn calculate_iou_arr(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    let union = area_a + area_b - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}
