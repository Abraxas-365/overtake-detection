// src/lane_legality.rs

use anyhow::Result;
use ort::{
    execution_providers::CUDAExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::collections::VecDeque;
use tracing::{debug, info, warn};

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

// Ego lane boundary estimation constants
const MERGE_THRESHOLD_RATIO: f32 = 0.06; // 6% of frame width — merge detections closer than this
const MIN_LANE_WIDTH: f32 = 200.0;
const MAX_LANE_WIDTH: f32 = 900.0;
const IDEAL_LANE_WIDTH: f32 = 450.0;
const DEFAULT_SINGLE_MARKING_WIDTH: f32 = 450.0;
const WIDTH_HISTORY_SIZE: usize = 20;
const SINGLE_BOUNDARY_CONFIDENCE_PENALTY: f32 = 0.65;

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

            let is_pair_double_dashed = (matches!(a.class_id, 7 | 8)
                && matches!(b.class_id, 9 | 10))
                || (matches!(a.class_id, 9 | 10) && matches!(b.class_id, 7 | 8));

            let is_pair_solid_dashed = (matches!(a.class_id, 4 | 5)
                && matches!(b.class_id, 9 | 10))
                || (matches!(a.class_id, 9 | 10) && matches!(b.class_id, 4 | 5));

            if is_pair_double_dashed || is_pair_solid_dashed {
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
// MERGED MARKING POSITION (for boundary estimation)
// ===========================================================================

/// A spatially-merged lane line position derived from one or more overlapping
/// YOLO-seg detections. When the model detects both "dashed_single_white" and
/// "solid_single_white" for the same physical center line, we merge them here
/// so the boundary estimator sees one logical marking, not two.
#[derive(Debug, Clone)]
struct MergedMarkingPosition {
    center_x: f32,
    confidence: f32,
    class_name: String,
    /// How many raw detections contributed to this position
    source_count: u32,
}

/// Merge spatially-close lane line detections into logical marking positions.
/// This prevents the same physical line (detected as e.g. solid + dashed) from
/// being treated as two separate lane boundaries.
fn merge_nearby_markings(
    markings: &[&DetectedRoadMarking],
    merge_threshold: f32,
) -> Vec<MergedMarkingPosition> {
    // Sort by center X
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
                // Same physical line — merge via weighted average
                let total_conf = last.confidence * last.source_count as f32 + conf;
                let new_count = last.source_count + 1;
                last.center_x =
                    (last.center_x * last.confidence + cx * conf) / (last.confidence + conf);
                last.confidence = total_conf / new_count as f32;
                last.source_count = new_count;
                // Keep the higher-confidence class name
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

    // Lane width memory for single-boundary fallback
    lane_width_history: VecDeque<f32>,
    last_valid_width: Option<f32>,
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
        })
    }

    // -----------------------------------------------------------------------
    // WIDTH MEMORY
    // -----------------------------------------------------------------------

    /// Record a successful two-boundary lane width measurement.
    fn record_lane_width(&mut self, width: f32) {
        if width >= MIN_LANE_WIDTH && width <= MAX_LANE_WIDTH {
            self.lane_width_history.push_back(width);
            if self.lane_width_history.len() > WIDTH_HISTORY_SIZE {
                self.lane_width_history.pop_front();
            }
            self.last_valid_width = Some(width);
        }
    }

    /// Get the best estimate for lane width based on history.
    /// Returns median of recent measurements, or default if no history.
    fn estimated_lane_width(&self) -> f32 {
        if self.lane_width_history.len() < 3 {
            return self
                .last_valid_width
                .unwrap_or(DEFAULT_SINGLE_MARKING_WIDTH);
        }
        let mut sorted: Vec<f32> = self.lane_width_history.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[sorted.len() / 2]
    }

    // -----------------------------------------------------------------------
    // EGO LANE BOUNDARY ESTIMATION (v2 — robust for single-line roads)
    // -----------------------------------------------------------------------
    //
    // Previous version split markings into left/right of frame center and
    // required one on each side. On Peruvian desert highways the center line
    // and the road edge often both sit LEFT of the camera center on curves,
    // causing a 100% miss.
    //
    // v2 approach:
    //   1. Merge spatially-close detections (same physical line detected
    //      as both solid and dashed).
    //   2. Try every pair of merged positions — pick the one that forms a
    //      valid lane width and best straddles the vehicle center.
    //   3. If no valid pair, fall back to single-marking estimation using
    //      width memory instead of a hardcoded 500px.
    //   4. No harsh edge-of-frame rejection (normal on curves).
    //
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

        // Filter for lane line classes only
        let lane_lines: Vec<&DetectedRoadMarking> = markings
            .iter()
            .filter(|m| is_lane_line_class(m.class_id))
            .collect();

        if lane_lines.is_empty() {
            return Ok(None);
        }

        let merge_threshold = width as f32 * MERGE_THRESHOLD_RATIO;
        let merged = merge_nearby_markings(&lane_lines, merge_threshold);

        debug!(
            "🔍 YOLO boundary: {} raw detections → {} merged positions [{}]",
            lane_lines.len(),
            merged.len(),
            merged
                .iter()
                .map(|m| format!("{:.0}({:.0}%)", m.center_x, m.confidence * 100.0))
                .collect::<Vec<_>>()
                .join(", ")
        );

        // ── ATTEMPT 1: Find the best pair of markings that form a lane ──

        if merged.len() >= 2 {
            if let Some(result) = self.find_best_lane_pair(&merged, vehicle_center_x, width) {
                return Ok(Some(result));
            }
        }

        // ── ATTEMPT 2: Single marking fallback ──

        if let Some(result) = self.single_marking_fallback(&merged, vehicle_center_x, width) {
            return Ok(Some(result));
        }

        debug!(
            "⚠️ YOLO boundary: no valid configuration from {} merged positions",
            merged.len()
        );
        Ok(None)
    }

    /// Try every pair of merged marking positions and pick the best lane.
    fn find_best_lane_pair(
        &mut self,
        merged: &[MergedMarkingPosition],
        vehicle_center_x: f32,
        frame_width: usize,
    ) -> Option<(f32, f32, f32)> {
        let mut best: Option<(f32, f32, f32, f32)> = None; // (left_x, right_x, conf, score)

        for i in 0..merged.len() {
            for j in (i + 1)..merged.len() {
                let left_x = merged[i].center_x;
                let right_x = merged[j].center_x;
                let lane_width = right_x - left_x;

                // Basic width sanity
                if lane_width < MIN_LANE_WIDTH || lane_width > MAX_LANE_WIDTH {
                    continue;
                }

                let lane_center = (left_x + right_x) / 2.0;
                let center_offset = (lane_center - vehicle_center_x).abs();

                // Score components:
                // 1. Width plausibility — prefer widths near ideal (450px)
                let width_score =
                    1.0 - ((lane_width - IDEAL_LANE_WIDTH).abs() / IDEAL_LANE_WIDTH).min(1.0);
                // 2. Center alignment — prefer pairs straddling vehicle center
                let center_score = 1.0 - (center_offset / (frame_width as f32 * 0.35)).min(1.0);
                // 3. Detection confidence
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

    /// When only one marking (or no valid pair), estimate the other boundary
    /// using width history. Pick the marking closest to vehicle center.
    fn single_marking_fallback(
        &self,
        merged: &[MergedMarkingPosition],
        vehicle_center_x: f32,
        frame_width: usize,
    ) -> Option<(f32, f32, f32)> {
        // Pick the marking closest to vehicle center
        let best = merged.iter().min_by(|a, b| {
            let da = (a.center_x - vehicle_center_x).abs();
            let db = (b.center_x - vehicle_center_x).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })?;

        let est_width = self.estimated_lane_width();
        let conf = best.confidence * SINGLE_BOUNDARY_CONFIDENCE_PENALTY;

        if best.center_x < vehicle_center_x {
            // Marking is LEFT of vehicle → it's the left boundary (center line)
            let right_x = best.center_x + est_width;

            // Soft sanity: warn but don't reject if estimated edge is far right.
            // On curves the road edge CAN be near the frame edge.
            if right_x > frame_width as f32 {
                debug!(
                    "⚠️ YOLO boundary (left-only): est_R={:.0} > frame_width={}, clamping",
                    right_x, frame_width
                );
                // Clamp but still return a result — better than nothing
                let right_x = (frame_width as f32 - 10.0).max(best.center_x + MIN_LANE_WIDTH);
                return Some((best.center_x, right_x, conf * 0.8));
            }

            debug!(
                "⚠️ YOLO boundary (left-only): L={:.0} ({}), est_R={:.0}, est_W={:.0}, conf={:.2}",
                best.center_x, best.class_name, right_x, est_width, conf
            );
            Some((best.center_x, right_x, conf))
        } else {
            // Marking is RIGHT of vehicle → it's the right boundary (road edge)
            let left_x = best.center_x - est_width;

            if left_x < 0.0 {
                debug!(
                    "⚠️ YOLO boundary (right-only): est_L={:.0} < 0, clamping",
                    left_x
                );
                let left_x = (10.0_f32).min(best.center_x - MIN_LANE_WIDTH);
                return Some((left_x.max(0.0), best.center_x, conf * 0.8));
            }

            debug!(
                "⚠️ YOLO boundary (right-only): est_L={:.0}, R={:.0} ({}), est_W={:.0}, conf={:.2}",
                left_x, best.center_x, best.class_name, est_width, conf
            );
            Some((left_x, best.center_x, conf))
        }
    }

    /// Get all detected road markings for a frame (for cross-validation)
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

        for marking in &mut detections {
            verify_line_color(frame_rgb, orig_w, orig_h, marking);
        }

        let detections = merge_composite_lines(detections);
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

