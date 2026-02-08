// src/lane_legality.rs

use anyhow::Result;
use ort::{
    execution_providers::CUDAExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
};
use serde::{Deserialize, Serialize};
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

// ============================================================================
// CROSSING SIDE (from lane model)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossingSide {
    None,
    Left,
    Right,
}

// ============================================================================
// TYPES
// ============================================================================

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

// ============================================================================
// FUSED RESULT (combines both models)
// ============================================================================

#[derive(Debug, Clone)]
pub struct FusedLegalityResult {
    /// Final verdict after fusion
    pub verdict: LineLegality,
    /// Whether the lane model confirms the vehicle is crossing
    pub crossing_confirmed_by_lane_model: bool,
    /// The line type detected by seg model at the crossing boundary
    pub line_type_from_seg_model: Option<DetectedRoadMarking>,
    /// Vehicle offset as percentage of lane width
    pub vehicle_offset_pct: f32,
    /// All detected markings (for visualization)
    pub all_markings: Vec<DetectedRoadMarking>,
    /// Whether ego bbox intersects any marking mask
    pub ego_intersects_marking: bool,
}

// ============================================================================
// TEMPORAL FILTER — prevents single-frame false positives
// ============================================================================

struct TemporalViolationFilter {
    /// Recent violation verdicts (ring buffer)
    recent_verdicts: Vec<(u64, LineLegality)>,
    /// How many consecutive illegal frames needed to confirm
    min_consecutive: usize,
    /// Max frame gap to consider "consecutive"
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

        // Keep only recent entries
        if self.recent_verdicts.len() > 20 {
            self.recent_verdicts.remove(0);
        }

        if !verdict.is_illegal() {
            return false;
        }

        // Count consecutive illegal frames (allowing small gaps)
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

    fn reset(&mut self) {
        self.recent_verdicts.clear();
    }
}

// ============================================================================
// CLASS ID MAPPING
// ============================================================================

fn class_id_to_name(class_id: usize) -> &'static str {
    match class_id {
        4 => "solid_single_white",
        5 => "solid_single_yellow",
        6 => "solid_single_red",
        7 => "solid_double_white",
        8 => "solid_double_yellow",
        9 => "dashed_single_white",
        10 => "dashed_single_yellow",
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

// ============================================================================
// COLOR VERIFICATION — Fixes yellow detected as white
// ============================================================================

/// Verify/correct line color by sampling actual pixel values from the marked region
/// This fixes the common issue where yellow center lines are misclassified as white
fn verify_line_color(
    frame_rgb: &[u8],
    frame_width: usize,
    frame_height: usize,
    marking: &mut DetectedRoadMarking,
) {
    // Only correct white classes that might be yellow
    let is_white_class = matches!(marking.class_id, 4 | 7 | 9);
    if !is_white_class {
        return;
    }

    // Sample pixels from the bbox center region
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

                // Brightness filter: keep > 60 to catch dim lines
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

    // R/B ratio calculation (safe division)
    let r_b_ratio = if avg_b > 1.0 { avg_r / avg_b } else { 2.0 };
    let g_b_ratio = if avg_b > 1.0 { avg_g / avg_b } else { 2.0 };

    // ✅ CALIBRATED THRESHOLDS based on your logs:
    // Yellow lines in logs: R/B ~1.15 to 1.25
    // White lines in logs:  R/B ~0.98 to 1.02
    // New Threshold:        > 1.08 (Safe gap)
    let is_yellow = avg_b > 1.0
        && r_b_ratio > 1.08      // Reduced from 1.3
        && g_b_ratio > 1.05      // Reduced from 1.1
        && avg_r > 90.0          // Reduced from 100.0
        && avg_g > 90.0          // Reduced from 100.0
        && (avg_r - avg_b) > 15.0; // Reduced difference check

    if is_yellow {
        let old_class = marking.class_id;
        let new_class = match marking.class_id {
            4 => 5,  // solid_single_white → solid_single_yellow
            7 => 8,  // solid_double_white → solid_double_yellow
            9 => 10, // dashed_single_white → dashed_single_yellow
            _ => marking.class_id,
        };

        if new_class != old_class {
            info!(
                "🎨 Color correction: {} → {} (R={:.0}, G={:.0}, B={:.0}, R/B={:.2})",
                class_id_to_name(old_class),
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
    } else {
        // Debug logging to keep verifying
        debug!(
            "Checked {}: Not yellow (R/B={:.2}, need >1.08)",
            marking.class_name, r_b_ratio
        );
    }
}

// ============================================================================
// LANE LEGALITY DETECTOR
// ============================================================================

pub struct LaneLegalityDetector {
    session: Session,
    ego_bbox_ratio: [f32; 4],
    temporal_filter: TemporalViolationFilter,
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
            temporal_filter: TemporalViolationFilter::new(
                2, // Need 2+ consecutive illegal frames
                6, // Max 6 frame gap (at interval=3, this means 2 detections)
            ),
        })
    }

    pub fn set_ego_bbox_ratio(&mut self, x1: f32, y1: f32, x2: f32, y2: f32) {
        self.ego_bbox_ratio = [x1, y1, x2, y2];
    }

    /// FUSED analysis: combines seg model + lane model for accurate results
    pub fn analyze_frame_fused(
        &mut self,
        frame: &[u8],
        width: usize,
        height: usize,
        frame_id: u64,
        confidence_threshold: f32,
        // From lane detection model (UFLDv2):
        vehicle_offset_px: f32,
        lane_width: Option<f32>,
        left_lane_x: Option<f32>,
        right_lane_x: Option<f32>,
        crossing_side: CrossingSide,
    ) -> Result<FusedLegalityResult> {
        // 1. Run seg model to detect all road markings
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
            frame, // ✅ Pass original frame for color verification
        )?;

        // 2. Calculate vehicle offset percentage
        let offset_pct = match lane_width {
            Some(w) if w > 50.0 => (vehicle_offset_px / w).abs(),
            _ => 0.0,
        };

        // 3. Determine if lane model confirms a crossing is happening
        //    Key insight: only flag illegal if the lane model says we're
        //    actually leaving our lane (offset > 25% = significant drift)
        let lane_model_confirms_crossing = crossing_side != CrossingSide::None && offset_pct > 0.25;

        // 4. If no crossing confirmed by lane model, return clean result
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

        // 5. Find the line being crossed based on WHICH SIDE we're crossing
        //    Use lane boundaries from UFLDv2 to identify the correct line
        let crossing_line = self.find_crossing_line(
            &all_markings,
            crossing_side,
            left_lane_x,
            right_lane_x,
            width,
            height,
        );

        // Store the boolean before consuming crossing_line
        let ego_intersects_marking = crossing_line.is_some();

        let (verdict, intersecting_line) = match crossing_line {
            Some(marking) => {
                let raw_verdict = marking.legality;

                // 6. Apply temporal filtering — need consecutive frames
                let confirmed = self.temporal_filter.update(frame_id, raw_verdict);

                if confirmed {
                    (raw_verdict, Some(marking))
                } else {
                    info!(
                        "Temporal filter: {} not yet confirmed (need {} consecutive)",
                        raw_verdict.as_str(),
                        2
                    );
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

    /// Find the line that corresponds to the boundary being crossed
    /// Instead of checking ego bbox overlap, match seg model detections
    /// to the lane boundary position from UFLDv2
    fn find_crossing_line(
        &self,
        markings: &[DetectedRoadMarking],
        crossing_side: CrossingSide,
        left_lane_x: Option<f32>,
        right_lane_x: Option<f32>,
        frame_width: usize,
        _frame_height: usize,
    ) -> Option<DetectedRoadMarking> {
        // Determine which lane boundary we're crossing
        let boundary_x = match crossing_side {
            CrossingSide::Left => left_lane_x,
            CrossingSide::Right => right_lane_x,
            CrossingSide::None => return None,
        };

        let boundary_x = match boundary_x {
            Some(x) => x,
            None => return None,
        };

        // Find the seg model detection whose bbox center is closest
        // to the lane boundary we're crossing
        let tolerance = frame_width as f32 * 0.12; // 12% of frame width

        let mut best_match: Option<(f32, &DetectedRoadMarking)> = None;

        for marking in markings {
            // Only consider lane line classes (not other markings)
            if !ILLEGAL_CLASS_IDS.contains(&marking.class_id)
                && !LEGAL_CLASS_IDS.contains(&marking.class_id)
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

        best_match.map(|(dist, marking)| {
            info!(
                "Matched seg detection '{}' (conf={:.0}%) to {} boundary at x={:.0} (dist={:.0}px)",
                marking.class_name,
                marking.confidence * 100.0,
                if crossing_side == CrossingSide::Left {
                    "LEFT"
                } else {
                    "RIGHT"
                },
                boundary_x,
                dist
            );
            marking.clone()
        })
    }

    // ========================================================================
    // Keep original analyze_frame for visualization (all markings)
    // ========================================================================

    pub fn analyze_frame(
        &mut self,
        frame: &[u8],
        width: usize,
        height: usize,
        frame_id: u64,
        confidence_threshold: f32,
    ) -> Result<LegalityResult> {
        let (input, scale, pad_x, pad_y) = self.preprocess(frame, width, height)?;
        let (box_output, mask_proto, _) = self.infer(&input)?;
        let markings = self.postprocess(
            &box_output,
            &mask_proto,
            &[],
            scale,
            pad_x,
            pad_y,
            width,
            height,
            confidence_threshold,
            frame, // ✅ Pass original frame for color verification
        )?;

        // For visualization only — no violation counting
        Ok(LegalityResult {
            verdict: LineLegality::Unknown,
            intersecting_line: None,
            all_markings: markings,
            ego_intersects_marking: false,
            frame_id,
        })
    }

    // ========================================================================
    // PREPROCESSING
    // ========================================================================

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

    // ========================================================================
    // INFERENCE
    // ========================================================================

    fn infer(&mut self, input: &[f32]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let shape = [1, 3, SEG_INPUT_SIZE, SEG_INPUT_SIZE];
        let input_value =
            ort::value::Value::from_array((shape.as_slice(), input.to_vec().into_boxed_slice()))?;

        let outputs = self.session.run(ort::inputs!["images" => input_value])?;

        let output0 = &outputs[0];
        let (_, data0) = output0.try_extract_tensor::<f32>()?;
        let box_output = data0.to_vec();

        let output1 = &outputs[1];
        let (_, data1) = output1.try_extract_tensor::<f32>()?;
        let mask_proto = data1.to_vec();

        Ok((box_output, mask_proto, Vec::new()))
    }

    // ========================================================================
    // POSTPROCESSING
    // ========================================================================

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
        frame_rgb: &[u8], // ✅ NEW: Original frame for color verification
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

        // ✅ Apply color verification BEFORE NMS
        // This corrects white→yellow misclassifications by sampling actual pixel colors
        for marking in &mut detections {
            verify_line_color(frame_rgb, orig_w, orig_h, marking);
        }

        let detections = nms_markings(detections, 0.45);
        info!("Detected {} road markings", detections.len());
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

// ============================================================================
// HELPERS
// ============================================================================

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
    while !dets.is_empty() {
        let current = dets.remove(0);
        keep.push(current.clone());
        dets.retain(|d| calculate_iou_arr(&current.bbox, &d.bbox) < iou_thresh);
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
