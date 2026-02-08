// src/lane_legality.rs
//
// Road Line Legality Detection via YOLOv8n-seg
// Fused with UFLDv2 lane detection for accurate crossing detection.
//
// Strategy:
//   - UFLDv2 determines IF the vehicle is crossing (lateral offset + state)
//   - YOLOv8-seg determines WHAT the vehicle is crossing (line type/color)
//   - Only flag a violation when BOTH models agree
//

use anyhow::Result;
use ort::{
    execution_providers::CUDAExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

// ============================================================================
// CONSTANTS
// ============================================================================

const SEG_INPUT_SIZE: usize = 640;
const NUM_CLASSES: usize = 25;
const MASK_SIZE: usize = 160; // YOLOv8-seg mask resolution (640/4)

/// Class IDs that make crossing ILLEGAL
const ILLEGAL_CLASS_IDS: [usize; 5] = [4, 5, 6, 7, 8];
/// Class IDs that make crossing LEGAL
const LEGAL_CLASS_IDS: [usize; 2] = [9, 10];
/// Class IDs considered CRITICAL violations
const CRITICAL_CLASS_IDS: [usize; 2] = [5, 8];

/// Minimum normalized offset (from UFLDv2) to consider a crossing
const FUSED_CROSSING_OFFSET_THRESHOLD: f32 = 0.38;
/// Minimum mask intersection pixels (at 160×160) for the seg model
const FUSED_MIN_MASK_PIXELS: usize = 20;
/// Search margin in pixels around a lane boundary when matching seg detections
const BOUNDARY_SEARCH_MARGIN: f32 = 100.0;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossingSide {
    Left,
    Right,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedRoadMarking {
    pub class_id: usize,
    pub class_name: String,
    pub confidence: f32,
    pub bbox: [f32; 4], // [x1, y1, x2, y2] in original coords
    pub legality: LineLegality,
    pub mask: Vec<u8>, // MASK_SIZE x MASK_SIZE binary mask
    pub mask_width: usize,
    pub mask_height: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalityResult {
    /// Overall legality verdict for this frame
    pub verdict: LineLegality,
    /// The specific line class that the ego vehicle is intersecting (if any)
    pub intersecting_line: Option<DetectedRoadMarking>,
    /// All detected road markings in the frame
    pub all_markings: Vec<DetectedRoadMarking>,
    /// Whether the ego vehicle bbox intersects any lane marking mask
    pub ego_intersects_marking: bool,
    /// Frame ID for tracking
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

/// Result from the fused analysis (UFLDv2 + YOLOv8-seg)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedLegalityResult {
    pub verdict: LineLegality,
    /// UFLDv2 confirmed the vehicle is crossing / near a boundary
    pub crossing_confirmed_by_lane_model: bool,
    /// The seg-model detection closest to the crossed boundary
    pub line_type_from_seg_model: Option<DetectedRoadMarking>,
    /// Normalized lateral offset from UFLDv2 (0.0 = center, 0.5 = on boundary)
    pub vehicle_offset_pct: f32,
    /// Legacy compat
    pub ego_intersects_marking: bool,
    pub frame_id: u64,
    pub all_markings: Vec<DetectedRoadMarking>,
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
// LANE LEGALITY DETECTOR
// ============================================================================

pub struct LaneLegalityDetector {
    session: Session,
    /// Ego vehicle bounding box in normalized coords [x1, y1, x2, y2]
    /// Used ONLY as fallback when lane model data is not available
    ego_bbox_ratio: [f32; 4],
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
            ego_bbox_ratio: [0.40, 0.80, 0.60, 0.98],
        })
    }

    /// Set custom ego vehicle bounding box ratios (fallback only)
    pub fn set_ego_bbox_ratio(&mut self, x1: f32, y1: f32, x2: f32, y2: f32) {
        self.ego_bbox_ratio = [x1, y1, x2, y2];
    }

    // ========================================================================
    // FUSED ANALYSIS (preferred — uses both models)
    // ========================================================================

    /// Run fused detection: YOLOv8-seg line classification + UFLDv2 position.
    ///
    /// Only flags a violation when the lane model confirms the vehicle is
    /// actually near / crossing a boundary AND the seg model identifies that
    /// boundary as an illegal line type.
    pub fn analyze_frame_fused(
        &mut self,
        frame: &[u8],
        width: usize,
        height: usize,
        frame_id: u64,
        confidence_threshold: f32,
        // ── from UFLDv2 / lane analyzer ──
        vehicle_lateral_offset: f32,
        lane_width: Option<f32>,
        left_lane_x: Option<f32>,
        right_lane_x: Option<f32>,
        crossing_side: CrossingSide,
    ) -> Result<FusedLegalityResult> {
        // 1. Run YOLOv8-seg to get all line classifications
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
        )?;

        // 2. Calculate normalised offset from UFLDv2
        let normalized_offset = lane_width
            .filter(|&w| w > 50.0)
            .map(|w| (vehicle_lateral_offset / w).abs())
            .unwrap_or(0.0);

        // 3. Does UFLDv2 say we're near / crossing a boundary?
        let is_crossing_per_lane_model = normalized_offset > FUSED_CROSSING_OFFSET_THRESHOLD
            && crossing_side != CrossingSide::None;

        // 4. Find the nearest seg-model detection to the boundary we're crossing
        let vehicle_x = width as f32 / 2.0;
        let nearest_line = if is_crossing_per_lane_model {
            self.find_nearest_line_to_crossing(
                &markings,
                vehicle_x,
                left_lane_x,
                right_lane_x,
                crossing_side,
                width,
                height,
            )
        } else {
            None
        };

        // 5. FUSED VERDICT — only flag when BOTH models agree
        let verdict = if is_crossing_per_lane_model {
            if let Some(ref line) = nearest_line {
                line.legality
            } else {
                // Crossing but can't classify the line → unknown, not illegal
                LineLegality::Unknown
            }
        } else {
            // Lane model says NOT crossing → never flag
            LineLegality::Legal
        };

        debug!(
            "F{}: fused offset={:.1}% side={:?} crossing={} nearest={} → {}",
            frame_id,
            normalized_offset * 100.0,
            crossing_side,
            is_crossing_per_lane_model,
            nearest_line
                .as_ref()
                .map(|l| l.class_name.as_str())
                .unwrap_or("none"),
            verdict.as_str(),
        );

        Ok(FusedLegalityResult {
            verdict,
            crossing_confirmed_by_lane_model: is_crossing_per_lane_model,
            line_type_from_seg_model: nearest_line,
            vehicle_offset_pct: normalized_offset,
            ego_intersects_marking: is_crossing_per_lane_model,
            frame_id,
            all_markings: markings,
        })
    }

    // ========================================================================
    // STANDALONE ANALYSIS (legacy fallback — no lane model data)
    // ========================================================================

    /// Run standalone detection + legality check on a frame.
    /// Use `analyze_frame_fused` when lane model data is available.
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
        )?;

        let ego_bbox = [
            self.ego_bbox_ratio[0] * width as f32,
            self.ego_bbox_ratio[1] * height as f32,
            self.ego_bbox_ratio[2] * width as f32,
            self.ego_bbox_ratio[3] * height as f32,
        ];

        let mut result = LegalityResult::no_detection(frame_id);
        result.all_markings = markings;

        let mut worst_legality = LineLegality::Unknown;

        for marking in &result.all_markings {
            if self.check_mask_intersection(marking, &ego_bbox, width, height) {
                result.ego_intersects_marking = true;

                let priority = legality_priority(marking.legality);
                let current_priority = legality_priority(worst_legality);

                if priority > current_priority {
                    worst_legality = marking.legality;
                    result.intersecting_line = Some(marking.clone());
                }
            }
        }

        result.verdict = worst_legality;
        Ok(result)
    }

    // ========================================================================
    // FIND NEAREST LINE TO CROSSING BOUNDARY
    // ========================================================================

    /// Given the side the vehicle is crossing toward (from UFLDv2), find
    /// the seg-model detection whose bbox center is closest to that lane
    /// boundary.  Returns `None` if no detection is within the search margin.
    fn find_nearest_line_to_crossing(
        &self,
        markings: &[DetectedRoadMarking],
        vehicle_x: f32,
        left_lane_x: Option<f32>,
        right_lane_x: Option<f32>,
        crossing_side: CrossingSide,
        _frame_w: usize,
        _frame_h: usize,
    ) -> Option<DetectedRoadMarking> {
        let boundary_x = match crossing_side {
            CrossingSide::Left => left_lane_x.unwrap_or(vehicle_x * 0.4),
            CrossingSide::Right => right_lane_x.unwrap_or(vehicle_x * 1.6),
            CrossingSide::None => return None,
        };

        let margin = BOUNDARY_SEARCH_MARGIN;

        // Filter markings whose bbox center-x is within ±margin of boundary
        let mut candidates: Vec<(f32, &DetectedRoadMarking)> = markings
            .iter()
            .filter_map(|m| {
                let bbox_cx = (m.bbox[0] + m.bbox[2]) / 2.0;
                let dist = (bbox_cx - boundary_x).abs();
                if dist <= margin {
                    Some((dist, m))
                } else {
                    None
                }
            })
            .collect();

        // Sort by distance (nearest first), break ties by confidence
        candidates.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    b.1.confidence
                        .partial_cmp(&a.1.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });

        candidates.first().map(|(_, m)| (*m).clone())
    }

    // ========================================================================
    // PREPROCESSING (Letterbox to 640×640)
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

        // Normalize to [0, 1] and HWC → CHW
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

        // YOLOv8-seg outputs:
        // output0: [1, 116, 8400] — 4 bbox + 25 classes + 32 mask coeffs + …
        // output1: [1, 32, 160, 160] — mask prototypes

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
    ) -> Result<Vec<DetectedRoadMarking>> {
        let mut detections = Vec::new();
        let num_detections = 8400;

        for i in 0..num_detections {
            let cx = output[i];
            let cy = output[num_detections + i];
            let w = output[num_detections * 2 + i];
            let h = output[num_detections * 3 + i];

            // Find best class among road marking classes
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

            // Extract 32 mask coefficients for this detection
            let mut mask_coeffs_det = [0.0f32; 32];
            for mc in 0..32 {
                mask_coeffs_det[mc] = output[num_detections * (4 + NUM_CLASSES + mc) + i];
            }

            // Reverse letterbox transform
            let x1 = ((cx - w / 2.0) - pad_x) / scale;
            let y1 = ((cy - h / 2.0) - pad_y) / scale;
            let x2 = ((cx + w / 2.0) - pad_x) / scale;
            let y2 = ((cy + h / 2.0) - pad_y) / scale;

            let x1 = x1.max(0.0).min(orig_w as f32);
            let y1 = y1.max(0.0).min(orig_h as f32);
            let x2 = x2.max(0.0).min(orig_w as f32);
            let y2 = y2.max(0.0).min(orig_h as f32);

            // Generate binary mask from mask_proto @ mask_coeffs
            let mask = self.generate_mask(
                mask_proto,
                &mask_coeffs_det,
                pad_x,
                pad_y,
                scale,
                orig_w,
                orig_h,
            );

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

        // Geometric correction for center lines misclassified as white
        Self::apply_geometric_correction(&mut detections, orig_w);

        // NMS
        let detections = nms_markings(detections, 0.45);

        debug!("Detected {} road markings", detections.len());
        Ok(detections)
    }

    // ========================================================================
    // GEOMETRIC CORRECTION FOR CENTER LINES
    // ========================================================================

    fn apply_geometric_correction(detections: &mut [DetectedRoadMarking], frame_width: usize) {
        let center_threshold = 0.15;
        let frame_center = frame_width as f32 / 2.0;

        for marking in detections.iter_mut() {
            let bbox_center_x = (marking.bbox[0] + marking.bbox[2]) / 2.0;
            let distance_from_center = (bbox_center_x - frame_center).abs() / frame_width as f32;

            if distance_from_center < center_threshold && marking.class_name.contains("white") {
                let original_class = marking.class_name.clone();
                let original_legality = marking.legality;

                marking.class_name = marking.class_name.replace("white", "yellow");

                marking.class_id = if marking.class_name.contains("dashed") {
                    10 // dashed_single_yellow
                } else if marking.class_name.contains("double") {
                    8 // solid_double_yellow (CRITICAL)
                } else {
                    5 // solid_single_yellow (CRITICAL)
                };

                marking.legality = class_id_to_legality(marking.class_id);

                if marking.legality != original_legality {
                    warn!(
                        "🔧 CENTER LINE CORRECTION: {} ({}) → {} ({}) | Distance from center: {:.1}%",
                        original_class,
                        original_legality.as_str(),
                        marking.class_name,
                        marking.legality.as_str(),
                        distance_from_center * 100.0
                    );
                }
            }
        }
    }

    // ========================================================================
    // MASK GENERATION
    // ========================================================================

    fn generate_mask(
        &self,
        mask_proto: &[f32], // [1, 32, 160, 160]
        mask_coeffs: &[f32; 32],
        _pad_x: f32,
        _pad_y: f32,
        _scale: f32,
        _orig_w: usize,
        _orig_h: usize,
    ) -> Vec<u8> {
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

    // ========================================================================
    // INTERSECTION CHECK (legacy / fallback)
    // ========================================================================

    fn check_mask_intersection(
        &self,
        marking: &DetectedRoadMarking,
        ego_bbox: &[f32; 4],
        frame_w: usize,
        frame_h: usize,
    ) -> bool {
        let scale_x = MASK_SIZE as f32 / frame_w as f32;
        let scale_y = MASK_SIZE as f32 / frame_h as f32;

        let mx1 = (ego_bbox[0] * scale_x) as usize;
        let my1 = (ego_bbox[1] * scale_y) as usize;
        let mx2 = (ego_bbox[2] * scale_x).min(MASK_SIZE as f32 - 1.0) as usize;
        let my2 = (ego_bbox[3] * scale_y).min(MASK_SIZE as f32 - 1.0) as usize;

        let mut intersection_pixels = 0;

        for y in my1..=my2.min(MASK_SIZE - 1) {
            for x in mx1..=mx2.min(MASK_SIZE - 1) {
                if marking.mask[y * MASK_SIZE + x] > 0 {
                    intersection_pixels += 1;
                }
            }
        }

        // Raised from 5 → FUSED_MIN_MASK_PIXELS to reduce false positives
        intersection_pixels >= FUSED_MIN_MASK_PIXELS
    }
}

// ============================================================================
// HELPERS
// ============================================================================

fn legality_priority(l: LineLegality) -> u8 {
    match l {
        LineLegality::CriticalIllegal => 4,
        LineLegality::Illegal => 3,
        LineLegality::Legal => 2,
        LineLegality::Caution => 1,
        LineLegality::Unknown => 0,
    }
}

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
