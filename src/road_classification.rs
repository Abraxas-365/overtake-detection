// src/road_classification.rs
//
// v5.1: Road type classification + Mixed line passing legality
//
// Implements Peru MTC rules (Manual de Dispositivos de Control de Tránsito
// Automotor para Calles y Carreteras, R.D. N° 16-2016-MTC/14, Cap. 3):
//
// ┌─────────────────────────────────────────────────────────────────────────┐
// │  LÍNEA CENTRAL (center line rules)                                      │
// │                                                                         │
// │  1. Doble continua amarilla (solid_double_yellow, class_id=8):          │
// │     Máxima restricción. Prohibido adelantar desde AMBOS sentidos.       │
// │     "Es el máximo nivel de restricción" — MTC                           │
// │                                                                         │
// │  2. Continua simple amarilla (solid_single_yellow, class_id=5):         │
// │     Prohibido cruzar. No adelantar.                                     │
// │                                                                         │
// │  3. Segmentada amarilla (dashed_single_yellow, class_id=10):            │
// │     Permitido adelantar con precaución.                                 │
// │                                                                         │
// │  4. LÍNEA MIXTA (mixed_double_yellow, class_id=99):                     │
// │     Una línea continua + una segmentada paralelas.                      │
// │     "Solo se puede pasar, girar o adelantar desde el carril que esté    │
// │      demarcado con la línea segmentada, pero si estás en el lado de     │
// │      la línea continua, está prohibido."                                │
// │                                                                         │
// │     For ego vehicle detection (Peru = right-hand traffic):              │
// │     - If dashed side is RIGHT → ego vehicle CAN pass (dashed faces us)  │
// │     - If solid side is RIGHT → ego vehicle CANNOT pass                  │
// │                                                                         │
// │  WHITE LINES (same-direction separation):                               │
// │  5. Continua blanca (solid_single/double_white, class_id=4/7):          │
// │     Prohibido cambiar de carril.                                        │
// │                                                                         │
// │  6. Segmentada blanca (dashed_single_white, class_id=9):                │
// │     Cambio de carril permitido.                                         │
// └─────────────────────────────────────────────────────────────────────────┘

use crate::color_analysis::MarkingColor;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::debug;

// ============================================================================
// TYPES
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoadType {
    TwoWay,
    OneWay,
    Unknown,
}

impl RoadType {
    pub fn as_str(&self) -> &'static str {
        match self {
            RoadType::TwoWay => "TWO_WAY",
            RoadType::OneWay => "ONE_WAY",
            RoadType::Unknown => "UNKNOWN",
        }
    }
    pub fn as_display_str(&self) -> &'static str {
        match self {
            RoadType::TwoWay => "DOBLE SENTIDO",
            RoadType::OneWay => "UN SENTIDO",
            RoadType::Unknown => "---",
        }
    }
}

impl std::fmt::Display for RoadType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PassingLegality {
    Allowed,
    Prohibited,
    MixedAllowed,
    MixedProhibited,
    Unknown,
}

impl PassingLegality {
    pub fn as_str(&self) -> &'static str {
        match self {
            PassingLegality::Allowed => "ADELANTAR PERMITIDO",
            PassingLegality::Prohibited => "ADELANTAR PROHIBIDO",
            PassingLegality::MixedAllowed => "MIXTA: PERMITIDO (lado segmentado)",
            PassingLegality::MixedProhibited => "MIXTA: PROHIBIDO (lado continuo)",
            PassingLegality::Unknown => "---",
        }
    }
    pub fn is_legal(&self) -> bool {
        matches!(
            self,
            PassingLegality::Allowed | PassingLegality::MixedAllowed
        )
    }
    pub fn is_illegal(&self) -> bool {
        matches!(
            self,
            PassingLegality::Prohibited | PassingLegality::MixedProhibited
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MixedLineSide {
    DashedRight,
    SolidRight,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadClassification {
    pub road_type: RoadType,
    pub passing_legality: PassingLegality,
    pub mixed_line_side: Option<MixedLineSide>,
    pub estimated_lanes: u32,
    pub center_marking_names: Vec<String>,
    pub confidence: f32,
}

impl Default for RoadClassification {
    fn default() -> Self {
        Self {
            road_type: RoadType::Unknown,
            passing_legality: PassingLegality::Unknown,
            mixed_line_side: None,
            estimated_lanes: 0,
            center_marking_names: Vec::new(),
            confidence: 0.0,
        }
    }
}

// ============================================================================
// MARKING INFO
// ============================================================================

#[derive(Debug, Clone)]
pub struct MarkingInfo {
    pub class_id: usize,
    pub class_name: String,
    pub center_x: f32,
    pub confidence: f32,
    pub detected_color: Option<MarkingColor>,
    /// Bounding box [x1, y1, x2, y2] in original image coordinates.
    /// Required for line crossing detector (checks ego/marking bbox overlap).
    pub bbox: [f32; 4],
    /// Segmentation mask data (from YOLO-seg model output).
    /// Used by mixed line side analysis and crossing detector.
    pub mask: Vec<u8>,
    pub mask_width: usize,
    pub mask_height: usize,
}

// ============================================================================
// CLASSIFIER
// ============================================================================

const TEMPORAL_WINDOW: usize = 30;
const MIN_CONSENSUS_FRAMES: usize = 5;
const CENTER_ZONE_LEFT_RATIO: f32 = 0.25;
const CENTER_ZONE_RIGHT_RATIO: f32 = 0.75;

pub struct RoadClassifier {
    road_type_history: VecDeque<RoadType>,
    passing_history: VecDeque<PassingLegality>,
    mixed_side_history: VecDeque<Option<MixedLineSide>>,
    last_stable: RoadClassification,
    frame_width: f32,
}

impl RoadClassifier {
    pub fn new(frame_width: f32) -> Self {
        Self {
            road_type_history: VecDeque::with_capacity(TEMPORAL_WINDOW),
            passing_history: VecDeque::with_capacity(TEMPORAL_WINDOW),
            mixed_side_history: VecDeque::with_capacity(TEMPORAL_WINDOW),
            last_stable: RoadClassification::default(),
            frame_width,
        }
    }

    pub fn update(&mut self, markings: &[MarkingInfo]) -> &RoadClassification {
        let instant = self.classify_single_frame(markings);
        self.push_history(
            instant.road_type,
            instant.passing_legality,
            instant.mixed_line_side,
        );

        self.last_stable = RoadClassification {
            road_type: self.majority_road_type(),
            passing_legality: self.majority_passing(),
            mixed_line_side: self.majority_mixed_side(),
            estimated_lanes: instant.estimated_lanes,
            center_marking_names: instant.center_marking_names,
            confidence: self.compute_confidence(),
        };
        &self.last_stable
    }

    pub fn current(&self) -> &RoadClassification {
        &self.last_stable
    }

    pub fn reset(&mut self) {
        self.road_type_history.clear();
        self.passing_history.clear();
        self.mixed_side_history.clear();
        self.last_stable = RoadClassification::default();
    }

    fn push_history(&mut self, rt: RoadType, pl: PassingLegality, ms: Option<MixedLineSide>) {
        self.road_type_history.push_back(rt);
        if self.road_type_history.len() > TEMPORAL_WINDOW {
            self.road_type_history.pop_front();
        }
        self.passing_history.push_back(pl);
        if self.passing_history.len() > TEMPORAL_WINDOW {
            self.passing_history.pop_front();
        }
        self.mixed_side_history.push_back(ms);
        if self.mixed_side_history.len() > TEMPORAL_WINDOW {
            self.mixed_side_history.pop_front();
        }
    }

    // ── Single-frame classification ──

    fn classify_single_frame(&self, markings: &[MarkingInfo]) -> RoadClassification {
        if markings.is_empty() {
            return RoadClassification::default();
        }

        let cl = self.frame_width * CENTER_ZONE_LEFT_RATIO;
        let cr = self.frame_width * CENTER_ZONE_RIGHT_RATIO;

        let center: Vec<&MarkingInfo> = markings
            .iter()
            .filter(|m| m.center_x >= cl && m.center_x <= cr && is_lane_line_class(m.class_id))
            .collect();

        let right_lines = markings
            .iter()
            .filter(|m| m.center_x > cr && is_lane_line_class(m.class_id))
            .count();

        let mixed = center
            .iter()
            .find(|m| m.class_id == 99 || m.class_name.contains("mixed"));
        let has_yellow = center.iter().any(|m| is_yellow_class(m));
        let has_white = center.iter().any(|m| is_white_class(m));

        let names: Vec<String> = center.iter().map(|m| m.class_name.clone()).collect();
        let max_conf = center
            .iter()
            .map(|m| m.confidence)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        if let Some(m) = mixed {
            // ── MIXED LINE (línea mixta) ──
            let (passing, side) = self.classify_mixed_line(m);
            RoadClassification {
                road_type: RoadType::TwoWay,
                passing_legality: passing,
                mixed_line_side: Some(side),
                estimated_lanes: (right_lines as u32 + 1).max(1),
                center_marking_names: names,
                confidence: max_conf,
            }
        } else if has_yellow {
            let passing = self.classify_yellow_center(&center);
            RoadClassification {
                road_type: RoadType::TwoWay,
                passing_legality: passing,
                mixed_line_side: None,
                estimated_lanes: (right_lines as u32 + 1).max(1),
                center_marking_names: names,
                confidence: max_conf,
            }
        } else if has_white && !center.is_empty() {
            let passing = self.classify_white_center(&center);
            RoadClassification {
                road_type: RoadType::OneWay,
                passing_legality: passing,
                mixed_line_side: None,
                estimated_lanes: (center.len() as u32 + 1).max(2),
                center_marking_names: names,
                confidence: max_conf,
            }
        } else {
            RoadClassification::default()
        }
    }

    /// Classify a mixed (solid+dashed) center line for passing legality.
    ///
    /// merge_composite_lines encodes which side is dashed in the class_name:
    ///   "mixed_double_yellow_dashed_right" → dashed is on the RIGHT
    ///   "mixed_double_yellow_solid_right"  → solid is on the RIGHT
    ///
    /// Peru = right-hand traffic. Ego vehicle is on the RIGHT of center.
    ///   dashed_right → dashed faces us → CAN overtake
    ///   solid_right  → solid faces us → CANNOT overtake
    fn classify_mixed_line(&self, m: &MarkingInfo) -> (PassingLegality, MixedLineSide) {
        let name = m.class_name.to_lowercase();
        if name.contains("dashed_right") {
            debug!("🚦 Línea mixta: segmentada DERECHA → ego PUEDE adelantar");
            (PassingLegality::MixedAllowed, MixedLineSide::DashedRight)
        } else if name.contains("solid_right") || name.contains("dashed_left") {
            debug!("🚦 Línea mixta: continua DERECHA → ego NO puede adelantar");
            (PassingLegality::MixedProhibited, MixedLineSide::SolidRight)
        } else {
            debug!(
                "🚦 Línea mixta detectada, lado desconocido: '{}'. Conservador → PROHIBIDO.",
                m.class_name
            );
            (PassingLegality::MixedProhibited, MixedLineSide::Unknown)
        }
    }

    fn classify_yellow_center(&self, center: &[&MarkingInfo]) -> PassingLegality {
        let has_double_solid = center.iter().any(|m| m.class_id == 8);
        let has_single_solid = center.iter().any(|m| m.class_id == 5);
        let has_dashed = center.iter().any(|m| m.class_id == 10);

        if has_double_solid {
            PassingLegality::Prohibited
        } else if has_single_solid && !has_dashed {
            PassingLegality::Prohibited
        } else if has_dashed && !has_single_solid {
            PassingLegality::Allowed
        } else {
            PassingLegality::Prohibited // Conservative when ambiguous
        }
    }

    fn classify_white_center(&self, center: &[&MarkingInfo]) -> PassingLegality {
        let has_solid = center.iter().any(|m| matches!(m.class_id, 4 | 7));
        let has_dashed = center.iter().any(|m| m.class_id == 9);

        if has_solid && !has_dashed {
            PassingLegality::Prohibited
        } else if has_dashed && !has_solid {
            PassingLegality::Allowed
        } else {
            PassingLegality::Unknown
        }
    }

    // ── Temporal smoothing ──

    fn majority_road_type(&self) -> RoadType {
        if self.road_type_history.len() < MIN_CONSENSUS_FRAMES {
            return RoadType::Unknown;
        }
        let (mut tw, mut ow) = (0u32, 0u32);
        for rt in &self.road_type_history {
            match rt {
                RoadType::TwoWay => tw += 1,
                RoadType::OneWay => ow += 1,
                _ => {}
            }
        }
        let th = self.road_type_history.len() as u32 / 2;
        if tw > th {
            RoadType::TwoWay
        } else if ow > th {
            RoadType::OneWay
        } else {
            RoadType::Unknown
        }
    }

    fn majority_passing(&self) -> PassingLegality {
        if self.passing_history.len() < MIN_CONSENSUS_FRAMES {
            return PassingLegality::Unknown;
        }
        let (mut a, mut p, mut ma, mut mp) = (0u32, 0u32, 0u32, 0u32);
        for pl in &self.passing_history {
            match pl {
                PassingLegality::Allowed => a += 1,
                PassingLegality::Prohibited => p += 1,
                PassingLegality::MixedAllowed => ma += 1,
                PassingLegality::MixedProhibited => mp += 1,
                _ => {}
            }
        }
        let th = self.passing_history.len() as u32 / 3;
        let counts = [
            (PassingLegality::Allowed, a),
            (PassingLegality::Prohibited, p),
            (PassingLegality::MixedAllowed, ma),
            (PassingLegality::MixedProhibited, mp),
        ];
        let best = counts.iter().max_by_key(|(_, c)| *c).unwrap();
        if best.1 > th {
            best.0
        } else {
            PassingLegality::Unknown
        }
    }

    fn majority_mixed_side(&self) -> Option<MixedLineSide> {
        let (mut dr, mut sr) = (0u32, 0u32);
        for ms in &self.mixed_side_history {
            match ms {
                Some(MixedLineSide::DashedRight) => dr += 1,
                Some(MixedLineSide::SolidRight) => sr += 1,
                _ => {}
            }
        }
        if dr == 0 && sr == 0 {
            None
        } else if dr > sr {
            Some(MixedLineSide::DashedRight)
        } else {
            Some(MixedLineSide::SolidRight)
        }
    }

    fn compute_confidence(&self) -> f32 {
        if self.road_type_history.is_empty() {
            return 0.0;
        }
        let cur = self.last_stable.road_type;
        let agree = self
            .road_type_history
            .iter()
            .filter(|&&rt| rt == cur)
            .count();
        agree as f32 / self.road_type_history.len() as f32
    }
}

// ============================================================================
// merge_composite_lines_v2 — improved version
// ============================================================================

pub fn proportional_merge_distance(frame_width: f32) -> f32 {
    (frame_width * 0.07).max(30.0).min(150.0)
}

#[derive(Debug, Clone)]
pub struct DetectedRoadMarkingCompat {
    pub class_id: usize,
    pub class_name: String,
    pub confidence: f32,
    pub bbox: [f32; 4],
    pub mask: Vec<u8>,
    pub mask_width: usize,
    pub mask_height: usize,
}

pub fn merge_composite_lines_v2(
    dets: Vec<DetectedRoadMarkingCompat>,
    frame_width: f32,
) -> Vec<DetectedRoadMarkingCompat> {
    use std::collections::HashSet;
    let merge_dist = proportional_merge_distance(frame_width);
    let mut merged = Vec::new();
    let mut used = HashSet::new();

    for i in 0..dets.len() {
        if used.contains(&i) {
            continue;
        }
        let a = &dets[i];
        let mut found = false;

        for j in (i + 1)..dets.len() {
            if used.contains(&j) {
                continue;
            }
            let b = &dets[j];

            let yellow_pair = (is_solid_yellow(a.class_id) && is_dashed_yellow(b.class_id))
                || (is_dashed_yellow(a.class_id) && is_solid_yellow(b.class_id));
            let white_pair = (is_solid_white(a.class_id) && is_dashed_white(b.class_id))
                || (is_dashed_white(a.class_id) && is_solid_white(b.class_id));

            if !yellow_pair && !white_pair {
                continue;
            }

            let cx_a = (a.bbox[0] + a.bbox[2]) / 2.0;
            let cx_b = (b.bbox[0] + b.bbox[2]) / 2.0;
            let dist_x = (cx_a - cx_b).abs();
            let v_overlap = vertical_overlap_ratio(&a.bbox, &b.bbox);

            if dist_x < merge_dist && v_overlap > 0.3 {
                let (solid, dashed) = if is_dashed_yellow(a.class_id) || is_dashed_white(a.class_id)
                {
                    (b, a)
                } else {
                    (a, b)
                };

                let solid_cx = (solid.bbox[0] + solid.bbox[2]) / 2.0;
                let dashed_cx = (dashed.bbox[0] + dashed.bbox[2]) / 2.0;
                let dashed_is_right = dashed_cx > solid_cx;

                let mut m = solid.clone();
                m.class_id = 99;
                m.confidence = (a.confidence + b.confidence) / 2.0;
                m.bbox = [
                    a.bbox[0].min(b.bbox[0]),
                    a.bbox[1].min(b.bbox[1]),
                    a.bbox[2].max(b.bbox[2]),
                    a.bbox[3].max(b.bbox[3]),
                ];

                let prefix = if yellow_pair { "yellow" } else { "white" };
                m.class_name = if dashed_is_right {
                    format!("mixed_double_{}_dashed_right", prefix)
                } else {
                    format!("mixed_double_{}_solid_right", prefix)
                };

                debug!(
                    "🧩 Merged línea mixta: {} (dashed_right={})",
                    m.class_name, dashed_is_right
                );
                merged.push(m);
                used.insert(i);
                used.insert(j);
                found = true;
                break;
            }
        }
        if !found {
            merged.push(a.clone());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // v6.1e: Suppress dashed detections that overlap with double-solid
    // detections. When the model sees one line of a double-solid as
    // "dashed" (common with worn paint at night), we keep the double-solid
    // and drop the false dashed detection.
    // ═══════════════════════════════════════════════════════════════════════

    let mut deduped = Vec::new();
    let mut suppressed = HashSet::new();

    for i in 0..merged.len() {
        let a = &merged[i];
        // Check if this is a dashed detection overlapping a double-solid
        if matches!(a.class_id, 9 | 10) {
            let cx_a = (a.bbox[0] + a.bbox[2]) / 2.0;
            for j in 0..merged.len() {
                if i == j {
                    continue;
                }
                let b = &merged[j];
                if matches!(b.class_id, 7 | 8) {
                    let cx_b = (b.bbox[0] + b.bbox[2]) / 2.0;
                    let dist = (cx_a - cx_b).abs();
                    let v_overlap = vertical_overlap_ratio(&a.bbox, &b.bbox);
                    if dist < merge_dist && v_overlap > 0.3 {
                        // Suppress dashed — double-solid takes priority
                        suppressed.insert(i);
                        debug!(
                            "🧹 Suppressed false {} near {} (dist={:.0}px)",
                            a.class_name, b.class_name, dist
                        );
                        break;
                    }
                }
            }
        }
    }

    for (i, m) in merged.into_iter().enumerate() {
        if !suppressed.contains(&i) {
            deduped.push(m);
        }
    }

    deduped
}

// ============================================================================
// HELPERS
// ============================================================================

fn is_lane_line_class(id: usize) -> bool {
    matches!(id, 4 | 5 | 6 | 7 | 8 | 9 | 10 | 99)
}

fn is_solid_yellow(id: usize) -> bool {
    id == 5
}

fn is_dashed_yellow(id: usize) -> bool {
    id == 10
}

fn is_solid_white(id: usize) -> bool {
    id == 4
}

fn is_dashed_white(id: usize) -> bool {
    id == 9
}

fn is_yellow_class(m: &MarkingInfo) -> bool {
    matches!(m.class_id, 5 | 8 | 10 | 99) || matches!(m.detected_color, Some(MarkingColor::Yellow))
}
fn is_white_class(m: &MarkingInfo) -> bool {
    matches!(m.class_id, 4 | 7 | 9) || matches!(m.detected_color, Some(MarkingColor::White))
}

fn vertical_overlap_ratio(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let top = a[1].max(b[1]);
    let bottom = a[3].min(b[3]);
    let overlap = (bottom - top).max(0.0);
    overlap / (a[3] - a[1]).min(b[3] - b[1]).max(1.0)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(id: usize, name: &str, cx: f32, conf: f32) -> MarkingInfo {
        MarkingInfo {
            class_id: id,
            class_name: name.into(),
            center_x: cx,
            confidence: conf,
            detected_color: None,
            bbox: [0.0, 0.0, 0.0, 0.0],
            mask: Vec::new(),
            mask_width: 0,
            mask_height: 0,
        }
    }

    #[test]
    fn test_double_solid_yellow_prohibited() {
        let c = RoadClassifier::new(1280.0);
        let r = c.classify_single_frame(&[mk(8, "solid_double_yellow", 640.0, 0.85)]);
        assert_eq!(r.road_type, RoadType::TwoWay);
        assert_eq!(r.passing_legality, PassingLegality::Prohibited);
    }

    #[test]
    fn test_dashed_yellow_allowed() {
        let c = RoadClassifier::new(1280.0);
        let r = c.classify_single_frame(&[mk(10, "dashed_single_yellow", 640.0, 0.80)]);
        assert_eq!(r.passing_legality, PassingLegality::Allowed);
    }

    #[test]
    fn test_mixed_dashed_right_ego_can_pass() {
        let c = RoadClassifier::new(1280.0);
        let r = c.classify_single_frame(&[mk(99, "mixed_double_yellow_dashed_right", 640.0, 0.85)]);
        assert_eq!(r.passing_legality, PassingLegality::MixedAllowed);
        assert_eq!(r.mixed_line_side, Some(MixedLineSide::DashedRight));
    }

    #[test]
    fn test_mixed_solid_right_ego_cannot_pass() {
        let c = RoadClassifier::new(1280.0);
        let r = c.classify_single_frame(&[mk(99, "mixed_double_yellow_solid_right", 640.0, 0.85)]);
        assert_eq!(r.passing_legality, PassingLegality::MixedProhibited);
        assert_eq!(r.mixed_line_side, Some(MixedLineSide::SolidRight));
    }

    #[test]
    fn test_one_way_white_dashed() {
        let c = RoadClassifier::new(1280.0);
        let r = c.classify_single_frame(&[mk(9, "dashed_single_white", 640.0, 0.80)]);
        assert_eq!(r.road_type, RoadType::OneWay);
        assert_eq!(r.passing_legality, PassingLegality::Allowed);
    }

    #[test]
    fn test_merge_creates_dashed_right() {
        let dets = vec![
            DetectedRoadMarkingCompat {
                class_id: 5,
                class_name: "solid_single_yellow".into(),
                confidence: 0.8,
                bbox: [600.0, 200.0, 630.0, 600.0],
                mask: vec![],
                mask_width: 0,
                mask_height: 0,
            },
            DetectedRoadMarkingCompat {
                class_id: 10,
                class_name: "dashed_single_yellow".into(),
                confidence: 0.75,
                bbox: [640.0, 200.0, 670.0, 600.0],
                mask: vec![],
                mask_width: 0,
                mask_height: 0,
            },
        ];
        let m = merge_composite_lines_v2(dets, 1280.0);
        assert_eq!(m.len(), 1);
        assert!(m[0].class_name.contains("dashed_right"));
    }

    #[test]
    fn test_merge_creates_solid_right() {
        let dets = vec![
            DetectedRoadMarkingCompat {
                class_id: 10,
                class_name: "dashed_single_yellow".into(),
                confidence: 0.75,
                bbox: [600.0, 200.0, 630.0, 600.0],
                mask: vec![],
                mask_width: 0,
                mask_height: 0,
            },
            DetectedRoadMarkingCompat {
                class_id: 5,
                class_name: "solid_single_yellow".into(),
                confidence: 0.8,
                bbox: [640.0, 200.0, 670.0, 600.0],
                mask: vec![],
                mask_width: 0,
                mask_height: 0,
            },
        ];
        let m = merge_composite_lines_v2(dets, 1280.0);
        assert_eq!(m.len(), 1);
        assert!(m[0].class_name.contains("solid_right"));
    }

    #[test]
    fn test_temporal_smoothing() {
        let mut c = RoadClassifier::new(1280.0);
        let mixed = [mk(99, "mixed_double_yellow_dashed_right", 640.0, 0.85)];
        for _ in 0..10 {
            c.update(&mixed);
        }
        assert_eq!(c.current().passing_legality, PassingLegality::MixedAllowed);
        assert!(c.current().confidence > 0.8);
    }
}
