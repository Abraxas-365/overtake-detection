// src/analysis/model_agreement.rs
//
// Cross-validation system that combines UFLDv2 (primary) and YOLOv8-seg (secondary)
// to improve lane detection robustness and reduce false positives.

use crate::lane_legality::DetectedRoadMarking;
use crate::types::VehicleState;
use tracing::{debug, info, warn};

// Agreement thresholds (in pixels at original resolution)
const STRONG_AGREEMENT_THRESHOLD: f32 = 15.0; // <15px diff = strong agreement
const WEAK_AGREEMENT_THRESHOLD: f32 = 25.0; // <25px diff = weak agreement
const DISAGREEMENT_THRESHOLD: f32 = 50.0; // >50px diff = critical disagreement

/// Result of comparing two model outputs
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AgreementLevel {
    StrongAgreement,      // Models agree closely (<15px)
    WeakAgreement,        // Models roughly agree (<25px)
    Disagreement,         // Models disagree moderately (25-50px)
    CriticalDisagreement, // Models strongly disagree (>50px)
}

/// Fused result combining both models
#[derive(Debug, Clone)]
pub struct FusedLaneEstimate {
    /// Final lateral offset (fused from both models)
    pub lateral_offset: f32,

    /// Final lane width
    pub lane_width: f32,

    /// Confidence in this estimate [0.0, 1.0]
    pub confidence: f32,

    /// How well the models agree
    pub agreement_level: AgreementLevel,

    /// Position difference between models (pixels)
    pub position_diff_px: f32,

    /// Which model contributed more to the result
    pub primary_source: EstimateSource,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EstimateSource {
    UFLDOnly,       // Only UFLDv2 available
    YOLOOnly,       // Only YOLOv8-seg available
    UFLDPrimary,    // UFLDv2 was trusted more
    YOLOPrimary,    // YOLOv8-seg was trusted more
    EqualWeighting, // Both equally weighted
}

impl EstimateSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            EstimateSource::UFLDOnly => "UFLD_ONLY",
            EstimateSource::YOLOOnly => "YOLO_ONLY",
            EstimateSource::UFLDPrimary => "UFLD_PRIMARY",
            EstimateSource::YOLOPrimary => "YOLO_PRIMARY",
            EstimateSource::EqualWeighting => "EQUAL_WEIGHT",
        }
    }
}

/// Cross-validation agreement checker
pub struct AgreementChecker {
    /// Recent agreement history for smoothing
    recent_agreements: Vec<AgreementLevel>,

    /// Consecutive disagreement counter
    consecutive_disagreements: u32,

    /// Total fusion attempts
    total_fusions: u64,

    /// Agreement statistics
    strong_agreements: u64,
    weak_agreements: u64,
    disagreements: u64,
    critical_disagreements: u64,
}

impl AgreementChecker {
    pub fn new() -> Self {
        Self {
            recent_agreements: Vec::with_capacity(10),
            consecutive_disagreements: 0,
            total_fusions: 0,
            strong_agreements: 0,
            weak_agreements: 0,
            disagreements: 0,
            critical_disagreements: 0,
        }
    }

    /// Fuse estimates from UFLDv2 and YOLOv8-seg
    ///
    /// # Arguments
    /// * `ufld_state` - Primary lane detector result (UFLDv2)
    /// * `yolo_offset` - Secondary estimate from road markings (YOLOv8-seg)
    /// * `yolo_confidence` - Confidence in YOLOv8-seg estimate
    pub fn fuse_estimates(
        &mut self,
        ufld_state: &VehicleState,
        yolo_offset: Option<f32>,
        yolo_confidence: f32,
    ) -> FusedLaneEstimate {
        self.total_fusions += 1;

        // Case 1: Only UFLDv2 available
        if yolo_offset.is_none() || !ufld_state.is_valid() {
            if ufld_state.is_valid() {
                return FusedLaneEstimate {
                    lateral_offset: ufld_state.lateral_offset,
                    lane_width: ufld_state.lane_width.unwrap_or(450.0),
                    confidence: ufld_state.detection_confidence,
                    agreement_level: AgreementLevel::StrongAgreement,
                    position_diff_px: 0.0,
                    primary_source: EstimateSource::UFLDOnly,
                };
            }

            // Case 2: Only YOLOv8-seg available
            if let Some(yolo_off) = yolo_offset {
                return FusedLaneEstimate {
                    lateral_offset: yolo_off,
                    lane_width: 450.0, // Default
                    confidence: yolo_confidence,
                    agreement_level: AgreementLevel::WeakAgreement,
                    position_diff_px: 0.0,
                    primary_source: EstimateSource::YOLOOnly,
                };
            }

            // Neither available - return invalid
            return FusedLaneEstimate {
                lateral_offset: 0.0,
                lane_width: 450.0,
                confidence: 0.0,
                agreement_level: AgreementLevel::CriticalDisagreement,
                position_diff_px: 0.0,
                primary_source: EstimateSource::UFLDOnly,
            };
        }

        // Case 3: Both models available - FUSION TIME!
        let yolo_off = yolo_offset.unwrap();
        let ufld_off = ufld_state.lateral_offset;

        // Calculate position difference
        let diff_px = (ufld_off - yolo_off).abs();

        // Determine agreement level
        let agreement = if diff_px < STRONG_AGREEMENT_THRESHOLD {
            self.strong_agreements += 1;
            self.consecutive_disagreements = 0;
            AgreementLevel::StrongAgreement
        } else if diff_px < WEAK_AGREEMENT_THRESHOLD {
            self.weak_agreements += 1;
            self.consecutive_disagreements = 0;
            AgreementLevel::WeakAgreement
        } else if diff_px < DISAGREEMENT_THRESHOLD {
            self.disagreements += 1;
            self.consecutive_disagreements += 1;
            AgreementLevel::Disagreement
        } else {
            self.critical_disagreements += 1;
            self.consecutive_disagreements += 1;
            AgreementLevel::CriticalDisagreement
        };

        // Track agreement history
        self.recent_agreements.push(agreement);
        if self.recent_agreements.len() > 10 {
            self.recent_agreements.remove(0);
        }

        // Warn about persistent disagreements
        if self.consecutive_disagreements == 5 {
            warn!(
                "⚠️  5 consecutive disagreements between UFLDv2 and YOLOv8-seg (diff={:.1}px)",
                diff_px
            );
        }

        // FUSION LOGIC based on agreement level
        match agreement {
            AgreementLevel::StrongAgreement => {
                // Trust both, but slightly favor UFLDv2 (it's faster/more specialized)
                let ufld_weight = 0.6;
                let yolo_weight = 0.4;

                let fused_offset = ufld_off * ufld_weight + yolo_off * yolo_weight;
                let fused_confidence = (ufld_state.detection_confidence * ufld_weight
                    + yolo_confidence * yolo_weight)
                    * 1.15; // Bonus for agreement

                debug!(
                    "✅ Strong agreement: UFLD={:.1}px, YOLO={:.1}px, diff={:.1}px → fused={:.1}px",
                    ufld_off, yolo_off, diff_px, fused_offset
                );

                FusedLaneEstimate {
                    lateral_offset: fused_offset,
                    lane_width: ufld_state.lane_width.unwrap_or(450.0),
                    confidence: fused_confidence.min(1.0),
                    agreement_level: agreement,
                    position_diff_px: diff_px,
                    primary_source: EstimateSource::EqualWeighting,
                }
            }

            AgreementLevel::WeakAgreement => {
                // Weighted average favoring higher confidence
                let ufld_conf = ufld_state.detection_confidence;
                let total_conf = ufld_conf + yolo_confidence;

                let ufld_weight = if total_conf > 0.01 {
                    ufld_conf / total_conf
                } else {
                    0.5
                };
                let yolo_weight = 1.0 - ufld_weight;

                let fused_offset = ufld_off * ufld_weight + yolo_off * yolo_weight;
                let fused_confidence = (ufld_conf + yolo_confidence) / 2.0 * 1.05; // Small bonus

                let primary = if ufld_weight > yolo_weight {
                    EstimateSource::UFLDPrimary
                } else {
                    EstimateSource::YOLOPrimary
                };

                debug!(
                    "⚖️  Weak agreement: UFLD={:.1}px (w={:.2}), YOLO={:.1}px (w={:.2}) → {:.1}px",
                    ufld_off, ufld_weight, yolo_off, yolo_weight, fused_offset
                );

                FusedLaneEstimate {
                    lateral_offset: fused_offset,
                    lane_width: ufld_state.lane_width.unwrap_or(450.0),
                    confidence: fused_confidence.min(1.0),
                    agreement_level: agreement,
                    position_diff_px: diff_px,
                    primary_source: primary,
                }
            }

            AgreementLevel::Disagreement | AgreementLevel::CriticalDisagreement => {
                // Trust the higher confidence source, but reduce confidence as penalty
                let trust_ufld = ufld_state.detection_confidence > yolo_confidence;

                let (chosen_offset, chosen_confidence, source) = if trust_ufld {
                    (
                        ufld_off,
                        ufld_state.detection_confidence,
                        EstimateSource::UFLDPrimary,
                    )
                } else {
                    (yolo_off, yolo_confidence, EstimateSource::YOLOPrimary)
                };

                // Confidence penalty for disagreement
                let penalty = match agreement {
                    AgreementLevel::Disagreement => 0.85,         // -15%
                    AgreementLevel::CriticalDisagreement => 0.70, // -30%
                    _ => 1.0,
                };

                let penalized_confidence = chosen_confidence * penalty;

                warn!(
                    "❌ {} UFLD={:.1}px (conf={:.2}), YOLO={:.1}px (conf={:.2}), diff={:.1}px → using {} at {:.2} confidence",
                    if matches!(agreement, AgreementLevel::CriticalDisagreement) { "CRITICAL" } else { "Disagreement:" },
                    ufld_off, ufld_state.detection_confidence,
                    yolo_off, yolo_confidence,
                    diff_px,
                    source.as_str(),
                    penalized_confidence
                );

                FusedLaneEstimate {
                    lateral_offset: chosen_offset,
                    lane_width: ufld_state.lane_width.unwrap_or(450.0),
                    confidence: penalized_confidence,
                    agreement_level: agreement,
                    position_diff_px: diff_px,
                    primary_source: source,
                }
            }
        }
    }

    /// Get agreement statistics
    pub fn get_stats(&self) -> AgreementStats {
        AgreementStats {
            total_fusions: self.total_fusions,
            strong_agreements: self.strong_agreements,
            weak_agreements: self.weak_agreements,
            disagreements: self.disagreements,
            critical_disagreements: self.critical_disagreements,
            agreement_rate: if self.total_fusions > 0 {
                (self.strong_agreements + self.weak_agreements) as f32 / self.total_fusions as f32
            } else {
                0.0
            },
        }
    }

    pub fn reset(&mut self) {
        self.recent_agreements.clear();
        self.consecutive_disagreements = 0;
        self.total_fusions = 0;
        self.strong_agreements = 0;
        self.weak_agreements = 0;
        self.disagreements = 0;
        self.critical_disagreements = 0;
    }
}

impl Default for AgreementChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct AgreementStats {
    pub total_fusions: u64,
    pub strong_agreements: u64,
    pub weak_agreements: u64,
    pub disagreements: u64,
    pub critical_disagreements: u64,
    pub agreement_rate: f32,
}

// ============================================================================
// HELPER: Extract lane position from YOLOv8-seg road markings
// ============================================================================

/// Estimate lateral offset from road markings detected by YOLOv8-seg
pub fn estimate_offset_from_markings(
    markings: &[DetectedRoadMarking],
    left_lane_x: Option<f32>,
    right_lane_x: Option<f32>,
    frame_width: f32,
) -> Option<(f32, f32)> {
    // If we have lane boundaries from UFLDv2, use them
    if let (Some(left_x), Some(right_x)) = (left_lane_x, right_lane_x) {
        let lane_center = (left_x + right_x) / 2.0;
        let vehicle_x = frame_width / 2.0;
        let offset = vehicle_x - lane_center;
        let lane_width = right_x - left_x;

        // Confidence based on number of markings detected
        let confidence = (markings.len() as f32 / 5.0).min(1.0) * 0.8;

        return Some((offset, confidence));
    }

    // Otherwise, try to estimate from markings themselves
    if markings.is_empty() {
        return None;
    }

    // Find left and right markings closest to vehicle center
    let vehicle_x = frame_width / 2.0;
    let mut left_marking: Option<f32> = None;
    let mut right_marking: Option<f32> = None;

    for marking in markings {
        let marking_x = (marking.bbox[0] + marking.bbox[2]) / 2.0;

        if marking_x < vehicle_x {
            if left_marking.is_none() || marking_x > left_marking.unwrap() {
                left_marking = Some(marking_x);
            }
        } else if marking_x > vehicle_x {
            if right_marking.is_none() || marking_x < right_marking.unwrap() {
                right_marking = Some(marking_x);
            }
        }
    }

    // Need both boundaries
    if let (Some(left_x), Some(right_x)) = (left_marking, right_marking) {
        let lane_center = (left_x + right_x) / 2.0;
        let offset = vehicle_x - lane_center;

        // Lower confidence when estimated from markings alone
        let confidence = 0.6;

        Some((offset, confidence))
    } else {
        None
    }
}
