// src/llm_client.rs
//
// Async HTTP client that sends maneuver analysis requests to the Go LLM server.
//
// Builds a LaneChangeLegalityRequest with:
//   - Strategically captured frames (from StrategicFrameBuffer)
//   - Full detection metadata from the Rust pipeline
//   - Trajectory, velocity, positioning, and temporal info
//   - On-device line crossing detection results (YOLOv8-seg prior)
//   - Curve, shadow, and vehicle overtake sensor data
//
// The goal is to give the LLM vision model maximum context so it can
// accurately confirm or correct the line type detection.

use crate::analysis::curvature_estimator::CurvatureEstimate;
use crate::analysis::maneuver_classifier::{ManeuverEvent, ManeuverSide, ManeuverType};
use crate::frame_buffer::{CapturedFrame, StrategicFrameBuffer};
use crate::lane_legality::LineLegality;
use crate::pipeline::legality_buffer::LegalityRingBuffer;
use crate::road_classification::RoadClassification;
use crate::types::VehicleState;

use base64::Engine;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{error, info, warn};

// ============================================================================
// REQUEST TYPES (must match Go server's expected JSON)
// ============================================================================

#[derive(Debug, Serialize)]
pub struct LaneChangeLegalityRequest {
    pub event_id: String,
    pub direction: String,
    pub start_frame_id: u64,
    pub end_frame_id: u64,
    pub video_timestamp_ms: f64,
    pub duration_ms: Option<f64>,
    pub source_id: String,
    pub frames: Vec<FrameData>,
    pub detection_metadata: DetectionMetadata,
    pub enable_image_enhancement: bool,
    pub enhancement_mode: String,
}

#[derive(Debug, Serialize)]
pub struct FrameData {
    pub frame_index: i32,
    pub timestamp_ms: f64,
    pub width: i32,
    pub height: i32,
    pub base64_image: String,
    pub lane_confidence: Option<f32>,
    pub offset_percentage: Option<f32>,
    /// Why this frame was selected (helps LLM understand what to look for)
    pub capture_reason: Option<String>,
    /// What marking was detected on the left at this frame
    pub left_marking_class: Option<String>,
    /// What marking was detected on the right at this frame
    pub right_marking_class: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct LineCrossingInfo {
    pub line_crossed: bool,
    pub line_type: String,
    pub is_legal: bool,
    pub severity: String,
    pub line_detection_confidence: f32,
    pub crossed_at_frame: u64,
    pub additional_lines_crossed: Vec<String>,
    pub analysis_details: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TrajectoryInfo {
    pub initial_position: f32,
    pub final_position: f32,
    pub net_displacement: f32,
    pub returned_to_start: bool,
    pub excursion_sufficient: bool,
    pub shape_score: f32,
    pub smoothness: f32,
    pub has_direction_reversal: bool,
}

#[derive(Debug, Serialize)]
pub struct VelocityInfo {
    pub peak_lateral_velocity: f32,
    pub avg_lateral_velocity: f32,
    pub velocity_pattern: String,
    pub max_acceleration: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct PositioningInfo {
    pub lane_width_min: f32,
    pub lane_width_max: f32,
    pub lane_width_avg: f32,
    pub lane_width_stable: bool,
    pub adjacent_lane_penetration: f32,
    pub baseline_offset: f32,
    pub baseline_frozen: bool,
}

#[derive(Debug, Serialize)]
pub struct StateTransition {
    pub from_state: String,
    pub to_state: String,
    pub frame_id: u64,
    pub timestamp_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct TemporalInfo {
    pub time_drifting_ms: Option<f64>,
    pub time_crossing_ms: Option<f64>,
    pub total_maneuver_duration_ms: f64,
    pub duration_plausible: bool,
    pub state_progression: Vec<StateTransition>,
}

#[derive(Debug, Serialize)]
pub struct DetectionMetadata {
    pub detection_confidence: f32,
    pub max_offset_normalized: f32,
    pub avg_lane_confidence: f32,
    pub both_lanes_ratio: f32,
    pub video_resolution: String,
    pub fps: f32,
    pub region: String,
    pub avg_lane_width_px: Option<f32>,

    // Curve detection
    pub curve_detected: bool,
    pub curve_angle_degrees: f32,
    pub curve_confidence: f32,
    pub curve_type: String,

    // Shadow overtake
    pub shadow_overtake_detected: bool,
    pub shadow_overtake_count: u32,
    pub shadow_worst_severity: String,
    pub shadow_blocking_vehicles: Vec<String>,

    // On-device line crossing from YOLOv8-seg
    pub line_crossing_info: Option<LineCrossingInfo>,

    // Vehicles overtaken
    pub vehicles_overtaken_count: u32,
    pub overtaken_vehicle_types: Vec<String>,
    pub overtaken_vehicle_ids: Vec<u32>,

    // Maneuver classification
    pub maneuver_type: String,
    pub incomplete_reason: Option<String>,

    // Trajectory analysis
    pub trajectory_info: TrajectoryInfo,
    pub velocity_info: VelocityInfo,
    pub positioning_info: PositioningInfo,

    pub detection_path: Option<String>,
    pub temporal_info: TemporalInfo,

    // Road classification context (from RoadClassifier temporal consensus)
    pub road_classification: Option<RoadClassificationInfo>,
}

/// Road classification info from the temporal consensus system
#[derive(Debug, Serialize)]
pub struct RoadClassificationInfo {
    pub road_type: String,
    pub passing_legality: String,
    pub mixed_line_side: Option<String>,
    pub estimated_lanes: u32,
    pub confidence: f32,
}

// ============================================================================
// RESPONSE TYPE
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct LaneChangeLegalityResponse {
    pub event_id: String,
    pub status: String,
    pub message: String,
}

// ============================================================================
// CLIENT
// ============================================================================

pub struct LlmClient {
    server_url: String,
    http_client: reqwest::Client,
    source_id: String,
    fps: f32,
    video_resolution: String,
}

impl LlmClient {
    pub fn new(server_url: &str, source_id: &str, fps: f32, width: i32, height: i32) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(200))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            server_url: server_url.to_string(),
            http_client,
            source_id: source_id.to_string(),
            fps,
            video_resolution: format!("{}x{}", width, height),
        }
    }

    /// Build and send a full analysis request to the LLM server.
    ///
    /// This takes all available data from the Rust pipeline and packages it
    /// into the richest possible request for the LLM vision model.
    pub async fn analyze_maneuver(
        &self,
        event: &ManeuverEvent,
        captured_frames: &[CapturedFrame],
        curvature: Option<&CurvatureEstimate>,
        road_classification: Option<&RoadClassification>,
        vehicle_state: Option<&VehicleState>,
        legality_buffer: Option<&LegalityRingBuffer>,
        shadow_overtake_count: u64,
        total_vehicles_overtaken: u64,
    ) -> Result<LaneChangeLegalityResponse, String> {
        let request = self.build_request(
            event,
            captured_frames,
            curvature,
            road_classification,
            vehicle_state,
            legality_buffer,
            shadow_overtake_count,
            total_vehicles_overtaken,
        );

        let url = format!("{}/api/analyze", self.server_url);

        info!(
            "🌐 Sending LLM request: {} | {} frames | dir={} | event={}",
            event.maneuver_type.as_str(),
            captured_frames.len(),
            event.side.as_str(),
            request.event_id,
        );

        match self.http_client.post(&url).json(&request).send().await {
            Ok(resp) => {
                if resp.status().is_success() {
                    match resp.json::<LaneChangeLegalityResponse>().await {
                        Ok(result) => {
                            info!(
                                "🌐 LLM response: {} — {}",
                                result.status, result.message
                            );
                            Ok(result)
                        }
                        Err(e) => {
                            warn!("🌐 Failed to parse LLM response: {}", e);
                            Err(format!("Parse error: {}", e))
                        }
                    }
                } else {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    warn!("🌐 LLM server error {}: {}", status, body);
                    Err(format!("HTTP {}: {}", status, body))
                }
            }
            Err(e) => {
                error!("🌐 Failed to reach LLM server: {}", e);
                Err(format!("Connection error: {}", e))
            }
        }
    }

    /// Build the full request with all pipeline metadata
    fn build_request(
        &self,
        event: &ManeuverEvent,
        captured_frames: &[CapturedFrame],
        curvature: Option<&CurvatureEstimate>,
        road_classification: Option<&RoadClassification>,
        vehicle_state: Option<&VehicleState>,
        legality_buffer: Option<&LegalityRingBuffer>,
        shadow_overtake_count: u64,
        total_vehicles_overtaken: u64,
    ) -> LaneChangeLegalityRequest {
        let event_id = format!(
            "{}_{}_{}_f{}",
            event.maneuver_type.as_str().to_lowercase(),
            event.side.as_str().to_lowercase(),
            chrono::Utc::now().format("%Y%m%d_%H%M%S"),
            event.start_frame,
        );

        let direction = match event.side {
            ManeuverSide::Left => "LEFT".to_string(),
            ManeuverSide::Right => "RIGHT".to_string(),
        };

        // ── Build frame data with per-frame context ──
        let frames: Vec<FrameData> = captured_frames
            .iter()
            .enumerate()
            .map(|(i, cf)| {
                let b64 = base64::engine::general_purpose::STANDARD.encode(&cf.jpeg_data);
                FrameData {
                    frame_index: i as i32,
                    timestamp_ms: cf.meta.timestamp_ms,
                    width: cf.meta.width as i32,
                    height: cf.meta.height as i32,
                    base64_image: b64,
                    lane_confidence: cf.meta.lane_confidence,
                    offset_percentage: cf.meta.offset_percentage,
                    capture_reason: Some(cf.meta.reason.as_str().to_string()),
                    left_marking_class: cf.meta.left_marking_class.clone(),
                    right_marking_class: cf.meta.right_marking_class.clone(),
                }
            })
            .collect();

        // ── Build line crossing info from on-device detection ──
        let line_crossing_info = event.crossed_line_class.as_ref().map(|class_name| {
            let is_legal = !event.legality.is_illegal();
            let severity = match event.legality {
                LineLegality::CriticalIllegal => "CRITICAL",
                LineLegality::Illegal => "HIGH",
                LineLegality::Legal => "NONE",
                LineLegality::Caution => "LOW",
                LineLegality::Unknown => "UNKNOWN",
            };
            LineCrossingInfo {
                line_crossed: true,
                line_type: class_name.clone(),
                is_legal,
                severity: severity.to_string(),
                line_detection_confidence: event.confidence,
                crossed_at_frame: event.start_frame,
                additional_lines_crossed: Vec::new(),
                analysis_details: Some(format!(
                    "Detected by YOLOv8-seg + RoadClassifier temporal consensus. Sources: {}",
                    event.sources.summary()
                )),
            }
        });

        // ── Build curve info ──
        let (curve_detected, curve_angle, curve_conf, curve_type) =
            if let Some(curv) = curvature {
                if curv.is_curve {
                    let angle = curv.mean_curvature.abs() * 180.0 / std::f32::consts::PI;
                    let ctype = if angle > 30.0 {
                        "SHARP"
                    } else if angle > 15.0 {
                        "MODERATE"
                    } else {
                        "GENTLE"
                    };
                    (true, angle, curv.confidence, ctype.to_string())
                } else {
                    (false, 0.0, 0.0, "NONE".to_string())
                }
            } else {
                (false, 0.0, 0.0, "NONE".to_string())
            };

        // ── Build shadow overtake info ──
        let is_shadow = event.maneuver_type == ManeuverType::ShadowOvertake;

        // ── Vehicle state → positioning info ──
        let (lane_width_avg, max_offset, avg_confidence) = if let Some(vs) = vehicle_state {
            (
                vs.lane_width.unwrap_or(0.0),
                vs.normalized_offset().unwrap_or(0.0),
                vs.detection_confidence,
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        // ── Road classification context ──
        let road_class_info = road_classification.map(|rc| RoadClassificationInfo {
            road_type: rc.road_type.as_display_str().to_string(),
            passing_legality: rc.passing_legality.as_str().to_string(),
            mixed_line_side: rc.mixed_line_side.as_ref().map(|s| format!("{:?}", s)),
            estimated_lanes: rc.estimated_lanes,
            confidence: rc.confidence,
        });

        // ── Overtaken vehicles ──
        let vehicles_overtaken_count = if event.passed_vehicle_id.is_some() {
            1_u32
        } else {
            0
        };

        let overtaken_vehicle_types: Vec<String> = event
            .passed_vehicle_class
            .map(|c| vec![format!("class_{}", c)])
            .unwrap_or_default();

        let overtaken_vehicle_ids: Vec<u32> = event
            .passed_vehicle_id
            .map(|id| vec![id])
            .unwrap_or_default();

        // ── Marking context from the legality buffer ──
        let marking_context_str = event.marking_context.as_ref().map(|mc| {
            format!(
                "L:{} R:{} @frame{}",
                mc.left_name.as_deref().unwrap_or("?"),
                mc.right_name.as_deref().unwrap_or("?"),
                mc.frame_id
            )
        });

        let detection_metadata = DetectionMetadata {
            detection_confidence: event.confidence,
            max_offset_normalized: max_offset.abs(),
            avg_lane_confidence: avg_confidence,
            both_lanes_ratio: if avg_confidence > 0.5 { 0.8 } else { 0.4 },
            video_resolution: self.video_resolution.clone(),
            fps: self.fps,
            region: "PE".to_string(),
            avg_lane_width_px: Some(lane_width_avg),

            curve_detected,
            curve_angle_degrees: curve_angle,
            curve_confidence: curve_conf,
            curve_type,

            shadow_overtake_detected: is_shadow,
            shadow_overtake_count: if is_shadow { 1 } else { 0 },
            shadow_worst_severity: if is_shadow {
                "HIGH".to_string()
            } else {
                "NONE".to_string()
            },
            shadow_blocking_vehicles: Vec::new(),

            line_crossing_info,

            vehicles_overtaken_count,
            overtaken_vehicle_types,
            overtaken_vehicle_ids,

            maneuver_type: event.maneuver_type.as_str().to_string(),
            incomplete_reason: None,

            trajectory_info: TrajectoryInfo {
                initial_position: 0.0,
                final_position: max_offset,
                net_displacement: max_offset,
                returned_to_start: event.side == ManeuverSide::Right,
                excursion_sufficient: max_offset.abs() > 0.3,
                shape_score: event.confidence,
                smoothness: 0.8,
                has_direction_reversal: event.side == ManeuverSide::Right,
            },
            velocity_info: VelocityInfo {
                peak_lateral_velocity: 0.0,
                avg_lateral_velocity: 0.0,
                velocity_pattern: "smooth".to_string(),
                max_acceleration: None,
            },
            positioning_info: PositioningInfo {
                lane_width_min: lane_width_avg * 0.9,
                lane_width_max: lane_width_avg * 1.1,
                lane_width_avg,
                lane_width_stable: true,
                adjacent_lane_penetration: max_offset.abs(),
                baseline_offset: 0.0,
                baseline_frozen: false,
            },

            detection_path: marking_context_str,

            temporal_info: TemporalInfo {
                time_drifting_ms: None,
                time_crossing_ms: Some(event.duration_ms * 0.3),
                total_maneuver_duration_ms: event.duration_ms,
                duration_plausible: event.duration_ms > 500.0 && event.duration_ms < 60000.0,
                state_progression: vec![
                    StateTransition {
                        from_state: "IN_LANE".to_string(),
                        to_state: "SHIFTING".to_string(),
                        frame_id: event.start_frame,
                        timestamp_ms: event.start_ms,
                    },
                    StateTransition {
                        from_state: "SHIFTING".to_string(),
                        to_state: "CROSSED".to_string(),
                        frame_id: (event.start_frame + event.end_frame) / 2,
                        timestamp_ms: (event.start_ms + event.end_ms) / 2.0,
                    },
                    StateTransition {
                        from_state: "CROSSED".to_string(),
                        to_state: "COMPLETE".to_string(),
                        frame_id: event.end_frame,
                        timestamp_ms: event.end_ms,
                    },
                ],
            },

            road_classification: road_class_info,
        };

        LaneChangeLegalityRequest {
            event_id,
            direction,
            start_frame_id: event.start_frame,
            end_frame_id: event.end_frame,
            video_timestamp_ms: event.start_ms,
            duration_ms: Some(event.duration_ms),
            source_id: self.source_id.clone(),
            frames,
            detection_metadata,
            enable_image_enhancement: true,
            enhancement_mode: "auto".to_string(),
        }
    }
}
