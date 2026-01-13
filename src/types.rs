// src/types.rs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Configuration Structs
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub model: ModelConfig,
    pub inference: InferenceConfig,
    pub detection: DetectionConfig,
    pub video: VideoConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub path: String,
    pub input_width: usize,
    pub input_height: usize,
    pub num_anchors: usize,
    pub num_lanes: usize,
    pub griding_num: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub use_tensorrt: bool,
    pub use_fp16: bool,
    pub enable_engine_cache: bool,
    pub engine_cache_path: String,
    pub num_threads: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    pub confidence_threshold: f32,
    pub min_points_per_lane: usize,
    pub smoother_window_size: usize,
    pub calibration_frames: usize,
    pub debounce_frames: u32,
    pub confirm_frames: u32,
    pub min_lane_confidence: f32,
    pub min_position_confidence: f32,
    #[serde(default = "default_drift_threshold")]
    pub drift_threshold: f32,
    #[serde(default = "default_crossing_threshold")]
    pub crossing_threshold: f32,
    #[serde(default = "default_cooldown_frames")]
    pub cooldown_frames: u32,
    #[serde(default = "default_min_duration")]
    pub min_lane_change_duration_ms: f64,
    #[serde(default = "default_max_duration")]
    pub max_lane_change_duration_ms: f64,
    #[serde(default = "default_skip_initial")]
    pub skip_initial_frames: u64,
}

fn default_drift_threshold() -> f32 {
    0.25
}
fn default_crossing_threshold() -> f32 {
    0.50
}
fn default_cooldown_frames() -> u32 {
    60
}
fn default_min_duration() -> f64 {
    800.0
}
fn default_max_duration() -> f64 {
    6000.0
}
fn default_skip_initial() -> u64 {
    90
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoConfig {
    pub input_dir: String,
    pub output_dir: String,
    pub source_width: usize,
    pub source_height: usize,
    pub target_fps: u32,
    pub save_annotated: bool,
    pub save_events_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
}

impl Config {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Config = serde_yaml::from_str(&contents)?;
        Ok(config)
    }
}

// ============================================================================
// Frame Type
// ============================================================================

#[derive(Debug, Clone)]
pub struct Frame {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub timestamp_ms: f64,
}

// ============================================================================
// Lane Detection Types
// ============================================================================

#[derive(Debug, Clone)]
pub struct DetectedLane {
    pub points: Vec<(f32, f32)>,
    pub confidence: f32,
}

// ============================================================================
// Analysis Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LaneChangeState {
    Centered,
    Drifting,
    Crossing,
    Completed,
}

impl LaneChangeState {
    pub fn as_str(&self) -> &'static str {
        match self {
            LaneChangeState::Centered => "CENTERED",
            LaneChangeState::Drifting => "DRIFTING",
            LaneChangeState::Crossing => "CROSSING",
            LaneChangeState::Completed => "COMPLETED",
        }
    }
}

impl std::fmt::Display for LaneChangeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn distance_to(&self, other: &Point) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LanePosition {
    LeftFar,
    LeftNear,
    RightNear,
    RightFar,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lane {
    pub lane_id: usize,
    pub points: Vec<Point>,
    pub confidence: f32,
    pub position: Option<LanePosition>,
}

impl Lane {
    pub fn from_detected(lane_id: usize, detected: &DetectedLane) -> Self {
        Self {
            lane_id,
            points: detected
                .points
                .iter()
                .map(|p| Point::new(p.0, p.1))
                .collect(),
            confidence: detected.confidence,
            position: None,
        }
    }

    pub fn get_x_at_y(&self, target_y: f32) -> Option<f32> {
        if self.points.len() < 2 {
            return None;
        }

        let mut sorted_points = self.points.clone();
        sorted_points.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal));

        for i in 0..sorted_points.len() - 1 {
            let p1 = &sorted_points[i];
            let p2 = &sorted_points[i + 1];

            if p1.y <= target_y && target_y <= p2.y {
                if (p2.y - p1.y).abs() < 1e-6 {
                    return Some(p1.x);
                }
                let ratio = (target_y - p1.y) / (p2.y - p1.y);
                return Some(p1.x + ratio * (p2.x - p1.x));
            }
        }
        None
    }

    pub fn avg_x(&self) -> f32 {
        if self.points.is_empty() {
            return 0.0;
        }
        self.points.iter().map(|p| p.x).sum::<f32>() / self.points.len() as f32
    }

    pub fn bottom_point(&self) -> Option<&Point> {
        self.points
            .iter()
            .max_by(|a, b| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
    }
}

// ============================================================================
// Vehicle State
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VehicleState {
    pub lateral_offset: f32,
    pub lane_width: Option<f32>,
    pub heading_offset: f32,
    pub frame_id: u64,
    pub timestamp_ms: f64,
    pub raw_offset: f32,
    pub detection_confidence: f32,
    /// True if both lanes were detected (more reliable)
    pub both_lanes_detected: bool,
}

impl VehicleState {
    pub fn invalid() -> Self {
        Self {
            lateral_offset: 0.0,
            lane_width: None,
            heading_offset: 0.0,
            frame_id: 0,
            timestamp_ms: 0.0,
            raw_offset: 0.0,
            detection_confidence: 0.0,
            both_lanes_detected: false,
        }
    }

    pub fn normalized_offset(&self) -> Option<f32> {
        match self.lane_width {
            Some(width) if width > 1.0 => Some(self.lateral_offset / width),
            _ => None,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.lane_width.map_or(false, |w| w > 50.0)
    }
}

// ============================================================================
// Direction
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Left = -1,
    Unknown = 0,
    Right = 1,
}

impl Direction {
    pub fn from_offset(offset: f32) -> Self {
        if offset > 0.0 {
            Direction::Right
        } else if offset < 0.0 {
            Direction::Left
        } else {
            Direction::Unknown
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Direction::Left => "LEFT",
            Direction::Right => "RIGHT",
            Direction::Unknown => "UNKNOWN",
        }
    }

    pub fn as_i32(&self) -> i32 {
        match self {
            Direction::Left => -1,
            Direction::Right => 1,
            Direction::Unknown => 0,
        }
    }
}

impl std::fmt::Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// Evidence Paths
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidencePaths {
    pub start_image_path: String,
    pub end_image_path: String,
}

// ============================================================================
// Lane Change Event
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneChangeEvent {
    pub event_id: String,
    pub timestamp: String,
    pub video_timestamp_ms: f64,
    pub start_frame_id: u64,
    pub end_frame_id: u64,
    pub direction: Direction,
    pub confidence: f32,
    pub duration_ms: Option<f64>,
    pub source_id: String,
    pub evidence_images: Option<EvidencePaths>,
    pub metadata: HashMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub legality: Option<LegalityInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalityInfo {
    pub is_legal: bool,
    pub lane_line_type: String,
    pub confidence: f32,
    pub analysis_details: Option<String>,
}

impl LaneChangeEvent {
    pub fn new(
        video_timestamp_ms: f64,
        start_frame_id: u64,
        end_frame_id: u64,
        direction: Direction,
        confidence: f32,
    ) -> Self {
        Self {
            event_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            video_timestamp_ms,
            start_frame_id,
            end_frame_id,
            direction,
            confidence,
            duration_ms: None,
            source_id: String::new(),
            evidence_images: None,
            metadata: HashMap::new(),
            legality: None,
        }
    }

    pub fn direction_name(&self) -> &'static str {
        self.direction.as_str()
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "event_id": self.event_id,
            "type": "lane_change",
            "direction": self.direction_name(),
            "timestamp_ms": self.video_timestamp_ms,
            "frames": {
                "start": self.start_frame_id,
                "end": self.end_frame_id
            },
            "evidence": self.evidence_images,
            "duration_ms": self.duration_ms,
            "source_id": self.source_id,
            "metadata": self.metadata,
            "legality": self.legality,
        })
    }
}

// ============================================================================
// Lane Change Config
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneChangeConfig {
    pub drift_threshold: f32,
    pub crossing_threshold: f32,
    pub min_frames_confirm: u32,
    pub cooldown_frames: u32,
    pub smoothing_alpha: f32,
    pub reference_y_ratio: f32,
    pub hysteresis_factor: f32,
    pub min_duration_ms: f64,
    pub max_duration_ms: f64,
    pub skip_initial_frames: u64,
    pub require_both_lanes: bool,
}

impl Default for LaneChangeConfig {
    fn default() -> Self {
        Self {
            drift_threshold: 0.30,
            crossing_threshold: 0.55,
            min_frames_confirm: 12,
            cooldown_frames: 90,
            smoothing_alpha: 0.3,
            reference_y_ratio: 0.8,
            hysteresis_factor: 0.5,
            min_duration_ms: 1500.0,
            max_duration_ms: 5000.0,
            skip_initial_frames: 150,
            require_both_lanes: true,
        }
    }
}

impl LaneChangeConfig {
    pub fn from_detection_config(detection: &DetectionConfig) -> Self {
        Self {
            drift_threshold: detection.drift_threshold,
            crossing_threshold: detection.crossing_threshold,
            min_frames_confirm: detection.confirm_frames,
            cooldown_frames: detection.cooldown_frames,
            smoothing_alpha: 0.3,
            reference_y_ratio: 0.8,
            hysteresis_factor: 0.5,
            min_duration_ms: detection.min_lane_change_duration_ms,
            max_duration_ms: detection.max_lane_change_duration_ms,
            skip_initial_frames: detection.skip_initial_frames,
            require_both_lanes: detection.require_both_lanes,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    pub confidence_threshold: f32,
    pub min_points_per_lane: usize,
    pub smoother_window_size: usize,
    pub calibration_frames: usize,
    pub debounce_frames: u32,
    pub confirm_frames: u32,
    pub min_lane_confidence: f32,
    pub min_position_confidence: f32,
    #[serde(default = "default_drift_threshold")]
    pub drift_threshold: f32,
    #[serde(default = "default_crossing_threshold")]
    pub crossing_threshold: f32,
    #[serde(default = "default_cooldown_frames")]
    pub cooldown_frames: u32,
    #[serde(default = "default_min_duration")]
    pub min_lane_change_duration_ms: f64,
    #[serde(default = "default_max_duration")]
    pub max_lane_change_duration_ms: f64,
    #[serde(default = "default_skip_initial")]
    pub skip_initial_frames: u64,
    #[serde(default = "default_require_both_lanes")]
    pub require_both_lanes: bool,
}

fn default_drift_threshold() -> f32 {
    0.30
}
fn default_crossing_threshold() -> f32 {
    0.55
}
fn default_cooldown_frames() -> u32 {
    90
}
fn default_min_duration() -> f64 {
    1500.0
}
fn default_max_duration() -> f64 {
    5000.0
}
fn default_skip_initial() -> u64 {
    150
}
fn default_require_both_lanes() -> bool {
    true
}
