// src/types.rs

use anyhow::Context;
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
    #[serde(default)]
    pub lane_legality: LaneLegalityConfig,
    #[serde(default)]
    pub processing: ProcessingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneLegalityConfig {
    pub enabled: bool,
    pub model_path: String,
    pub confidence_threshold: f32,
    /// Run every N frames (3 = every 3rd frame)
    pub inference_interval: u64,
    /// Ego vehicle bounding box ratios [x1, y1, x2, y2]
    pub ego_bbox_ratio: [f32; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Run YOLO vehicle detection every N frames (1 = every frame, 2 = every other)
    pub vehicle_detection_interval: u64,
    /// Run YOLO-seg lane boundary estimation every N frames (1 = every frame)
    pub lane_detection_interval: u64,
    /// Write annotated video every N frames (1 = every frame, 2 = half, 0 = disable)
    /// Skipped frames are still written (last annotated frame is repeated) to keep video in sync.
    pub annotation_interval: u64,
    /// Print per-stage timing every N frames (0 = disabled)
    pub timing_log_interval: u64,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            vehicle_detection_interval: 2,
            lane_detection_interval: 2,
            annotation_interval: 1,
            timing_log_interval: 0,
        }
    }
}

impl Default for LaneLegalityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model_path: "models/lane_legality.onnx".to_string(),
            confidence_threshold: 0.25,
            inference_interval: 3,
            ego_bbox_ratio: [0.30, 0.75, 0.70, 0.98],
        }
    }
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
    #[serde(default = "default_require_both_lanes")]
    pub require_both_lanes: bool,

    // NEW: Blackout recovery fields added here
    #[serde(default = "default_blackout_frames")]
    pub blackout_detection_frames: u32,
    #[serde(default = "default_blackout_jump")]
    pub blackout_jump_threshold: f32,
    #[serde(default = "default_blackout_enable")]
    pub enable_blackout_recovery: bool,
}

// Default value helper functions
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

// Defaults for new blackout fields
fn default_blackout_frames() -> u32 {
    10
}
fn default_blackout_jump() -> f32 {
    0.25
}
fn default_blackout_enable() -> bool {
    true
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
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path))?;
        let config: Config = serde_yaml::from_str(&contents)
            .with_context(|| format!("Failed to parse config YAML: {}", path))?;
        config.validate()?;
        Ok(config)
    }

    /// Validate configuration at startup — fail fast on bad values
    pub fn validate(&self) -> anyhow::Result<()> {
        // Model file must exist
        if !std::path::Path::new(&self.model.path).exists() {
            anyhow::bail!(
                "Lane model not found: {}. Download it first.",
                self.model.path
            );
        }

        // Lane legality model (if enabled)
        if self.lane_legality.enabled
            && !std::path::Path::new(&self.lane_legality.model_path).exists()
        {
            anyhow::bail!(
                "Lane legality model not found: {}. \
                 Set lane_legality.enabled=false to disable.",
                self.lane_legality.model_path
            );
        }

        // Detection thresholds must make sense
        if self.detection.drift_threshold >= self.detection.crossing_threshold {
            anyhow::bail!(
                "drift_threshold ({}) must be < crossing_threshold ({})",
                self.detection.drift_threshold,
                self.detection.crossing_threshold
            );
        }

        if self.detection.drift_threshold <= 0.0 || self.detection.drift_threshold >= 1.0 {
            anyhow::bail!(
                "drift_threshold must be in (0, 1), got {}",
                self.detection.drift_threshold
            );
        }

        if self.detection.crossing_threshold <= 0.0 || self.detection.crossing_threshold >= 1.0 {
            anyhow::bail!(
                "crossing_threshold must be in (0, 1), got {}",
                self.detection.crossing_threshold
            );
        }

        // Duration sanity
        if self.detection.min_lane_change_duration_ms >= self.detection.max_lane_change_duration_ms
        {
            anyhow::bail!(
                "min_duration ({}) must be < max_duration ({})",
                self.detection.min_lane_change_duration_ms,
                self.detection.max_lane_change_duration_ms
            );
        }

        // Video directory
        if !std::path::Path::new(&self.video.input_dir).exists() {
            anyhow::bail!("Video input directory not found: {}", self.video.input_dir);
        }

        // Ego bbox ratios
        let bbox = &self.lane_legality.ego_bbox_ratio;
        if bbox[0] >= bbox[2] || bbox[1] >= bbox[3] {
            anyhow::bail!(
                "ego_bbox_ratio invalid: x1 < x2 and y1 < y2 required, got {:?}",
                bbox
            );
        }
        for &v in bbox.iter() {
            if !(0.0..=1.0).contains(&v) {
                anyhow::bail!("ego_bbox_ratio values must be in [0, 1], got {:?}", bbox);
            }
        }

        Ok(())
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvidencePaths {
    pub start_image_path: String,
    pub end_image_path: String,
}

// ============================================================================
// Curve Information
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CurveInfo {
    pub is_curve: bool,
    pub angle_degrees: f32,
    pub confidence: f32,
    pub curve_type: CurveType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurveType {
    None,
    Moderate, // 5-15 degrees
    Sharp,    // > 15 degrees
}

impl CurveInfo {
    pub fn none() -> Self {
        Self {
            is_curve: false,
            angle_degrees: 0.0,
            confidence: 0.0,
            curve_type: CurveType::None,
        }
    }
}

// ============================================================================
// Lane Change Event
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

    // Blackout recovery - detects lane changes during lane detection failures
    pub blackout_detection_frames: u32,
    pub blackout_jump_threshold: f32,
    pub enable_blackout_recovery: bool,
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

            // Blackout recovery defaults
            blackout_detection_frames: 10,
            blackout_jump_threshold: 0.25,
            enable_blackout_recovery: true,
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

            // Blackout recovery mapping
            blackout_detection_frames: detection.blackout_detection_frames,
            blackout_jump_threshold: detection.blackout_jump_threshold,
            enable_blackout_recovery: detection.enable_blackout_recovery,
        }
    }
}

// ============================================================================
// Vehicle Position (for smoother)
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub struct VehiclePosition {
    pub lane_index: i32,
    pub lateral_offset: f32,
    pub confidence: f32,
    pub timestamp: f64,
}
