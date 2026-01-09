// src/types.rs

use serde::{Deserialize, Serialize};

// ============================================================================
// Configuration Structs
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub model: ModelConfig,
    pub inference: InferenceConfig,
    pub detection: DetectionConfig,
    pub overtake: OvertakeConfig,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OvertakeConfig {
    pub lane_change_offset_threshold: f32,
    pub debounce_frames: u32,
    pub confirm_frames: u32,
    pub max_window_seconds: f64,
    pub min_interval_seconds: f64,
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

// ============================================================================
// Video Processing Types
// ============================================================================

#[derive(Debug, Clone)]
pub struct Frame {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub timestamp: f64,
}

#[derive(Debug, Clone)]
pub struct Lane {
    pub points: Vec<(f32, f32)>,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct LaneDetection {
    pub lanes: Vec<Lane>,
    pub timestamp: f64,
}

// ============================================================================
// Vehicle Position and Direction
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VehiclePosition {
    pub lane_index: i32,
    pub lateral_offset: f32,
    pub confidence: f32,
    pub timestamp: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Direction {
    Left,
    Right,
}

// ============================================================================
// Event Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneChangeEvent {
    pub timestamp: f64,
    pub direction: Direction,
    pub from_lane: i32,
    pub to_lane: i32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OvertakeEvent {
    pub start_timestamp: f64,
    pub end_timestamp: f64,
    pub first_direction: Direction,
    pub second_direction: Direction,
    pub start_lane: i32,
    pub end_lane: i32,
    pub is_complete: bool,
    pub confidence: f32,
}

// ============================================================================
// Implementation: Config Loading
// ============================================================================

impl Config {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Config = serde_yaml::from_str(&contents)?;
        Ok(config)
    }
}

// ============================================================================
// Implementation: VehiclePosition Helpers
// ============================================================================

impl VehiclePosition {
    /// Create an invalid position (used as default/placeholder)
    pub fn invalid() -> Self {
        Self {
            lane_index: -1,
            lateral_offset: 0.0,
            confidence: 0.0,
            timestamp: 0.0,
        }
    }

    /// Check if this position is valid
    pub fn is_valid(&self) -> bool {
        self.lane_index >= 0 && self.confidence > 0.0
    }
}

// ============================================================================
// Implementation: Direction Helpers
// ============================================================================

impl Direction {
    /// Convert to string for display
    pub fn as_str(&self) -> &'static str {
        match self {
            Direction::Left => "Left",
            Direction::Right => "Right",
        }
    }

    /// Get opposite direction
    pub fn opposite(&self) -> Self {
        match self {
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }
}

impl std::fmt::Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
