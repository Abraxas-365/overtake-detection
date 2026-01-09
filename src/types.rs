use serde::{Deserialize, Serialize};

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

#[derive(Debug, Clone)]
pub struct LaneChangeEvent {
    pub timestamp: f64,
    pub direction: Direction,
    pub from_lane: i32,
    pub to_lane: i32,
    pub confidence: f32,
}

// Make sure these have Serialize
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum Direction {
    None,
    Left,
    Right,
}

#[derive(Debug, Clone, Serialize)]
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

// Add to src/types.rs

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct VehiclePosition {
    pub lane_index: i32,
    pub lateral_offset: f32,
    pub confidence: f32,
    pub timestamp: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Direction {
    Left,
    Right,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LaneChangeEvent {
    pub timestamp: f64,
    pub direction: Direction,
    pub from_lane: i32,
    pub to_lane: i32,
    pub confidence: f32,
}
