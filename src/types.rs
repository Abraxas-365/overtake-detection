// src/types.rs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Lane Change State Machine States (matching Python)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LaneChangeState {
    /// Vehicle is centered in lane
    Centered,
    /// Vehicle is drifting toward lane boundary
    Drifting,
    /// Vehicle is crossing lane boundary
    Crossing,
    /// Lane change has completed
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

// ============================================================================
// Point and Lane Types (matching Python)
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Calculate Euclidean distance to another point
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
    /// Interpolate X coordinate at a given Y coordinate
    pub fn get_x_at_y(&self, target_y: f32) -> Option<f32> {
        if self.points.len() < 2 {
            return None;
        }

        // Sort points by Y coordinate
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

    /// Get the bottom-most point (highest Y value, closest to vehicle)
    pub fn bottom_point(&self) -> Option<&Point> {
        self.points
            .iter()
            .max_by(|a, b| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get the top-most point (lowest Y value, farthest from vehicle)
    pub fn top_point(&self) -> Option<&Point> {
        self.points
            .iter()
            .min_by(|a, b| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Calculate average X position of the lane
    pub fn avg_x(&self) -> f32 {
        if self.points.is_empty() {
            return 0.0;
        }
        self.points.iter().map(|p| p.x).sum::<f32>() / self.points.len() as f32
    }
}

// ============================================================================
// Vehicle State (matching Python's VehicleState)
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VehicleState {
    /// Distance from lane center in pixels. Negative = left, Positive = right
    pub lateral_offset: f32,
    /// Estimated lane width in pixels (None if unknown)
    pub lane_width: Option<f32>,
    /// Angular offset from lane direction in radians
    pub heading_offset: f32,
    /// Frame ID when this state was computed
    pub frame_id: u64,
    /// Timestamp in milliseconds
    pub timestamp_ms: f64,
}

impl VehicleState {
    /// Create an invalid/default vehicle state
    pub fn invalid() -> Self {
        Self {
            lateral_offset: 0.0,
            lane_width: None,
            heading_offset: 0.0,
            frame_id: 0,
            timestamp_ms: 0.0,
        }
    }

    /// Lateral offset as fraction of lane width (-0.5 to 0.5, 0 = centered)
    pub fn normalized_offset(&self) -> Option<f32> {
        match self.lane_width {
            Some(width) if width > 1.0 => Some(self.lateral_offset / width),
            _ => None,
        }
    }

    /// Check if we have enough lane information for analysis
    pub fn is_valid(&self) -> bool {
        self.lane_width.map_or(false, |w| w > 0.0)
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

    pub fn opposite(&self) -> Self {
        match self {
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
            Direction::Unknown => Direction::Unknown,
        }
    }
}

impl std::fmt::Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// Lane Change Event (matching Python's LaneChangeEvent)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneChangeEvent {
    /// Unique event identifier (UUID)
    pub event_id: String,
    /// Event timestamp (ISO format string)
    pub timestamp: String,
    /// Position in video in milliseconds
    pub video_timestamp_ms: f64,
    /// Frame ID when event was detected
    pub frame_id: u64,
    /// Direction of lane change: -1 = left, 1 = right, 0 = unknown
    pub direction: Direction,
    /// Confidence in the event detection [0.0, 1.0]
    pub confidence: f32,
    /// Duration of lane change maneuver in ms (for COMPLETED events)
    pub duration_ms: Option<f64>,
    /// Identifier of the video source
    pub source_id: String,
    /// Additional event metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl LaneChangeEvent {
    pub fn new(
        video_timestamp_ms: f64,
        frame_id: u64,
        direction: Direction,
        confidence: f32,
    ) -> Self {
        Self {
            event_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            video_timestamp_ms,
            frame_id,
            direction,
            confidence,
            duration_ms: None,
            source_id: String::new(),
            metadata: HashMap::new(),
        }
    }

    /// Human-readable direction name
    pub fn direction_name(&self) -> &'static str {
        self.direction.as_str()
    }

    /// Convert to dictionary/map for JSON serialization
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "event_id": self.event_id,
            "event_type": "lane_change",
            "timestamp": self.timestamp,
            "video_timestamp_ms": self.video_timestamp_ms,
            "frame_id": self.frame_id,
            "direction": self.direction_name(),
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
            "source_id": self.source_id,
            "metadata": self.metadata,
        })
    }
}

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneChangeConfig {
    /// Threshold (as fraction of lane width) to trigger drift (default: 0.2)
    pub drift_threshold: f32,
    /// Threshold to detect lane crossing (default: 0.4)
    pub crossing_threshold: f32,
    /// Frames required to confirm state change (default: 5)
    pub min_frames_confirm: u32,
    /// Cooldown period after completed lane change (default: 30)
    pub cooldown_frames: u32,
    /// EMA smoothing factor (higher = less smoothing) (default: 0.3)
    pub smoothing_alpha: f32,
    /// Vertical position ratio for lane measurements (default: 0.8)
    pub reference_y_ratio: f32,
}

impl Default for LaneChangeConfig {
    fn default() -> Self {
        Self {
            drift_threshold: 0.2,
            crossing_threshold: 0.4,
            min_frames_confirm: 5,
            cooldown_frames: 30,
            smoothing_alpha: 0.3,
            reference_y_ratio: 0.8,
        }
    }
}

// ============================================================================
// Frame type
// ============================================================================

#[derive(Debug, Clone)]
pub struct Frame {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub frame_id: u64,
    pub timestamp_ms: f64,
    pub source_fps: f64,
}
