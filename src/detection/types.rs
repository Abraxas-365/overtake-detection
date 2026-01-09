// src/detection/types.rs
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub struct VehiclePosition {
    pub lane_index: i32,
    pub lateral_offset: f32,
    pub left_boundary: f32,
    pub right_boundary: f32,
    pub confidence: f32,
    pub timestamp: Instant,
}

impl VehiclePosition {
    pub fn invalid() -> Self {
        Self {
            lane_index: -1,
            lateral_offset: 0.0,
            left_boundary: 0.0,
            right_boundary: 0.0,
            confidence: 0.0,
            timestamp: Instant::now(),
        }
    }

    pub fn is_valid(&self) -> bool {
        self.lane_index >= 0 && self.confidence > 0.5
    }
}

#[derive(Debug, Clone)]
pub struct Lane {
    pub points: Vec<LanePoint>,
    pub lane_id: usize,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct LanePoint {
    pub x: f32,
    pub y: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    None,
    Left,
    Right,
}

#[derive(Debug, Clone, Copy)]
pub struct LaneChangeEvent {
    pub timestamp: Instant,
    pub direction: Direction,
    pub from_lane: i32,
    pub to_lane: i32,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct OvertakeEvent {
    pub start_timestamp: Instant,
    pub end_timestamp: Instant,
    pub first_direction: Direction,
    pub second_direction: Direction,
    pub start_lane: i32,
    pub end_lane: i32,
    pub is_complete: bool,
    pub confidence: f32,
}
