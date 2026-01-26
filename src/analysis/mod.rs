// src/analysis/mod.rs

mod boundary_detector;
mod curve_detector;
mod fallback_detector;
mod lane_analyzer;
mod position_estimator;
mod state_machine;
mod velocity_tracker;

pub use fallback_detector::{FallbackDetection, FallbackLaneChangeDetector, FallbackMethod}; // ← ADD THIS
pub use lane_analyzer::LaneChangeAnalyzer;
