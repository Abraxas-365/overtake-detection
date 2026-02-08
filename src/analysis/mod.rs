// src/analysis/mod.rs

mod boundary_detector;
mod curve_detector;
mod lane_analyzer;
mod position_estimator;
mod state_machine;
pub mod velocity_tracker;

pub use lane_analyzer::LaneChangeAnalyzer;
pub use velocity_tracker::LateralVelocityTracker; // 🆕 Export
