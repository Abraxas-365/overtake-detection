// src/analysis/mod.rs

pub mod adaptive;
pub mod baseline_confidence; // 🆕 NEW
pub mod boundary_detector;
pub mod curve_detector;
pub mod fallback_estimator;
pub mod inference_scheduler; // 🆕 NEW
pub mod lane_analyzer;
pub mod model_agreement; // 🆕 NEW
pub mod position_estimator;
pub mod state_machine;
pub mod velocity_tracker; // 🆕 Add this

// Re-exports
pub use baseline_confidence::BaselineConfidence; // 🆕 NEW
pub use inference_scheduler::InferenceScheduler;
pub use lane_analyzer::LaneChangeAnalyzer;
pub use model_agreement::{AgreementChecker, EstimateSource, FusedLaneEstimate}; // 🆕 NEW // 🆕 NEW
