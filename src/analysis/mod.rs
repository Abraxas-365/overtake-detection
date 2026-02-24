// src/analysis/mod.rs
//
// Analysis modules for maneuver detection pipeline.
//
// v4.13: Added curvature_estimator for polynomial curve detection from YOLO-seg masks.
// v5.0:  Added polynomial_tracker for Kalman-based boundary tracking.

pub mod curvature_estimator;
pub mod ego_motion;
pub mod lateral_detector;
pub mod maneuver_classifier;
pub mod maneuver_pipeline;
pub mod pass_detector;
pub mod polynomial_tracker;
pub mod vehicle_tracker;

// Re-exports for ergonomic access from main.rs
pub use ego_motion::{EgoMotionEstimate, GrayFrame};
pub use lateral_detector::{LaneMeasurement, LateralShiftEvent, ShiftDirection};
pub use maneuver_classifier::{DetectionSources, ManeuverEvent, ManeuverSide, ManeuverType};
pub use maneuver_pipeline::{ManeuverFrameInput, ManeuverPipeline, ManeuverPipelineConfig};
pub use pass_detector::{PassDirection, PassEvent, PassSide};
pub use polynomial_tracker::{
    GeometricLaneChangeSignals, PolynomialBoundaryTracker, PolynomialTrackerConfig,
};
pub use vehicle_tracker::{DetectionInput, Track, TrackState, VehicleZone};
