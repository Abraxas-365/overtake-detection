// src/analysis/mod.rs
//
// Maneuver detection v2 pipeline modules.
//
// Signal flow:
//   YOLO Detections → vehicle_tracker → pass_detector ─┐
//   UFLDv2 Lanes    → lateral_detector ─────────────────┼→ maneuver_classifier → ManeuverEvent
//   Raw Frame       → ego_motion ───────────────────────┘
//   YOLO-seg Markings → maneuver_classifier.update_markings()
//
// Orchestrated by maneuver_pipeline::ManeuverPipeline.

pub mod ego_motion;
pub mod lateral_detector;
pub mod maneuver_classifier;
pub mod maneuver_pipeline;
pub mod pass_detector;
pub mod vehicle_tracker;

// Re-exports for ergonomic access from main.rs
pub use ego_motion::{EgoMotionEstimate, GrayFrame};
pub use lateral_detector::{LaneMeasurement, LateralShiftEvent, ShiftDirection};
pub use maneuver_classifier::{DetectionSources, ManeuverEvent, ManeuverSide, ManeuverType};
pub use maneuver_pipeline::{
    ManeuverFrameInput, ManeuverFrameResult, ManeuverPipeline, ManeuverPipelineConfig,
};
pub use pass_detector::{PassDirection, PassEvent, PassSide};
pub use vehicle_tracker::{DetectionInput, Track, TrackState, VehicleZone};

