// src/detection/mod.rs

mod overtake_detector;
mod position_calculator;
mod smoother;
mod state_machine;
mod types;

// Re-export public APIs
pub use overtake_detector::OvertakeDetector;
pub use position_calculator::calculate_vehicle_position;
pub use smoother::LanePositionSmoother;
pub use state_machine::StateMachine;
pub use types::*;
