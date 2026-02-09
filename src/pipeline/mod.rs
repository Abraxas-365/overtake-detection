// src/pipeline/mod.rs

pub mod event_bus;
pub mod frame_context;
pub mod metrics;
pub mod orchestrator;

pub use event_bus::{EventBus, PipelineEvent};
pub use frame_context::FrameContext;
pub use metrics::PipelineMetrics;
pub use orchestrator::PipelineOrchestrator;
