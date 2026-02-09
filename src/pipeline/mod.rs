// src/pipeline/mod.rs

pub mod event_bus;
pub mod frame_context;
pub mod legality_buffer;
pub mod metrics;

pub use event_bus::{EventBus, PipelineEvent};
pub use frame_context::FrameContext;
pub use metrics::PipelineMetrics;
