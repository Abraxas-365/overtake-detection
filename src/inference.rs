// src/inference.rs - CORRECT ort 2.0.0-rc.11 API

use crate::types::Config;
use anyhow::{Context, Result};
use ndarray::Array;
use ort::{
    execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use tracing::{debug, info};

pub struct InferenceEngine {
    session: Session,
    config: Config,
}

impl InferenceEngine {
    pub fn new(config: Config) -> Result<Self> {
        info!("Initializing inference engine");
        info!("Model path: {}", config.model.path);

        let mut session_builder = Session::builder()?;

        if config.inference.use_tensorrt {
            info!("Enabling TensorRT execution provider");

            let mut trt_options = TensorRTExecutionProvider::default();
            if config.inference.use_fp16 {
                trt_options = trt_options.with_fp16(true);
                info!("FP16 precision enabled");
            }
            if config.inference.enable_engine_cache {
                trt_options = trt_options.with_engine_cache(true).with_timing_cache(true);
                info!(
                    "Engine cache enabled at: {}",
                    config.inference.engine_cache_path
                );
            }

            session_builder = session_builder.with_execution_providers([trt_options.build()])?;
        }

        // Add CUDA as fallback
        session_builder =
            session_builder.with_execution_providers([CUDAExecutionProvider::default().build()])?;

        info!("Building ONNX Runtime session...");
        let session = session_builder
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(config.inference.num_threads)?
            .with_inter_threads(1)?
            .commit_from_file(&config.model.path)
            .context("Failed to load model")?;

        info!("✓ Inference engine initialized successfully");

        Ok(Self { session, config })
    }

    pub fn infer(&self, input: &[f32]) -> Result<Vec<f32>> {
        debug!("Running inference");

        // Create input array
        let input_array = Array::from_shape_vec(
            (
                1,
                3,
                self.config.model.input_height,
                self.config.model.input_width,
            ),
            input.to_vec(),
        )?;

        // Create Value from ndarray
        let input_value = Value::from_array(input_array)?;

        // Run inference with inputs! macro
        let outputs = self.session.run(ort::inputs!["input" => input_value])?;

        // Extract output tensor
        let output = &outputs["output"];
        let (_, data) = output.try_extract_raw_tensor::<f32>()?;

        Ok(data.to_vec())
    }
}
