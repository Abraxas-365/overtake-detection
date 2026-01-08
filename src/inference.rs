use crate::types::{Config, LaneDetection};
use anyhow::{Context, Result};
use ndarray::{Array, IxDyn};
use ort::{
    CUDAExecutionProvider, ExecutionProvider, GraphOptimizationLevel, Session,
    TensorRTExecutionProvider,
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

        // Build execution providers
        let mut providers: Vec<ExecutionProvider> = Vec::new();

        if config.inference.use_tensorrt {
            info!("Enabling TensorRT execution provider");
            let mut trt_provider = TensorRTExecutionProvider::default();

            if config.inference.use_fp16 {
                trt_provider = trt_provider.with_fp16(true);
                info!("FP16 precision enabled");
            }

            if config.inference.enable_engine_cache {
                trt_provider = trt_provider.with_engine_cache(true).with_timing_cache(true);
                info!(
                    "Engine cache enabled at: {}",
                    config.inference.engine_cache_path
                );
            }

            providers.push(trt_provider.build());
        }

        // Add CUDA as fallback
        providers.push(CUDAExecutionProvider::default().build());

        info!("Building ONNX Runtime session...");
        let session = Session::builder()?
            .with_execution_providers(providers)?
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

        // Create input tensor
        let input_shape = vec![
            1,
            3,
            self.config.model.input_height,
            self.config.model.input_width,
        ];
        let input_array = Array::from_shape_vec(IxDyn(&input_shape), input.to_vec())
            .context("Failed to create input array")?;

        // Run inference
        let outputs = self
            .session
            .run(ort::inputs![input_array]?)
            .context("Inference failed")?;

        // Extract output
        let output = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;
        let output_slice = output
            .view()
            .as_slice()
            .context("Failed to get output slice")?;

        Ok(output_slice.to_vec())
    }
}
