// src/analysis/inference_scheduler.rs
//
// Adaptive scheduler for YOLOv8-seg inference.
// Runs YOLOv8-seg only when needed to balance performance and robustness.

use tracing::debug;

/// Decides when to run YOLOv8-seg based on system state
pub struct InferenceScheduler {
    /// Frames since last YOLOv8-seg run
    frames_since_last_yolo: u32,

    /// Current scheduling strategy
    current_strategy: SchedulingStrategy,

    /// Total frames processed
    total_frames: u64,

    /// Times YOLOv8-seg was invoked
    yolo_invocations: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SchedulingStrategy {
    /// Run every frame (during maneuvers)
    EveryFrame,

    /// Run every 2nd frame (high uncertainty)
    EveryOther,

    /// Run every 3rd frame (medium uncertainty)
    EveryThird,

    /// Run every 5th frame (stable driving)
    EveryFifth,

    /// Skip (during cooldown or high confidence)
    Skip,
}

impl InferenceScheduler {
    pub fn new() -> Self {
        Self {
            frames_since_last_yolo: 0,
            current_strategy: SchedulingStrategy::EveryThird,
            total_frames: 0,
            yolo_invocations: 0,
        }
    }

    /// Determine if YOLOv8-seg should run this frame
    ///
    /// # Arguments
    /// * `ufld_confidence` - UFLDv2's confidence in current detection
    /// * `baseline_confidence` - Baseline tracker confidence
    /// * `is_maneuvering` - Is vehicle in DRIFTING or CROSSING state?
    /// * `frames_since_occlusion` - Frames since lane recovery (0 if no recent occlusion)
    pub fn should_run_yolo(
        &mut self,
        ufld_confidence: f32,
        baseline_confidence: f32,
        is_maneuvering: bool,
        frames_since_occlusion: u32,
    ) -> bool {
        self.total_frames += 1;
        self.frames_since_last_yolo += 1;

        // ALWAYS run during maneuvers
        if is_maneuvering {
            self.current_strategy = SchedulingStrategy::EveryFrame;
            self.frames_since_last_yolo = 0;
            self.yolo_invocations += 1;
            return true;
        }

        // ALWAYS run shortly after occlusion recovery
        if frames_since_occlusion > 0 && frames_since_occlusion < 30 {
            self.current_strategy = SchedulingStrategy::EveryOther;
            if self.frames_since_last_yolo >= 2 {
                self.frames_since_last_yolo = 0;
                self.yolo_invocations += 1;
                return true;
            }
            return false;
        }

        // Choose strategy based on confidence levels
        let strategy = if ufld_confidence < 0.5 || baseline_confidence < 0.4 {
            // Low confidence → validate more often
            SchedulingStrategy::EveryOther
        } else if ufld_confidence < 0.7 || baseline_confidence < 0.6 {
            // Medium confidence → validate occasionally
            SchedulingStrategy::EveryThird
        } else {
            // High confidence → validate rarely
            SchedulingStrategy::EveryFifth
        };

        self.current_strategy = strategy;

        // Check if enough frames have passed
        let should_run = match strategy {
            SchedulingStrategy::EveryFrame => true,
            SchedulingStrategy::EveryOther => self.frames_since_last_yolo >= 2,
            SchedulingStrategy::EveryThird => self.frames_since_last_yolo >= 3,
            SchedulingStrategy::EveryFifth => self.frames_since_last_yolo >= 5,
            SchedulingStrategy::Skip => false,
        };

        if should_run {
            self.frames_since_last_yolo = 0;
            self.yolo_invocations += 1;
            info!(
                "🔍 YOLOv8-seg scheduled (strategy={:?}, UFLD_conf={:.2}, baseline_conf={:.2})",
                strategy, ufld_confidence, baseline_confidence
            );
        }

        should_run
    }

    /// Get average FPS estimate based on scheduling
    pub fn estimated_fps(&self, base_ufld_fps: f32) -> f32 {
        if self.total_frames == 0 {
            return base_ufld_fps;
        }

        let yolo_frequency = self.yolo_invocations as f32 / self.total_frames as f32;

        // Rough FPS calculation:
        // - UFLDv2 alone: ~300 FPS
        // - UFLDv2 + YOLOv8-seg: ~80 FPS
        // Weighted average based on how often we run YOLOv8-seg

        let ufld_only_fps = 300.0;
        let both_models_fps = 80.0;

        ufld_only_fps * (1.0 - yolo_frequency) + both_models_fps * yolo_frequency
    }

    /// Get utilization statistics
    pub fn get_stats(&self) -> SchedulerStats {
        SchedulerStats {
            total_frames: self.total_frames,
            yolo_invocations: self.yolo_invocations,
            yolo_frequency: if self.total_frames > 0 {
                self.yolo_invocations as f32 / self.total_frames as f32
            } else {
                0.0
            },
            current_strategy: self.current_strategy,
        }
    }

    pub fn reset(&mut self) {
        self.frames_since_last_yolo = 0;
        self.current_strategy = SchedulingStrategy::EveryThird;
        self.total_frames = 0;
        self.yolo_invocations = 0;
    }
}

impl Default for InferenceScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub total_frames: u64,
    pub yolo_invocations: u64,
    pub yolo_frequency: f32,
    pub current_strategy: SchedulingStrategy,
}
