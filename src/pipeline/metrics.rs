// src/pipeline/metrics.rs
//
// Production observability. Tracks timing, counts, and rates
// for every subsystem. Export via /metrics endpoint or logs.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct PipelineMetrics {
    pub total_frames: Arc<AtomicU64>,
    pub frames_with_lanes: Arc<AtomicU64>,
    pub frames_with_vehicles: Arc<AtomicU64>,
    pub lane_changes_detected: Arc<AtomicU64>,
    pub complete_overtakes: Arc<AtomicU64>,
    pub incomplete_overtakes: Arc<AtomicU64>,
    pub illegal_crossings: Arc<AtomicU64>,
    pub critical_violations: Arc<AtomicU64>,
    pub shadow_overtakes: Arc<AtomicU64>,
    pub api_successes: Arc<AtomicU64>,
    pub api_failures: Arc<AtomicU64>,
    pub inference_time_us: Arc<AtomicU64>,
    pub legality_time_us: Arc<AtomicU64>,
    pub yolo_time_us: Arc<AtomicU64>,
    pub started_at: Instant,
}

impl PipelineMetrics {
    pub fn new() -> Self {
        Self {
            total_frames: Arc::new(AtomicU64::new(0)),
            frames_with_lanes: Arc::new(AtomicU64::new(0)),
            frames_with_vehicles: Arc::new(AtomicU64::new(0)),
            lane_changes_detected: Arc::new(AtomicU64::new(0)),
            complete_overtakes: Arc::new(AtomicU64::new(0)),
            incomplete_overtakes: Arc::new(AtomicU64::new(0)),
            illegal_crossings: Arc::new(AtomicU64::new(0)),
            critical_violations: Arc::new(AtomicU64::new(0)),
            shadow_overtakes: Arc::new(AtomicU64::new(0)),
            api_successes: Arc::new(AtomicU64::new(0)),
            api_failures: Arc::new(AtomicU64::new(0)),
            inference_time_us: Arc::new(AtomicU64::new(0)),
            legality_time_us: Arc::new(AtomicU64::new(0)),
            yolo_time_us: Arc::new(AtomicU64::new(0)),
            started_at: Instant::now(),
        }
    }

    pub fn inc(&self, counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_timing(&self, counter: &AtomicU64, duration_us: u64) {
        counter.store(duration_us, Ordering::Relaxed);
    }

    pub fn fps(&self) -> f64 {
        let frames = self.total_frames.load(Ordering::Relaxed);
        let elapsed = self.started_at.elapsed().as_secs_f64();
        if elapsed > 0.01 {
            frames as f64 / elapsed
        } else {
            0.0
        }
    }

    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            total_frames: self.total_frames.load(Ordering::Relaxed),
            fps: self.fps(),
            lane_changes: self.lane_changes_detected.load(Ordering::Relaxed),
            complete_overtakes: self.complete_overtakes.load(Ordering::Relaxed),
            incomplete_overtakes: self.incomplete_overtakes.load(Ordering::Relaxed),
            illegal_crossings: self.illegal_crossings.load(Ordering::Relaxed),
            critical_violations: self.critical_violations.load(Ordering::Relaxed),
            shadow_overtakes: self.shadow_overtakes.load(Ordering::Relaxed),
            api_successes: self.api_successes.load(Ordering::Relaxed),
            api_failures: self.api_failures.load(Ordering::Relaxed),
            avg_inference_us: self.inference_time_us.load(Ordering::Relaxed),
            avg_legality_us: self.legality_time_us.load(Ordering::Relaxed),
            avg_yolo_us: self.yolo_time_us.load(Ordering::Relaxed),
            elapsed_secs: self.started_at.elapsed().as_secs_f64(),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MetricsSummary {
    pub total_frames: u64,
    pub fps: f64,
    pub lane_changes: u64,
    pub complete_overtakes: u64,
    pub incomplete_overtakes: u64,
    pub illegal_crossings: u64,
    pub critical_violations: u64,
    pub shadow_overtakes: u64,
    pub api_successes: u64,
    pub api_failures: u64,
    pub avg_inference_us: u64,
    pub avg_legality_us: u64,
    pub avg_yolo_us: u64,
    pub elapsed_secs: f64,
}
