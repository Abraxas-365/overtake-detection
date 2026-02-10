// src/analysis/adaptive/context_detector.rs

use super::thresholds::RoadContext;
use crate::types::VehicleState;
use std::collections::VecDeque;
use tracing::{debug, info};

/// Detects current road context from sensor data
pub struct ContextDetector {
    // Detection confidence history
    confidence_history: VecDeque<f32>,

    // Lane width history (paved roads have consistent width)
    lane_width_history: VecDeque<f32>,

    // Frames without lanes (indicates dust/occlusion)
    frames_without_lanes: u32,

    // Current context
    pub current_context: RoadContext,

    // Confidence in current context classification
    context_confidence: f32,

    // Hysteresis to avoid rapid switching
    frames_in_context: u32,
    min_frames_before_switch: u32,
}

impl ContextDetector {
    pub fn new() -> Self {
        Self {
            confidence_history: VecDeque::with_capacity(90), // 3 seconds
            lane_width_history: VecDeque::with_capacity(90),
            frames_without_lanes: 0,
            current_context: RoadContext::Unknown,
            context_confidence: 0.0,
            frames_in_context: 0,
            min_frames_before_switch: 60, // 2 seconds hysteresis
        }
    }

    /// Update context based on current frame data
    pub fn update(&mut self, vehicle_state: &VehicleState) -> RoadContext {
        // Update histories
        if vehicle_state.is_valid() {
            self.confidence_history
                .push_back(vehicle_state.detection_confidence);
            if let Some(width) = vehicle_state.lane_width {
                self.lane_width_history.push_back(width);
            }
            self.frames_without_lanes = 0;
        } else {
            self.frames_without_lanes += 1;
        }

        // Keep histories at max 3 seconds
        if self.confidence_history.len() > 90 {
            self.confidence_history.pop_front();
        }
        if self.lane_width_history.len() > 90 {
            self.lane_width_history.pop_front();
        }

        // Calculate metrics
        let avg_confidence = self.average_confidence();
        let confidence_stability = self.confidence_stability();
        let has_consistent_width = self.has_consistent_lane_width();
        let occlusion_rate = self.occlusion_rate();

        // Classify context
        let detected_context = self.classify_context(
            avg_confidence,
            confidence_stability,
            has_consistent_width,
            occlusion_rate,
        );

        // Apply hysteresis (don't switch too quickly)
        if detected_context == self.current_context {
            self.frames_in_context += 1;
            self.context_confidence = (self.context_confidence + 0.05).min(1.0);
        } else if self.frames_in_context < self.min_frames_before_switch {
            // Not enough evidence to switch yet
            self.frames_in_context += 1;
        } else {
            // Confident switch
            info!(
                "🔄 Context switch: {:?} → {:?} (conf={:.2})",
                self.current_context, detected_context, self.context_confidence
            );
            self.current_context = detected_context;
            self.frames_in_context = 0;
            self.context_confidence = 0.5;
        }

        self.current_context
    }

    fn classify_context(
        &self,
        avg_confidence: f32,
        stability: f32,
        has_consistent_width: bool,
        occlusion_rate: f32,
    ) -> RoadContext {
        // MINING ROUTE - DUST CONDITIONS
        // - Low average confidence (<0.35)
        // - High occlusion rate (>20%)
        // - Unstable detections
        if avg_confidence < 0.35 && occlusion_rate > 0.20 {
            debug!(
                "🏜️ Detected: MiningRouteDust (conf={:.2}, occl={:.1}%)",
                avg_confidence,
                occlusion_rate * 100.0
            );
            return RoadContext::MiningRouteDust;
        }

        // MINING ROUTE - UNPAVED
        // - Moderate confidence (0.35-0.50)
        // - No consistent lane width (unpaved roads vary)
        // - Moderate occlusion
        if avg_confidence >= 0.35 && avg_confidence < 0.50 && !has_consistent_width {
            debug!(
                "🛤️ Detected: MiningRouteUnpaved (conf={:.2}, width_stable={})",
                avg_confidence, has_consistent_width
            );
            return RoadContext::MiningRouteUnpaved;
        }

        // MINING ROUTE - PAVED
        // - Good confidence (0.50-0.65)
        // - Consistent lane width
        // - Low occlusion
        if avg_confidence >= 0.50
            && avg_confidence < 0.65
            && has_consistent_width
            && occlusion_rate < 0.10
        {
            debug!("🛣️ Detected: MiningRoutePaved (conf={:.2})", avg_confidence);
            return RoadContext::MiningRoutePaved;
        }

        // HIGHWAY - PAVED
        // - High confidence (>0.65)
        // - Very stable
        // - Consistent width
        if avg_confidence >= 0.65 && stability > 0.85 && has_consistent_width {
            debug!(
                "🛣️ Detected: HighwayPaved (conf={:.2}, stab={:.2})",
                avg_confidence, stability
            );
            return RoadContext::HighwayPaved;
        }

        // Default: Unknown (need more data)
        RoadContext::Unknown
    }

    fn average_confidence(&self) -> f32 {
        if self.confidence_history.is_empty() {
            return 0.0;
        }
        self.confidence_history.iter().sum::<f32>() / self.confidence_history.len() as f32
    }

    fn confidence_stability(&self) -> f32 {
        if self.confidence_history.len() < 10 {
            return 0.0;
        }

        let values: Vec<f32> = self.confidence_history.iter().copied().collect();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std = variance.sqrt();

        // Stability = 1 - normalized_std (higher is more stable)
        (1.0 - (std / 0.3)).max(0.0)
    }

    fn has_consistent_lane_width(&self) -> bool {
        if self.lane_width_history.len() < 20 {
            return false;
        }

        let widths: Vec<f32> = self.lane_width_history.iter().copied().collect();
        let mean = widths.iter().sum::<f32>() / widths.len() as f32;
        let variance = widths.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / widths.len() as f32;
        let std = variance.sqrt();

        // Consistent if std < 15% of mean
        (std / mean) < 0.15
    }

    fn occlusion_rate(&self) -> f32 {
        // Recent occlusion rate (last 3 seconds)
        let total_frames = self.confidence_history.len() + self.frames_without_lanes as usize;
        if total_frames == 0 {
            return 0.0;
        }

        self.frames_without_lanes as f32 / total_frames as f32
    }

    pub fn get_context(&self) -> RoadContext {
        self.current_context
    }

    pub fn get_confidence(&self) -> f32 {
        self.context_confidence
    }

    pub fn reset(&mut self) {
        self.confidence_history.clear();
        self.lane_width_history.clear();
        self.frames_without_lanes = 0;
        self.current_context = RoadContext::Unknown;
        self.context_confidence = 0.0;
        self.frames_in_context = 0;
    }
}

impl Default for ContextDetector {
    fn default() -> Self {
        Self::new()
    }
}
