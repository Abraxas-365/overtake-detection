// src/analysis/adaptive/thresholds.rs

use std::collections::VecDeque;
use tracing::{debug, info, warn};

/// Single adaptive threshold with noise-based adjustment
#[derive(Clone, Debug)]
pub struct AdaptiveThreshold {
    pub name: String,

    // Threshold values
    pub nominal_value: f32, // Default/baseline
    pub current_value: f32, // Active (adapted)
    pub min_value: f32,     // Floor
    pub max_value: f32,     // Ceiling

    // Statistics
    noise_mean: f32,
    noise_std: f32,
    recent_stable: VecDeque<f32>,

    // Configuration
    sigma_multiplier: f32,    // Threshold = noise + σ_mult * σ
    confidence_scaling: bool, // Scale based on detection confidence
}

impl AdaptiveThreshold {
    pub fn new(name: &str, nominal: f32, min: f32, max: f32) -> Self {
        Self {
            name: name.to_string(),
            nominal_value: nominal,
            current_value: nominal,
            min_value: min,
            max_value: max,
            noise_mean: 0.0,
            noise_std: 0.0,
            recent_stable: VecDeque::with_capacity(100),
            sigma_multiplier: 2.5, // 2.5σ above noise = 98.8% confidence
            confidence_scaling: true,
        }
    }

    /// Update threshold from stable period measurements
    pub fn update_from_stable_period(&mut self, measurements: &[f32], detection_confidence: f32) {
        if measurements.len() < 15 {
            debug!(
                "🔧 [{}] Insufficient data ({} samples)",
                self.name,
                measurements.len()
            );
            return;
        }

        // Calculate statistics
        self.noise_mean = measurements.iter().sum::<f32>() / measurements.len() as f32;

        let variance: f32 = measurements
            .iter()
            .map(|x| (x - self.noise_mean).powi(2))
            .sum::<f32>()
            / measurements.len() as f32;

        self.noise_std = variance.sqrt();

        // Base adaptive value: mean + sigma_multiplier * std
        let noise_threshold = self.noise_mean.abs() + self.sigma_multiplier * self.noise_std;

        // Confidence scaling
        let confidence_factor = if self.confidence_scaling {
            // High confidence (good detections) → lower threshold (more sensitive)
            // Low confidence (poor detections) → higher threshold (more conservative)
            if detection_confidence > 0.75 {
                0.80 // Can afford to be 20% more aggressive
            } else if detection_confidence > 0.60 {
                0.90 // Slightly more aggressive
            } else if detection_confidence > 0.40 {
                1.00 // Nominal
            } else if detection_confidence > 0.25 {
                1.20 // More conservative
            } else {
                1.50 // Very conservative in poor conditions
            }
        } else {
            1.0
        };

        // Apply scaling and clamp
        let adapted = (noise_threshold * confidence_factor)
            .max(self.min_value)
            .min(self.max_value);

        // Smooth transition (exponential moving average)
        self.current_value = 0.3 * adapted + 0.7 * self.current_value;

        info!(
            "🔧 [{}] Adapted: {:.3} → {:.3} | noise(μ={:.3}, σ={:.3}) | conf={:.2} | factor={:.2}",
            self.name,
            self.nominal_value,
            self.current_value,
            self.noise_mean,
            self.noise_std,
            detection_confidence,
            confidence_factor
        );
    }

    /// Quick online update with single measurement
    pub fn feed_measurement(&mut self, value: f32) {
        self.recent_stable.push_back(value);
        if self.recent_stable.len() > 100 {
            self.recent_stable.pop_front();
        }
    }

    /// Check if we have stable period and update
    pub fn try_auto_update(&mut self, detection_confidence: f32) {
        if self.recent_stable.len() < 30 {
            return;
        }

        // Check if recent measurements are stable (low variance)
        let measurements: Vec<f32> = self.recent_stable.iter().copied().collect();
        let mean = measurements.iter().sum::<f32>() / measurements.len() as f32;
        let variance = measurements.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
            / measurements.len() as f32;

        // Only update during stable periods (variance < 0.01)
        if variance < 0.01 {
            self.update_from_stable_period(&measurements, detection_confidence);
            self.recent_stable.clear(); // Reset for next window
        }
    }

    pub fn get(&self) -> f32 {
        self.current_value
    }

    pub fn reset_to_nominal(&mut self) {
        self.current_value = self.nominal_value;
        self.recent_stable.clear();
    }
}

/// Duration threshold (milliseconds)
#[derive(Clone, Debug)]
pub struct AdaptiveDuration {
    pub name: String,
    pub nominal_value: f64,
    pub current_value: f64,
    pub min_value: f64,
    pub max_value: f64,
}

impl AdaptiveDuration {
    pub fn new(name: &str, nominal: f64, min: f64, max: f64) -> Self {
        Self {
            name: name.to_string(),
            nominal_value: nominal,
            current_value: nominal,
            min_value: min,
            max_value: max,
        }
    }

    pub fn set(&mut self, value: f64) {
        self.current_value = value.max(self.min_value).min(self.max_value);
    }

    pub fn get(&self) -> f64 {
        self.current_value
    }
}

/// Complete set of adaptive thresholds
pub struct AdaptiveThresholdSet {
    pub drift_threshold: AdaptiveThreshold,
    pub crossing_threshold: AdaptiveThreshold,
    pub consistency_threshold: AdaptiveThreshold,
    pub min_duration: AdaptiveDuration,

    // Counters
    frames_processed: u64,
    last_adaptation_frame: u64,
}

impl AdaptiveThresholdSet {
    pub fn new() -> Self {
        Self {
            // Drift detection (when to start considering movement)
            drift_threshold: AdaptiveThreshold::new(
                "drift_start",
                0.28, // Nominal: 28% deviation
                0.18, // Min: 18% (aggressive)
                0.45, // Max: 45% (conservative)
            ),

            // Crossing threshold (when to confirm lane change)
            crossing_threshold: AdaptiveThreshold::new(
                "crossing", 0.40, // Nominal: 40%
                0.30, // Min: 30%
                0.60, // Max: 60%
            ),

            // Direction consistency (% of frames agreeing on direction)
            consistency_threshold: AdaptiveThreshold::new(
                "consistency",
                0.70, // Nominal: 70%
                0.60, // Min: 60%
                0.90, // Max: 90%
            ),

            // Minimum duration
            min_duration: AdaptiveDuration::new(
                "min_duration",
                1500.0, // Nominal: 1.5s
                800.0,  // Min: 0.8s
                5000.0, // Max: 5s (for slow trucks)
            ),

            frames_processed: 0,
            last_adaptation_frame: 0,
        }
    }

    /// Adapt all thresholds based on current conditions
    pub fn adapt(&mut self, detection_confidence: f32, recent_deviations: &[f32], frame_id: u64) {
        self.frames_processed = frame_id;

        // Only adapt every 30 frames (1 second)
        if frame_id - self.last_adaptation_frame < 30 {
            return;
        }

        self.last_adaptation_frame = frame_id;

        // Update drift threshold from stable deviations
        if recent_deviations.len() >= 15 {
            self.drift_threshold
                .update_from_stable_period(recent_deviations, detection_confidence);
        }
    }

    /// Apply context-specific overrides (called by context detector)
    pub fn apply_context_overrides(&mut self, context: &RoadContext) {
        match context {
            RoadContext::MiningRouteDust => {
                // Dust conditions: very conservative
                self.drift_threshold.current_value = self.drift_threshold.current_value.max(0.35);
                self.consistency_threshold.current_value = 0.85;
                self.min_duration.current_value = 2500.0;
                info!("🏜️ Dust mode: Stricter thresholds");
            }

            RoadContext::MiningRouteUnpaved => {
                // Unpaved: rely more on YOLOv8, longer durations
                self.min_duration.current_value = 3000.0;
                self.consistency_threshold.current_value = 0.80;
                info!("🛤️ Unpaved mode: Longer duration required");
            }

            RoadContext::HighwayPaved => {
                // Good conditions: can be more aggressive
                self.drift_threshold.current_value = self.drift_threshold.current_value.min(0.25);
                self.min_duration.current_value = 1200.0;
                info!("🛣️ Highway mode: Standard thresholds");
            }

            _ => {}
        }
    }

    pub fn reset(&mut self) {
        self.drift_threshold.reset_to_nominal();
        self.crossing_threshold.reset_to_nominal();
        self.consistency_threshold.reset_to_nominal();
        self.min_duration.current_value = self.min_duration.nominal_value;
    }
}

impl Default for AdaptiveThresholdSet {
    fn default() -> Self {
        Self::new()
    }
}

// Road context enum (defined in context_detector.rs but used here)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoadContext {
    MiningRouteDust,    // Desert, dust, poor visibility
    MiningRouteUnpaved, // No markings, gravel/dirt
    MiningRoutePaved,   // Paved with markings
    HighwayPaved,       // Clear highway
    Urban,              // City roads
    Unknown,
}
