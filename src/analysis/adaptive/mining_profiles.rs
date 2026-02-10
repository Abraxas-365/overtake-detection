// src/analysis/adaptive/mining_profiles.rs

use super::thresholds::RoadContext;

/// Peru mining route characteristics
pub struct MiningRouteProfile {
    pub name: &'static str,
    pub description: &'static str,

    // Detection parameters
    pub min_lane_confidence: f32,
    pub min_duration_ms: f64,
    pub drift_threshold: f32,
    pub crossing_threshold: f32,
    pub consistency_threshold: f32,

    // Validation requirements
    pub require_yolo_validation: bool,
    pub require_boundary_crossing: bool,
    pub allow_sustained_path: bool,

    // Post-occlusion behavior
    pub post_occlusion_freeze_frames: u32,
    pub max_occlusion_before_reset: u32,
}

impl MiningRouteProfile {
    /// Dust storm / heavy dust conditions
    pub fn dust_heavy() -> Self {
        Self {
            name: "MiningDustHeavy",
            description: "Heavy dust, very poor visibility",
            min_lane_confidence: 0.60,         // Only trust high-confidence
            min_duration_ms: 3500.0,           // 3.5s minimum
            drift_threshold: 0.40,             // Large deviation required
            crossing_threshold: 0.50,          // Clear crossing
            consistency_threshold: 0.90,       // Very consistent
            require_yolo_validation: true,     // Must have YOLOv8 confirmation
            require_boundary_crossing: true,   // Physical boundary cross
            allow_sustained_path: false,       // Too risky in dust
            post_occlusion_freeze_frames: 150, // 5 seconds freeze
            max_occlusion_before_reset: 120,   // 4 seconds max blind
        }
    }

    /// Moderate dust (common in Peru mining)
    pub fn dust_moderate() -> Self {
        Self {
            name: "MiningDustModerate",
            description: "Moderate dust, intermittent visibility",
            min_lane_confidence: 0.50,
            min_duration_ms: 2500.0, // 2.5s
            drift_threshold: 0.35,
            crossing_threshold: 0.45,
            consistency_threshold: 0.85,
            require_yolo_validation: true,
            require_boundary_crossing: false,
            allow_sustained_path: false,      // Disabled in dust
            post_occlusion_freeze_frames: 90, // 3 seconds
            max_occlusion_before_reset: 90,
        }
    }

    /// Unpaved road (gravel/dirt, no markings)
    pub fn unpaved() -> Self {
        Self {
            name: "MiningUnpaved",
            description: "Unpaved route, relying on road edges/YOLOv8",
            min_lane_confidence: 0.45,
            min_duration_ms: 1800.0,
            // Trucks are slow
            drift_threshold: 0.30, // More permissive
            crossing_threshold: 0.40,
            consistency_threshold: 0.80,
            require_yolo_validation: true, // Critical on unpaved
            require_boundary_crossing: false,
            allow_sustained_path: true, // Can use gradual changes
            post_occlusion_freeze_frames: 60,
            max_occlusion_before_reset: 60,
        }
    }

    /// Paved mining route (best case)
    pub fn paved() -> Self {
        Self {
            name: "MiningPaved",
            description: "Paved route with markings",
            min_lane_confidence: 0.40,
            min_duration_ms: 2000.0,
            drift_threshold: 0.28,
            crossing_threshold: 0.40,
            consistency_threshold: 0.75,
            require_yolo_validation: false, // Optional
            require_boundary_crossing: false,
            allow_sustained_path: true,
            post_occlusion_freeze_frames: 60,
            max_occlusion_before_reset: 90,
        }
    }

    /// Highway (rare in mining, but for reference)
    pub fn highway() -> Self {
        Self {
            name: "Highway",
            description: "Standard highway conditions",
            min_lane_confidence: 0.35,
            min_duration_ms: 1500.0,
            drift_threshold: 0.25,
            crossing_threshold: 0.40,
            consistency_threshold: 0.70,
            require_yolo_validation: false,
            require_boundary_crossing: false,
            allow_sustained_path: true,
            post_occlusion_freeze_frames: 45,
            max_occlusion_before_reset: 120,
        }
    }

    /// Get profile for context
    pub fn for_context(context: RoadContext) -> Self {
        match context {
            RoadContext::MiningRouteDust => Self::dust_moderate(),
            RoadContext::MiningRouteUnpaved => Self::unpaved(),
            RoadContext::MiningRoutePaved => Self::paved(),
            RoadContext::HighwayPaved => Self::highway(),
            _ => Self::paved(), // Default
        }
    }
}
