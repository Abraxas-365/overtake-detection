// src/analysis/baseline_confidence.rs
//
// Tracks the reliability of the lane position baseline to prevent false
// lane change detections after occlusion periods.
//
// Key insight: Don't immediately trust lane detections after long periods
// without detection. Require them to prove stability before triggering maneuvers.

use tracing::{debug, info, warn};

/// Tracks the reliability of the current baseline offset
pub struct BaselineConfidence {
    /// Current confidence score [0.0, 1.0]
    confidence: f32,

    /// Frame when baseline was last established
    established_at_frame: u64,

    /// Number of consecutive frames confirming this baseline
    confirming_frames: u32,

    /// Number of frames without lane detection since baseline
    frames_without_lanes: u32,

    /// Was baseline established after occlusion?
    post_occlusion: bool,

    /// Minimum frames required to trust post-occlusion baseline
    post_occlusion_stabilization_frames: u32,

    /// Frames since last occlusion recovery
    frames_since_recovery: u32,

    /// Last known offset value for stability checking
    last_offset: f32,

    /// Running average of offset during stabilization
    stabilization_offset_sum: f32,
    stabilization_offset_count: u32,
}

impl BaselineConfidence {
    pub fn new() -> Self {
        Self {
            confidence: 0.0,
            established_at_frame: 0,
            confirming_frames: 0,
            frames_without_lanes: 0,
            post_occlusion: false,
            post_occlusion_stabilization_frames: 60, // 2 seconds at 30fps
            frames_since_recovery: 0,
            last_offset: 0.0,
            stabilization_offset_sum: 0.0,
            stabilization_offset_count: 0,
        }
    }

    /// Establish a new baseline with quality metrics
    ///
    /// # Arguments
    /// * `frame_id` - Frame number where baseline is established
    /// * `both_lanes_detected` - Were both ego lanes detected?
    /// * `detection_confidence` - Lane detector's confidence score
    /// * `lane_width` - Detected lane width (helps validate reasonableness)
    /// * `after_occlusion` - Was this baseline set after a period without lanes?
    /// * `occlusion_duration_frames` - How many frames was occlusion? (for adaptive stabilization)
    pub fn establish_baseline(
        &mut self,
        frame_id: u64,
        offset: f32,
        both_lanes_detected: bool,
        detection_confidence: f32,
        lane_width: Option<f32>,
        after_occlusion: bool,
        occlusion_duration_frames: u32,
    ) {
        // Calculate initial confidence based on detection quality
        let mut initial_confidence = 0.5;

        // Bonus for both lanes detected
        if both_lanes_detected {
            initial_confidence += 0.2;
        }

        // Bonus for high detection confidence
        if detection_confidence > 0.8 {
            initial_confidence += 0.15;
        } else if detection_confidence > 0.6 {
            initial_confidence += 0.1;
        }

        // Bonus for reasonable lane width
        if let Some(width) = lane_width {
            if width > 200.0 && width < 800.0 {
                initial_confidence += 0.15;
            } else if width > 100.0 && width < 1000.0 {
                initial_confidence += 0.05;
            }
        }

        // CRITICAL: Reduce confidence if right after occlusion
        if after_occlusion {
            // Longer occlusion → more confidence reduction
            let occlusion_penalty = if occlusion_duration_frames > 300 {
                0.2 // 80% reduction for very long occlusion (10+ seconds)
            } else if occlusion_duration_frames > 100 {
                0.3 // 70% reduction for long occlusion (3-10 seconds)
            } else if occlusion_duration_frames > 30 {
                0.4 // 60% reduction for medium occlusion (1-3 seconds)
            } else {
                0.5 // 50% reduction for short occlusion (<1 second)
            };

            initial_confidence *= occlusion_penalty;
            self.post_occlusion = true;
            self.frames_since_recovery = 0;

            // Adaptive stabilization period based on occlusion length
            self.post_occlusion_stabilization_frames = if occlusion_duration_frames > 300 {
                90 // 3 seconds for very long occlusion
            } else if occlusion_duration_frames > 100 {
                60 // 2 seconds for medium occlusion
            } else {
                30 // 1 second for short occlusion
            };

            warn!(
                "🔶 Post-occlusion baseline at frame {} after {:.1}s blind",
                frame_id,
                occlusion_duration_frames as f32 / 30.0
            );
            info!(
                "   Initial confidence: {:.2} (needs {} stabilization frames)",
                initial_confidence, self.post_occlusion_stabilization_frames
            );
        } else {
            self.post_occlusion = false;
            info!(
                "✅ Baseline established at frame {} with confidence {:.2}",
                frame_id, initial_confidence
            );
        }

        self.confidence = initial_confidence.min(1.0);
        self.established_at_frame = frame_id;
        self.confirming_frames = 1;
        self.frames_without_lanes = 0;
        self.last_offset = offset;
        self.stabilization_offset_sum = offset;
        self.stabilization_offset_count = 1;
    }

    /// Update confidence based on new lane detection
    ///
    /// # Arguments
    /// * `offset` - Current lateral offset
    /// * `both_lanes_detected` - Were both lanes detected this frame?
    /// * `detection_confidence` - Lane detector confidence
    /// * `baseline_offset` - The current baseline value we're checking against
    pub fn update_with_detection(
        &mut self,
        offset: f32,
        both_lanes_detected: bool,
        detection_confidence: f32,
        baseline_offset: f32,
    ) {
        self.frames_without_lanes = 0;
        self.frames_since_recovery += 1;

        // Check if offset is stable (close to baseline or our running average)
        let tolerance = if self.post_occlusion { 20.0 } else { 15.0 };

        let offset_stable = if self.post_occlusion && self.stabilization_offset_count > 0 {
            // During stabilization, check against running average
            let avg_offset = self.stabilization_offset_sum / self.stabilization_offset_count as f32;
            (offset - avg_offset).abs() < tolerance
        } else {
            // Normal operation, check against baseline
            (offset - baseline_offset).abs() < tolerance
        };

        // Track offset during stabilization
        if self.post_occlusion {
            self.stabilization_offset_sum += offset;
            self.stabilization_offset_count += 1;
        }

        // If offset is stable, this confirms the baseline
        if offset_stable {
            self.confirming_frames += 1;

            // Gradually increase confidence with confirmations
            let confirmation_bonus = 0.02 * (self.confirming_frames as f32 / 10.0).min(1.0);
            self.confidence = (self.confidence + confirmation_bonus).min(1.0);

            // Special handling for post-occlusion recovery
            if self.post_occlusion {
                // Check if we've met stabilization requirements
                if self.frames_since_recovery >= self.post_occlusion_stabilization_frames {
                    // Additional requirement: offset must be consistently stable
                    if self.confirming_frames >= (self.post_occlusion_stabilization_frames / 2) {
                        info!(
                            "✅ Post-occlusion baseline STABILIZED after {} frames",
                            self.frames_since_recovery
                        );
                        info!(
                            "   Final confidence: {:.2} → {:.2}",
                            self.confidence,
                            self.confidence.max(0.8)
                        );
                        self.post_occlusion = false;
                        self.confidence = self.confidence.max(0.8); // Boost to high confidence
                        self.stabilization_offset_sum = 0.0;
                        self.stabilization_offset_count = 0;
                    } else {
                        debug!(
                            "⏳ Stabilization period complete but not enough confirmations ({}/{})",
                            self.confirming_frames,
                            self.post_occlusion_stabilization_frames / 2
                        );
                    }
                } else if self.frames_since_recovery % 30 == 0 {
                    // Progress update every second
                    debug!(
                        "🔄 Stabilizing: {}/{} frames ({:.0}% confidence)",
                        self.frames_since_recovery,
                        self.post_occlusion_stabilization_frames,
                        self.confidence * 100.0
                    );
                }
            }
        } else {
            // Offset changed significantly
            // Could be a real maneuver or just noise
            if (offset - self.last_offset).abs() > 30.0 {
                // Large change - might be real maneuver
                // Don't penalize if we're not in post-occlusion mode
                if !self.post_occlusion {
                    self.confirming_frames = self.confirming_frames.saturating_sub(2);
                } else {
                    // During stabilization, large changes are suspicious
                    self.confirming_frames = self.confirming_frames.saturating_sub(5);
                    self.confidence = (self.confidence - 0.05).max(0.0);
                    debug!(
                        "⚠️  Large offset change during stabilization: {:.1}px → {:.1}px",
                        self.last_offset, offset
                    );
                }
            } else {
                // Small change - probably just noise
                self.confirming_frames = self.confirming_frames.saturating_sub(1);
            }
        }

        // Bonus for quality detections
        if both_lanes_detected && detection_confidence > 0.7 {
            self.confidence = (self.confidence + 0.01).min(1.0);
        }

        self.last_offset = offset;
    }

    /// Update confidence during occlusion (no lane detection)
    pub fn update_without_detection(&mut self) {
        self.frames_without_lanes += 1;

        // Decay confidence during occlusion
        let decay_per_frame = 0.003; // Loses ~0.3% per frame
        self.confidence = (self.confidence - decay_per_frame).max(0.0);

        // Log warnings at key thresholds
        match self.frames_without_lanes {
            30 => {
                debug!(
                    "⚠️  Short occlusion: {} frames without lanes (conf: {:.2})",
                    self.frames_without_lanes, self.confidence
                );
            }
            90 => {
                warn!(
                    "⚠️  Extended occlusion: {} frames without lanes (conf: {:.2})",
                    self.frames_without_lanes, self.confidence
                );
            }
            180 => {
                warn!(
                    "⚠️  Long occlusion: {} frames without lanes (conf: {:.2})",
                    self.frames_without_lanes, self.confidence
                );
            }
            _ => {}
        }

        // After very long occlusion, invalidate baseline completely
        if self.frames_without_lanes == 180 {
            // 6 seconds
            warn!(
                "🗑️  Baseline invalidated after {:.1}s occlusion (was {:.0}% confidence)",
                self.frames_without_lanes as f32 / 30.0,
                self.confidence * 100.0
            );
            self.confidence = 0.0;
            self.confirming_frames = 0;
        }
    }

    /// Can we trust this baseline enough to DETECT NEW maneuvers?
    ///
    /// Returns false during post-occlusion stabilization period
    /// or when confidence is too low.
    pub fn is_reliable_for_detection(&self) -> bool {
        // During post-occlusion stabilization, don't trust for NEW maneuvers
        if self.is_stabilizing() {
            return false;
        }

        // Need minimum confidence
        self.confidence >= 0.6
    }

    /// Can we use this baseline for TRACKING existing maneuvers?
    ///
    /// More lenient than detection - allows continuing to track
    /// even with lower confidence.
    pub fn is_usable_for_tracking(&self) -> bool {
        self.confidence >= 0.25
    }

    /// Get current confidence value
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Is system currently stabilizing after occlusion?
    pub fn is_stabilizing(&self) -> bool {
        self.post_occlusion && self.frames_since_recovery < self.post_occlusion_stabilization_frames
    }

    /// Frames remaining in stabilization period
    pub fn stabilization_frames_remaining(&self) -> u32 {
        if self.is_stabilizing() {
            self.post_occlusion_stabilization_frames
                .saturating_sub(self.frames_since_recovery)
        } else {
            0
        }
    }

    /// Get how many frames we've been without lane detection
    pub fn frames_without_lanes(&self) -> u32 {
        self.frames_without_lanes
    }

    /// Should we re-establish the baseline?
    ///
    /// Returns true if baseline is completely lost and needs fresh data
    pub fn needs_reestablishment(&self) -> bool {
        self.confidence < 0.1
    }

    pub fn reset(&mut self) {
        self.confidence = 0.0;
        self.established_at_frame = 0;
        self.confirming_frames = 0;
        self.frames_without_lanes = 0;
        self.post_occlusion = false;
        self.frames_since_recovery = 0;
        self.last_offset = 0.0;
        self.stabilization_offset_sum = 0.0;
        self.stabilization_offset_count = 0;
    }
}

impl Default for BaselineConfidence {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_post_occlusion_low_confidence() {
        let mut bc = BaselineConfidence::new();

        // Establish baseline after 300 frame occlusion
        bc.establish_baseline(500, 0.0, true, 0.9, Some(450.0), true, 300);

        // Should have low confidence due to occlusion
        assert!(
            bc.confidence() < 0.3,
            "Post-occlusion should have low initial confidence"
        );
        assert!(bc.is_stabilizing(), "Should be in stabilization mode");
        assert!(
            !bc.is_reliable_for_detection(),
            "Should not be reliable for detection"
        );
    }

    #[test]
    fn test_stabilization_period() {
        let mut bc = BaselineConfidence::new();

        bc.establish_baseline(100, 0.0, true, 0.9, Some(450.0), true, 100);

        // Simulate 60 frames of stable detections
        for i in 0..60 {
            bc.update_with_detection(0.0, true, 0.9, 0.0);

            if i < 59 {
                assert!(bc.is_stabilizing(), "Should still be stabilizing");
            }
        }

        // After 60 frames with good confirmations, should be stable
        assert!(!bc.is_stabilizing(), "Should have completed stabilization");
        assert!(bc.is_reliable_for_detection(), "Should now be reliable");
        assert!(
            bc.confidence() >= 0.8,
            "Confidence should be high after stabilization"
        );
    }

    #[test]
    fn test_confidence_decay_during_occlusion() {
        let mut bc = BaselineConfidence::new();

        bc.establish_baseline(100, 0.0, true, 0.9, Some(450.0), false, 0);
        let initial_conf = bc.confidence();

        // Simulate 100 frames without detection
        for _ in 0..100 {
            bc.update_without_detection();
        }

        assert!(
            bc.confidence() < initial_conf,
            "Confidence should decay during occlusion"
        );
    }

    #[test]
    fn test_high_quality_baseline_without_occlusion() {
        let mut bc = BaselineConfidence::new();

        // Establish baseline with perfect conditions, no occlusion
        bc.establish_baseline(100, 0.0, true, 0.95, Some(450.0), false, 0);

        // Should have high initial confidence
        assert!(
            bc.confidence() >= 0.8,
            "High quality detection should have high confidence"
        );
        assert!(
            bc.is_reliable_for_detection(),
            "Should be immediately reliable"
        );
        assert!(!bc.is_stabilizing(), "Should not need stabilization");
    }
}
