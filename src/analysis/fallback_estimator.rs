// src/analysis/fallback_estimator.rs
//
// Provides lateral position estimation when the primary lane detector
// (UFLDv2) is unable to detect lanes. Uses three fallback strategies
// in priority order.

use crate::lane_legality::DetectedRoadMarking;
use crate::vehicle_detection::Detection;
use std::collections::VecDeque;
use tracing::{debug, info, warn};

// ============================================================================
// TYPES
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EstimationSource {
    /// UFLDv2 lane detection (primary, highest confidence)
    LaneModel,
    /// Road markings from YOLOv8-seg used as lane proxies
    RoadMarkingProxy,
    /// Collective lateral drift of tracked vehicles (inverse = ego motion)
    VehicleFleetFlow,
    /// Holding last known position with decaying confidence
    DeadReckoning,
    /// No estimation possible
    None,
}

impl EstimationSource {
    pub fn base_confidence(&self) -> f32 {
        match self {
            EstimationSource::LaneModel => 0.90,
            EstimationSource::RoadMarkingProxy => 0.60,
            EstimationSource::VehicleFleetFlow => 0.40,
            EstimationSource::DeadReckoning => 0.25,
            EstimationSource::None => 0.0,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            EstimationSource::LaneModel => "LANE_MODEL",
            EstimationSource::RoadMarkingProxy => "ROAD_MARKING_PROXY",
            EstimationSource::VehicleFleetFlow => "VEHICLE_FLEET_FLOW",
            EstimationSource::DeadReckoning => "DEAD_RECKONING",
            EstimationSource::None => "NONE",
        }
    }
}

#[derive(Debug, Clone)]
pub struct FallbackEstimate {
    /// Lateral offset in pixels (positive = right of center)
    pub lateral_offset: f32,
    /// Estimated lane width (may be from history)
    pub lane_width: f32,
    /// Confidence in this estimate [0, 1]
    pub confidence: f32,
    /// Which estimation tier produced this
    pub source: EstimationSource,
    /// Frame ID
    pub frame_id: u64,
}

// ============================================================================
// TIER 2: Road Marking Proxy
// ============================================================================
// Uses bounding boxes from YOLOv8-seg detections as pseudo-lane boundaries.
// If we see a solid/dashed line on the left and right, we can triangulate
// the vehicle's position relative to them.

struct RoadMarkingProxy {
    /// Recent left-side marking X positions
    left_marking_history: VecDeque<f32>,
    /// Recent right-side marking X positions
    right_marking_history: VecDeque<f32>,
    history_size: usize,
}

impl RoadMarkingProxy {
    fn new() -> Self {
        Self {
            left_marking_history: VecDeque::with_capacity(15),
            right_marking_history: VecDeque::with_capacity(15),
            history_size: 15,
        }
    }

    /// Attempt to estimate position from road marking detections
    fn estimate(
        &mut self,
        markings: &[DetectedRoadMarking],
        frame_width: f32,
        frame_height: f32,
    ) -> Option<(f32, f32)> {
        // Only use relevant classes (solid/dashed lines, not other markings)
        let lane_markings: Vec<&DetectedRoadMarking> = markings
            .iter()
            .filter(|m| matches!(m.class_id, 4 | 5 | 6 | 7 | 8 | 9 | 10 | 99) && m.confidence > 0.3)
            .collect();

        if lane_markings.is_empty() {
            return None;
        }

        let vehicle_x = frame_width / 2.0;
        // Use markings in the lower portion of the frame (closer to vehicle)
        let reference_y = frame_height * 0.75;

        let mut left_x: Option<f32> = None;
        let mut right_x: Option<f32> = None;

        for marking in &lane_markings {
            let cx = (marking.bbox[0] + marking.bbox[2]) / 2.0;
            let cy = (marking.bbox[1] + marking.bbox[3]) / 2.0;

            // Only consider markings in the lower half (near ego)
            if cy < frame_height * 0.4 {
                continue;
            }

            if cx < vehicle_x {
                // Left-side marking
                if left_x.is_none() || cx > left_x.unwrap() {
                    left_x = Some(cx);
                }
            } else {
                // Right-side marking
                if right_x.is_none() || cx < right_x.unwrap() {
                    right_x = Some(cx);
                }
            }
        }

        // Update histories
        if let Some(lx) = left_x {
            self.left_marking_history.push_back(lx);
            if self.left_marking_history.len() > self.history_size {
                self.left_marking_history.pop_front();
            }
        }
        if let Some(rx) = right_x {
            self.right_marking_history.push_back(rx);
            if self.right_marking_history.len() > self.history_size {
                self.right_marking_history.pop_front();
            }
        }

        // Use median of recent values for stability
        let effective_left = self.median_or_value(&self.left_marking_history, left_x);
        let effective_right = self.median_or_value(&self.right_marking_history, right_x);

        match (effective_left, effective_right) {
            (Some(lx), Some(rx)) => {
                let width = rx - lx;
                if width > 80.0 && width < 900.0 {
                    let center = (lx + rx) / 2.0;
                    let offset = vehicle_x - center;
                    Some((offset, width))
                } else {
                    None
                }
            }
            (Some(lx), None) => {
                // Estimate with default width
                let width = 450.0;
                let center = lx + width / 2.0;
                Some((vehicle_x - center, width))
            }
            (None, Some(rx)) => {
                let width = 450.0;
                let center = rx - width / 2.0;
                Some((vehicle_x - center, width))
            }
            _ => None,
        }
    }

    fn median_or_value(&self, history: &VecDeque<f32>, current: Option<f32>) -> Option<f32> {
        if history.len() >= 3 {
            let mut sorted: Vec<f32> = history.iter().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            Some(sorted[sorted.len() / 2])
        } else {
            current
        }
    }

    fn reset(&mut self) {
        self.left_marking_history.clear();
        self.right_marking_history.clear();
    }
}

// ============================================================================
// TIER 3: Vehicle Fleet Flow
// ============================================================================
// When the ego vehicle changes lanes, ALL tracked vehicles appear to
// shift laterally in the OPPOSITE direction. By measuring the collective
// lateral drift of tracked vehicle centroids, we can infer ego motion.
//
// Key insight: If 3+ vehicles all shift LEFT by ~50px over 10 frames,
// the ego vehicle moved RIGHT by ~50px.

#[derive(Debug, Clone)]
struct VehicleSnapshot {
    frame_id: u64,
    centroids: Vec<(u32, f32, f32)>, // (vehicle_id, center_x, center_y)
}

struct VehicleFleetFlow {
    snapshots: VecDeque<VehicleSnapshot>,
    max_snapshots: usize,
    /// Accumulated lateral ego offset (pixels)
    accumulated_ego_offset: f32,
    /// Last known lane width from primary detector
    last_known_lane_width: f32,
    /// Minimum vehicles required for reliable estimate
    min_vehicles: usize,
    /// Frame window for computing flow
    flow_window_frames: usize,
}

impl VehicleFleetFlow {
    fn new() -> Self {
        Self {
            snapshots: VecDeque::with_capacity(30),
            max_snapshots: 30,
            accumulated_ego_offset: 0.0,
            last_known_lane_width: 450.0,
            min_vehicles: 2,
            flow_window_frames: 10,
        }
    }

    /// Record current vehicle positions
    fn record_snapshot(&mut self, detections: &[Detection], frame_id: u64) {
        let centroids: Vec<(u32, f32, f32)> = detections
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let cx = (d.bbox[0] + d.bbox[2]) / 2.0;
                let cy = (d.bbox[1] + d.bbox[3]) / 2.0;
                (i as u32, cx, cy)
            })
            .collect();

        self.snapshots.push_back(VehicleSnapshot {
            frame_id,
            centroids,
        });

        if self.snapshots.len() > self.max_snapshots {
            self.snapshots.pop_front();
        }
    }

    /// Estimate ego lateral velocity from vehicle fleet motion
    fn estimate_ego_lateral_shift(&self) -> Option<f32> {
        if self.snapshots.len() < 2 {
            return None;
        }

        let recent = self.snapshots.back()?;
        let window_start_idx = self.snapshots.len().saturating_sub(self.flow_window_frames);
        let older = self.snapshots.get(window_start_idx)?;

        if recent.centroids.len() < self.min_vehicles || older.centroids.len() < self.min_vehicles {
            return None;
        }

        // Match vehicles between frames by proximity (simple nearest-neighbor)
        let mut lateral_shifts: Vec<f32> = Vec::new();

        for &(_, new_cx, new_cy) in &recent.centroids {
            let mut best_match: Option<(f32, f32)> = None; // (distance, dx)

            for &(_, old_cx, old_cy) in &older.centroids {
                let dist = ((new_cx - old_cx).powi(2) + (new_cy - old_cy).powi(2)).sqrt();
                // Only match if reasonably close (same vehicle)
                if dist < 200.0 {
                    if best_match.is_none() || dist < best_match.unwrap().0 {
                        best_match = Some((dist, new_cx - old_cx));
                    }
                }
            }

            if let Some((_, dx)) = best_match {
                lateral_shifts.push(dx);
            }
        }

        if lateral_shifts.len() < self.min_vehicles {
            return None;
        }

        // Check consensus: most vehicles should agree on direction
        let positive = lateral_shifts.iter().filter(|&&s| s > 5.0).count();
        let negative = lateral_shifts.iter().filter(|&&s| s < -5.0).count();
        let total = lateral_shifts.len();

        let consensus_ratio = positive.max(negative) as f32 / total as f32;
        if consensus_ratio < 0.6 {
            return None; // No clear consensus → vehicles moving independently
        }

        // Median of shifts (robust to outliers)
        lateral_shifts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_shift = lateral_shifts[lateral_shifts.len() / 2];

        // Ego moved in OPPOSITE direction of vehicle fleet shift
        Some(-median_shift)
    }

    /// Update the accumulated ego offset and return current estimate
    fn update_and_estimate(
        &mut self,
        detections: &[Detection],
        frame_id: u64,
    ) -> Option<(f32, f32)> {
        self.record_snapshot(detections, frame_id);

        let ego_shift = self.estimate_ego_lateral_shift()?;

        // Only accumulate if shift is meaningful
        if ego_shift.abs() > 2.0 {
            self.accumulated_ego_offset += ego_shift;
        }

        Some((self.accumulated_ego_offset, self.last_known_lane_width))
    }

    fn set_reference_position(&mut self, offset: f32, lane_width: f32) {
        self.accumulated_ego_offset = offset;
        self.last_known_lane_width = lane_width;
    }

    fn reset(&mut self) {
        self.snapshots.clear();
        self.accumulated_ego_offset = 0.0;
    }
}

// ============================================================================
// TIER 4: Dead Reckoning
// ============================================================================

struct DeadReckoning {
    last_valid_offset: f32,
    last_valid_width: f32,
    last_valid_frame: u64,
    last_valid_source: EstimationSource,
    decay_rate_per_frame: f32, // Confidence decays this much per frame
}

impl DeadReckoning {
    fn new() -> Self {
        Self {
            last_valid_offset: 0.0,
            last_valid_width: 450.0,
            last_valid_frame: 0,
            last_valid_source: EstimationSource::None,
            decay_rate_per_frame: 0.005, // ~0 confidence after 50 frames (1.7s)
        }
    }

    fn save(&mut self, offset: f32, width: f32, frame_id: u64, source: EstimationSource) {
        self.last_valid_offset = offset;
        self.last_valid_width = width;
        self.last_valid_frame = frame_id;
        self.last_valid_source = source;
    }

    fn estimate(&self, current_frame: u64) -> Option<(f32, f32, f32)> {
        if self.last_valid_frame == 0 {
            return None;
        }

        let frames_elapsed = current_frame.saturating_sub(self.last_valid_frame) as f32;
        let base_conf = self.last_valid_source.base_confidence();
        let confidence = (base_conf - frames_elapsed * self.decay_rate_per_frame).max(0.05);

        // Don't use dead reckoning for more than 3 seconds
        if frames_elapsed > 90.0 {
            return None;
        }

        Some((self.last_valid_offset, self.last_valid_width, confidence))
    }

    fn reset(&mut self) {
        self.last_valid_frame = 0;
    }
}

// ============================================================================
// MAIN FALLBACK ESTIMATOR
// ============================================================================

pub struct FallbackPositionEstimator {
    road_marking_proxy: RoadMarkingProxy,
    vehicle_fleet_flow: VehicleFleetFlow,
    dead_reckoning: DeadReckoning,

    frame_width: f32,
    frame_height: f32,

    /// Current active source
    active_source: EstimationSource,
    /// Frames since last primary (UFLDv2) detection
    frames_without_primary: u32,
    /// Whether a lane change is suspected from fallback signals
    pub fallback_lane_change_suspected: bool,
    /// Lateral velocity from fallback (px/s)
    pub fallback_lateral_velocity: f32,
}

impl FallbackPositionEstimator {
    pub fn new(frame_width: f32, frame_height: f32) -> Self {
        Self {
            road_marking_proxy: RoadMarkingProxy::new(),
            vehicle_fleet_flow: VehicleFleetFlow::new(),
            dead_reckoning: DeadReckoning::new(),
            frame_width,
            frame_height,
            active_source: EstimationSource::None,
            frames_without_primary: 0,
            fallback_lane_change_suspected: false,
            fallback_lateral_velocity: 0.0,
        }
    }

    /// Called when the primary lane detector succeeds.
    /// Syncs all fallback systems to the known-good position.
    pub fn sync_from_primary(&mut self, lateral_offset: f32, lane_width: f32, frame_id: u64) {
        self.frames_without_primary = 0;
        self.active_source = EstimationSource::LaneModel;

        // Sync fleet flow reference
        self.vehicle_fleet_flow
            .set_reference_position(lateral_offset, lane_width);

        // Save for dead reckoning
        self.dead_reckoning.save(
            lateral_offset,
            lane_width,
            frame_id,
            EstimationSource::LaneModel,
        );
    }

    /// Called when the primary lane detector fails.
    /// Tries fallback tiers in order and returns the best estimate.
    pub fn estimate_fallback(
        &mut self,
        road_markings: &[DetectedRoadMarking],
        vehicle_detections: &[Detection],
        frame_id: u64,
    ) -> Option<FallbackEstimate> {
        self.frames_without_primary += 1;

        // ── Tier 2: Road Marking Proxy ─────────────────────────
        if let Some((offset, width)) =
            self.road_marking_proxy
                .estimate(road_markings, self.frame_width, self.frame_height)
        {
            self.active_source = EstimationSource::RoadMarkingProxy;
            self.dead_reckoning
                .save(offset, width, frame_id, EstimationSource::RoadMarkingProxy);

            // Also sync fleet flow so it stays calibrated
            self.vehicle_fleet_flow
                .set_reference_position(offset, width);

            debug!(
                "📍 Fallback Tier 2 (Road Markings): offset={:.1}px, width={:.0}px",
                offset, width
            );

            return Some(FallbackEstimate {
                lateral_offset: offset,
                lane_width: width,
                confidence: EstimationSource::RoadMarkingProxy.base_confidence(),
                source: EstimationSource::RoadMarkingProxy,
                frame_id,
            });
        }

        // ── Tier 3: Vehicle Fleet Flow ─────────────────────────
        if let Some((offset, width)) = self
            .vehicle_fleet_flow
            .update_and_estimate(vehicle_detections, frame_id)
        {
            self.active_source = EstimationSource::VehicleFleetFlow;
            self.dead_reckoning
                .save(offset, width, frame_id, EstimationSource::VehicleFleetFlow);

            // Detect rapid lateral change from fleet flow
            let ego_shift = self
                .vehicle_fleet_flow
                .estimate_ego_lateral_shift()
                .unwrap_or(0.0);
            self.fallback_lateral_velocity = ego_shift * 30.0; // Approximate px/s

            if ego_shift.abs() > 15.0 {
                self.fallback_lane_change_suspected = true;
                info!(
                    "🔶 Fleet flow detects lateral ego motion: {:.1}px/frame ({:.0}px/s)",
                    ego_shift, self.fallback_lateral_velocity
                );
            }

            debug!(
                "📍 Fallback Tier 3 (Fleet Flow): offset={:.1}px, width={:.0}px",
                offset, width
            );

            return Some(FallbackEstimate {
                lateral_offset: offset,
                lane_width: width,
                confidence: EstimationSource::VehicleFleetFlow.base_confidence(),
                source: EstimationSource::VehicleFleetFlow,
                frame_id,
            });
        }

        // ── Tier 4: Dead Reckoning ─────────────────────────────
        if let Some((offset, width, confidence)) = self.dead_reckoning.estimate(frame_id) {
            self.active_source = EstimationSource::DeadReckoning;

            return Some(FallbackEstimate {
                lateral_offset: offset,
                lane_width: width,
                confidence,
                source: EstimationSource::DeadReckoning,
                frame_id,
            });
        }

        self.active_source = EstimationSource::None;
        None
    }

    pub fn active_source(&self) -> EstimationSource {
        self.active_source
    }

    pub fn frames_without_primary(&self) -> u32 {
        self.frames_without_primary
    }

    pub fn reset(&mut self) {
        self.road_marking_proxy.reset();
        self.vehicle_fleet_flow.reset();
        self.dead_reckoning.reset();
        self.active_source = EstimationSource::None;
        self.frames_without_primary = 0;
        self.fallback_lane_change_suspected = false;
        self.fallback_lateral_velocity = 0.0;
    }
}
