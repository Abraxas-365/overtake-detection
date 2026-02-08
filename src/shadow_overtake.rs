// src/shadow_overtake.rs
//
// Shadow Overtake Detection
//
// Detects when another vehicle ahead is simultaneously occupying the overtaking
// lane during your overtake maneuver, blocking forward visibility into oncoming
// traffic. This is extremely dangerous and illegal in Peru (DS 016-2009-MTC).
//

use crate::overtake_analyzer::TrackedVehicle;
use crate::types::Direction;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

// ============================================================================
// TYPES
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShadowSeverity {
    Warning,
    Dangerous,
    Critical,
}

impl ShadowSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            ShadowSeverity::Warning => "WARNING",
            ShadowSeverity::Dangerous => "DANGEROUS",
            ShadowSeverity::Critical => "CRITICAL",
        }
    }

    pub fn rank(&self) -> u8 {
        match self {
            ShadowSeverity::Warning => 1,
            ShadowSeverity::Dangerous => 2,
            ShadowSeverity::Critical => 3,
        }
    }
}

impl std::fmt::Display for ShadowSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A confirmed shadow overtake event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowOvertakeEvent {
    /// ID of the vehicle blocking visibility
    pub blocking_vehicle_id: u32,
    /// Class of the blocking vehicle (car, truck, bus, motorcycle)
    pub blocking_vehicle_class: String,
    /// Frame when shadow was first confirmed
    pub detected_at_frame: u64,
    /// Timestamp when shadow was first confirmed
    pub detected_at_timestamp_ms: f64,
    /// Number of frames the vehicle was blocking
    pub frames_blocked: u32,
    /// Severity of the shadow overtake
    pub severity: ShadowSeverity,
    /// Closest distance ratio (0.0 = directly ahead, 1.0 = far ahead)
    pub closest_distance_ratio: f32,
    /// Last frame the shadow was active
    pub last_active_frame: u64,
}

// ============================================================================
// CONFIGURATION
// ============================================================================

#[derive(Debug, Clone)]
pub struct ShadowOvertakeConfig {
    /// Minimum consecutive frames in overtaking lane to confirm a shadow
    pub min_confirm_frames: u32,
    /// Vehicle with center_y above this ratio × frame_height is "ahead"
    /// (lower Y = further ahead in perspective)
    pub max_ahead_y_ratio: f32,
    /// Vehicles above this are too far to be relevant
    pub min_ahead_y_ratio: f32,
    /// Pixel margin for lane boundary check
    pub lane_margin_px: f32,
    /// Distance ratio threshold for DANGEROUS
    pub dangerous_distance_ratio: f32,
    /// Distance ratio threshold for CRITICAL
    pub critical_distance_ratio: f32,
    /// Frames without seeing the vehicle before dropping the candidate
    pub stale_candidate_frames: u64,
    /// Check frequency (every N frames)
    pub check_every_n_frames: u64,
}

impl Default for ShadowOvertakeConfig {
    fn default() -> Self {
        Self {
            min_confirm_frames: 4,
            max_ahead_y_ratio: 0.65,
            min_ahead_y_ratio: 0.03,
            lane_margin_px: 80.0,
            dangerous_distance_ratio: 0.45,
            critical_distance_ratio: 0.22,
            stale_candidate_frames: 12,
            check_every_n_frames: 3,
        }
    }
}

// ============================================================================
// INTERNAL CANDIDATE TRACKING
// ============================================================================

#[derive(Debug, Clone)]
struct ShadowCandidate {
    vehicle_id: u32,
    class_name: String,
    first_detected_frame: u64,
    last_detected_frame: u64,
    frames_in_overtaking_lane: u32,
    min_distance_ratio: f32,
    avg_center_x: f32,
    samples: u32,
    confirmed: bool,
}

// ============================================================================
// SHADOW OVERTAKE DETECTOR
// ============================================================================

pub struct ShadowOvertakeDetector {
    config: ShadowOvertakeConfig,
    frame_width: f32,
    frame_height: f32,

    /// Active candidates being evaluated
    candidates: HashMap<u32, ShadowCandidate>,

    /// Confirmed shadow events for the current overtake
    current_overtake_shadows: Vec<ShadowOvertakeEvent>,

    /// All shadow events across the entire video
    all_events: Vec<ShadowOvertakeEvent>,

    /// Whether we're actively monitoring
    is_monitoring: bool,

    /// Current overtake direction
    overtake_direction: Direction,

    /// Frame when monitoring started
    monitoring_start_frame: u64,

    /// Total shadow overtakes detected
    total_detected: usize,
}

impl ShadowOvertakeDetector {
    pub fn new(frame_width: f32, frame_height: f32) -> Self {
        Self::with_config(frame_width, frame_height, ShadowOvertakeConfig::default())
    }

    pub fn with_config(frame_width: f32, frame_height: f32, config: ShadowOvertakeConfig) -> Self {
        Self {
            config,
            frame_width,
            frame_height,
            candidates: HashMap::new(),
            current_overtake_shadows: Vec::new(),
            all_events: Vec::new(),
            is_monitoring: false,
            overtake_direction: Direction::Unknown,
            monitoring_start_frame: 0,
            total_detected: 0,
        }
    }

    // ========================================================================
    // LIFECYCLE
    // ========================================================================

    /// Begin monitoring for shadow overtakes. Called when an overtake starts
    /// (first lane change detected by OvertakeTracker).
    pub fn start_monitoring(&mut self, direction: Direction, frame_id: u64) {
        self.is_monitoring = true;
        self.overtake_direction = direction;
        self.monitoring_start_frame = frame_id;
        self.candidates.clear();
        self.current_overtake_shadows.clear();

        info!(
            "👁️  Shadow monitoring STARTED: direction={}, frame={}",
            direction.as_str(),
            frame_id
        );
    }

    /// Stop monitoring and return all shadow events detected during this overtake.
    /// Called when overtake completes, times out, or is cancelled.
    pub fn stop_monitoring(&mut self) -> Vec<ShadowOvertakeEvent> {
        if !self.is_monitoring {
            return Vec::new();
        }

        self.is_monitoring = false;
        self.candidates.clear();

        let events = std::mem::take(&mut self.current_overtake_shadows);

        if !events.is_empty() {
            info!(
                "👁️  Shadow monitoring STOPPED: {} shadow event(s) detected",
                events.len()
            );
            for ev in &events {
                info!(
                    "  ⚫ {} (ID #{}) — severity={}, dist={:.0}%, blocked {} frames",
                    ev.blocking_vehicle_class,
                    ev.blocking_vehicle_id,
                    ev.severity.as_str(),
                    ev.closest_distance_ratio * 100.0,
                    ev.frames_blocked
                );
            }
        } else {
            debug!("👁️  Shadow monitoring STOPPED: no shadows detected");
        }

        self.all_events.extend(events.clone());
        events
    }

    // ========================================================================
    // MAIN UPDATE
    // ========================================================================

    /// Evaluate all tracked vehicles against shadow overtake criteria.
    ///
    /// Returns a new `ShadowOvertakeEvent` the first time a shadow is confirmed.
    /// Subsequent frames update the existing event internally.
    ///
    /// # Arguments
    /// * `tracked_vehicles` - Current tracked vehicles from `OvertakeAnalyzer`
    /// * `left_lane_x` - Left lane boundary X at reference Y (if detected)
    /// * `right_lane_x` - Right lane boundary X at reference Y (if detected)
    /// * `frame_id` - Current frame number
    /// * `timestamp_ms` - Current timestamp in milliseconds
    pub fn update(
        &mut self,
        tracked_vehicles: &HashMap<u32, TrackedVehicle>,
        left_lane_x: Option<f32>,
        right_lane_x: Option<f32>,
        frame_id: u64,
        timestamp_ms: f64,
    ) -> Option<ShadowOvertakeEvent> {
        if !self.is_monitoring {
            return None;
        }

        // Only check at configured frequency
        if frame_id % self.config.check_every_n_frames != 0 {
            return None;
        }

        let ego_y = self.frame_height * 0.75;
        let frame_center_x = self.frame_width / 2.0;
        let ahead_max_y = self.frame_height * self.config.max_ahead_y_ratio;
        let ahead_min_y = self.frame_height * self.config.min_ahead_y_ratio;

        let mut newly_confirmed: Option<ShadowOvertakeEvent> = None;

        for (&vehicle_id, vehicle) in tracked_vehicles {
            // Get the latest position that's recent enough to be relevant
            let latest_pos = match vehicle
                .position_history
                .iter()
                .rev()
                .find(|p| frame_id.saturating_sub(p.frame_id) < 15)
            {
                Some(pos) => pos,
                None => {
                    self.candidates.remove(&vehicle_id);
                    continue;
                }
            };

            // ── Check 1: Is the vehicle AHEAD of us? ──────────────────────
            if latest_pos.center_y > ahead_max_y || latest_pos.center_y < ahead_min_y {
                self.candidates.remove(&vehicle_id);
                continue;
            }

            // ── Check 2: Is the vehicle in the OVERTAKING LANE? ───────────
            let in_overtaking_lane = self.check_in_overtaking_lane(
                latest_pos.center_x,
                left_lane_x,
                right_lane_x,
                frame_center_x,
            );

            if !in_overtaking_lane {
                self.candidates.remove(&vehicle_id);
                continue;
            }

            // ── Check 3: Compute distance ratio ──────────────────────────
            let distance_ratio = if ego_y > ahead_min_y {
                ((ego_y - latest_pos.center_y) / (ego_y - ahead_min_y)).clamp(0.0, 1.0)
            } else {
                1.0
            };

            // ── Update or create candidate ───────────────────────────────
            let candidate = self
                .candidates
                .entry(vehicle_id)
                .or_insert_with(|| ShadowCandidate {
                    vehicle_id,
                    class_name: vehicle.class_name.clone(),
                    first_detected_frame: frame_id,
                    last_detected_frame: frame_id,
                    frames_in_overtaking_lane: 0,
                    min_distance_ratio: distance_ratio,
                    avg_center_x: 0.0,
                    samples: 0,
                    confirmed: false,
                });

            candidate.last_detected_frame = frame_id;
            candidate.frames_in_overtaking_lane += 1;
            candidate.samples += 1;
            candidate.avg_center_x = candidate.avg_center_x
                + (latest_pos.center_x - candidate.avg_center_x) / candidate.samples as f32;

            if distance_ratio < candidate.min_distance_ratio {
                candidate.min_distance_ratio = distance_ratio;
            }

            // ── Confirm if enough evidence ───────────────────────────────
            if !candidate.confirmed
                && candidate.frames_in_overtaking_lane >= self.config.min_confirm_frames
            {
                candidate.confirmed = true;
                self.total_detected += 1;

                let severity = self.compute_severity(
                    candidate.min_distance_ratio,
                    candidate.frames_in_overtaking_lane,
                );

                let event = ShadowOvertakeEvent {
                    blocking_vehicle_id: vehicle_id,
                    blocking_vehicle_class: candidate.class_name.clone(),
                    detected_at_frame: frame_id,
                    detected_at_timestamp_ms: timestamp_ms,
                    frames_blocked: candidate.frames_in_overtaking_lane,
                    severity,
                    closest_distance_ratio: candidate.min_distance_ratio,
                    last_active_frame: frame_id,
                };

                warn!(
                    "⚫ SHADOW OVERTAKE: {} (ID #{}) blocking in overtaking lane | \
                     severity={} | distance={:.0}%",
                    event.blocking_vehicle_class,
                    event.blocking_vehicle_id,
                    severity.as_str(),
                    distance_ratio * 100.0,
                );

                self.current_overtake_shadows.push(event.clone());
                newly_confirmed = Some(event);
            }

            // ── Upgrade severity of existing confirmed shadow ────────────
            if candidate.confirmed {
                let new_severity = self.compute_severity(
                    candidate.min_distance_ratio,
                    candidate.frames_in_overtaking_lane,
                );

                if let Some(existing) = self
                    .current_overtake_shadows
                    .iter_mut()
                    .find(|e| e.blocking_vehicle_id == vehicle_id)
                {
                    existing.last_active_frame = frame_id;
                    existing.frames_blocked = candidate.frames_in_overtaking_lane;

                    if new_severity.rank() > existing.severity.rank() {
                        warn!(
                            "⚫ SHADOW SEVERITY UPGRADE: {} → {} for {} (ID #{})",
                            existing.severity.as_str(),
                            new_severity.as_str(),
                            existing.blocking_vehicle_class,
                            vehicle_id,
                        );
                        existing.severity = new_severity;
                        existing.closest_distance_ratio = candidate.min_distance_ratio;
                    }
                }
            }
        }

        // Purge stale candidates
        let stale_threshold = self.config.stale_candidate_frames;
        self.candidates
            .retain(|_, c| frame_id.saturating_sub(c.last_detected_frame) < stale_threshold);

        newly_confirmed
    }

    // ========================================================================
    // HELPERS
    // ========================================================================

    fn check_in_overtaking_lane(
        &self,
        vehicle_x: f32,
        left_lane_x: Option<f32>,
        right_lane_x: Option<f32>,
        frame_center_x: f32,
    ) -> bool {
        let margin = self.config.lane_margin_px;

        match self.overtake_direction {
            Direction::Left => {
                // Overtaking left → oncoming lane is to the LEFT
                // Vehicle is in overtaking lane if its X < left lane boundary
                match left_lane_x {
                    Some(lx) => vehicle_x < lx + margin,
                    None => vehicle_x < frame_center_x - margin,
                }
            }
            Direction::Right => {
                // Overtaking right → overtaking lane is to the RIGHT
                match right_lane_x {
                    Some(rx) => vehicle_x > rx - margin,
                    None => vehicle_x > frame_center_x + margin,
                }
            }
            Direction::Unknown => false,
        }
    }

    fn compute_severity(&mut self, distance_ratio: f32, frames_blocked: u32) -> ShadowSeverity {
        // Multiple blockers → always critical
        let confirmed_count = self
            .current_overtake_shadows
            .iter()
            .filter(|e| e.severity != ShadowSeverity::Warning)
            .count();

        if confirmed_count >= 2 {
            return ShadowSeverity::Critical;
        }

        if distance_ratio <= self.config.critical_distance_ratio || frames_blocked > 30 {
            ShadowSeverity::Critical
        } else if distance_ratio <= self.config.dangerous_distance_ratio || frames_blocked > 15 {
            ShadowSeverity::Dangerous
        } else {
            ShadowSeverity::Warning
        }
    }

    // ========================================================================
    // ACCESSORS
    // ========================================================================

    pub fn is_monitoring(&self) -> bool {
        self.is_monitoring
    }

    pub fn active_shadow_count(&self) -> usize {
        self.current_overtake_shadows.len()
    }

    pub fn total_shadow_events(&self) -> usize {
        self.total_detected
    }

    pub fn worst_active_severity(&self) -> Option<ShadowSeverity> {
        self.current_overtake_shadows
            .iter()
            .map(|e| e.severity)
            .max_by_key(|s| s.rank())
    }

    pub fn get_all_events(&self) -> &[ShadowOvertakeEvent] {
        &self.all_events
    }
}
