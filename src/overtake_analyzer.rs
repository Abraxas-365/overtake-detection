// src/overtake_analyzer.rs

use crate::vehicle_detection::Detection;
use std::collections::HashMap;
use tracing::info;

#[derive(Debug, Clone)]
pub struct TrackedVehicle {
    pub id: u32,
    pub bbox: [f32; 4],
    pub class_name: String,
    pub first_seen_frame: u64,
    pub last_seen_frame: u64,
    pub position_history: Vec<VehiclePosition>,
}

#[derive(Debug, Clone)]
pub struct VehiclePosition {
    pub frame_id: u64,
    pub center_y: f32,
    pub center_x: f32,
}

pub struct OvertakeAnalyzer {
    next_id: u32,
    tracked_vehicles: HashMap<u32, TrackedVehicle>,
    iou_threshold: f32,
    frame_width: f32,
    frame_height: f32,
}

#[derive(Debug, Clone)]
pub struct OvertakeEvent {
    pub vehicle_id: u32,
    pub class_name: String,
    pub overtaken_at_frame: u64,
}

impl OvertakeAnalyzer {
    pub fn new(frame_width: f32, frame_height: f32) -> Self {
        Self {
            next_id: 0,
            tracked_vehicles: HashMap::new(),
            iou_threshold: 0.3,
            frame_width,
            frame_height,
        }
    }

    pub fn get_tracked_vehicles(&self) -> &HashMap<u32, TrackedVehicle> {
        &self.tracked_vehicles
    }

    pub fn update(&mut self, detections: Vec<Detection>, frame_id: u64) {
        let mut matched_track_ids = Vec::new();
        let mut unmatched_detections = Vec::new();

        for det in detections {
            let mut best_match: Option<(u32, f32)> = None;

            for (track_id, track) in &self.tracked_vehicles {
                if track.class_name != det.class_name {
                    continue;
                }

                let iou = calculate_iou(&track.bbox, &det.bbox);
                if iou > self.iou_threshold {
                    if best_match.is_none() || iou > best_match.unwrap().1 {
                        best_match = Some((*track_id, iou));
                    }
                }
            }

            if let Some((track_id, _)) = best_match {
                if let Some(track) = self.tracked_vehicles.get_mut(&track_id) {
                    track.bbox = det.bbox;
                    track.last_seen_frame = frame_id;
                    track.position_history.push(VehiclePosition {
                        frame_id,
                        center_x: (det.bbox[0] + det.bbox[2]) / 2.0,
                        center_y: (det.bbox[1] + det.bbox[3]) / 2.0,
                    });
                    matched_track_ids.push(track_id);
                }
            } else {
                unmatched_detections.push(det);
            }
        }

        for det in unmatched_detections {
            let center_x = (det.bbox[0] + det.bbox[2]) / 2.0;
            let center_y = (det.bbox[1] + det.bbox[3]) / 2.0;

            self.tracked_vehicles.insert(
                self.next_id,
                TrackedVehicle {
                    id: self.next_id,
                    bbox: det.bbox,
                    class_name: det.class_name,
                    first_seen_frame: frame_id,
                    last_seen_frame: frame_id,
                    position_history: vec![VehiclePosition {
                        frame_id,
                        center_x,
                        center_y,
                    }],
                },
            );

            info!(
                "🆕 New vehicle tracked: ID #{}, type: {}",
                self.next_id, self.tracked_vehicles[&self.next_id].class_name
            );

            self.next_id += 1;
        }

        // 🆕 MUCH longer retention: 300 frames = ~10 seconds at 30fps
        // This allows vehicles to be tracked even after they disappear from view during overtake
        let removed = self.tracked_vehicles.len();
        self.tracked_vehicles.retain(|id, track| {
            let should_keep = frame_id - track.last_seen_frame < 300;
            if !should_keep {
                info!(
                    "🗑️  Removing vehicle ID #{} ({}) - not seen for {} frames",
                    id,
                    track.class_name,
                    frame_id - track.last_seen_frame
                );
            }
            should_keep
        });

        let removed = removed - self.tracked_vehicles.len();
        if removed > 0 {
            info!("🗑️  Removed {} stale vehicle(s)", removed);
        }
    }

    pub fn analyze_overtake(
        &self,
        start_frame: u64,
        end_frame: u64,
        direction: &str,
    ) -> Vec<OvertakeEvent> {
        let mut overtaken = Vec::new();

        // Ego vehicle is at bottom of frame
        let ego_y = self.frame_height * 0.70; // 🆕 Changed from 0.75 to 0.70

        info!(
            "🔍 Analyzing overtake: frames {}-{}, direction: {}, active vehicles: {}",
            start_frame,
            end_frame,
            direction,
            self.tracked_vehicles.len()
        );

        for (vehicle_id, track) in &self.tracked_vehicles {
            info!(
                "  📋 Vehicle ID #{} ({}): first_seen={}, last_seen={}, positions={}",
                vehicle_id,
                track.class_name,
                track.first_seen_frame,
                track.last_seen_frame,
                track.position_history.len()
            );
            // Only consider vehicles visible during the maneuver
            if track.last_seen_frame < start_frame || track.first_seen_frame > end_frame {
                continue;
            }

            // Get position at start and end
            let start_pos = track
                .position_history
                .iter()
                .find(|p| p.frame_id >= start_frame);

            let end_pos = track
                .position_history
                .iter()
                .rev()
                .find(|p| p.frame_id <= end_frame);

            if let (Some(start), Some(end)) = (start_pos, end_pos) {
                // 🆕 More lenient detection
                // Vehicle was ahead (higher in frame = smaller Y)
                let was_in_front = start.center_y < ego_y - 30.0; // 🆕 Changed from 50 to 30

                // Vehicle is now behind/alongside (lower in frame = larger Y)
                let is_behind = end.center_y > ego_y - 50.0; // 🆕 Changed from -20 to -50

                // 🆕 Also check if vehicle moved DOWN in frame (relative motion)
                let moved_down = end.center_y > start.center_y + 20.0;

                info!(
                    "  Vehicle ID #{} ({}): start_y={:.1}, end_y={:.1}, ego_y={:.1}",
                    vehicle_id, track.class_name, start.center_y, end.center_y, ego_y
                );
                info!(
                    "    was_in_front={}, is_behind={}, moved_down={}",
                    was_in_front, is_behind, moved_down
                );

                // 🆕 More flexible detection: either classic logic OR relative motion
                if (was_in_front && is_behind) || (moved_down && start.center_y < ego_y) {
                    // Check if vehicle is in the target lane
                    // 🆕 CORRECTED: When overtaking LEFT, vehicle is on the RIGHT (your original lane)
                    // When overtaking RIGHT, vehicle is on the LEFT (your original lane)
                    let is_in_target_lane = if direction == "LEFT" {
                        // Overtaking by going left → vehicle is on the right
                        start.center_x > self.frame_width / 2.0 - 100.0 // ✅ RIGHT side
                    } else {
                        // Overtaking by going right → vehicle is on the left
                        start.center_x < self.frame_width / 2.0 + 100.0 // ✅ LEFT side
                    };

                    info!(
                        "    is_in_target_lane={}, center_x={:.1}, frame_width={:.1}",
                        is_in_target_lane, start.center_x, self.frame_width
                    );

                    if is_in_target_lane {
                        overtaken.push(OvertakeEvent {
                            vehicle_id: *vehicle_id,
                            class_name: track.class_name.clone(),
                            overtaken_at_frame: end.frame_id,
                        });

                        info!(
                            "✅ Overtook {} (ID #{}) during frames {}-{}",
                            track.class_name, vehicle_id, start_frame, end_frame
                        );
                    } else {
                        info!("    ❌ Not in target lane");
                    }
                } else {
                    info!("    ❌ Didn't meet overtake criteria");
                }
            }
        }

        if overtaken.is_empty() {
            info!("⚠️  No vehicles detected as overtaken");
        } else {
            info!("🎯 Total vehicles overtaken: {}", overtaken.len());
        }

        overtaken
    }

    pub fn get_active_vehicle_count(&self) -> usize {
        self.tracked_vehicles.len()
    }

    pub fn get_total_unique_vehicles(&self) -> u32 {
        self.next_id
    }
}

fn calculate_iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
    let x1 = box1[0].max(box2[0]);
    let y1 = box1[1].max(box2[1]);
    let x2 = box1[2].min(box2[2]);
    let y2 = box1[3].min(box2[3]);

    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    let area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    let union = area1 + area2 - intersection;

    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}
