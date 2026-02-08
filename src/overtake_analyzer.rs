// src/overtake_analyzer.rs

#[derive(Debug, Clone)]
pub enum LaneChangeClassification {
    Overtaking {
        vehicles_passed: usize,
        vehicle_types: Vec<String>,
        is_legal: bool,
        reason: String,
    },
    NormalLaneChange {
        is_legal: bool,
        reason: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LaneLineType {
    DashedWhite,       // Discontinua blanca
    SolidWhite,        // Continua blanca
    DashedYellow,      // Discontinua amarilla (center)
    SolidYellow,       // Continua amarilla (no passing)
    DoubleSolidYellow, // Doble continua (never cross)
    Unknown,
}

impl OvertakeAnalyzer {
    /// Classify and analyze legality of the maneuver
    pub fn classify_maneuver(
        &self,
        start_frame: u64,
        end_frame: u64,
        direction: &str,
        lane_line_type: LaneLineType,
        is_in_curve: bool,
        has_oncoming_traffic: bool,
    ) -> LaneChangeClassification {
        // 1. Check if we overtook any vehicles
        let overtakes = self.analyze_overtake(start_frame, end_frame, direction);

        if !overtakes.is_empty() {
            // This is an OVERTAKING maneuver
            let (is_legal, reason) = self.evaluate_overtaking_legality(
                lane_line_type,
                is_in_curve,
                has_oncoming_traffic,
                overtakes.len(),
            );

            LaneChangeClassification::Overtaking {
                vehicles_passed: overtakes.len(),
                vehicle_types: overtakes.iter().map(|o| o.class_name.clone()).collect(),
                is_legal,
                reason,
            }
        } else {
            // This is a NORMAL LANE CHANGE (no vehicle ahead)
            let (is_legal, reason) =
                self.evaluate_lane_change_legality(lane_line_type, is_in_curve);

            LaneChangeClassification::NormalLaneChange { is_legal, reason }
        }
    }

    /// Evaluate legality of OVERTAKING (stricter rules)
    fn evaluate_overtaking_legality(
        &self,
        lane_line: LaneLineType,
        is_in_curve: bool,
        has_oncoming: bool,
        vehicles_count: usize,
    ) -> (bool, String) {
        use LaneLineType::*;

        // Rule 1: Overtaking in curves is ILLEGAL (DS 016-2009-MTC Art. 215)
        if is_in_curve {
            return (
                false,
                "Overtaking in curve - ILLEGAL per Art. 215".to_string(),
            );
        }

        // Rule 2: Solid yellow or double yellow = ILLEGAL overtaking
        match lane_line {
            SolidYellow => {
                return (
                    false,
                    "Overtaking on solid yellow line - ILLEGAL".to_string(),
                );
            }
            DoubleSolidYellow => {
                return (
                    false,
                    "Overtaking on double solid yellow - ILLEGAL".to_string(),
                );
            }
            _ => {}
        }

        // Rule 3: Oncoming traffic = ILLEGAL
        if has_oncoming {
            return (
                false,
                "Overtaking with oncoming traffic - ILLEGAL".to_string(),
            );
        }

        // Rule 4: Multiple vehicles at once = ILLEGAL (Peru)
        if vehicles_count > 1 {
            return (
                false,
                format!("Overtaking {} vehicles at once - ILLEGAL", vehicles_count),
            );
        }

        // Rule 5: Check line type
        match lane_line {
            DashedWhite | DashedYellow => (true, "Overtaking on dashed line - LEGAL".to_string()),
            SolidWhite => {
                // Technically allowed but discouraged
                (
                    true,
                    "Overtaking on solid white - LEGAL but discouraged".to_string(),
                )
            }
            Unknown => (
                true,
                "Overtaking - line type unknown, assuming legal".to_string(),
            ),
            _ => (false, "Overtaking conditions not met".to_string()),
        }
    }

    /// Evaluate legality of NORMAL LANE CHANGE (less strict)
    fn evaluate_lane_change_legality(
        &self,
        lane_line: LaneLineType,
        is_in_curve: bool,
    ) -> (bool, String) {
        use LaneLineType::*;

        // Normal lane changes are generally allowed, even in curves
        // Exception: Double solid yellow
        match lane_line {
            DoubleSolidYellow => (false, "Crossing double solid yellow - ILLEGAL".to_string()),
            SolidYellow => {
                // Allowed if repositioning to turn/exit, but flag it
                (
                    true,
                    "Lane change on solid yellow - LEGAL if turning, otherwise risky".to_string(),
                )
            }
            _ => {
                if is_in_curve {
                    (
                        true,
                        "Lane change in curve - LEGAL (not overtaking)".to_string(),
                    )
                } else {
                    (true, "Normal lane change - LEGAL".to_string())
                }
            }
        }
    }

    /// Detect if there's oncoming traffic
    pub fn detect_oncoming_traffic(&self, start_frame: u64, end_frame: u64) -> bool {
        // Check for vehicles in opposite direction
        // Vehicles in top-left quadrant moving right-to-left
        for (_, track) in &self.tracked_vehicles {
            if track.last_seen_frame < start_frame || track.first_seen_frame > end_frame {
                continue;
            }

            // Check if vehicle is in opposite lane (left side, top of frame)
            if let Some(start_pos) = track.position_history.first() {
                let is_opposite_lane = start_pos.center_x < self.frame_width / 3.0;
                let is_distant = start_pos.center_y < self.frame_height / 3.0;

                if is_opposite_lane && is_distant {
                    return true;
                }
            }
        }
        false
    }
}
