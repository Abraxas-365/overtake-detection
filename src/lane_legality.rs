// src/lane_legality.rs — NEW: Fused crossing detection

/// Result from fusing both models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedLegalityResult {
    pub verdict: LineLegality,
    pub crossing_confirmed_by_lane_model: bool,
    pub line_type_from_seg_model: Option<DetectedRoadMarking>,
    pub vehicle_offset_pct: f32,
    pub ego_intersects_marking: bool,
    pub frame_id: u64,
    pub all_markings: Vec<DetectedRoadMarking>,
}

impl LaneLegalityDetector {
    /// Fused analysis: uses lane model position + seg model classification
    pub fn analyze_frame_fused(
        &mut self,
        frame: &[u8],
        width: usize,
        height: usize,
        frame_id: u64,
        confidence_threshold: f32,
        // FROM UFLDv2 lane detection:
        vehicle_lateral_offset: f32, // lateral offset in pixels
        lane_width: Option<f32>,     // detected lane width
        left_lane_x: Option<f32>,    // left boundary X at reference Y
        right_lane_x: Option<f32>,   // right boundary X at reference Y
        crossing_side: CrossingSide, // which boundary is being approached
    ) -> Result<FusedLegalityResult> {
        // 1. Run YOLOv8-seg to get all line classifications
        let (input, scale, pad_x, pad_y) = self.preprocess(frame, width, height)?;
        let (box_output, mask_proto, _) = self.infer(&input)?;
        let markings = self.postprocess(
            &box_output,
            &mask_proto,
            &[],
            scale,
            pad_x,
            pad_y,
            width,
            height,
            confidence_threshold,
        )?;

        // 2. Calculate normalized offset from UFLDv2
        let normalized_offset = lane_width
            .map(|w| (vehicle_lateral_offset / w).abs())
            .unwrap_or(0.0);

        // 3. Determine if UFLDv2 says we're actually crossing
        let is_crossing_per_lane_model = normalized_offset > 0.40; // 40%+ = near boundary

        // 4. Find the NEAREST line to the crossing side
        let vehicle_x = width as f32 / 2.0;
        let nearest_line = self.find_nearest_line_to_crossing(
            &markings,
            vehicle_x,
            left_lane_x,
            right_lane_x,
            crossing_side,
            width,
            height,
        );

        // 5. FUSED VERDICT: Only flag illegal if BOTH models agree
        let verdict = if is_crossing_per_lane_model {
            if let Some(ref line) = nearest_line {
                // Lane model says crossing + seg model identified the line type
                line.legality
            } else {
                LineLegality::Unknown // Crossing but can't classify line
            }
        } else {
            // Lane model says NOT crossing → don't flag regardless of seg model
            LineLegality::Legal
        };

        Ok(FusedLegalityResult {
            verdict,
            crossing_confirmed_by_lane_model: is_crossing_per_lane_model,
            line_type_from_seg_model: nearest_line,
            vehicle_offset_pct: normalized_offset,
            ego_intersects_marking: is_crossing_per_lane_model,
            frame_id,
            all_markings: markings,
        })
    }

    /// Find the line closest to the side the vehicle is crossing toward
    fn find_nearest_line_to_crossing(
        &self,
        markings: &[DetectedRoadMarking],
        vehicle_x: f32,
        left_lane_x: Option<f32>,
        right_lane_x: Option<f32>,
        crossing_side: CrossingSide,
        frame_w: usize,
        _frame_h: usize,
    ) -> Option<DetectedRoadMarking> {
        // Determine the X region where we expect the crossed line to be
        let (target_x_min, target_x_max) = match crossing_side {
            CrossingSide::Left => {
                let lx = left_lane_x.unwrap_or(vehicle_x * 0.4);
                (lx - 80.0, lx + 80.0) // ±80px around left lane boundary
            }
            CrossingSide::Right => {
                let rx = right_lane_x.unwrap_or(vehicle_x * 1.6);
                (rx - 80.0, rx + 80.0) // ±80px around right lane boundary
            }
            CrossingSide::None => return None,
        };

        // Find the best matching detection near that boundary
        markings
            .iter()
            .filter(|m| {
                let bbox_cx = (m.bbox[0] + m.bbox[2]) / 2.0;
                bbox_cx >= target_x_min && bbox_cx <= target_x_max
            })
            .max_by(|a, b| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossingSide {
    Left,
    Right,
    None,
}
