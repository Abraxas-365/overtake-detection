// src/vehicle_detection.rs

use anyhow::Result;
use ort::{
    execution_providers::CUDAExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
};
use tracing::{debug, info};

const YOLO_INPUT_SIZE: usize = 640;
const YOLO_CLASSES: usize = 80;

// COCO class IDs for vehicles
const VEHICLE_CLASSES: [usize; 4] = [2, 3, 5, 7]; // car, motorcycle, bus, truck

#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: [f32; 4], // [x1, y1, x2, y2] in original image coordinates
    pub confidence: f32,
    pub class_id: usize,
    pub class_name: String,
}

pub struct YoloDetector {
    session: Session,
}

impl YoloDetector {
    pub fn new(model_path: &str) -> Result<Self> {
        info!("Loading YOLO model: {}", model_path);

        let session = Session::builder()?
            .with_execution_providers([CUDAExecutionProvider::default().with_device_id(0).build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        info!("✓ YOLO detector initialized");
        Ok(Self { session })
    }

    pub fn detect(
        &mut self,
        frame: &[u8],
        width: usize,
        height: usize,
        confidence_threshold: f32,
    ) -> Result<Vec<Detection>> {
        // 1. Preprocess (letterbox + normalize)
        let (input, scale, pad_x, pad_y) = self.preprocess(frame, width, height)?;

        // 2. Run inference
        let output = self.infer(&input)?;

        // 3. Postprocess (parse detections + NMS)
        let mut detections =
            self.postprocess(&output, scale, pad_x, pad_y, confidence_threshold)?;

        // 4. 🔧 NEW: Merge adjacent vehicles (cab + trailer, convoy fragments)
        detections = merge_adjacent_vehicles(detections, width as f32);

        debug!("Detected {} vehicles (after merging)", detections.len());
        Ok(detections)
    }

    fn preprocess(
        &self,
        src: &[u8],
        src_w: usize,
        src_h: usize,
    ) -> Result<(Vec<f32>, f32, f32, f32)> {
        let target_size = YOLO_INPUT_SIZE;

        // Calculate scale to fit image inside 640x640 while maintaining aspect ratio
        let scale = (target_size as f32 / src_w as f32).min(target_size as f32 / src_h as f32);
        let scaled_w = (src_w as f32 * scale) as usize;
        let scaled_h = (src_h as f32 * scale) as usize;

        // Padding to center the image
        let pad_x = (target_size - scaled_w) as f32 / 2.0;
        let pad_y = (target_size - scaled_h) as f32 / 2.0;

        // Resize
        let resized = resize_bilinear(src, src_w, src_h, scaled_w, scaled_h);

        // Create padded 640x640 canvas (gray background)
        let mut canvas = vec![114u8; target_size * target_size * 3];

        // Copy resized image to center
        for y in 0..scaled_h {
            for x in 0..scaled_w {
                let src_idx = (y * scaled_w + x) * 3;
                let dst_x = x + pad_x as usize;
                let dst_y = y + pad_y as usize;
                let dst_idx = (dst_y * target_size + dst_x) * 3;
                canvas[dst_idx..dst_idx + 3].copy_from_slice(&resized[src_idx..src_idx + 3]);
            }
        }

        // Normalize [0, 255] -> [0, 1] and convert HWC -> CHW
        let mut input = vec![0.0f32; 3 * target_size * target_size];
        for c in 0..3 {
            for h in 0..target_size {
                for w in 0..target_size {
                    let hwc_idx = (h * target_size + w) * 3 + c;
                    let chw_idx = c * target_size * target_size + h * target_size + w;
                    input[chw_idx] = canvas[hwc_idx] as f32 / 255.0;
                }
            }
        }

        Ok((input, scale, pad_x, pad_y))
    }

    fn infer(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let shape = [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE];
        let input_value =
            ort::value::Value::from_array((shape.as_slice(), input.to_vec().into_boxed_slice()))?;

        let outputs = self.session.run(ort::inputs!["images" => input_value])?;
        let output = &outputs[0];
        let (_, data) = output.try_extract_tensor::<f32>()?;

        Ok(data.to_vec())
    }

    fn postprocess(
        &self,
        output: &[f32],
        scale: f32,
        pad_x: f32,
        pad_y: f32,
        conf_thresh: f32,
    ) -> Result<Vec<Detection>> {
        let mut detections = Vec::new();

        // YOLO output: [1, 84, 8400] -> we work with 8400 predictions
        // Each prediction: [x, y, w, h, class0_conf, class1_conf, ..., class79_conf]

        for i in 0..8400 {
            // Extract bbox (center format)
            let cx = output[i];
            let cy = output[8400 + i];
            let w = output[8400 * 2 + i];
            let h = output[8400 * 3 + i];

            // Find best class
            let mut max_conf = 0.0f32;
            let mut best_class = 0;

            for c in 0..YOLO_CLASSES {
                let conf = output[8400 * (4 + c) + i];
                if conf > max_conf {
                    max_conf = conf;
                    best_class = c;
                }
            }

            // Filter by confidence and vehicle classes
            if max_conf < conf_thresh || !VEHICLE_CLASSES.contains(&best_class) {
                continue;
            }

            // Convert center format to corner format
            let x1 = cx - w / 2.0;
            let y1 = cy - h / 2.0;
            let x2 = cx + w / 2.0;
            let y2 = cy + h / 2.0;

            // Reverse letterbox transformation to get original image coordinates
            let x1 = (x1 - pad_x) / scale;
            let y1 = (y1 - pad_y) / scale;
            let x2 = (x2 - pad_x) / scale;
            let y2 = (y2 - pad_y) / scale;

            detections.push(Detection {
                bbox: [x1, y1, x2, y2],
                confidence: max_conf,
                class_id: best_class,
                class_name: class_id_to_name(best_class),
            });
        }

        // Apply Non-Maximum Suppression
        let detections = nms(detections, 0.45);

        Ok(detections)
    }
}

fn resize_bilinear(src: &[u8], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<u8> {
    let mut dst = vec![0u8; dst_h * dst_w * 3];
    let x_ratio = src_w as f32 / dst_w as f32;
    let y_ratio = src_h as f32 / dst_h as f32;

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = dx as f32 * x_ratio;
            let sy = dy as f32 * y_ratio;
            let sx0 = sx.floor() as usize;
            let sy0 = sy.floor() as usize;
            let sx1 = (sx0 + 1).min(src_w - 1);
            let sy1 = (sy0 + 1).min(src_h - 1);
            let fx = sx - sx0 as f32;
            let fy = sy - sy0 as f32;

            for c in 0..3 {
                let p00 = src[(sy0 * src_w + sx0) * 3 + c] as f32;
                let p10 = src[(sy0 * src_w + sx1) * 3 + c] as f32;
                let p01 = src[(sy1 * src_w + sx0) * 3 + c] as f32;
                let p11 = src[(sy1 * src_w + sx1) * 3 + c] as f32;

                let val = p00 * (1.0 - fx) * (1.0 - fy)
                    + p10 * fx * (1.0 - fy)
                    + p01 * (1.0 - fx) * fy
                    + p11 * fx * fy;

                dst[(dy * dst_w + dx) * 3 + c] = val.round() as u8;
            }
        }
    }
    dst
}

fn class_id_to_name(class_id: usize) -> String {
    match class_id {
        2 => "car",
        3 => "motorcycle",
        5 => "bus",
        7 => "truck",
        _ => "unknown",
    }
    .to_string()
}

fn nms(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    if detections.is_empty() {
        return detections;
    }

    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut keep = Vec::new();

    while !detections.is_empty() {
        let current = detections.remove(0);
        keep.push(current.clone());

        detections.retain(|det| {
            let iou = calculate_iou(&current.bbox, &det.bbox);
            iou < iou_threshold
        });
    }

    keep
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

// ============================================================================
// ADJACENT VEHICLE MERGING (v1.0 — Truck Cab + Trailer Fix)
// ============================================================================

/// Merge adjacent vehicle detections that are likely part of the same physical vehicle
/// (e.g., truck cab + trailer detected as separate objects)
fn merge_adjacent_vehicles(detections: Vec<Detection>, frame_width: f32) -> Vec<Detection> {
    if detections.is_empty() {
        return detections;
    }

    let mut merged = Vec::new();
    let mut used = vec![false; detections.len()];

    for i in 0..detections.len() {
        if used[i] {
            continue;
        }

        let mut current = detections[i].clone();
        used[i] = true;

        // Find all detections that should merge with this one
        for j in (i + 1)..detections.len() {
            if used[j] {
                continue;
            }

            if should_merge(&current, &detections[j], frame_width) {
                // Merge j into current
                current = merge_detections(&current, &detections[j]);
                used[j] = true;
            }
        }

        merged.push(current);
    }

    if merged.len() < detections.len() {
        debug!(
            "Merged {} detections into {} vehicles",
            detections.len(),
            merged.len()
        );
    }

    merged
}

/// Determine if two detections should be merged (same vehicle split into parts)
fn should_merge(det1: &Detection, det2: &Detection, frame_width: f32) -> bool {
    // Only merge vehicles (not mixing car + truck with other classes)
    if !VEHICLE_CLASSES.contains(&det1.class_id) || !VEHICLE_CLASSES.contains(&det2.class_id) {
        return false;
    }

    let [x1_1, y1_1, x2_1, y2_1] = det1.bbox;
    let [x1_2, y1_2, x2_2, y2_2] = det2.bbox;

    // Calculate centers and dimensions
    let cy1 = (y1_1 + y2_1) / 2.0;
    let cy2 = (y1_2 + y2_2) / 2.0;
    let h1 = y2_1 - y1_1;
    let h2 = y2_2 - y1_2;

    // 1. Vertical alignment check — must be roughly at same height
    let vertical_overlap = (cy1 - cy2).abs() < (h1.max(h2) * 0.6);
    if !vertical_overlap {
        return false;
    }

    // 2. Horizontal proximity — gap between boxes must be small
    let gap = if x1_2 > x2_1 {
        x1_2 - x2_1 // det2 is to the right
    } else if x1_1 > x2_2 {
        x1_1 - x2_2 // det1 is to the right
    } else {
        0.0 // They overlap horizontally
    };

    // Allow gap up to 15% of frame width (handles cab-trailer separation)
    let max_gap = frame_width * 0.15;
    if gap > max_gap {
        return false;
    }

    // 3. Size similarity — prevent merging distant small car with huge truck
    let area1 = (x2_1 - x1_1) * h1;
    let area2 = (x2_2 - x1_2) * h2;
    let area_ratio = area1.min(area2) / area1.max(area2);

    // If one is >4x bigger, don't merge (unless gap is tiny)
    if area_ratio < 0.25 && gap > frame_width * 0.05 {
        return false;
    }

    true
}

/// Merge two detections into one (create bounding box that encompasses both)
fn merge_detections(det1: &Detection, det2: &Detection) -> Detection {
    let [x1_1, y1_1, x2_1, y2_1] = det1.bbox;
    let [x1_2, y1_2, x2_2, y2_2] = det2.bbox;

    // Create union bounding box
    let merged_bbox = [
        x1_1.min(x1_2),
        y1_1.min(y1_2),
        x2_1.max(x2_2),
        y2_1.max(y2_2),
    ];

    // Use higher confidence and prioritize truck > bus > car
    let class_priority = |id: usize| match id {
        7 => 3, // truck
        5 => 2, // bus
        3 => 1, // motorcycle
        2 => 0, // car
        _ => -1,
    };

    let (class_id, class_name, confidence) =
        if class_priority(det1.class_id) >= class_priority(det2.class_id) {
            (
                det1.class_id,
                det1.class_name.clone(),
                det1.confidence.max(det2.confidence),
            )
        } else {
            (
                det2.class_id,
                det2.class_name.clone(),
                det1.confidence.max(det2.confidence),
            )
        };

    Detection {
        bbox: merged_bbox,
        confidence,
        class_id,
        class_name,
    }
}
