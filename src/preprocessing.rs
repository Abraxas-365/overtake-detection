// src/preprocessing.rs

use anyhow::Result;

// ============================================================================
// CONTRAST ENHANCEMENT FOR LANE DETECTION
// ============================================================================

/// Enhanced preprocessing with contrast improvement for faded lane markings
pub fn preprocess_with_enhancement(
    src: &[u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Result<Vec<f32>> {
    // Step 1: Apply CLAHE-like contrast enhancement
    let enhanced = apply_clahe(src, src_width, src_height, 2.0);

    // Step 2: Enhance white lane markings specifically
    let lane_enhanced = enhance_lane_markings(&enhanced, src_width, src_height);

    // Step 3: Resize
    let resized = resize_bilinear(&lane_enhanced, src_width, src_height, dst_width, dst_height);

    // Step 4: Normalize and convert HWC -> CHW
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];

    let mut output = vec![0.0f32; 3 * dst_height * dst_width];

    for c in 0..3 {
        for h in 0..dst_height {
            for w in 0..dst_width {
                let hwc_idx = (h * dst_width + w) * 3 + c;
                let chw_idx = c * dst_height * dst_width + h * dst_width + w;

                let pixel = resized[hwc_idx] as f32 / 255.0;
                output[chw_idx] = (pixel - MEAN[c]) / STD[c];
            }
        }
    }

    Ok(output)
}

/// Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
/// This helps with faded lane markings by enhancing local contrast
fn apply_clahe(src: &[u8], width: usize, height: usize, clip_limit: f32) -> Vec<u8> {
    let mut enhanced = src.to_vec();
    let tile_size = 64; // Size of each tile for local histogram equalization

    // Process in tiles
    let tiles_x = (width + tile_size - 1) / tile_size;
    let tiles_y = (height + tile_size - 1) / tile_size;

    // Calculate histogram for each tile
    let mut tile_cdfs: Vec<Vec<[f32; 256]>> = vec![vec![[0.0; 256]; tiles_x]; tiles_y];

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let start_y = ty * tile_size;
            let start_x = tx * tile_size;
            let end_y = (start_y + tile_size).min(height);
            let end_x = (start_x + tile_size).min(width);

            // Build histogram for this tile (using luminance)
            let mut hist = [0u32; 256];
            let mut pixel_count = 0u32;

            for y in start_y..end_y {
                for x in start_x..end_x {
                    let idx = (y * width + x) * 3;
                    // Convert to grayscale for histogram
                    let gray = (0.299 * src[idx] as f32
                        + 0.587 * src[idx + 1] as f32
                        + 0.114 * src[idx + 2] as f32) as u8;
                    hist[gray as usize] += 1;
                    pixel_count += 1;
                }
            }

            // Clip histogram
            let clip_threshold = (clip_limit * pixel_count as f32 / 256.0) as u32;
            let mut excess = 0u32;

            for h in hist.iter_mut() {
                if *h > clip_threshold {
                    excess += *h - clip_threshold;
                    *h = clip_threshold;
                }
            }

            // Redistribute excess uniformly
            let increment = excess / 256;
            let remainder = (excess % 256) as usize;

            for (i, h) in hist.iter_mut().enumerate() {
                *h += increment;
                if i < remainder {
                    *h += 1;
                }
            }

            // Build CDF
            let mut cdf = [0.0f32; 256];
            cdf[0] = hist[0] as f32;
            for i in 1..256 {
                cdf[i] = cdf[i - 1] + hist[i] as f32;
            }

            // Normalize CDF
            let cdf_max = cdf[255];
            if cdf_max > 0.0 {
                for c in cdf.iter_mut() {
                    *c = (*c / cdf_max) * 255.0;
                }
            }

            tile_cdfs[ty][tx] = cdf;
        }
    }

    // Apply equalization with bilinear interpolation between tiles
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;

            // Find surrounding tiles
            let tile_y = (y as f32 / tile_size as f32).min((tiles_y - 1) as f32);
            let tile_x = (x as f32 / tile_size as f32).min((tiles_x - 1) as f32);

            let ty0 = tile_y.floor() as usize;
            let tx0 = tile_x.floor() as usize;
            let ty1 = (ty0 + 1).min(tiles_y - 1);
            let tx1 = (tx0 + 1).min(tiles_x - 1);

            let fy = tile_y - ty0 as f32;
            let fx = tile_x - tx0 as f32;

            // Process each color channel
            for c in 0..3 {
                let pixel = src[idx + c];

                // Get equalized values from surrounding tiles
                let v00 = tile_cdfs[ty0][tx0][pixel as usize];
                let v01 = tile_cdfs[ty0][tx1][pixel as usize];
                let v10 = tile_cdfs[ty1][tx0][pixel as usize];
                let v11 = tile_cdfs[ty1][tx1][pixel as usize];

                // Bilinear interpolation
                let v0 = v00 * (1.0 - fx) + v01 * fx;
                let v1 = v10 * (1.0 - fx) + v11 * fx;
                let value = v0 * (1.0 - fy) + v1 * fy;

                enhanced[idx + c] = value.clamp(0.0, 255.0) as u8;
            }
        }
    }

    enhanced
}

/// Enhance white lane markings specifically
/// This boosts pixels that are likely to be lane markings (bright, grayish-white)
fn enhance_lane_markings(src: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut enhanced = src.to_vec();

    // Only process the bottom 60% of the image (where lanes typically are)
    let start_row = (height as f32 * 0.4) as usize;

    for y in start_row..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;

            let r = src[idx] as f32;
            let g = src[idx + 1] as f32;
            let b = src[idx + 2] as f32;

            // Calculate brightness and saturation
            let brightness = (r + g + b) / 3.0;
            let max_rgb = r.max(g).max(b);
            let min_rgb = r.min(g).min(b);
            let saturation = if max_rgb > 0.0 {
                (max_rgb - min_rgb) / max_rgb
            } else {
                0.0
            };

            // Detect potential lane marking pixels:
            // - Relatively bright (brightness > 100)
            // - Low saturation (grayish-white, not colored)
            // - Similar R, G, B values
            let is_potential_lane = brightness > 100.0
                && saturation < 0.3
                && (r - g).abs() < 30.0
                && (g - b).abs() < 30.0;

            if is_potential_lane {
                // Boost brightness of potential lane pixels
                let boost_factor = 1.3 + (brightness - 100.0) / 255.0 * 0.5;
                enhanced[idx] = (r * boost_factor).min(255.0) as u8;
                enhanced[idx + 1] = (g * boost_factor).min(255.0) as u8;
                enhanced[idx + 2] = (b * boost_factor).min(255.0) as u8;
            }

            // Also darken very dark pixels (road surface) to increase contrast
            if brightness < 80.0 && saturation < 0.2 {
                let darken_factor = 0.85;
                enhanced[idx] = (r * darken_factor) as u8;
                enhanced[idx + 1] = (g * darken_factor) as u8;
                enhanced[idx + 2] = (b * darken_factor) as u8;
            }
        }
    }

    enhanced
}

/// Simple global contrast stretch
pub fn enhance_contrast_simple(src: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut enhanced = src.to_vec();

    // Find min/max in bottom half of image (where lanes are)
    let start_row = height / 2;
    let mut min_vals = [255u8; 3];
    let mut max_vals = [0u8; 3];

    for y in start_row..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            for c in 0..3 {
                min_vals[c] = min_vals[c].min(src[idx + c]);
                max_vals[c] = max_vals[c].max(src[idx + c]);
            }
        }
    }

    // Apply contrast stretch to entire image
    for i in 0..(width * height) {
        for c in 0..3 {
            let idx = i * 3 + c;
            let range = (max_vals[c] - min_vals[c]).max(1) as f32;
            let val = src[idx] as f32;
            let stretched = ((val - min_vals[c] as f32) / range * 255.0).clamp(0.0, 255.0);
            enhanced[idx] = stretched as u8;
        }
    }

    enhanced
}

// ============================================================================
// ORIGINAL PREPROCESSING FUNCTIONS
// ============================================================================

/// Preprocess raw RGB image for model input (original without enhancement)
pub fn preprocess(
    src: &[u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Result<Vec<f32>> {
    // Resize
    let resized = resize_bilinear(src, src_width, src_height, dst_width, dst_height);

    // Normalize and convert HWC -> CHW
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];

    let mut output = vec![0.0f32; 3 * dst_height * dst_width];

    for c in 0..3 {
        for h in 0..dst_height {
            for w in 0..dst_width {
                let hwc_idx = (h * dst_width + w) * 3 + c;
                let chw_idx = c * dst_height * dst_width + h * dst_width + w;

                let pixel = resized[hwc_idx] as f32 / 255.0;
                output[chw_idx] = (pixel - MEAN[c]) / STD[c];
            }
        }
    }

    Ok(output)
}

/// Bilinear image resize
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess() {
        let src = vec![128u8; 640 * 480 * 3];
        let result = preprocess(&src, 640, 480, 1600, 320);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3 * 320 * 1600);
    }

    #[test]
    fn test_preprocess_with_enhancement() {
        let src = vec![128u8; 640 * 480 * 3];
        let result = preprocess_with_enhancement(&src, 640, 480, 1600, 320);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3 * 320 * 1600);
    }

    #[test]
    fn test_resize() {
        let src = vec![255u8; 100 * 100 * 3];
        let dst = resize_bilinear(&src, 100, 100, 50, 50);
        assert_eq!(dst.len(), 50 * 50 * 3);
    }

    #[test]
    fn test_clahe() {
        let src = vec![128u8; 100 * 100 * 3];
        let result = apply_clahe(&src, 100, 100, 2.0);
        assert_eq!(result.len(), src.len());
    }
}
