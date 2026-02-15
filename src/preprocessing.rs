// src/preprocessing.rs
//
// v5.0: Enhanced preprocessing pipeline with adaptive contrast enhancement (CLAHE)
//       for improved lane detection on degraded Peruvian road markings.
//
// Key improvements:
//   - CLAHE on luminance channel before resize (handles fog, glare, faded paint)
//   - Configurable enhancement strength
//   - Pure-Rust implementation (no OpenCV dependency for preprocessing)

use anyhow::Result;
use tracing::debug;

// ============================================================================
// PUBLIC API
// ============================================================================

/// Preprocess raw RGB image for UFLDv2 model input.
///
/// Pipeline: RGB → CLAHE on luminance → Resize → Normalize (ImageNet) → CHW
pub fn preprocess(
    src: &[u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Result<Vec<f32>> {
    // 1. Apply CLAHE on the luminance channel for contrast enhancement.
    //    This dramatically improves detection of faded markings common on
    //    Peruvian roads, and handles garúa (coastal fog) and highland glare.
    let enhanced = apply_clahe_on_luminance(src, src_width, src_height, 8, 2.0);

    // 2. Bilinear resize to model input dimensions
    let resized = resize_bilinear(&enhanced, src_width, src_height, dst_width, dst_height);

    // 3. Normalize (ImageNet stats) and convert HWC → CHW
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

/// Preprocess WITHOUT CLAHE — for cases where you want the original behavior
/// (e.g., YOLO vehicle detection which was trained on unenhanced images).
pub fn preprocess_raw(
    src: &[u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Result<Vec<f32>> {
    let resized = resize_bilinear(src, src_width, src_height, dst_width, dst_height);

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

// ============================================================================
// CLAHE (Contrast Limited Adaptive Histogram Equalization)
// ============================================================================

/// Apply CLAHE on the Y (luminance) channel of a YUV conversion.
///
/// This preserves color information while boosting local contrast,
/// making faded lane markings much more visible to the model.
///
/// * `tile_size` — Number of tiles per axis (8 = 8x8 grid). Higher = more local.
/// * `clip_limit` — Contrast amplification limit (2.0 is conservative, 4.0 aggressive).
fn apply_clahe_on_luminance(
    rgb: &[u8],
    width: usize,
    height: usize,
    tile_size: usize,
    clip_limit: f32,
) -> Vec<u8> {
    let pixel_count = width * height;

    // --- Step 1: Extract Y channel (BT.601 luma) ---
    let mut y_channel = vec![0u8; pixel_count];
    let mut cb_channel = vec![0i16; pixel_count]; // stored as offset from 128
    let mut cr_channel = vec![0i16; pixel_count];

    for i in 0..pixel_count {
        let r = rgb[i * 3] as f32;
        let g = rgb[i * 3 + 1] as f32;
        let b = rgb[i * 3 + 2] as f32;

        // BT.601 conversion
        let y = (0.299 * r + 0.587 * g + 0.114 * b).round();
        let cb = (-0.169 * r - 0.331 * g + 0.500 * b).round();
        let cr = (0.500 * r - 0.419 * g - 0.081 * b).round();

        y_channel[i] = y.clamp(0.0, 255.0) as u8;
        cb_channel[i] = cb as i16;
        cr_channel[i] = cr as i16;
    }

    // --- Step 2: Apply CLAHE to Y channel ---
    let enhanced_y = clahe_equalize(&y_channel, width, height, tile_size, clip_limit);

    // --- Step 3: Reconstruct RGB ---
    let mut output = vec![0u8; pixel_count * 3];

    for i in 0..pixel_count {
        let y = enhanced_y[i] as f32;
        let cb = cb_channel[i] as f32;
        let cr = cr_channel[i] as f32;

        let r = (y + 1.402 * cr).round().clamp(0.0, 255.0) as u8;
        let g = (y - 0.344 * cb - 0.714 * cr).round().clamp(0.0, 255.0) as u8;
        let b = (y + 1.772 * cb).round().clamp(0.0, 255.0) as u8;

        output[i * 3] = r;
        output[i * 3 + 1] = g;
        output[i * 3 + 2] = b;
    }

    output
}

/// Core CLAHE algorithm on a single-channel (grayscale) image.
fn clahe_equalize(
    src: &[u8],
    width: usize,
    height: usize,
    tile_grid: usize,
    clip_limit: f32,
) -> Vec<u8> {
    let tile_grid = tile_grid.max(2); // minimum 2x2 grid
    let tile_w = width / tile_grid;
    let tile_h = height / tile_grid;

    if tile_w == 0 || tile_h == 0 {
        return src.to_vec(); // image too small for tiling
    }

    let num_bins = 256usize;
    let pixels_per_tile = tile_w * tile_h;

    // Actual clip limit in histogram counts
    let actual_clip = ((clip_limit * pixels_per_tile as f32) / num_bins as f32).max(1.0) as u32;

    // --- Build per-tile CDFs ---
    // cdfs[ty][tx] = [256 values], each in [0, 255]
    let mut cdfs: Vec<Vec<[u8; 256]>> = Vec::with_capacity(tile_grid);

    for ty in 0..tile_grid {
        let mut row = Vec::with_capacity(tile_grid);
        for tx in 0..tile_grid {
            let x0 = tx * tile_w;
            let y0 = ty * tile_h;
            let x1 = if tx == tile_grid - 1 {
                width
            } else {
                x0 + tile_w
            };
            let y1 = if ty == tile_grid - 1 {
                height
            } else {
                y0 + tile_h
            };

            let actual_pixels = (x1 - x0) * (y1 - y0);

            // Histogram
            let mut hist = [0u32; 256];
            for y in y0..y1 {
                for x in x0..x1 {
                    hist[src[y * width + x] as usize] += 1;
                }
            }

            // Clip & redistribute
            let mut excess = 0u32;
            for bin in hist.iter_mut() {
                if *bin > actual_clip {
                    excess += *bin - actual_clip;
                    *bin = actual_clip;
                }
            }
            let per_bin = excess / num_bins as u32;
            let remainder = excess - per_bin * num_bins as u32;
            for (i, bin) in hist.iter_mut().enumerate() {
                *bin += per_bin;
                if (i as u32) < remainder {
                    *bin += 1;
                }
            }

            // CDF
            let mut cdf = [0u8; 256];
            let mut cumsum = 0u64;
            for i in 0..256 {
                cumsum += hist[i] as u64;
                cdf[i] = ((cumsum * 255) / actual_pixels as u64).min(255) as u8;
            }

            row.push(cdf);
        }
        cdfs.push(row);
    }

    // --- Bilinear interpolation between tile CDFs ---
    let mut dst = vec![0u8; width * height];

    for y in 0..height {
        for x in 0..width {
            let val = src[y * width + x] as usize;

            // Which tile center is this pixel near?
            // Tile centers are at (tx * tile_w + tile_w/2, ty * tile_h + tile_h/2)
            let fx = (x as f32 - tile_w as f32 / 2.0) / tile_w as f32;
            let fy = (y as f32 - tile_h as f32 / 2.0) / tile_h as f32;

            let tx0 = (fx.floor() as isize).clamp(0, tile_grid as isize - 1) as usize;
            let ty0 = (fy.floor() as isize).clamp(0, tile_grid as isize - 1) as usize;
            let tx1 = (tx0 + 1).min(tile_grid - 1);
            let ty1 = (ty0 + 1).min(tile_grid - 1);

            let alpha = (fx - tx0 as f32).clamp(0.0, 1.0);
            let beta = (fy - ty0 as f32).clamp(0.0, 1.0);

            let v00 = cdfs[ty0][tx0][val] as f32;
            let v10 = cdfs[ty0][tx1][val] as f32;
            let v01 = cdfs[ty1][tx0][val] as f32;
            let v11 = cdfs[ty1][tx1][val] as f32;

            let interpolated = v00 * (1.0 - alpha) * (1.0 - beta)
                + v10 * alpha * (1.0 - beta)
                + v01 * (1.0 - alpha) * beta
                + v11 * alpha * beta;

            dst[y * width + x] = interpolated.round().clamp(0.0, 255.0) as u8;
        }
    }

    dst
}

// ============================================================================
// BILINEAR RESIZE
// ============================================================================

/// Bilinear image resize for RGB images.
pub fn resize_bilinear(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
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

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_dimensions() {
        let src = vec![128u8; 640 * 480 * 3];
        let result = preprocess(&src, 640, 480, 1600, 320);
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
    fn test_clahe_preserves_dimensions() {
        let src = vec![128u8; 64 * 64 * 3];
        let result = apply_clahe_on_luminance(&src, 64, 64, 4, 2.0);
        assert_eq!(result.len(), 64 * 64 * 3);
    }

    #[test]
    fn test_clahe_boosts_contrast() {
        // Create a low-contrast image: all pixels between 100-110
        let width = 64;
        let height = 64;
        let mut src = vec![0u8; width * height * 3];
        for i in 0..width * height {
            let v = 100 + (i % 10) as u8;
            src[i * 3] = v;
            src[i * 3 + 1] = v;
            src[i * 3 + 2] = v;
        }

        let result = apply_clahe_on_luminance(&src, width, height, 4, 2.0);

        // After CLAHE, the range should be expanded
        let min_val = result.chunks(3).map(|c| c[0]).min().unwrap();
        let max_val = result.chunks(3).map(|c| c[0]).max().unwrap();
        let range = max_val - min_val;

        assert!(
            range > 10,
            "CLAHE should expand contrast range, got {}",
            range
        );
    }

    #[test]
    fn test_preprocess_raw_matches_original() {
        let src = vec![128u8; 100 * 100 * 3];
        let raw = preprocess_raw(&src, 100, 100, 50, 50).unwrap();
        // Just verify it doesn't crash and produces correct size
        assert_eq!(raw.len(), 3 * 50 * 50);
    }
}
