// src/preprocessing.rs

use anyhow::Result;

/// Preprocess raw RGB image for model input
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
    fn test_resize() {
        let src = vec![255u8; 100 * 100 * 3];
        let dst = resize_bilinear(&src, 100, 100, 50, 50);
        assert_eq!(dst.len(), 50 * 50 * 3);
    }
}
