use opencv::{core, imgproc, prelude::*};

pub fn visualize_lanes(frame: &Mat, lanes: &[Lane], position: &VehiclePosition) -> Result<Mat> {
    let mut debug_frame = frame.clone();

    // Draw detected lanes with different colors
    let colors = [
        core::Scalar::new(255.0, 0.0, 0.0, 0.0),   // Blue - Lane 0
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),   // Green - Lane 1
        core::Scalar::new(0.0, 0.0, 255.0, 0.0),   // Red - Lane 2
        core::Scalar::new(255.0, 255.0, 0.0, 0.0), // Cyan - Lane 3
    ];

    for (i, lane) in lanes.iter().enumerate() {
        // Draw lane points
        for window in lane.points.windows(2) {
            let p1 = core::Point::new(window[0].x as i32, window[0].y as i32);
            let p2 = core::Point::new(window[1].x as i32, window[1].y as i32);
            imgproc::line(
                &mut debug_frame,
                p1,
                p2,
                colors[i % 4],
                3,
                imgproc::LINE_8,
                0,
            )?;
        }

        // Draw lane ID
        if let Some(first_point) = lane.points.first() {
            imgproc::put_text(
                &mut debug_frame,
                &format!("L{} ({:.2})", i, lane.confidence),
                core::Point::new(first_point.x as i32, first_point.y as i32 - 10),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                colors[i % 4],
                2,
                imgproc::LINE_8,
                false,
            )?;
        }
    }

    // Draw vehicle position indicator
    let vehicle_x = frame.cols() / 2;
    let vehicle_y = (frame.rows() as f32 * 0.85) as i32;

    imgproc::circle(
        &mut debug_frame,
        core::Point::new(vehicle_x, vehicle_y),
        10,
        core::Scalar::new(0.0, 255.0, 255.0, 0.0), // Yellow
        -1,
        imgproc::LINE_8,
        0,
    )?;

    // Draw current lane info
    let info = format!(
        "Lane: {} | Offset: {:.2} | Conf: {:.2}",
        position.lane_index, position.lateral_offset, position.confidence
    );
    imgproc::put_text(
        &mut debug_frame,
        &info,
        core::Point::new(10, 30),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.8,
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        false,
    )?;

    Ok(debug_frame)
}
