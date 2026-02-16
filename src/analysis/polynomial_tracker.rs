// src/analysis/polynomial_tracker.rs
//
// v5.0: Polynomial Kalman Tracker for lane boundary geometry.
//
// Tracks lane boundary polynomial coefficients (a, b, c) and their rates of
// change (ȧ, ḃ, ċ) through a 6-state linear Kalman filter per boundary.
//
// Why this exists:
//   - DetectionCache caches scalar positions (left_x, right_x) and compensates
//     with ego-motion. This loses curve shape information during dropout.
//   - The polynomial tracker caches the FUNCTION (x = ay² + by + c) and
//     extrapolates the full curve geometry during dropout using the Kalman
//     predict step. This preserves road shape for 15-20 frames.
//   - The rate-of-change state (ȧ, ḃ, ċ) provides geometric lane change signals:
//     on a curve both boundaries evolve together, on a lane change they diverge.
//
// Coordinate system:
//   Same as curvature_estimator.rs — polynomials are in IMAGE y-space with
//   y normalized to [0, 1] over the spine's vertical extent. The 'c' coefficient
//   is the boundary's x-position at the top of the visible region.
//
// Integration:
//   Sits between curvature_estimator (measurement source) and lateral_detector
//   (consumer of lane geometry). The PolynomialBoundaryTracker wraps both
//   left and right KFs and provides smoothed/predicted boundary positions,
//   lane widths, and geometric lane change signals.
//
// Performance:
//   ~200 bytes per boundary. Two 6×6 matrix ops per frame per boundary.
//   Negligible vs YOLO inference.

use super::curvature_estimator::LanePolynomial;
use tracing::{debug, info, warn};

// ============================================================================
// 6×6 MATRIX MATH (inline, no external dependency)
// ============================================================================
//
// We only need: multiply, add, transpose, and a simple inversion for the
// 3×3 innovation covariance. Everything is stack-allocated.

/// 6×6 matrix stored row-major.
#[derive(Debug, Clone, Copy)]
struct Mat6([f32; 36]);

impl Mat6 {
    const ZERO: Self = Self([0.0; 36]);

    fn identity() -> Self {
        let mut m = Self::ZERO;
        for i in 0..6 {
            m.0[i * 6 + i] = 1.0;
        }
        m
    }

    #[inline]
    fn get(&self, r: usize, c: usize) -> f32 {
        self.0[r * 6 + c]
    }

    #[inline]
    fn set(&mut self, r: usize, c: usize, v: f32) {
        self.0[r * 6 + c] = v;
    }

    fn mul(&self, rhs: &Mat6) -> Mat6 {
        let mut out = Mat6::ZERO;
        for i in 0..6 {
            for j in 0..6 {
                let mut sum = 0.0f32;
                for k in 0..6 {
                    sum += self.get(i, k) * rhs.get(k, j);
                }
                out.set(i, j, sum);
            }
        }
        out
    }

    fn add(&self, rhs: &Mat6) -> Mat6 {
        let mut out = Mat6::ZERO;
        for i in 0..36 {
            out.0[i] = self.0[i] + rhs.0[i];
        }
        out
    }

    fn sub(&self, rhs: &Mat6) -> Mat6 {
        let mut out = Mat6::ZERO;
        for i in 0..36 {
            out.0[i] = self.0[i] - rhs.0[i];
        }
        out
    }

    fn transpose(&self) -> Mat6 {
        let mut out = Mat6::ZERO;
        for i in 0..6 {
            for j in 0..6 {
                out.set(i, j, self.get(j, i));
            }
        }
        out
    }

    fn trace(&self) -> f32 {
        let mut s = 0.0;
        for i in 0..6 {
            s += self.get(i, i);
        }
        s
    }

    /// Multiply 6×6 * 6×1 → 6×1
    fn mul_vec(&self, v: &Vec6) -> Vec6 {
        let mut out = Vec6::ZERO;
        for i in 0..6 {
            let mut sum = 0.0f32;
            for j in 0..6 {
                sum += self.get(i, j) * v.0[j];
            }
            out.0[i] = sum;
        }
        out
    }
}

/// 6×3 matrix (for H' and K computation).
#[derive(Debug, Clone, Copy)]
struct Mat6x3([f32; 18]);

impl Mat6x3 {
    const ZERO: Self = Self([0.0; 18]);

    #[inline]
    fn get(&self, r: usize, c: usize) -> f32 {
        self.0[r * 3 + c]
    }

    #[inline]
    fn set(&mut self, r: usize, c: usize, v: f32) {
        self.0[r * 3 + c] = v;
    }
}

/// 3×6 matrix (for H).
#[derive(Debug, Clone, Copy)]
struct Mat3x6([f32; 18]);

impl Mat3x6 {
    const ZERO: Self = Self([0.0; 18]);

    #[inline]
    fn get(&self, r: usize, c: usize) -> f32 {
        self.0[r * 6 + c]
    }

    #[inline]
    fn set(&mut self, r: usize, c: usize, v: f32) {
        self.0[r * 6 + c] = v;
    }

    /// H (3×6) * P (6×6) → 3×6
    fn mul_mat6(&self, rhs: &Mat6) -> Mat3x6 {
        let mut out = Mat3x6::ZERO;
        for i in 0..3 {
            for j in 0..6 {
                let mut sum = 0.0f32;
                for k in 0..6 {
                    sum += self.get(i, k) * rhs.get(k, j);
                }
                out.set(i, j, sum);
            }
        }
        out
    }

    /// H (3×6) * v (6×1) → 3×1
    fn mul_vec(&self, v: &Vec6) -> Vec3 {
        let mut out = Vec3::ZERO;
        for i in 0..3 {
            let mut sum = 0.0f32;
            for j in 0..6 {
                sum += self.get(i, j) * v.0[j];
            }
            out.0[i] = sum;
        }
        out
    }

    fn transpose(&self) -> Mat6x3 {
        let mut out = Mat6x3::ZERO;
        for i in 0..3 {
            for j in 0..6 {
                out.set(j, i, self.get(i, j));
            }
        }
        out
    }
}

/// 3×3 matrix for innovation covariance S.
#[derive(Debug, Clone, Copy)]
struct Mat3([f32; 9]);

impl Mat3 {
    const ZERO: Self = Self([0.0; 9]);

    #[inline]
    fn get(&self, r: usize, c: usize) -> f32 {
        self.0[r * 3 + c]
    }

    #[inline]
    fn set(&mut self, r: usize, c: usize, v: f32) {
        self.0[r * 3 + c] = v;
    }

    /// Invert 3×3 via cofactor method. Returns None if singular.
    fn invert(&self) -> Option<Self> {
        let m = &self.0;
        let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6]);

        if det.abs() < 1e-12 {
            return None;
        }

        let inv_det = 1.0 / det;
        Some(Mat3([
            (m[4] * m[8] - m[5] * m[7]) * inv_det,
            (m[2] * m[7] - m[1] * m[8]) * inv_det,
            (m[1] * m[5] - m[2] * m[4]) * inv_det,
            (m[5] * m[6] - m[3] * m[8]) * inv_det,
            (m[0] * m[8] - m[2] * m[6]) * inv_det,
            (m[2] * m[3] - m[0] * m[5]) * inv_det,
            (m[3] * m[7] - m[4] * m[6]) * inv_det,
            (m[1] * m[6] - m[0] * m[7]) * inv_det,
            (m[0] * m[4] - m[1] * m[3]) * inv_det,
        ]))
    }

    /// (3×6_transposed as 6×3) * (3×3) → (6×3)
    /// Actually: Mat6x3 * Mat3 → Mat6x3
    fn right_mul_6x3(&self, lhs: &Mat6x3) -> Mat6x3 {
        let mut out = Mat6x3::ZERO;
        for i in 0..6 {
            for j in 0..3 {
                let mut sum = 0.0f32;
                for k in 0..3 {
                    sum += lhs.get(i, k) * self.get(k, j);
                }
                out.set(i, j, sum);
            }
        }
        out
    }
}

/// 6-element state vector.
#[derive(Debug, Clone, Copy)]
struct Vec6([f32; 6]);

impl Vec6 {
    const ZERO: Self = Self([0.0; 6]);

    fn sub(&self, rhs: &Vec6) -> Vec6 {
        Vec6([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
            self.0[4] - rhs.0[4],
            self.0[5] - rhs.0[5],
        ])
    }
}

/// 3-element measurement/innovation vector.
#[derive(Debug, Clone, Copy)]
struct Vec3([f32; 3]);

impl Vec3 {
    const ZERO: Self = Self([0.0; 3]);

    fn sub(&self, rhs: &Vec3) -> Vec3 {
        Vec3([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
        ])
    }
}

/// Multiply Mat6x3 (6×3) * Vec3 (3×1) → Vec6 (6×1)
fn mat6x3_mul_vec3(m: &Mat6x3, v: &Vec3) -> Vec6 {
    let mut out = Vec6::ZERO;
    for i in 0..6 {
        let mut sum = 0.0f32;
        for j in 0..3 {
            sum += m.get(i, j) * v.0[j];
        }
        out.0[i] = sum;
    }
    out
}

/// Compute (3×6) * (6×6) * (6×3) → 3×3 — needed for S = H P H' + R
fn hpht(h: &Mat3x6, p: &Mat6, ht: &Mat6x3) -> Mat3 {
    // First: hp = H * P (3×6)
    let hp = h.mul_mat6(p);
    // Then: hp * H' (3×6 × 6×3 → 3×3)
    let mut out = Mat3::ZERO;
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0f32;
            for k in 0..6 {
                sum += hp.get(i, k) * ht.get(k, j);
            }
            out.set(i, j, sum);
        }
    }
    out
}

/// K (6×3) * H (3×6) → 6×6
fn mat6x3_mul_mat3x6(lhs: &Mat6x3, rhs: &Mat3x6) -> Mat6 {
    let mut out = Mat6::ZERO;
    for i in 0..6 {
        for j in 0..6 {
            let mut sum = 0.0f32;
            for k in 0..3 {
                sum += lhs.get(i, k) * rhs.get(k, j);
            }
            out.set(i, j, sum);
        }
    }
    out
}

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for a single boundary Kalman filter.
#[derive(Debug, Clone)]
pub struct PolyKFConfig {
    /// Process noise for polynomial coefficients (a, b, c) per frame.
    /// Controls how quickly the filter expects the road geometry to change.
    /// Higher = more responsive to real changes, noisier.
    /// Lower = smoother tracking, slower to adapt.
    pub q_coefficient: f32,

    /// Process noise for rate-of-change states (ȧ, ḃ, ċ) per frame.
    /// Should be larger than q_coefficient since rates are less constrained.
    pub q_rate: f32,

    /// Base measurement noise for the 'a' (curvature) coefficient.
    /// Scaled by inverse fit quality (higher RMSE → more noise).
    pub r_base_a: f32,

    /// Base measurement noise for the 'b' (slope) coefficient.
    pub r_base_b: f32,

    /// Base measurement noise for the 'c' (intercept/position) coefficient.
    /// Usually smallest since intercept is the most reliably measured.
    pub r_base_c: f32,

    /// Maximum frames to extrapolate via predict-only before expiring.
    pub max_predict_frames: u32,

    /// Initial covariance diagonal for coefficients (a, b, c).
    pub initial_p_coeff: f32,

    /// Initial covariance diagonal for rates (ȧ, ḃ, ċ).
    pub initial_p_rate: f32,
}

impl Default for PolyKFConfig {
    fn default() -> Self {
        Self {
            q_coefficient: 1e-5,
            q_rate: 5e-4,
            r_base_a: 1.0,
            r_base_b: 0.1,
            r_base_c: 0.05,
            max_predict_frames: 20,
            initial_p_coeff: 100.0,
            initial_p_rate: 10.0,
        }
    }
}

/// Configuration for the full boundary tracker (both sides).
#[derive(Debug, Clone)]
pub struct PolynomialTrackerConfig {
    pub kf: PolyKFConfig,

    /// Minimum fit RMSE to accept a measurement. Below this, something
    /// is wrong (perfect fits on noisy data → overfitting).
    pub min_fit_rmse: f32,

    /// Maximum fit RMSE to accept. Above this, the polynomial doesn't
    /// describe the boundary shape well.
    pub max_fit_rmse: f32,

    /// Minimum spine points in the polynomial fit to trust it.
    pub min_fit_points: usize,

    /// Divergence threshold for boundary velocity to signal lane change.
    /// |ċ_left - ċ_right| above this = boundaries moving apart/together.
    pub boundary_divergence_threshold: f32,

    /// Lane width rate-of-change threshold (px/frame) to signal lane change.
    pub lane_width_rate_threshold: f32,

    /// Near-far offset divergence threshold for lane change signal.
    pub offset_divergence_threshold: f32,

    /// Y-normalized positions for near/far sampling.
    pub near_y: f32,
    pub far_y: f32,
}

impl Default for PolynomialTrackerConfig {
    fn default() -> Self {
        Self {
            kf: PolyKFConfig::default(),
            min_fit_rmse: 0.0,
            max_fit_rmse: 25.0,
            min_fit_points: 8,
            boundary_divergence_threshold: 3.0,
            lane_width_rate_threshold: 2.0,
            offset_divergence_threshold: 0.08,
            near_y: 0.82,
            far_y: 0.45,
        }
    }
}

// ============================================================================
// SINGLE BOUNDARY KALMAN FILTER
// ============================================================================

/// Kalman filter tracking polynomial coefficients for one lane boundary.
///
/// State: [a, b, c, ȧ, ḃ, ċ]
///   - (a, b, c): polynomial coefficients, x(y) = a·y² + b·y + c
///   - (ȧ, ḃ, ċ): rates of change per frame
///
/// Measurement: [a_meas, b_meas, c_meas] from curvature_estimator
///
/// Transition: constant-velocity model on coefficients
///   x_{k+1} = F * x_k + w,  F = [[I₃, I₃], [0₃, I₃]]
#[derive(Debug, Clone)]
pub struct LanePolyKF {
    /// State vector [a, b, c, ȧ, ḃ, ċ]
    x: Vec6,
    /// State covariance (6×6)
    p: Mat6,
    /// Process noise (6×6, diagonal)
    q: Mat6,
    /// State transition matrix (6×6)
    f: Mat6,
    /// Observation matrix H (3×6): extracts [a, b, c] from state
    h: Mat3x6,
    /// H transposed (6×3)
    ht: Mat6x3,
    /// Covariance trace when last measurement was applied.
    /// Used for principled confidence decay during prediction.
    p_trace_at_measurement: f32,
    /// Whether the filter has been initialized with at least one measurement.
    initialized: bool,
    /// Frames since last measurement update (predict-only count).
    frames_without_measurement: u32,
    /// Last innovation vector (measurement − prediction) for lane change detection.
    last_innovation: Vec3,
    /// Last innovation magnitude (L2 norm of innovation).
    pub last_innovation_magnitude: f32,
}

impl LanePolyKF {
    pub fn new(config: &PolyKFConfig) -> Self {
        // State transition: constant-rate model
        // [a, b, c, ȧ, ḃ, ċ]_{k+1} = F * [a, b, c, ȧ, ḃ, ċ]_k
        // where F = [[I₃, I₃], [0₃, I₃]]
        let mut f = Mat6::identity();
        // Upper-right 3×3 block = I₃ (coefficient += rate * dt, dt=1)
        f.set(0, 3, 1.0);
        f.set(1, 4, 1.0);
        f.set(2, 5, 1.0);

        // Process noise: diagonal
        let mut q = Mat6::ZERO;
        q.set(0, 0, config.q_coefficient);
        q.set(1, 1, config.q_coefficient);
        q.set(2, 2, config.q_coefficient);
        q.set(3, 3, config.q_rate);
        q.set(4, 4, config.q_rate);
        q.set(5, 5, config.q_rate);

        // Observation: H = [I₃ | 0₃]
        let mut h = Mat3x6::ZERO;
        h.set(0, 0, 1.0); // observe a
        h.set(1, 1, 1.0); // observe b
        h.set(2, 2, 1.0); // observe c

        let ht = h.transpose();

        // Initial covariance: high uncertainty
        let mut p = Mat6::ZERO;
        p.set(0, 0, config.initial_p_coeff);
        p.set(1, 1, config.initial_p_coeff);
        p.set(2, 2, config.initial_p_coeff);
        p.set(3, 3, config.initial_p_rate);
        p.set(4, 4, config.initial_p_rate);
        p.set(5, 5, config.initial_p_rate);

        Self {
            x: Vec6::ZERO,
            p,
            q,
            f,
            h,
            ht,
            p_trace_at_measurement: p.trace(),
            initialized: false,
            frames_without_measurement: 0,
            last_innovation: Vec3::ZERO,
            last_innovation_magnitude: 0.0,
        }
    }

    /// Kalman predict step. Advances state by one frame.
    pub fn predict(&mut self) {
        // x = F * x
        self.x = self.f.mul_vec(&self.x);
        // P = F * P * F' + Q
        let fp = self.f.mul(&self.p);
        let ft = self.f.transpose();
        self.p = fp.mul(&ft).add(&self.q);
    }

    /// Kalman update step with a new polynomial measurement.
    ///
    /// Returns the innovation vector (measurement − prediction) which is
    /// used as a lane change signal.
    pub fn update(&mut self, poly: &LanePolynomial, config: &PolyKFConfig) -> Vec3 {
        let z = Vec3([poly.a, poly.b, poly.c]);

        if !self.initialized {
            // First measurement: initialize state directly
            self.x = Vec6([poly.a, poly.b, poly.c, 0.0, 0.0, 0.0]);
            self.initialized = true;
            self.frames_without_measurement = 0;
            self.p_trace_at_measurement = self.p.trace();
            self.last_innovation = Vec3::ZERO;
            self.last_innovation_magnitude = 0.0;
            return Vec3::ZERO;
        }

        // Innovation: y = z - H * x
        let predicted_z = self.h.mul_vec(&self.x);
        let innovation = z.sub(&predicted_z);

        // Adaptive measurement noise based on fit quality.
        // Worse fits (higher RMSE, fewer points) → trust prediction more.
        let rmse_factor = (poly.rmse_px / 5.0).max(1.0); // normalize around 5px RMSE
        let point_factor = (20.0 / poly.num_points as f32).max(1.0); // normalize around 20 points
        let noise_scale = rmse_factor * point_factor;

        let mut r = Mat3::ZERO;
        r.set(0, 0, config.r_base_a * noise_scale);
        r.set(1, 1, config.r_base_b * noise_scale);
        r.set(2, 2, config.r_base_c * noise_scale);

        // Innovation covariance: S = H * P * H' + R
        let s = hpht(&self.h, &self.p, &self.ht);
        let s_with_r = Mat3([
            s.0[0] + r.0[0],
            s.0[1] + r.0[1],
            s.0[2] + r.0[2],
            s.0[3] + r.0[3],
            s.0[4] + r.0[4],
            s.0[5] + r.0[5],
            s.0[6] + r.0[6],
            s.0[7] + r.0[7],
            s.0[8] + r.0[8],
        ]);

        // Kalman gain: K = P * H' * S^{-1}
        let s_inv = match s_with_r.invert() {
            Some(inv) => inv,
            None => {
                warn!("📐 Poly KF: singular innovation covariance, skipping update");
                return innovation;
            }
        };

        // P * H' (6×3)
        let pht = {
            let mut out = Mat6x3::ZERO;
            for i in 0..6 {
                for j in 0..3 {
                    let mut sum = 0.0f32;
                    for k in 0..6 {
                        sum += self.p.get(i, k) * self.ht.get(k, j);
                    }
                    out.set(i, j, sum);
                }
            }
            out
        };

        // K = P*H' * S^{-1} (6×3)
        let k = s_inv.right_mul_6x3(&pht);

        // State update: x = x + K * innovation
        let correction = mat6x3_mul_vec3(&k, &innovation);
        self.x = Vec6([
            self.x.0[0] + correction.0[0],
            self.x.0[1] + correction.0[1],
            self.x.0[2] + correction.0[2],
            self.x.0[3] + correction.0[3],
            self.x.0[4] + correction.0[4],
            self.x.0[5] + correction.0[5],
        ]);

        // Covariance update: P = (I - K*H) * P
        let kh = mat6x3_mul_mat3x6(&k, &self.h);
        let i_kh = Mat6::identity().sub(&kh);
        self.p = i_kh.mul(&self.p);

        // Bookkeeping
        self.frames_without_measurement = 0;
        self.p_trace_at_measurement = self.p.trace();
        self.last_innovation = innovation;
        self.last_innovation_magnitude =
            (innovation.0[0].powi(2) + innovation.0[1].powi(2) + innovation.0[2].powi(2)).sqrt();

        innovation
    }

    /// Predict-only step with ego-motion compensation.
    ///
    /// Adjusts the 'c' coefficient (boundary x-position) by the ego lateral
    /// displacement. When the ego vehicle moves right, lane boundaries appear
    /// to shift left in the camera frame.
    pub fn predict_with_ego(&mut self, ego_lateral_velocity_px: f32) {
        self.predict();
        // Ego compensation: vehicle moves right → boundaries shift left
        self.x.0[2] -= ego_lateral_velocity_px;
        self.frames_without_measurement += 1;
    }

    /// Confidence based on covariance growth since last measurement.
    ///
    /// Returns [0, 1]. When P grows (predict-only), confidence drops.
    /// More principled than fixed exponential decay.
    pub fn confidence(&self) -> f32 {
        if !self.initialized || self.p_trace_at_measurement < 1e-10 {
            return 0.0;
        }
        let trace_ratio = self.p.trace() / self.p_trace_at_measurement;
        // Ratio ≥ 1.0 (covariance only grows during predict-only steps).
        // Map to [0, 1]: confidence = 1/sqrt(ratio)
        (1.0 / trace_ratio.sqrt()).clamp(0.0, 1.0)
    }

    /// Current polynomial coefficients (smoothed or predicted).
    pub fn coefficients(&self) -> (f32, f32, f32) {
        (self.x.0[0], self.x.0[1], self.x.0[2])
    }

    /// Current rate of change of polynomial coefficients.
    pub fn rates(&self) -> (f32, f32, f32) {
        (self.x.0[3], self.x.0[4], self.x.0[5])
    }

    /// Evaluate the polynomial at a normalized y position.
    /// y ∈ [0, 1] where 0 = top of visible region, 1 = bottom.
    pub fn x_at(&self, y_norm: f32) -> f32 {
        let (a, b, c) = self.coefficients();
        a * y_norm * y_norm + b * y_norm + c
    }

    /// Rate of change of x-position at the reference point (c_dot).
    /// This is how fast the boundary's x-position is changing per frame.
    pub fn position_rate(&self) -> f32 {
        self.x.0[5] // ċ
    }

    /// Rate of change of curvature (a_dot).
    pub fn curvature_rate(&self) -> f32 {
        self.x.0[3] // ȧ
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub fn frames_stale(&self) -> u32 {
        self.frames_without_measurement
    }

    /// Reset the filter state (e.g., on scene change).
    pub fn reset(&mut self, config: &PolyKFConfig) {
        *self = Self::new(config);
    }
}

// ============================================================================
// GEOMETRIC LANE CHANGE SIGNALS
// ============================================================================

/// Geometric signals derived from the polynomial tracker that indicate
/// lane changes independently of ego lateral offset.
///
/// These signals discriminate curves from lane changes by comparing
/// how the left and right boundaries evolve relative to each other.
#[derive(Debug, Clone, Copy, Default)]
pub struct GeometricLaneChangeSignals {
    /// Boundary position-rate divergence: ċ_left − ċ_right (px/frame).
    /// On a curve: both ċ have same sign → divergence ≈ 0.
    /// On a lane change: one approaches, one recedes → |divergence| > 0.
    pub boundary_velocity_divergence: f32,

    /// Rate of change of lane width at reference y (px/frame).
    /// On a curve: lane width is ~constant → rate ≈ 0.
    /// On a lane change: width changes → |rate| > 0.
    pub lane_width_rate: f32,

    /// Curvature rate divergence: ȧ_left − ȧ_right.
    /// On a curve: both curvatures evolve together → low divergence.
    /// On a lane change: curvatures diverge → high divergence.
    pub curvature_rate_divergence: f32,

    /// Near-field vs far-field offset divergence.
    /// On a curve: offset is consistent at different y-positions.
    /// On a lane change: near offset differs from far offset.
    pub near_far_offset_divergence: f32,

    /// Innovation magnitude when lanes returned after dropout.
    /// Large asymmetric innovation = lane change happened during dropout.
    pub recovery_innovation_left: f32,
    pub recovery_innovation_right: f32,

    /// Whether the geometric signals collectively suggest a lane change.
    /// This is a convenience flag — consumers can also check individual signals.
    pub suggests_lane_change: bool,

    /// Confidence in the geometric signals [0, 1].
    /// Low when one or both trackers are in predict-only mode.
    pub confidence: f32,
}

// ============================================================================
// TRACKER STATE
// ============================================================================

/// State of the polynomial tracker for one side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryState {
    /// No data yet
    Uninitialized,
    /// Tracking with fresh measurements
    Tracking,
    /// Predicting (no measurement for 1+ frames)
    Predicting,
    /// Expired (too many predict-only frames)
    Expired,
}

impl BoundaryState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Uninitialized => "UNINIT",
            Self::Tracking => "TRACKING",
            Self::Predicting => "PREDICTING",
            Self::Expired => "EXPIRED",
        }
    }

    pub fn has_estimate(&self) -> bool {
        matches!(self, Self::Tracking | Self::Predicting)
    }
}

// ============================================================================
// POLYNOMIAL BOUNDARY TRACKER (wraps both left + right KFs)
// ============================================================================

/// Tracks both lane boundaries using polynomial Kalman filters.
///
/// Provides:
/// - Smoothed boundary polynomials when detections are present
/// - Predicted boundary polynomials during detection dropout (curve-aware)
/// - Geometric lane change signals (boundary divergence, width rate, etc.)
/// - Innovation-based lane change detection on recovery from dropout
#[derive(Debug, Clone)]
pub struct PolynomialBoundaryTracker {
    config: PolynomialTrackerConfig,

    left_kf: LanePolyKF,
    right_kf: LanePolyKF,

    left_state: BoundaryState,
    right_state: BoundaryState,

    /// Last computed geometric signals (updated every frame).
    signals: GeometricLaneChangeSignals,

    /// Previous frame's lane width at reference_y (for rate computation).
    prev_lane_width: f32,

    /// Rolling history of boundary velocity divergence for smoothing.
    divergence_history: [f32; 8],
    divergence_idx: usize,

    /// Total frames processed.
    frame_count: u64,

    // ── Metrics ──
    pub total_left_measurements: u64,
    pub total_right_measurements: u64,
    pub total_left_predictions: u64,
    pub total_right_predictions: u64,
}

impl PolynomialBoundaryTracker {
    pub fn new(config: PolynomialTrackerConfig) -> Self {
        Self {
            left_kf: LanePolyKF::new(&config.kf),
            right_kf: LanePolyKF::new(&config.kf),
            left_state: BoundaryState::Uninitialized,
            right_state: BoundaryState::Uninitialized,
            signals: GeometricLaneChangeSignals::default(),
            prev_lane_width: 0.0,
            divergence_history: [0.0; 8],
            divergence_idx: 0,
            frame_count: 0,
            total_left_measurements: 0,
            total_right_measurements: 0,
            total_left_predictions: 0,
            total_right_predictions: 0,
            config,
        }
    }

    /// Process one frame. Call every frame with available polynomial fits.
    ///
    /// # Arguments
    /// * `left_poly` — Polynomial fit for the left boundary (None if not detected)
    /// * `right_poly` — Polynomial fit for the right boundary (None if not detected)
    /// * `ego_lateral_velocity_px` — Ego lateral velocity in px/frame (from optical flow)
    /// * `ego_x` — Ego vehicle x-position in image coordinates (typically frame_width/2)
    ///
    /// # Returns
    /// Updated geometric lane change signals.
    pub fn update(
        &mut self,
        left_poly: Option<&LanePolynomial>,
        right_poly: Option<&LanePolynomial>,
        ego_lateral_velocity_px: f32,
        ego_x: f32,
    ) -> &GeometricLaneChangeSignals {
        self.frame_count += 1;

        // ── Left boundary ────────────────────────────────────────
        let left_innovation = if let Some(lp) = left_poly {
            if self.is_fit_acceptable(lp) {
                self.left_kf.predict();
                let innov = self.left_kf.update(lp, &self.config.kf);
                self.left_state = BoundaryState::Tracking;
                self.total_left_measurements += 1;
                Some(innov)
            } else {
                self.left_kf.predict_with_ego(ego_lateral_velocity_px);
                self.update_boundary_state_predict(
                    &mut self.left_state,
                    self.left_kf.frames_stale(),
                );
                self.total_left_predictions += 1;
                None
            }
        } else {
            self.left_kf.predict_with_ego(ego_lateral_velocity_px);
            self.update_boundary_state_predict_left();
            self.total_left_predictions += 1;
            None
        };

        // ── Right boundary ───────────────────────────────────────
        let right_innovation = if let Some(rp) = right_poly {
            if self.is_fit_acceptable(rp) {
                self.right_kf.predict();
                let innov = self.right_kf.update(rp, &self.config.kf);
                self.right_state = BoundaryState::Tracking;
                self.total_right_measurements += 1;
                Some(innov)
            } else {
                self.right_kf.predict_with_ego(ego_lateral_velocity_px);
                self.update_boundary_state_predict_right();
                self.total_right_predictions += 1;
                None
            }
        } else {
            self.right_kf.predict_with_ego(ego_lateral_velocity_px);
            self.update_boundary_state_predict_right();
            self.total_right_predictions += 1;
            None
        };

        // ── Compute geometric signals ────────────────────────────
        self.compute_signals(ego_x, left_innovation, right_innovation);

        // ── Periodic logging ─────────────────────────────────────
        if self.frame_count % 150 == 0 {
            debug!(
                "📐 PolyTracker F{}: L={} ({:.2}conf, stale={}f) R={} ({:.2}conf, stale={}f) | \
                 div={:.2} wrate={:.2} nf_div={:.2} | lc={}",
                self.frame_count,
                self.left_state.as_str(),
                self.left_kf.confidence(),
                self.left_kf.frames_stale(),
                self.right_state.as_str(),
                self.right_kf.confidence(),
                self.right_kf.frames_stale(),
                self.signals.boundary_velocity_divergence,
                self.signals.lane_width_rate,
                self.signals.near_far_offset_divergence,
                self.signals.suggests_lane_change,
            );
        }

        &self.signals
    }

    // ── Boundary state management ────────────────────────────────

    fn update_boundary_state_predict_left(&mut self) {
        let stale = self.left_kf.frames_stale();
        self.left_state = if !self.left_kf.is_initialized() {
            BoundaryState::Uninitialized
        } else if stale > self.config.kf.max_predict_frames {
            BoundaryState::Expired
        } else {
            BoundaryState::Predicting
        };
    }

    fn update_boundary_state_predict_right(&mut self) {
        let stale = self.right_kf.frames_stale();
        self.right_state = if !self.right_kf.is_initialized() {
            BoundaryState::Uninitialized
        } else if stale > self.config.kf.max_predict_frames {
            BoundaryState::Expired
        } else {
            BoundaryState::Predicting
        };
    }

    fn update_boundary_state_predict(state: &mut BoundaryState, stale_frames: u32) {
        // Static helper — not used for left/right since we need &mut self for kf access
        *state = if stale_frames > 20 {
            BoundaryState::Expired
        } else {
            BoundaryState::Predicting
        };
    }

    fn is_fit_acceptable(&self, poly: &LanePolynomial) -> bool {
        poly.rmse_px >= self.config.min_fit_rmse
            && poly.rmse_px <= self.config.max_fit_rmse
            && poly.num_points >= self.config.min_fit_points
    }

    // ── Signal computation ───────────────────────────────────────

    fn compute_signals(
        &mut self,
        ego_x: f32,
        left_innovation: Option<Vec3>,
        right_innovation: Option<Vec3>,
    ) {
        let both_active = self.left_state.has_estimate() && self.right_state.has_estimate();

        if !both_active {
            self.signals = GeometricLaneChangeSignals::default();
            return;
        }

        let (_, _, left_c_rate) = self.left_kf.rates();
        let (_, _, right_c_rate) = self.right_kf.rates();

        // 1. Boundary velocity divergence
        //    Curve: both ċ same sign, similar magnitude → divergence ≈ 0
        //    Lane change: one approaches, one recedes → divergence large
        let raw_divergence = left_c_rate - right_c_rate;
        self.divergence_history[self.divergence_idx % 8] = raw_divergence;
        self.divergence_idx += 1;

        let filled = self.divergence_idx.min(8);
        let smoothed_divergence: f32 =
            self.divergence_history[..filled].iter().sum::<f32>() / filled as f32;

        // 2. Lane width rate of change
        let ref_y = self.config.near_y;
        let left_x = self.left_kf.x_at(ref_y);
        let right_x = self.right_kf.x_at(ref_y);
        let current_width = right_x - left_x;
        let width_rate = if self.prev_lane_width > 0.0 {
            current_width - self.prev_lane_width
        } else {
            0.0
        };
        self.prev_lane_width = current_width;

        // 3. Curvature rate divergence
        let left_a_rate = self.left_kf.curvature_rate();
        let right_a_rate = self.right_kf.curvature_rate();
        let curvature_rate_div = left_a_rate - right_a_rate;

        // 4. Near-far offset divergence
        let near_y = self.config.near_y;
        let far_y = self.config.far_y;
        let near_offset = self.normalized_offset_at(ego_x, near_y);
        let far_offset = self.normalized_offset_at(ego_x, far_y);
        let nf_divergence = (near_offset - far_offset).abs();

        // 5. Recovery innovation
        let recovery_left = left_innovation
            .map(|i| (i.0[0].powi(2) + i.0[1].powi(2) + i.0[2].powi(2)).sqrt())
            .unwrap_or(0.0);
        let recovery_right = right_innovation
            .map(|i| (i.0[0].powi(2) + i.0[1].powi(2) + i.0[2].powi(2)).sqrt())
            .unwrap_or(0.0);

        // 6. Confidence: geometric mean of both boundary confidences
        let left_conf = self.left_kf.confidence();
        let right_conf = self.right_kf.confidence();
        let signal_confidence = (left_conf * right_conf).sqrt();

        // 7. Lane change suggestion
        let div_exceeds = smoothed_divergence.abs() > self.config.boundary_divergence_threshold;
        let width_exceeds = width_rate.abs() > self.config.lane_width_rate_threshold;
        let nf_exceeds = nf_divergence > self.config.offset_divergence_threshold;

        // Require at least 2 of 3 signals to suggest lane change
        let vote_count = div_exceeds as u8 + width_exceeds as u8 + nf_exceeds as u8;
        let suggests_lc = vote_count >= 2 && signal_confidence > 0.3;

        self.signals = GeometricLaneChangeSignals {
            boundary_velocity_divergence: smoothed_divergence,
            lane_width_rate: width_rate,
            curvature_rate_divergence: curvature_rate_div,
            near_far_offset_divergence: nf_divergence,
            recovery_innovation_left: recovery_left,
            recovery_innovation_right: recovery_right,
            suggests_lane_change: suggests_lc,
            confidence: signal_confidence,
        };
    }

    // ════════════════════════════════════════════════════════════════
    // PUBLIC QUERY API
    // ════════════════════════════════════════════════════════════════

    /// Left boundary x-position at a normalized y.
    pub fn left_x_at(&self, y_norm: f32) -> f32 {
        self.left_kf.x_at(y_norm)
    }

    /// Right boundary x-position at a normalized y.
    pub fn right_x_at(&self, y_norm: f32) -> f32 {
        self.right_kf.x_at(y_norm)
    }

    /// Lane width at a normalized y.
    pub fn lane_width_at(&self, y_norm: f32) -> f32 {
        self.right_kf.x_at(y_norm) - self.left_kf.x_at(y_norm)
    }

    /// Normalized lateral offset at a specific y.
    /// Returns value in [-1, 1] where 0 = centered.
    pub fn normalized_offset_at(&self, ego_x: f32, y_norm: f32) -> f32 {
        let left = self.left_kf.x_at(y_norm);
        let right = self.right_kf.x_at(y_norm);
        let width = (right - left).max(1.0);
        let center = (left + right) / 2.0;
        (ego_x - center) / (width / 2.0)
    }

    /// Get the smoothed/predicted left boundary as a LanePolynomial.
    pub fn left_polynomial(&self) -> Option<LanePolynomial> {
        if !self.left_state.has_estimate() {
            return None;
        }
        let (a, b, c) = self.left_kf.coefficients();
        Some(LanePolynomial {
            a,
            b,
            c,
            rmse_px: 0.0, // smoothed, no residual
            num_points: 0,
            vertical_span_px: 0.0,
        })
    }

    /// Get the smoothed/predicted right boundary as a LanePolynomial.
    pub fn right_polynomial(&self) -> Option<LanePolynomial> {
        if !self.right_state.has_estimate() {
            return None;
        }
        let (a, b, c) = self.right_kf.coefficients();
        Some(LanePolynomial {
            a,
            b,
            c,
            rmse_px: 0.0,
            num_points: 0,
            vertical_span_px: 0.0,
        })
    }

    /// Effective left boundary x-position at reference y (comparable to DetectionCache.left_x).
    pub fn left_x(&self) -> Option<f32> {
        if self.left_state.has_estimate() {
            Some(self.left_kf.x_at(self.config.near_y))
        } else {
            None
        }
    }

    /// Effective right boundary x-position at reference y.
    pub fn right_x(&self) -> Option<f32> {
        if self.right_state.has_estimate() {
            Some(self.right_kf.x_at(self.config.near_y))
        } else {
            None
        }
    }

    /// Effective confidence (min of both boundaries).
    pub fn confidence(&self) -> f32 {
        if !self.left_state.has_estimate() || !self.right_state.has_estimate() {
            return 0.0;
        }
        self.left_kf.confidence().min(self.right_kf.confidence())
    }

    /// Left boundary confidence.
    pub fn left_confidence(&self) -> f32 {
        if self.left_state.has_estimate() {
            self.left_kf.confidence()
        } else {
            0.0
        }
    }

    /// Right boundary confidence.
    pub fn right_confidence(&self) -> f32 {
        if self.right_state.has_estimate() {
            self.right_kf.confidence()
        } else {
            0.0
        }
    }

    /// Current geometric lane change signals.
    pub fn signals(&self) -> &GeometricLaneChangeSignals {
        &self.signals
    }

    /// Whether both boundaries have usable estimates (tracking or predicting).
    pub fn both_active(&self) -> bool {
        self.left_state.has_estimate() && self.right_state.has_estimate()
    }

    /// Current state of the left boundary.
    pub fn left_state(&self) -> BoundaryState {
        self.left_state
    }

    /// Current state of the right boundary.
    pub fn right_state(&self) -> BoundaryState {
        self.right_state
    }

    /// Left boundary innovation magnitude (0 when predicting).
    pub fn left_innovation(&self) -> f32 {
        self.left_kf.last_innovation_magnitude
    }

    /// Right boundary innovation magnitude.
    pub fn right_innovation(&self) -> f32 {
        self.right_kf.last_innovation_magnitude
    }

    /// Left boundary stale frame count (for diagnostics).
    pub fn left_kf_stale(&self) -> u32 {
        self.left_kf.frames_stale()
    }

    /// Right boundary stale frame count (for diagnostics).
    pub fn right_kf_stale(&self) -> u32 {
        self.right_kf.frames_stale()
    }

    /// Reset both filters (e.g., on scene change / video cut).
    pub fn reset(&mut self) {
        self.left_kf.reset(&self.config.kf);
        self.right_kf.reset(&self.config.kf);
        self.left_state = BoundaryState::Uninitialized;
        self.right_state = BoundaryState::Uninitialized;
        self.signals = GeometricLaneChangeSignals::default();
        self.prev_lane_width = 0.0;
        self.divergence_history = [0.0; 8];
        self.divergence_idx = 0;
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_poly(a: f32, b: f32, c: f32) -> LanePolynomial {
        LanePolynomial {
            a,
            b,
            c,
            rmse_px: 3.0,
            num_points: 30,
            vertical_span_px: 300.0,
        }
    }

    #[test]
    fn test_kf_initialization() {
        let config = PolyKFConfig::default();
        let mut kf = LanePolyKF::new(&config);
        assert!(!kf.is_initialized());

        let poly = make_poly(0.5, 2.0, 300.0);
        kf.predict();
        kf.update(&poly, &config);
        assert!(kf.is_initialized());

        let (a, b, c) = kf.coefficients();
        assert!((a - 0.5).abs() < 0.1);
        assert!((b - 2.0).abs() < 0.5);
        assert!((c - 300.0).abs() < 10.0);
    }

    #[test]
    fn test_kf_smoothing() {
        // Feed noisy measurements; filter should smooth them
        let config = PolyKFConfig::default();
        let mut kf = LanePolyKF::new(&config);

        let true_c = 300.0;
        for i in 0..50 {
            let noise = ((i * 7 + 3) % 11) as f32 - 5.0; // deterministic noise
            let poly = make_poly(0.5, 2.0, true_c + noise);
            kf.predict();
            kf.update(&poly, &config);
        }

        let (_, _, c) = kf.coefficients();
        // After 50 frames the filter should converge close to the true value
        assert!(
            (c - true_c).abs() < 3.0,
            "Expected c ≈ {}, got {} (error={})",
            true_c,
            c,
            (c - true_c).abs()
        );
    }

    #[test]
    fn test_prediction_holds_shape() {
        // Initialize with a curved boundary, then predict for several frames
        let config = PolyKFConfig::default();
        let mut kf = LanePolyKF::new(&config);

        // Feed stable measurements to converge
        for _ in 0..30 {
            let poly = make_poly(10.0, -5.0, 300.0);
            kf.predict();
            kf.update(&poly, &config);
        }

        let (a0, b0, c0) = kf.coefficients();

        // Now predict-only for 10 frames (no ego motion)
        for _ in 0..10 {
            kf.predict_with_ego(0.0);
        }

        let (a1, b1, c1) = kf.coefficients();

        // Curvature and slope should be approximately preserved
        assert!(
            (a1 - a0).abs() < 1.0,
            "Curvature should be preserved: {} vs {}",
            a0,
            a1
        );
        assert!(
            (b1 - b0).abs() < 1.0,
            "Slope should be preserved: {} vs {}",
            b0,
            b1
        );
        // Position might drift slightly due to rate terms
    }

    #[test]
    fn test_confidence_decay_during_prediction() {
        let config = PolyKFConfig::default();
        let mut kf = LanePolyKF::new(&config);

        // Initialize and stabilize
        for _ in 0..30 {
            let poly = make_poly(0.0, 0.0, 300.0);
            kf.predict();
            kf.update(&poly, &config);
        }

        let conf_fresh = kf.confidence();
        assert!(
            conf_fresh > 0.8,
            "Fresh confidence should be high: {}",
            conf_fresh
        );

        // Predict-only for 10 frames
        for _ in 0..10 {
            kf.predict_with_ego(0.0);
        }

        let conf_stale = kf.confidence();
        assert!(
            conf_stale < conf_fresh,
            "Stale confidence ({}) should be < fresh ({})",
            conf_stale,
            conf_fresh
        );
        assert!(
            conf_stale > 0.0,
            "Should still have some confidence after 10 frames"
        );
    }

    #[test]
    fn test_ego_compensation() {
        let config = PolyKFConfig::default();
        let mut kf = LanePolyKF::new(&config);

        for _ in 0..30 {
            let poly = make_poly(0.0, 0.0, 300.0);
            kf.predict();
            kf.update(&poly, &config);
        }

        let (_, _, c_before) = kf.coefficients();

        // Ego moves right by 5px/frame for 5 frames
        for _ in 0..5 {
            kf.predict_with_ego(5.0);
        }

        let (_, _, c_after) = kf.coefficients();

        // Boundary should appear to shift left (c decreases)
        let shift = c_before - c_after;
        assert!(shift > 20.0, "Expected ~25px leftward shift, got {}", shift);
    }

    #[test]
    fn test_innovation_on_lane_change() {
        let config = PolyKFConfig::default();
        let mut kf = LanePolyKF::new(&config);

        // Stabilize at c = 300
        for _ in 0..30 {
            let poly = make_poly(0.0, 0.0, 300.0);
            kf.predict();
            kf.update(&poly, &config);
        }

        // Sudden jump to c = 350 (boundary moved 50px)
        let poly = make_poly(0.0, 0.0, 350.0);
        kf.predict();
        let innovation = kf.update(&poly, &config);

        // Innovation should be large in the 'c' component
        assert!(
            innovation.0[2].abs() > 30.0,
            "Expected large c-innovation on jump, got {}",
            innovation.0[2]
        );
    }

    #[test]
    fn test_tracker_curve_no_false_lane_change() {
        // Both boundaries curving right → should NOT suggest lane change
        let config = PolynomialTrackerConfig::default();
        let mut tracker = PolynomialBoundaryTracker::new(config);

        let ego_x = 640.0;

        // Feed matching curves for both boundaries
        for i in 0..50 {
            // Both boundaries curving identically, just offset in x
            let left = make_poly(10.0, -3.0, 400.0);
            let right = make_poly(10.0, -3.0, 900.0);
            tracker.update(Some(&left), Some(&right), 0.0, ego_x);
        }

        let signals = tracker.signals();
        assert!(
            !signals.suggests_lane_change,
            "Matching curves should NOT suggest lane change. \
             div={:.2} wrate={:.2} nf={:.2}",
            signals.boundary_velocity_divergence,
            signals.lane_width_rate,
            signals.near_far_offset_divergence,
        );
    }

    #[test]
    fn test_tracker_lane_change_divergence() {
        // Left boundary approaching, right boundary receding → lane change
        let config = PolynomialTrackerConfig {
            boundary_divergence_threshold: 1.0,
            lane_width_rate_threshold: 1.0,
            ..PolynomialTrackerConfig::default()
        };
        let mut tracker = PolynomialBoundaryTracker::new(config);

        let ego_x = 640.0;

        // First stabilize
        for _ in 0..30 {
            let left = make_poly(0.0, 0.0, 400.0);
            let right = make_poly(0.0, 0.0, 900.0);
            tracker.update(Some(&left), Some(&right), 0.0, ego_x);
        }

        // Now simulate lane change: left boundary approaches (c increases),
        // right boundary recedes (c increases too but faster)
        for i in 0..20 {
            let drift = i as f32 * 5.0;
            let left = make_poly(0.0, 0.0, 400.0 + drift * 0.3);
            let right = make_poly(0.0, 0.0, 900.0 + drift);
            tracker.update(Some(&left), Some(&right), 0.0, ego_x);
        }

        let signals = tracker.signals();
        // The divergence should be non-trivial (right boundary moving faster)
        assert!(
            signals.boundary_velocity_divergence.abs() > 0.5 || signals.lane_width_rate.abs() > 0.5,
            "Diverging boundaries should produce non-trivial signals. \
             div={:.2} wrate={:.2}",
            signals.boundary_velocity_divergence,
            signals.lane_width_rate,
        );
    }

    #[test]
    fn test_mat6_identity_mul() {
        let i = Mat6::identity();
        let result = i.mul(&i);
        for r in 0..6 {
            for c in 0..6 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (result.get(r, c) - expected).abs() < 1e-6,
                    "I*I [{},{}] = {} expected {}",
                    r,
                    c,
                    result.get(r, c),
                    expected
                );
            }
        }
    }

    #[test]
    fn test_mat3_invert() {
        // Simple diagonal matrix
        let m = Mat3([2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]);
        let inv = m.invert().unwrap();
        assert!((inv.get(0, 0) - 0.5).abs() < 1e-6);
        assert!((inv.get(1, 1) - 1.0 / 3.0).abs() < 1e-6);
        assert!((inv.get(2, 2) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_mat3_singular() {
        let m = Mat3([1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 1.0, 1.0]);
        assert!(m.invert().is_none());
    }
}

