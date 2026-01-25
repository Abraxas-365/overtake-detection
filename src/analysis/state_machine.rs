// src/analysis/state_machine.rs

use super::boundary_detector::CrossingType;
use super::curve_detector::CurveDetector;
use super::velocity_tracker::LateralVelocityTracker;
use crate::types::{Direction, LaneChangeConfig, LaneChangeEvent, LaneChangeState, VehicleState};
use tracing::{debug, info, warn};

pub struct LaneChangeStateMachine {
    config: LaneChangeConfig,
    source_id: String,
    state: LaneChangeState,
    frames_in_state: u32,
    pending_state: Option<LaneChangeState>,
    pending_frames: u32,
    change_direction: Direction,
    change_start_frame: Option<u64>,
    change_start_time: Option<f64>,
    cooldown_remaining: u32,
    max_offset_in_change: f32,
    total_frames_processed: u64,

    // Baseline tracking
    offset_history: Vec<f32>,
    baseline_offset: f32,
    baseline_samples: Vec<f32>,
    is_baseline_established: bool,
    frames_since_baseline: u32,
    stable_centered_frames: u32,

    // Enhanced detectors
    curve_detector: CurveDetector,
    velocity_tracker: LateralVelocityTracker,

    // ============================================================================
    // 🆕 NUEVOS CAMPOS PARA DETECCIÓN DE ESTABILIZACIÓN
    // ============================================================================
    /// Contador de frames consecutivos donde la desviación se mantiene estable
    stable_deviation_frames: u32,
    /// Última desviación registrada para calcular el cambio
    last_deviation: f32,
    /// Historial de desviaciones recientes para detección de estabilización
    recent_deviations: Vec<f32>,
}

impl LaneChangeStateMachine {
    pub fn new(config: LaneChangeConfig) -> Self {
        Self {
            config,
            source_id: String::new(),
            state: LaneChangeState::Centered,
            frames_in_state: 0,
            pending_state: None,
            pending_frames: 0,
            change_direction: Direction::Unknown,
            change_start_frame: None,
            change_start_time: None,
            cooldown_remaining: 0,
            max_offset_in_change: 0.0,
            total_frames_processed: 0,
            offset_history: Vec::with_capacity(60),
            baseline_offset: 0.0,
            baseline_samples: Vec::with_capacity(90),
            is_baseline_established: false,
            frames_since_baseline: 0,
            stable_centered_frames: 0,
            curve_detector: CurveDetector::new(),
            velocity_tracker: LateralVelocityTracker::new(),
            // 🆕 Inicializar nuevos campos
            stable_deviation_frames: 0,
            last_deviation: 0.0,
            recent_deviations: Vec::with_capacity(30),
        }
    }

    pub fn current_state(&self) -> &str {
        self.state.as_str()
    }

    pub fn update_curve_detector(&mut self, lanes: &[crate::types::Lane]) -> bool {
        self.curve_detector.is_in_curve(lanes)
    }

    pub fn update(
        &mut self,
        vehicle_state: &VehicleState,
        frame_id: u64,
        timestamp_ms: f64,
        crossing_type: CrossingType,
    ) -> Option<LaneChangeEvent> {
        self.total_frames_processed += 1;

        // Skip initial frames
        if self.total_frames_processed < self.config.skip_initial_frames {
            return None;
        }

        // Handle cooldown
        if self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
            if self.cooldown_remaining == 0 {
                self.state = LaneChangeState::Centered;
                self.frames_in_state = 0;
            }
            return None;
        }

        // Check timeout
        if self.state == LaneChangeState::Drifting || self.state == LaneChangeState::Crossing {
            if let Some(start_time) = self.change_start_time {
                let elapsed = timestamp_ms - start_time;
                if elapsed > self.config.max_duration_ms {
                    warn!("⏰ Lane change timeout after {:.0}ms", elapsed);
                    self.reset_lane_change();
                    self.cooldown_remaining = 30;
                    return None;
                }
            }
        }

        if !vehicle_state.is_valid() {
            return None;
        }

        let lane_width = vehicle_state.lane_width.unwrap();
        let normalized_offset = vehicle_state.lateral_offset / lane_width;
        let abs_offset = normalized_offset.abs();

        // Update velocity tracker
        let lateral_velocity = self
            .velocity_tracker
            .get_velocity(vehicle_state.lateral_offset, timestamp_ms);

        // Update offset history
        self.offset_history.push(normalized_offset);
        if self.offset_history.len() > 60 {
            self.offset_history.remove(0);
        }

        // PHASE 1: Establish baseline
        if !self.is_baseline_established {
            if abs_offset < 0.15 {
                self.baseline_samples.push(normalized_offset);
                self.stable_centered_frames += 1;
            } else {
                if self.stable_centered_frames < 30 {
                    self.baseline_samples.clear();
                    self.stable_centered_frames = 0;
                }
            }

            if self.baseline_samples.len() >= 60 && self.stable_centered_frames >= 60 {
                let mut sorted = self.baseline_samples.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                self.baseline_offset = sorted[sorted.len() / 2];
                self.is_baseline_established = true;
                self.frames_since_baseline = 0;
                info!(
                    "✅ Baseline established: {:.3} ({:.1}%) at frame {} ({:.1}s)",
                    self.baseline_offset,
                    self.baseline_offset * 100.0,
                    frame_id,
                    timestamp_ms / 1000.0
                );
            }
            return None;
        }

        self.frames_since_baseline += 1;

        if self.frames_since_baseline < 30 {
            return None;
        }

        // Calculate deviation from baseline
        let deviation = (normalized_offset - self.baseline_offset).abs();

        // Track max offset during lane change
        if self.state == LaneChangeState::Drifting || self.state == LaneChangeState::Crossing {
            if deviation > self.max_offset_in_change {
                self.max_offset_in_change = deviation;
            }
        }

        // 🆕 Actualizar historial de desviaciones para detección de estabilización
        self.recent_deviations.push(deviation);
        if self.recent_deviations.len() > 30 {
            self.recent_deviations.remove(0);
        }

        // Track stable centered
        if deviation < 0.10 {
            self.stable_centered_frames += 1;
        } else {
            self.stable_centered_frames = 0;
        }

        let direction = Direction::from_offset(normalized_offset - self.baseline_offset);

        // Determine target state with enhanced logic
        let target_state = self.determine_target_state(deviation, crossing_type, lateral_velocity);

        debug!(
            "F{}: offset={:.1}%, dev={:.1}%, vel={:.1}px/s, cross={:?}, state={:?}→{:?}",
            frame_id,
            normalized_offset * 100.0,
            deviation * 100.0,
            lateral_velocity,
            crossing_type,
            self.state,
            target_state
        );

        self.check_transition(target_state, direction, frame_id, timestamp_ms)
    }

    fn reset_lane_change(&mut self) {
        self.state = LaneChangeState::Centered;
        self.frames_in_state = 0;
        self.pending_state = None;
        self.pending_frames = 0;
        self.change_direction = Direction::Unknown;
        self.change_start_frame = None;
        self.change_start_time = None;
        self.max_offset_in_change = 0.0;
        // 🆕 Resetear campos de estabilización
        self.stable_deviation_frames = 0;
        self.last_deviation = 0.0;
        self.recent_deviations.clear();
    }

    // ============================================================================
    // 🆕 NUEVA FUNCIÓN: Detectar si la desviación se ha estabilizado
    // ============================================================================
    /// Determina si la desviación se ha mantenido estable durante suficientes frames
    ///
    /// Criterios:
    /// 1. Suficiente historial (mínimo 15 frames)
    /// 2. La desviación no cambia más de STABLE_THRESHOLD entre frames consecutivos
    /// 3. El rango (max - min) en la ventana reciente es pequeño
    fn is_deviation_stable(&self) -> bool {
        const MIN_HISTORY_SIZE: usize = 15;
        const STABLE_THRESHOLD: f32 = 0.03; // 3% de cambio máximo
        const MAX_RANGE: f32 = 0.08; // 8% de rango máximo

        if self.recent_deviations.len() < MIN_HISTORY_SIZE {
            return false;
        }

        // Tomar los últimos N frames
        let window_size = 15;
        let recent = &self.recent_deviations[self.recent_deviations.len() - window_size..];

        // Calcular rango (max - min)
        let max_dev = recent
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let min_dev = recent
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let range = max_dev - min_dev;

        // Verificar que el rango sea pequeño
        if range > MAX_RANGE {
            return false;
        }

        // Verificar que los cambios frame-a-frame sean pequeños
        let mut large_changes = 0;
        for window in recent.windows(2) {
            let change = (window[1] - window[0]).abs();
            if change > STABLE_THRESHOLD {
                large_changes += 1;
            }
        }

        // Permitir máximo 2 cambios grandes en la ventana
        large_changes <= 2
    }

    // ============================================================================
    // 🔧 FUNCIÓN MODIFICADA: Lógica mejorada para CROSSING → COMPLETED
    // ============================================================================
    fn determine_target_state(
        &mut self,
        deviation: f32,
        crossing_type: CrossingType,
        lateral_velocity: f32,
    ) -> LaneChangeState {
        // Minimum lateral velocity required (pixels per second)
        const MIN_LATERAL_VELOCITY: f32 = 30.0;

        match self.state {
            LaneChangeState::Centered => {
                // PRIMARY: Boundary crossing detection with velocity
                if crossing_type != CrossingType::None
                    && lateral_velocity.abs() > MIN_LATERAL_VELOCITY
                {
                    if self.is_deviation_sustained(self.config.drift_threshold * 0.9) {
                        info!(
                            "🚨 Lane change trigger: boundary crossing {:?} + velocity {:.1}px/s",
                            crossing_type, lateral_velocity
                        );
                        return LaneChangeState::Drifting;
                    }
                }

                // FALLBACK: Very large deviation even without boundary crossing
                if deviation >= 0.55 {
                    if self.is_deviation_sustained(self.config.drift_threshold)
                        && lateral_velocity.abs() > 20.0
                    {
                        info!(
                            "🚨 Lane change trigger (fallback): large deviation {:.1}% + velocity {:.1}px/s",
                            deviation * 100.0, lateral_velocity
                        );
                        return LaneChangeState::Drifting;
                    }
                }

                // SECONDARY: Medium-high deviation with sustained movement
                if deviation >= self.config.drift_threshold + 0.10
                    && lateral_velocity.abs() > MIN_LATERAL_VELOCITY
                {
                    if self.is_deviation_sustained(self.config.drift_threshold) {
                        info!(
                            "🚨 Lane change trigger (deviation-based): dev={:.1}% + velocity {:.1}px/s",
                            deviation * 100.0, lateral_velocity
                        );
                        return LaneChangeState::Drifting;
                    }
                }

                LaneChangeState::Centered
            }
            LaneChangeState::Drifting => {
                // Transición a CROSSING si supera el umbral
                if deviation >= self.config.crossing_threshold {
                    return LaneChangeState::Crossing;
                }

                // Regresar a CENTERED si la desviación cae mucho (cancelación)
                if deviation < self.config.drift_threshold * 0.5 {
                    if self.max_offset_in_change >= self.config.crossing_threshold {
                        // Ya cruzó, completar
                        return LaneChangeState::Completed;
                    } else {
                        // No cruzó lo suficiente, cancelar
                        warn!(
                            "❌ Lane change cancelled: max dev {:.1}% < threshold {:.1}%",
                            self.max_offset_in_change * 100.0,
                            self.config.crossing_threshold * 100.0
                        );
                        return LaneChangeState::Centered;
                    }
                }

                LaneChangeState::Drifting
            }
            // ============================================================================
            // 🆕 LÓGICA MEJORADA: CROSSING → COMPLETED con detección de estabilización
            // ============================================================================
            LaneChangeState::Crossing => {
                // Actualizar contador de estabilización
                let deviation_change = (deviation - self.last_deviation).abs();
                const FRAME_TO_FRAME_THRESHOLD: f32 = 0.03; // 3%

                if deviation_change < FRAME_TO_FRAME_THRESHOLD {
                    self.stable_deviation_frames += 1;
                } else {
                    self.stable_deviation_frames = 0;
                }

                self.last_deviation = deviation;

                // CRITERIO 1: Estabilización detectada + desviación razonable
                // El vehículo se ha estabilizado en el nuevo carril
                if self.is_deviation_stable() && deviation < 0.35 {
                    // Máximo 35% de desviación permitida
                    info!(
                        "✅ Lane change completing: stabilized at {:.1}% deviation (stable for {} frames)",
                        deviation * 100.0,
                        self.stable_deviation_frames
                    );
                    return LaneChangeState::Completed;
                }

                // CRITERIO 2: Volvió muy cerca del centro (retorno rápido)
                // Esto cubre casos donde el vehículo vuelve casi al centro original
                if deviation < self.config.drift_threshold * 0.5 {
                    info!(
                        "✅ Lane change completing: returned close to center ({:.1}%)",
                        deviation * 100.0
                    );
                    return LaneChangeState::Completed;
                }

                // CRITERIO 3: Estabilización prolongada incluso con desviación mayor
                // Si ha estado estable por MUCHO tiempo (30+ frames), probablemente completó
                if self.stable_deviation_frames >= 30 && deviation < 0.45 {
                    // Máximo 45%
                    info!(
                        "✅ Lane change completing: prolonged stability ({} frames) at {:.1}%",
                        self.stable_deviation_frames,
                        deviation * 100.0
                    );
                    return LaneChangeState::Completed;
                }

                // Seguir en CROSSING
                LaneChangeState::Crossing
            }
            LaneChangeState::Completed => LaneChangeState::Centered,
        }
    }

    fn is_deviation_sustained(&self, threshold: f32) -> bool {
        if self.offset_history.len() < 8 {
            return false;
        }

        let high_count = self
            .offset_history
            .iter()
            .rev()
            .take(6)
            .filter(|o| (*o - self.baseline_offset).abs() >= threshold)
            .count();

        high_count >= 5
    }

    fn check_transition(
        &mut self,
        target_state: LaneChangeState,
        direction: Direction,
        frame_id: u64,
        timestamp_ms: f64,
    ) -> Option<LaneChangeEvent> {
        if target_state == self.state {
            self.pending_state = None;
            self.pending_frames = 0;
            self.frames_in_state += 1;
            return None;
        }

        if self.pending_state == Some(target_state) {
            self.pending_frames += 1;
        } else {
            self.pending_state = Some(target_state);
            self.pending_frames = 1;
        }

        if self.pending_frames < self.config.min_frames_confirm {
            return None;
        }

        self.execute_transition(target_state, direction, frame_id, timestamp_ms)
    }

    fn execute_transition(
        &mut self,
        target_state: LaneChangeState,
        direction: Direction,
        frame_id: u64,
        timestamp_ms: f64,
    ) -> Option<LaneChangeEvent> {
        let from_state = self.state;

        info!(
            "State: {:?} → {:?} at frame {} ({:.2}s)",
            from_state,
            target_state,
            frame_id,
            timestamp_ms / 1000.0
        );

        // Starting a lane change
        if target_state == LaneChangeState::Drifting && from_state == LaneChangeState::Centered {
            self.change_direction = direction;
            self.change_start_frame = Some(frame_id);
            self.change_start_time = Some(timestamp_ms);
            self.max_offset_in_change = 0.0;
            // 🆕 Resetear contadores de estabilización al iniciar
            self.stable_deviation_frames = 0;
            self.last_deviation = 0.0;
            info!(
                "🚗 Lane change started: {} at {:.2}s",
                direction.as_str(),
                timestamp_ms / 1000.0
            );
        }

        // Cancellation
        if target_state == LaneChangeState::Centered && from_state == LaneChangeState::Drifting {
            info!("↩️ Lane change cancelled (returned to center)");
            self.reset_lane_change();
            self.cooldown_remaining = 30;
            return None;
        }

        let duration_ms = if target_state == LaneChangeState::Completed {
            self.change_start_time.map(|start| timestamp_ms - start)
        } else {
            None
        };

        self.state = target_state;
        self.frames_in_state = 0;
        self.pending_state = None;
        self.pending_frames = 0;

        // Handle completion
        if target_state == LaneChangeState::Completed {
            // Validation 1: Duration
            if let Some(dur) = duration_ms {
                if dur < self.config.min_duration_ms {
                    warn!(
                        "❌ Rejected: too short ({:.0}ms < {:.0}ms)",
                        dur, self.config.min_duration_ms
                    );
                    self.reset_lane_change();
                    self.cooldown_remaining = 60;
                    return None;
                }
            }

            // Validation 2: Max offset must have crossed threshold
            if self.max_offset_in_change < self.config.crossing_threshold {
                warn!(
                    "❌ Rejected: max deviation {:.1}% < crossing threshold {:.1}%",
                    self.max_offset_in_change * 100.0,
                    self.config.crossing_threshold * 100.0
                );
                self.reset_lane_change();
                self.cooldown_remaining = 60;
                return None;
            }

            self.cooldown_remaining = self.config.cooldown_frames;

            let start_frame = self.change_start_frame.unwrap_or(frame_id);
            let start_time = self.change_start_time.unwrap_or(timestamp_ms);
            let confidence = self.calculate_confidence(duration_ms);

            let mut event = LaneChangeEvent::new(
                start_time,
                start_frame,
                frame_id,
                self.change_direction,
                confidence,
            );
            event.duration_ms = duration_ms;
            event.source_id = self.source_id.clone();

            info!(
                "✅ LANE CHANGE CONFIRMED: {} started at {:.2}s, ended at {:.2}s (duration: {:.0}ms, max_dev: {:.1}%)",
                event.direction_name(),
                start_time / 1000.0,
                timestamp_ms / 1000.0,
                duration_ms.unwrap_or(0.0),
                self.max_offset_in_change * 100.0
            );

            // Reset baseline after lane change
            self.baseline_offset = 0.0;
            self.baseline_samples.clear();
            self.is_baseline_established = false;
            self.stable_centered_frames = 0;
            self.frames_since_baseline = 0;

            self.reset_lane_change();
            return Some(event);
        }

        None
    }

    fn calculate_confidence(&self, duration_ms: Option<f64>) -> f32 {
        let mut confidence: f32 = 0.5;

        if self.max_offset_in_change > 0.60 {
            confidence += 0.25;
        } else if self.max_offset_in_change > 0.50 {
            confidence += 0.15;
        } else {
            confidence += 0.05;
        }

        if let Some(dur) = duration_ms {
            if dur > 1500.0 && dur < 4000.0 {
                confidence += 0.15;
            } else if dur > 1200.0 && dur < 6000.0 {
                confidence += 0.05;
            }
        }

        confidence.min(0.95)
    }

    pub fn reset(&mut self) {
        self.reset_lane_change();
        self.cooldown_remaining = 0;
        self.total_frames_processed = 0;
        self.offset_history.clear();
        self.baseline_offset = 0.0;
        self.baseline_samples.clear();
        self.is_baseline_established = false;
        self.frames_since_baseline = 0;
        self.stable_centered_frames = 0;
        self.curve_detector.reset();
        self.velocity_tracker.reset();
        // 🆕 Resetear campos de estabilización
        self.stable_deviation_frames = 0;
        self.last_deviation = 0.0;
        self.recent_deviations.clear();
    }

    pub fn set_source_id(&mut self, source_id: String) {
        self.source_id = source_id;
    }
}
