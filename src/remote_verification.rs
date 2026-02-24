// src/remote_verification.rs
//
// v9.0: Remote server verification for maneuver events.
//
// When a maneuver is detected, this module selects 5 strategic frames
// from a ring buffer (covering the maneuver window) and sends them
// along with event metadata to a remote server for verification using
// a more capable model (e.g., a VLM).
//
// The server response can confirm or override the local legality decision.

use anyhow::{Context, Result};
use base64::Engine;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

use crate::analysis::maneuver_classifier::ManeuverEvent;

// ============================================================================
// FRAME RING BUFFER
// ============================================================================

/// Stores recent frames in a fixed-size ring buffer so that when an event fires
/// we can look back and pick frames at specific frame IDs.
pub struct FrameRingBuffer {
    /// (frame_id, timestamp_ms, JPEG-encoded bytes)
    entries: Vec<Option<FrameEntry>>,
    capacity: usize,
    /// Index where the next frame will be written.
    write_idx: usize,
}

#[derive(Clone)]
struct FrameEntry {
    frame_id: u64,
    timestamp_ms: f64,
    jpeg_bytes: Vec<u8>,
    width: usize,
    height: usize,
}

impl FrameRingBuffer {
    pub fn new(capacity: usize) -> Self {
        let mut entries = Vec::with_capacity(capacity);
        entries.resize_with(capacity, || None);
        Self {
            entries,
            capacity,
            write_idx: 0,
        }
    }

    /// Push a raw RGB frame into the buffer, encoding it to JPEG first.
    /// Returns false if JPEG encoding fails (frame is skipped).
    pub fn push_frame(
        &mut self,
        frame_id: u64,
        timestamp_ms: f64,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> bool {
        let jpeg_bytes = match encode_rgb_to_jpeg(rgb_data, width, height) {
            Some(bytes) => bytes,
            None => return false,
        };
        self.entries[self.write_idx] = Some(FrameEntry {
            frame_id,
            timestamp_ms,
            jpeg_bytes,
            width,
            height,
        });
        self.write_idx = (self.write_idx + 1) % self.capacity;
        true
    }

    /// Select 5 strategic frames for the given maneuver window.
    ///
    /// Strategy: pick frames closest to these temporal positions:
    ///   1. Maneuver start
    ///   2. 25% through
    ///   3. Midpoint (50%)
    ///   4. 75% through
    ///   5. Maneuver end
    ///
    /// If the maneuver is very short (< 5 frames), returns whatever is available.
    pub fn select_strategic_frames(&self, start_frame: u64, end_frame: u64) -> Vec<SelectedFrame> {
        // Collect all entries that fall within [start_frame, end_frame]
        let mut in_range: Vec<&FrameEntry> = self
            .entries
            .iter()
            .filter_map(|e| e.as_ref())
            .filter(|e| e.frame_id >= start_frame && e.frame_id <= end_frame)
            .collect();

        in_range.sort_by_key(|e| e.frame_id);

        if in_range.is_empty() {
            // Fallback: grab the most recent frames available
            return self.most_recent_frames(5);
        }

        if in_range.len() <= 5 {
            return in_range
                .iter()
                .map(|e| SelectedFrame {
                    frame_id: e.frame_id,
                    timestamp_ms: e.timestamp_ms,
                    jpeg_bytes: e.jpeg_bytes.clone(),
                    width: e.width,
                    height: e.height,
                })
                .collect();
        }

        // Pick 5 strategic positions
        let target_frames: Vec<u64> = (0..5)
            .map(|i| {
                let frac = i as f64 / 4.0;
                let f = start_frame as f64 + frac * (end_frame - start_frame) as f64;
                f.round() as u64
            })
            .collect();

        let mut selected = Vec::with_capacity(5);
        let mut used_indices = std::collections::HashSet::new();

        for target in &target_frames {
            // Find closest frame to this target
            let (best_idx, _best) = in_range
                .iter()
                .enumerate()
                .filter(|(idx, _)| !used_indices.contains(idx))
                .min_by_key(|(_, e)| (e.frame_id as i64 - *target as i64).unsigned_abs())
                .unwrap();

            used_indices.insert(best_idx);
            let e = &in_range[best_idx];
            selected.push(SelectedFrame {
                frame_id: e.frame_id,
                timestamp_ms: e.timestamp_ms,
                jpeg_bytes: e.jpeg_bytes.clone(),
                width: e.width,
                height: e.height,
            });
        }

        selected.sort_by_key(|s| s.frame_id);
        selected
    }

    fn most_recent_frames(&self, count: usize) -> Vec<SelectedFrame> {
        let mut entries: Vec<&FrameEntry> =
            self.entries.iter().filter_map(|e| e.as_ref()).collect();

        entries.sort_by_key(|e| std::cmp::Reverse(e.frame_id));
        entries.truncate(count);
        entries.reverse();

        entries
            .iter()
            .map(|e| SelectedFrame {
                frame_id: e.frame_id,
                timestamp_ms: e.timestamp_ms,
                jpeg_bytes: e.jpeg_bytes.clone(),
                width: e.width,
                height: e.height,
            })
            .collect()
    }
}

pub struct SelectedFrame {
    pub frame_id: u64,
    pub timestamp_ms: f64,
    pub jpeg_bytes: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

// ============================================================================
// REQUEST / RESPONSE TYPES
// ============================================================================

#[derive(Debug, Serialize)]
pub struct VerificationRequest {
    /// Unique event ID for correlation
    pub event_id: String,
    /// Maneuver metadata
    pub maneuver: ManeuverSummary,
    /// 5 strategic frames as base64 JPEG
    pub frames: Vec<FramePayload>,
}

#[derive(Debug, Serialize)]
pub struct ManeuverSummary {
    pub maneuver_type: String,
    pub side: String,
    pub local_legality: String,
    pub confidence: f32,
    pub start_ms: f64,
    pub end_ms: f64,
    pub duration_ms: f64,
    pub start_frame: u64,
    pub end_frame: u64,
    pub crossed_line_class: Option<String>,
    pub crossed_line_class_id: Option<usize>,
    pub is_on_curve: bool,
    pub road_classification: Option<RoadClassSummary>,
}

#[derive(Debug, Serialize)]
pub struct RoadClassSummary {
    pub center_line_class: Option<String>,
    pub passing_legality: String,
    pub is_passing_legal: bool,
    pub confidence: f32,
}

#[derive(Debug, Serialize)]
pub struct FramePayload {
    /// Position label: "start", "early", "mid", "late", "end"
    pub position: String,
    pub frame_id: u64,
    pub timestamp_ms: f64,
    /// Base64-encoded JPEG
    pub image_base64: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VerificationResponse {
    /// The event_id echoed back for correlation
    pub event_id: String,
    /// Server's determined line type (e.g., "solid_double_yellow", "dashed_single_yellow")
    pub verified_line_type: Option<String>,
    /// Server's determined class ID (4-10, 99)
    pub verified_class_id: Option<usize>,
    /// Server's legality verdict: "LEGAL", "ILLEGAL", "CRITICAL_ILLEGAL", "UNKNOWN"
    pub verified_legality: String,
    /// Server's confidence in its determination (0.0 - 1.0)
    pub confidence: f32,
    /// Whether the server agrees with the local detection
    pub agrees_with_local: bool,
    /// Optional explanation / reasoning from the model
    pub reasoning: Option<String>,
}

// ============================================================================
// REMOTE VERIFICATION CLIENT
// ============================================================================

pub struct RemoteVerificationClient {
    http_client: reqwest::Client,
    server_url: String,
    /// If true, send requests but don't block — fire & forget with logging.
    async_mode: bool,
    /// Minimum confidence to accept server's override.
    min_override_confidence: f32,
}

impl RemoteVerificationClient {
    pub fn new(
        server_url: String,
        timeout_secs: u64,
        async_mode: bool,
        min_override_confidence: f32,
    ) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self {
            http_client,
            server_url,
            async_mode,
            min_override_confidence,
        })
    }

    /// Build a VerificationRequest from a ManeuverEvent and selected frames.
    pub fn build_request(event: &ManeuverEvent, frames: &[SelectedFrame]) -> VerificationRequest {
        let position_labels = ["start", "early", "mid", "late", "end"];

        let frame_payloads: Vec<FramePayload> = frames
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let label = if i < position_labels.len() {
                    position_labels[i].to_string()
                } else {
                    format!("extra_{}", i)
                };
                FramePayload {
                    position: label,
                    frame_id: f.frame_id,
                    timestamp_ms: f.timestamp_ms,
                    image_base64: base64::engine::general_purpose::STANDARD.encode(&f.jpeg_bytes),
                }
            })
            .collect();

        let road_class =
            event
                .road_classification_at_maneuver
                .as_ref()
                .map(|rc| RoadClassSummary {
                    center_line_class: rc.center_line_class.clone(),
                    passing_legality: rc.passing_legality.clone(),
                    is_passing_legal: rc.is_passing_legal,
                    confidence: rc.confidence,
                });

        VerificationRequest {
            event_id: uuid::Uuid::new_v4().to_string(),
            maneuver: ManeuverSummary {
                maneuver_type: event.maneuver_type.as_str().to_string(),
                side: event.side.as_str().to_string(),
                local_legality: format!("{:?}", event.legality),
                confidence: event.confidence,
                start_ms: event.start_ms,
                end_ms: event.end_ms,
                duration_ms: event.duration_ms,
                start_frame: event.start_frame,
                end_frame: event.end_frame,
                crossed_line_class: event.crossed_line_class.clone(),
                crossed_line_class_id: event.crossed_line_class_id,
                is_on_curve: event.is_on_curve,
                road_classification: road_class,
            },
            frames: frame_payloads,
        }
    }

    /// Send verification request to the remote server.
    ///
    /// In async_mode, spawns a tokio task and returns None immediately.
    /// In sync mode, blocks until the server responds.
    pub async fn verify(&self, request: &VerificationRequest) -> Option<VerificationResponse> {
        let url = format!("{}/verify", self.server_url.trim_end_matches('/'));
        let frame_count = request.frames.len();
        let event_id = request.event_id.clone();

        info!(
            "🌐 Sending verification request {} ({} frames) to {}",
            event_id, frame_count, url,
        );

        match self.http_client.post(&url).json(request).send().await {
            Ok(response) => {
                if !response.status().is_success() {
                    error!(
                        "🌐 Verification server returned {}: {}",
                        response.status(),
                        response
                            .text()
                            .await
                            .unwrap_or_else(|_| "<no body>".to_string()),
                    );
                    return None;
                }

                match response.json::<VerificationResponse>().await {
                    Ok(vr) => {
                        info!(
                            "🌐 Verification response for {}: legality={}, confidence={:.2}, agrees={}",
                            vr.event_id, vr.verified_legality, vr.confidence, vr.agrees_with_local,
                        );
                        if let Some(ref reason) = vr.reasoning {
                            debug!("🌐 Server reasoning: {}", reason);
                        }
                        Some(vr)
                    }
                    Err(e) => {
                        error!("🌐 Failed to parse verification response: {}", e);
                        None
                    }
                }
            }
            Err(e) => {
                error!("🌐 Verification request failed: {}", e);
                None
            }
        }
    }

    /// Whether the server's response should override the local legality.
    pub fn should_override(&self, response: &VerificationResponse) -> bool {
        !response.agrees_with_local && response.confidence >= self.min_override_confidence
    }

    /// Parse the server's legality string into a LineLegality.
    pub fn parse_legality(legality_str: &str) -> crate::lane_legality::LineLegality {
        match legality_str.to_uppercase().as_str() {
            "LEGAL" => crate::lane_legality::LineLegality::Legal,
            "ILLEGAL" => crate::lane_legality::LineLegality::Illegal,
            "CRITICAL_ILLEGAL" => crate::lane_legality::LineLegality::CriticalIllegal,
            "CAUTION" => crate::lane_legality::LineLegality::Caution,
            _ => crate::lane_legality::LineLegality::Unknown,
        }
    }

    pub fn is_async(&self) -> bool {
        self.async_mode
    }

    pub fn server_url(&self) -> &str {
        &self.server_url
    }
}

// ============================================================================
// JPEG ENCODING HELPER
// ============================================================================

/// Encode raw RGB bytes into a JPEG. Returns None on failure.
fn encode_rgb_to_jpeg(rgb_data: &[u8], width: usize, height: usize) -> Option<Vec<u8>> {
    use image::{ImageBuffer, RgbImage};

    let img: RgbImage = ImageBuffer::from_raw(width as u32, height as u32, rgb_data.to_vec())?;

    let mut buf = std::io::Cursor::new(Vec::new());
    // Quality 80 is a good balance of size/quality for network transfer
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 80);
    img.write_with_encoder(encoder).ok()?;

    Some(buf.into_inner())
}

// ============================================================================
// VERIFIED EVENT (for JSONL output enrichment)
// ============================================================================

/// Enriches a ManeuverEvent with remote verification data for output.
#[derive(Debug, Clone, Serialize)]
pub struct VerifiedManeuverEvent {
    /// The original maneuver event (flattened)
    #[serde(flatten)]
    pub event: ManeuverEvent,
    /// Remote verification result (if available)
    pub remote_verification: Option<VerificationResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub event_id: String,
    pub server_legality: String,
    pub server_line_type: Option<String>,
    pub server_class_id: Option<usize>,
    pub server_confidence: f32,
    pub agrees_with_local: bool,
    pub was_overridden: bool,
    pub reasoning: Option<String>,
}

impl VerificationResult {
    pub fn from_response(response: &VerificationResponse, was_overridden: bool) -> Self {
        Self {
            event_id: response.event_id.clone(),
            server_legality: response.verified_legality.clone(),
            server_line_type: response.verified_line_type.clone(),
            server_class_id: response.verified_class_id,
            server_confidence: response.confidence,
            agrees_with_local: response.agrees_with_local,
            was_overridden,
            reasoning: response.reasoning.clone(),
        }
    }
}
