// server/main.go
// Lane Change Legality Analysis Server - Peru Traffic Rules v8.0
// ENHANCED: Strategic frame selection awareness + full pipeline metadata
// LLM Vision focused on LINE ANALYSIS — sensor data trusted as-is
// NEW: Per-frame capture reason context, road classification prior,
//      enriched prompts with trajectory/velocity/positioning data

package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/disintegration/imaging"
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
	openai "github.com/sashabaranov/go-openai"
)

// ============================================================================
// IMAGE PREPROCESSING
// ============================================================================

type ImageEnhancementConfig struct {
	ApplyBrightnessAdjustment bool
	BrightnessIncrease        float64
	ApplyContrastAdjustment   bool
	ContrastIncrease          float64
	ApplySharpen              bool
	SharpenAmount             float64
	ApplySaturation           bool
	SaturationIncrease        float64
	ApplyGammaCorrection      bool
	GammaValue                float64
	SendMultipleVersions      bool
}

func GetDefaultEnhancementConfig() ImageEnhancementConfig {
	return ImageEnhancementConfig{
		ApplyBrightnessAdjustment: true,
		BrightnessIncrease:        15.0,
		ApplyContrastAdjustment:   true,
		ContrastIncrease:          20.0,
		ApplySharpen:              true,
		SharpenAmount:             2.0,
		ApplySaturation:           true,
		SaturationIncrease:        25.0,
		ApplyGammaCorrection:      true,
		GammaValue:                1.2,
		SendMultipleVersions:      false,
	}
}

func GetNightEnhancementConfig() ImageEnhancementConfig {
	return ImageEnhancementConfig{
		ApplyBrightnessAdjustment: true,
		BrightnessIncrease:        30.0,
		ApplyContrastAdjustment:   true,
		ContrastIncrease:          35.0,
		ApplySharpen:              true,
		SharpenAmount:             3.0,
		ApplySaturation:           true,
		SaturationIncrease:        40.0,
		ApplyGammaCorrection:      true,
		GammaValue:                1.5,
		SendMultipleVersions:      true,
	}
}

func GetBrightEnhancementConfig() ImageEnhancementConfig {
	return ImageEnhancementConfig{
		ApplyBrightnessAdjustment: true,
		BrightnessIncrease:        -10.0,
		ApplyContrastAdjustment:   true,
		ContrastIncrease:          15.0,
		ApplySharpen:              true,
		SharpenAmount:             1.5,
		ApplySaturation:           true,
		SaturationIncrease:        15.0,
		ApplyGammaCorrection:      true,
		GammaValue:                0.9,
		SendMultipleVersions:      false,
	}
}

type ImageProcessor struct{}

func NewImageProcessor() *ImageProcessor {
	return &ImageProcessor{}
}

func (p *ImageProcessor) DecodeBase64Image(base64Str string) (image.Image, error) {
	if strings.HasPrefix(base64Str, "data:image") {
		parts := strings.Split(base64Str, ",")
		if len(parts) > 1 {
			base64Str = parts[1]
		}
	}
	imageData, err := base64.StdEncoding.DecodeString(base64Str)
	if err != nil {
		return nil, fmt.Errorf("failed to decode base64: %w", err)
	}
	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %w", err)
	}
	return img, nil
}

func (p *ImageProcessor) EncodeImageToBase64(img image.Image, quality int) (string, error) {
	buf := new(bytes.Buffer)
	opts := &jpeg.Options{Quality: quality}
	if err := jpeg.Encode(buf, img, opts); err != nil {
		return "", fmt.Errorf("failed to encode image: %w", err)
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes()), nil
}

func (p *ImageProcessor) EnhanceImage(img image.Image, config ImageEnhancementConfig) image.Image {
	enhanced := img
	if config.ApplyBrightnessAdjustment {
		enhanced = imaging.AdjustBrightness(enhanced, config.BrightnessIncrease)
	}
	if config.ApplyContrastAdjustment {
		enhanced = imaging.AdjustContrast(enhanced, config.ContrastIncrease)
	}
	if config.ApplySaturation {
		enhanced = imaging.AdjustSaturation(enhanced, config.SaturationIncrease)
	}
	if config.ApplyGammaCorrection {
		enhanced = imaging.AdjustGamma(enhanced, config.GammaValue)
	}
	if config.ApplySharpen {
		enhanced = imaging.Sharpen(enhanced, config.SharpenAmount)
	}
	return enhanced
}

func (p *ImageProcessor) ProcessFrameWithEnhancements(base64Image string, config ImageEnhancementConfig) ([]string, error) {
	img, err := p.DecodeBase64Image(base64Image)
	if err != nil {
		return nil, err
	}
	results := []string{}
	if !config.ApplyBrightnessAdjustment && !config.ApplyContrastAdjustment {
		results = append(results, base64Image)
		return results, nil
	}
	enhanced := p.EnhanceImage(img, config)
	enhancedBase64, err := p.EncodeImageToBase64(enhanced, 90)
	if err != nil {
		return nil, err
	}
	results = append(results, enhancedBase64)
	if config.SendMultipleVersions {
		results = append(results, base64Image)
	}
	return results, nil
}

func (p *ImageProcessor) DetectSceneType(base64Image string) string {
	img, err := p.DecodeBase64Image(base64Image)
	if err != nil {
		return "unknown"
	}
	bounds := img.Bounds()
	var totalBrightness uint64
	var pixelCount uint64
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			brightness := (0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)) / 256
			totalBrightness += uint64(brightness)
			pixelCount++
		}
	}
	avgBrightness := float64(totalBrightness) / float64(pixelCount)
	if avgBrightness < 80 {
		return "night"
	} else if avgBrightness > 180 {
		return "bright"
	}
	return "day"
}

// ============================================================================
// MODELS — COMPLETE MATCH WITH RUST PIPELINE v8.0
// ============================================================================

type LineCrossingInfo struct {
	LineCrossed             bool     `json:"line_crossed"`
	LineType                string   `json:"line_type"`
	IsLegal                 bool     `json:"is_legal"`
	Severity                string   `json:"severity"`
	LineDetectionConfidence float32  `json:"line_detection_confidence"`
	CrossedAtFrame          uint64   `json:"crossed_at_frame"`
	AdditionalLinesCrossed  []string `json:"additional_lines_crossed,omitempty"`
	AnalysisDetails         *string  `json:"analysis_details,omitempty"`
}

type TrajectoryInfo struct {
	InitialPosition      float32 `json:"initial_position"`
	FinalPosition        float32 `json:"final_position"`
	NetDisplacement      float32 `json:"net_displacement"`
	ReturnedToStart      bool    `json:"returned_to_start"`
	ExcursionSufficient  bool    `json:"excursion_sufficient"`
	ShapeScore           float32 `json:"shape_score"`
	Smoothness           float32 `json:"smoothness"`
	HasDirectionReversal bool    `json:"has_direction_reversal"`
}

type VelocityInfo struct {
	PeakLateralVelocity float32  `json:"peak_lateral_velocity"`
	AvgLateralVelocity  float32  `json:"avg_lateral_velocity"`
	VelocityPattern     string   `json:"velocity_pattern"`
	MaxAcceleration     *float32 `json:"max_acceleration,omitempty"`
}

type PositioningInfo struct {
	LaneWidthMin            float32 `json:"lane_width_min"`
	LaneWidthMax            float32 `json:"lane_width_max"`
	LaneWidthAvg            float32 `json:"lane_width_avg"`
	LaneWidthStable         bool    `json:"lane_width_stable"`
	AdjacentLanePenetration float32 `json:"adjacent_lane_penetration"`
	BaselineOffset          float32 `json:"baseline_offset"`
	BaselineFrozen          bool    `json:"baseline_frozen"`
}

type StateTransition struct {
	FromState   string  `json:"from_state"`
	ToState     string  `json:"to_state"`
	FrameID     uint64  `json:"frame_id"`
	TimestampMs float64 `json:"timestamp_ms"`
}

type TemporalInfo struct {
	TimeDriftingMs          *float64          `json:"time_drifting_ms,omitempty"`
	TimeCrossingMs          *float64          `json:"time_crossing_ms,omitempty"`
	TotalManeuverDurationMs float64           `json:"total_maneuver_duration_ms"`
	DurationPlausible       bool              `json:"duration_plausible"`
	StateProgression        []StateTransition `json:"state_progression"`
}

// v8.0: Road classification from temporal consensus
type RoadClassificationInfo struct {
	RoadType         string  `json:"road_type"`
	PassingLegality  string  `json:"passing_legality"`
	MixedLineSide    *string `json:"mixed_line_side,omitempty"`
	EstimatedLanes   uint32  `json:"estimated_lanes"`
	Confidence       float32 `json:"confidence"`
}

type DetectionMetadata struct {
	DetectionConfidence float32  `json:"detection_confidence"`
	MaxOffsetNormalized float32  `json:"max_offset_normalized"`
	AvgLaneConfidence   float32  `json:"avg_lane_confidence"`
	BothLanesRatio      float32  `json:"both_lanes_ratio"`
	VideoResolution     string   `json:"video_resolution"`
	FPS                 float32  `json:"fps"`
	Region              string   `json:"region"`
	AvgLaneWidthPx      *float32 `json:"avg_lane_width_px,omitempty"`

	CurveDetected     bool    `json:"curve_detected"`
	CurveAngleDegrees float32 `json:"curve_angle_degrees"`
	CurveConfidence   float32 `json:"curve_confidence"`
	CurveType         string  `json:"curve_type"`

	ShadowOvertakeDetected bool     `json:"shadow_overtake_detected"`
	ShadowOvertakeCount    uint32   `json:"shadow_overtake_count"`
	ShadowWorstSeverity    string   `json:"shadow_worst_severity"`
	ShadowBlockingVehicles []string `json:"shadow_blocking_vehicles"`

	LineCrossingInfo *LineCrossingInfo `json:"line_crossing_info,omitempty"`

	VehiclesOvertakenCount uint32   `json:"vehicles_overtaken_count"`
	OvertakenVehicleTypes  []string `json:"overtaken_vehicle_types,omitempty"`
	OvertakenVehicleIDs    []uint32 `json:"overtaken_vehicle_ids,omitempty"`

	ManeuverType     string  `json:"maneuver_type"`
	IncompleteReason *string `json:"incomplete_reason,omitempty"`

	TrajectoryInfo  TrajectoryInfo  `json:"trajectory_info"`
	VelocityInfo    VelocityInfo    `json:"velocity_info"`
	PositioningInfo PositioningInfo `json:"positioning_info"`

	DetectionPath *string `json:"detection_path,omitempty"`

	TemporalInfo TemporalInfo `json:"temporal_info"`

	// v8.0: Road classification from temporal consensus system
	RoadClassification *RoadClassificationInfo `json:"road_classification,omitempty"`
}

type LaneChangeLegalityRequest struct {
	EventID                string            `json:"event_id"`
	Direction              string            `json:"direction"`
	StartFrameID           uint64            `json:"start_frame_id"`
	EndFrameID             uint64            `json:"end_frame_id"`
	VideoTimestampMs       float64           `json:"video_timestamp_ms"`
	DurationMs             *float64          `json:"duration_ms,omitempty"`
	SourceID               string            `json:"source_id"`
	Frames                 []FrameData       `json:"frames"`
	DetectionMetadata      DetectionMetadata `json:"detection_metadata"`
	EnableImageEnhancement bool              `json:"enable_image_enhancement,omitempty"`
	EnhancementMode        string            `json:"enhancement_mode,omitempty"`
}

type FrameData struct {
	FrameIndex        int      `json:"frame_index"`
	TimestampMs       float64  `json:"timestamp_ms"`
	Width             int      `json:"width"`
	Height            int      `json:"height"`
	Base64Image       string   `json:"base64_image"`
	LaneConfidence    *float32 `json:"lane_confidence,omitempty"`
	OffsetPercentage  *float32 `json:"offset_percentage,omitempty"`
	// v8.0: Strategic capture metadata
	CaptureReason     *string  `json:"capture_reason,omitempty"`
	LeftMarkingClass  *string  `json:"left_marking_class,omitempty"`
	RightMarkingClass *string  `json:"right_marking_class,omitempty"`
}

type LaneChangeLegalityResponse struct {
	EventID string `json:"event_id"`
	Status  string `json:"status"`
	Message string `json:"message"`
}

type LaneAnalysisResult struct {
	IsLegal      bool    `json:"is_legal"`
	ManeuverType string  `json:"maneuver_type"`
	Confidence   float32 `json:"confidence"`
	Reasoning    string  `json:"reasoning"`

	LineColor         string `json:"line_color"`
	LinePattern       string `json:"line_pattern"`
	LineStructure     string `json:"line_structure"`
	NearLanePattern   string `json:"near_lane_pattern"`
	FarLanePattern    string `json:"far_lane_pattern"`
	ApplicablePattern string `json:"applicable_pattern"`
	CrossingLine      string `json:"crossing_line"`

	IsShadowOvertaking   bool   `json:"is_shadow_overtaking"`
	ShadowVehicleAhead   string `json:"shadow_vehicle_ahead,omitempty"`
	ShadowDistanceMeters int    `json:"shadow_distance_meters,omitempty"`

	VehiclesOvertaken    int    `json:"vehicles_overtaken"`
	VehicleCountEvidence string `json:"vehicle_count_evidence,omitempty"`

	RoadCurvature struct {
		IsBlindCurve      bool   `json:"is_blind_curve"`
		VisualDescription string `json:"visual_description"`
	} `json:"road_curvature"`

	IllegalReasons []string `json:"illegal_reasons"`
	IsInCurve      bool     `json:"is_in_curve"`

	AgreesWithOnDeviceModel bool   `json:"agrees_with_on_device_model"`
	OnDeviceLineType        string `json:"on_device_line_type,omitempty"`
	DisagreementReason      string `json:"disagreement_reason,omitempty"`

	// v8.0: Road classification agreement
	AgreesWithRoadClassifier bool   `json:"agrees_with_road_classifier,omitempty"`
	RoadClassifierDisagreement string `json:"road_classifier_disagreement,omitempty"`

	ImageEnhancementApplied bool   `json:"image_enhancement_applied,omitempty"`
	SceneType               string `json:"scene_type,omitempty"`
	VersionsAnalyzed        int    `json:"versions_analyzed,omitempty"`
}

type AnalysisLogEntry struct {
	EventID           string    `json:"event_id"`
	Timestamp         time.Time `json:"timestamp"`
	ProcessingTimeSec float64   `json:"processing_time_sec"`
	ModelUsed         string    `json:"model_used"`
	Direction         string    `json:"direction"`

	LaneAnalysisResult

	OnDeviceDetection  *LineCrossingInfo `json:"on_device_detection,omitempty"`
	SensorVehicleCount uint32            `json:"sensor_vehicle_count"`
	SensorShadow       bool              `json:"sensor_shadow"`
	SensorCurve        bool              `json:"sensor_curve"`
	SensorCurveAngle   float32           `json:"sensor_curve_angle"`
	ManeuverTypeInput  string            `json:"maneuver_type_input"`
	DurationMs         *float64          `json:"duration_ms,omitempty"`
	MaxOffset          float32           `json:"max_offset"`
}

// ============================================================================
// OPENAI SERVICE
// ============================================================================

type OpenAIService struct {
	client         *openai.Client
	model          string
	imageProcessor *ImageProcessor
}

func NewOpenAIService(apiKey, model string) *OpenAIService {
	return &OpenAIService{
		client:         openai.NewClient(apiKey),
		model:          model,
		imageProcessor: NewImageProcessor(),
	}
}

func (s *OpenAIService) AnalyzeLaneChange(ctx context.Context, req *LaneChangeLegalityRequest) (*LaneAnalysisResult, error) {
	content := []openai.ChatMessagePart{
		{
			Type: openai.ChatMessagePartTypeText,
			Text: s.buildPrompt(req),
		},
	}

	var enhancementConfig ImageEnhancementConfig
	sceneType := "day"

	if req.EnableImageEnhancement {
		if len(req.Frames) > 0 {
			sceneType = s.imageProcessor.DetectSceneType(req.Frames[0].Base64Image)
			log.Printf("Scene detected: %s", sceneType)
		}

		switch req.EnhancementMode {
		case "night":
			enhancementConfig = GetNightEnhancementConfig()
		case "bright":
			enhancementConfig = GetBrightEnhancementConfig()
		case "day":
			enhancementConfig = GetDefaultEnhancementConfig()
		default:
			switch sceneType {
			case "night":
				enhancementConfig = GetNightEnhancementConfig()
			case "bright":
				enhancementConfig = GetBrightEnhancementConfig()
			default:
				enhancementConfig = GetDefaultEnhancementConfig()
			}
		}
	}

	maxImages := 7
	if len(req.Frames) < maxImages {
		maxImages = len(req.Frames)
	}

	totalVersions := 0

	for i := 0; i < maxImages; i++ {
		frame := req.Frames[i]

		// v8.0: Add per-frame context label so LLM knows what each image shows
		frameLabel := ""
		if frame.CaptureReason != nil {
			frameLabel = fmt.Sprintf("[Frame %d — %s]", i, *frame.CaptureReason)
		} else {
			frameLabel = fmt.Sprintf("[Frame %d]", i)
		}

		// Add per-frame marking context if available
		if frame.LeftMarkingClass != nil || frame.RightMarkingClass != nil {
			left := "?"
			right := "?"
			if frame.LeftMarkingClass != nil {
				left = *frame.LeftMarkingClass
			}
			if frame.RightMarkingClass != nil {
				right = *frame.RightMarkingClass
			}
			frameLabel += fmt.Sprintf(" Marcas detectadas: Izq=%s, Der=%s", left, right)
		}

		if frame.LaneConfidence != nil {
			frameLabel += fmt.Sprintf(" (confianza: %.0f%%)", *frame.LaneConfidence*100)
		}

		content = append(content, openai.ChatMessagePart{
			Type: openai.ChatMessagePartTypeText,
			Text: frameLabel,
		})

		var imagesToSend []string
		if req.EnableImageEnhancement {
			processed, err := s.imageProcessor.ProcessFrameWithEnhancements(frame.Base64Image, enhancementConfig)
			if err != nil {
				imagesToSend = []string{frame.Base64Image}
			} else {
				imagesToSend = processed
			}
		} else {
			imagesToSend = []string{frame.Base64Image}
		}

		for _, imgBase64 := range imagesToSend {
			imageURL := fmt.Sprintf("data:image/jpeg;base64,%s", imgBase64)
			content = append(content, openai.ChatMessagePart{
				Type: openai.ChatMessagePartTypeImageURL,
				ImageURL: &openai.ChatMessageImageURL{
					URL:    imageURL,
					Detail: openai.ImageURLDetailHigh,
				},
			})
			totalVersions++
		}
	}

	log.Printf("Total images sent: %d (Model: %s)", totalVersions, s.model)

	chatReq := openai.ChatCompletionRequest{
		Model: s.model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: s.getSystemPrompt(),
			},
			{
				Role:         openai.ChatMessageRoleUser,
				MultiContent: content,
			},
		},
		ResponseFormat: &openai.ChatCompletionResponseFormat{
			Type: openai.ChatCompletionResponseFormatTypeJSONObject,
		},
	}

	resp, err := s.client.CreateChatCompletion(ctx, chatReq)
	if err != nil {
		return nil, fmt.Errorf("OpenAI API error: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no response from OpenAI")
	}

	log.Printf("Tokens - Input: %d, Output: %d", resp.Usage.PromptTokens, resp.Usage.CompletionTokens)

	result, err := s.parseResponse(resp.Choices[0].Message.Content)
	if err != nil {
		return nil, err
	}

	result = s.mergeWithSensorData(result, req)
	result.ImageEnhancementApplied = req.EnableImageEnhancement
	result.SceneType = sceneType
	result.VersionsAnalyzed = totalVersions

	return result, nil
}

// ============================================================================
// MERGE SENSOR DATA — trust mathematical detections
// ============================================================================

func (s *OpenAIService) mergeWithSensorData(result *LaneAnalysisResult, req *LaneChangeLegalityRequest) *LaneAnalysisResult {
	meta := req.DetectionMetadata

	if meta.ShadowOvertakeDetected {
		result.IsShadowOvertaking = true
		if len(meta.ShadowBlockingVehicles) > 0 {
			result.ShadowVehicleAhead = strings.Join(meta.ShadowBlockingVehicles, ", ")
		}
		shadowReason := fmt.Sprintf("Shadow overtake: %d vehiculo(s) bloqueando visibilidad (%s)",
			meta.ShadowOvertakeCount, meta.ShadowWorstSeverity)
		if !containsReason(result.IllegalReasons, "shadow") && !containsReason(result.IllegalReasons, "sombra") {
			result.IllegalReasons = append(result.IllegalReasons, shadowReason)
		}
		result.IsLegal = false
	}

	if meta.VehiclesOvertakenCount > 0 {
		result.VehiclesOvertaken = int(meta.VehiclesOvertakenCount)
	}

	if meta.CurveDetected {
		result.IsInCurve = true
		result.RoadCurvature.IsBlindCurve = meta.CurveType == "SHARP"
		result.RoadCurvature.VisualDescription = fmt.Sprintf(
			"Curva %s detectada por sensor: %.1f grados (confianza %.0f%%)",
			meta.CurveType, meta.CurveAngleDegrees, meta.CurveConfidence*100)

		if meta.CurveType == "SHARP" && strings.ToUpper(req.Direction) == "LEFT" {
			curveReason := fmt.Sprintf("Adelantamiento en curva %s de %.1f grados (Art. 90 DS 016-2009-MTC)",
				meta.CurveType, meta.CurveAngleDegrees)
			if !containsReason(result.IllegalReasons, "curva") {
				result.IllegalReasons = append(result.IllegalReasons, curveReason)
			}
			result.IsLegal = false
		}
	}

	if meta.LineCrossingInfo != nil {
		result.OnDeviceLineType = meta.LineCrossingInfo.LineType
	}

	if len(result.IllegalReasons) > 0 {
		result.IsLegal = false
	}

	return result
}

func containsReason(reasons []string, keyword string) bool {
	lower := strings.ToLower(keyword)
	for _, r := range reasons {
		if strings.Contains(strings.ToLower(r), lower) {
			return true
		}
	}
	return false
}

// ============================================================================
// SYSTEM PROMPT — v8.0: Enriched with full pipeline context awareness
// ============================================================================

func (s *OpenAIService) getSystemPrompt() string {
	return `Eres un experto perito en normativa vial del Peru (DS 016-2009-MTC).
Tu trabajo PRINCIPAL es analizar las LINEAS DE LA VIA en imagenes de dashcam.

Los datos de curva, vehiculos adelantados y shadow overtake ya vienen confirmados
por sensores matematicos. NO los re-evalues. Solo analiza las LINEAS.

CONTEXTO DE LAS IMAGENES:
Las imagenes que recibes fueron capturadas ESTRATEGICAMENTE en momentos clave:
- "pre_crossing": Justo ANTES de cruzar la linea (vista limpia desde carril original)
- "at_crossing": DURANTE el cruce de la linea (contacto con la marca vial)
- "peak_offset": En el punto de MAXIMA desviacion lateral (vista desde carril contrario)
- "return_crossing": Durante el RETORNO al carril original
- "post_maneuver": Despues de completar la maniobra
- "curve_context": Cuando se detecto curvatura en la via
- "vehicle_context": Mostrando vehiculos adelantados
- "periodic": Captura periodica durante la maniobra

IMPORTANTE: Cada imagen viene etiquetada con su razon de captura y las marcas viales
detectadas por el modelo YOLOv8-seg en ese frame exacto. Usa esta informacion como
contexto adicional para tu analisis visual.

REGLAS PERU (conduccion por la DERECHA):
- Linea doble mixta: el conductor se rige por la linea de SU lado (DERECHO).
- Linea SEGMENTADA de tu lado = LEGAL cruzar.
- Linea CONTINUA de tu lado = ILEGAL cruzar.
- Linea amarilla doble continua = SIEMPRE ILEGAL.
- Adelantamiento en curva cerrada = ILEGAL (Art. 90 DS 016-2009-MTC).

CLASIFICACION DE LINEAS (nombres del modelo YOLOv8-seg):
- solid_single_yellow (class 5): Linea continua amarilla simple -> ILEGAL
- solid_single_white (class 4): Linea continua blanca simple -> ILEGAL
- solid_double_yellow (class 8): Doble linea continua amarilla -> ILEGAL CRITICO
- dashed_single_white (class 9): Linea segmentada blanca -> LEGAL
- dashed_single_yellow (class 10): Linea segmentada amarilla -> LEGAL
- mixed_double_yellow_dashed_right (class 99): Doble mixta, segmentada de tu lado -> LEGAL
- mixed_double_yellow_solid_right (class 99): Doble mixta, continua de tu lado -> ILEGAL

SISTEMA DE CLASIFICACION TEMPORAL (RoadClassifier):
El pipeline Rust mantiene un consenso temporal de 30 frames sobre el tipo de via.
Si se proporciona, usa esta informacion como PRIOR fuerte:
- Si el clasificador dice "mixed_dashed_right" con alta confianza, la linea
  probablemente ES mixta con segmentada de tu lado, incluso si en una imagen
  individual parece solida (por perspectiva, sombras, o angulo de camara).

FORMATO JSON OBLIGATORIO:
{
  "is_legal": boolean,
  "maneuver_type": "ADELANTAMIENTO" | "RETORNO_CARRIL",
  "confidence": 0.0-1.0,
  "reasoning": "Analisis visual detallado de las lineas",

  "line_color": "AMARILLA" | "BLANCA",
  "line_pattern": "CONTINUA" | "SEGMENTADA",
  "line_structure": "SIMPLE" | "DOBLE" | "DOBLE_MIXTA",

  "near_lane_pattern": "Descripcion de la LINEA DEL LADO DERECHO (tu lado)",
  "far_lane_pattern": "Descripcion de la LINEA DEL LADO IZQUIERDO (contrario)",
  "applicable_pattern": "La linea que rige al conductor (la de tu lado derecho)",
  "crossing_line": "Tipo exacto de linea cruzada segun la clasificacion YOLOv8",

  "road_curvature": {
     "is_blind_curve": boolean,
     "visual_description": "lo que ves de la curvatura en las imagenes"
  },

  "illegal_reasons": [],

  "agrees_with_on_device_model": boolean,
  "on_device_line_type": "tipo que reporto el sensor YOLOv8-seg",
  "disagreement_reason": "si no coincide, explica que ves diferente en las imagenes",

  "agrees_with_road_classifier": boolean,
  "road_classifier_disagreement": "si no coincide con el clasificador temporal"
}`
}

// ============================================================================
// PROMPT BUILDER — v8.0: Full pipeline context
// ============================================================================

func (s *OpenAIService) buildPrompt(req *LaneChangeLegalityRequest) string {
	meta := req.DetectionMetadata

	// ── Maneuver context ──
	maneuverCtx := ""
	lineInstruction := ""
	if strings.ToUpper(req.Direction) == "LEFT" {
		maneuverCtx = "El vehiculo ADELANTA (sale del carril derecho al izquierdo para rebasar)."
		lineInstruction = "Analiza la LINEA DERECHA del eje central (la de tu carril original). Esta es la que determina la legalidad."
	} else {
		maneuverCtx = "El vehiculo RETORNA (vuelve del carril izquierdo al derecho despues de rebasar)."
		lineInstruction = "Analiza la LINEA DERECHA del eje central (la del carril al que vuelve). Para el retorno, la legalidad ya fue determinada por el cruce inicial."
	}

	// ── On-device line detection prior ──
	onDevicePrior := ""
	if meta.LineCrossingInfo != nil {
		lci := meta.LineCrossingInfo
		legalWord := "ILEGAL"
		if lci.IsLegal {
			legalWord = "LEGAL"
		}
		analysisDetail := ""
		if lci.AnalysisDetails != nil {
			analysisDetail = *lci.AnalysisDetails
		}
		onDevicePrior = fmt.Sprintf(`
DETECCION DEL MODELO ON-DEVICE (YOLOv8-seg, confianza %.0f%%):
   Tipo de linea detectado: "%s"
   Veredicto automatico: %s (%s)
   Detalles: %s

   -> Si CONFIRMAS este tipo de linea visualmente, pon agrees_with_on_device_model=true.
   -> Si ves algo DIFERENTE, pon agrees_with_on_device_model=false y explica en disagreement_reason.

   TIPOS CONOCIDOS del modelo (referencia):
   - solid_single_yellow / solid_double_yellow -> ILEGAL
   - dashed_single_white / dashed_single_yellow -> LEGAL
   - mixed_double_yellow_dashed_right -> LEGAL (segmentada de tu lado)
   - mixed_double_yellow_solid_right -> ILEGAL (continua de tu lado)`,
			lci.LineDetectionConfidence*100, lci.LineType, legalWord, lci.Severity, analysisDetail)
	} else {
		onDevicePrior = `
DETECCION ON-DEVICE: El modelo de segmentacion NO pudo identificar la linea.
   Debes hacer el analisis visual COMPLETO. Pon agrees_with_on_device_model=false.`
	}

	// ── v8.0: Road classification prior from temporal consensus ──
	roadClassPrior := ""
	if meta.RoadClassification != nil {
		rc := meta.RoadClassification
		mixedSide := "N/A"
		if rc.MixedLineSide != nil {
			mixedSide = *rc.MixedLineSide
		}
		roadClassPrior = fmt.Sprintf(`
CLASIFICACION TEMPORAL DE VIA (consenso de 30 frames, confianza %.0f%%):
   Tipo de via: %s
   Legalidad de paso: %s
   Lado de linea mixta: %s
   Carriles estimados: %d

   Este clasificador analiza multiples frames temporalmente. Si su confianza es
   alta (>70%%), es un PRIOR FUERTE. Usalo para resolver ambiguedades visuales.
   Si la imagen individual parece contradecir el clasificador temporal, considera
   que puede ser un efecto de perspectiva, sombra, o angulo de camara.`,
			rc.Confidence*100, rc.RoadType, rc.PassingLegality, mixedSide, rc.EstimatedLanes)
	}

	// ── Sensor facts (NOT for LLM to dispute) ──
	sensorFacts := ""

	if meta.CurveDetected {
		sensorFacts += fmt.Sprintf(`
CURVA CONFIRMADA POR SENSOR: %s, angulo %.1f grados (confianza %.0f%%).
   En curvas, las lineas pueden parecer distorsionadas por la perspectiva.
   Presta atencion especial a la forma de la linea en las imagenes "curve_context".
   Si es curva SHARP y el vehiculo adelanta, es ILEGAL por Art. 90.`,
			meta.CurveType, meta.CurveAngleDegrees, meta.CurveConfidence*100)
	}

	if meta.ShadowOvertakeDetected {
		vehicles := strings.Join(meta.ShadowBlockingVehicles, ", ")
		sensorFacts += fmt.Sprintf(`
SHADOW OVERTAKE CONFIRMADO POR SENSOR: %d vehiculo(s) bloqueando (%s). Severidad: %s.
   Esto ya marca la maniobra como ilegal. No lo disputes.`,
			meta.ShadowOvertakeCount, vehicles, meta.ShadowWorstSeverity)
	}

	if meta.VehiclesOvertakenCount > 0 {
		types := strings.Join(meta.OvertakenVehicleTypes, ", ")
		sensorFacts += fmt.Sprintf(`
VEHICULOS ADELANTADOS (YOLO tracking): %d -> %s. Confirmado por tracking.`,
			meta.VehiclesOvertakenCount, types)
	}

	// ── v8.0: Trajectory context ──
	trajectoryCtx := ""
	ti := meta.TrajectoryInfo
	if ti.ExcursionSufficient {
		trajectoryCtx = fmt.Sprintf(`
TRAYECTORIA:
   Posicion inicial: %.2f | Posicion final: %.2f | Desplazamiento neto: %.2f
   Retorno al inicio: %v | Excursion suficiente: %v
   Puntuacion de forma: %.2f | Suavidad: %.2f | Cambio de direccion: %v`,
			ti.InitialPosition, ti.FinalPosition, ti.NetDisplacement,
			ti.ReturnedToStart, ti.ExcursionSufficient,
			ti.ShapeScore, ti.Smoothness, ti.HasDirectionReversal)
	}

	// ── v8.0: Temporal context ──
	temporalCtx := ""
	temp := meta.TemporalInfo
	if temp.TotalManeuverDurationMs > 0 {
		temporalCtx = fmt.Sprintf(`
TEMPORAL:
   Duracion total: %.0fms | Plausible: %v
   Progresion de estados: %d transiciones`,
			temp.TotalManeuverDurationMs, temp.DurationPlausible, len(temp.StateProgression))
	}

	// ── Quality context ──
	qualityCtx := fmt.Sprintf(`
CALIDAD DE DETECCION:
   Confianza general: %.0f%% | Ambos carriles vistos: %.0f%% | Max offset: %.0f%%
   Ancho de carril promedio: %.0fpx | Resolucion: %s | FPS: %.0f`,
		meta.DetectionConfidence*100, meta.BothLanesRatio*100, meta.MaxOffsetNormalized*100,
		ptrToFloat(meta.AvgLaneWidthPx), meta.VideoResolution, meta.FPS)

	if meta.DetectionConfidence < 0.5 || meta.BothLanesRatio < 0.5 {
		qualityCtx += "\n   ADVERTENCIA: Calidad baja. Confia MAS en tu analisis visual que en el modelo."
	}

	// ── v8.0: Frame capture strategy explanation ──
	frameCtx := fmt.Sprintf(`
IMAGENES PROPORCIONADAS: %d frames capturados estrategicamente.
   Cada imagen esta etiquetada con su razon de captura y las marcas viales
   detectadas en ese momento exacto por YOLOv8-seg.

   ESTRATEGIA DE ANALISIS RECOMENDADA:
   1. Empieza con "pre_crossing" — vista limpia de la linea desde el carril original
   2. Confirma con "at_crossing" — la linea durante el cruce
   3. Verifica con "peak_offset" — vista desde el carril contrario (perspectiva diferente)
   4. Si hay "curve_context" — evalua como la curvatura afecta la apariencia de la linea
   5. Compara todas las vistas para determinar el tipo REAL de linea`, len(req.Frames))

	return fmt.Sprintf(`Eres perito de transito en Peru. Se conduce por la DERECHA.

CONTEXTO DE LA MANIOBRA:
%s
%s

%s

%s

%s
%s
%s

%s

%s

TU TAREA — ANALISIS DE LINEAS EN MULTIPLES PERSPECTIVAS:

1. Revisa TODAS las imagenes en orden. Cada una muestra un momento diferente de la maniobra.
2. Ubica el EJE CENTRAL de la via. En carreteras peruanas suele estar marcado con linea amarilla.
3. Si es DOBLE LINEA, identifica:
   A. [LINEA DERECHA]: La que colinda con el carril derecho (sentido normal del conductor).
   B. [LINEA IZQUIERDA]: La que colinda con el carril izquierdo (sentido contrario).
4. Para cada imagen, determina si la [LINEA DERECHA] es CONTINUA o SEGMENTADA.
   - SEGMENTADA (con espacios/interrupciones) -> LEGAL cruzar
   - CONTINUA (sin interrupciones) -> ILEGAL cruzar
5. Si diferentes imagenes muestran resultados diferentes (ej: una parece continua, otra segmentada),
   da MAS PESO a las imagenes "pre_crossing" y "at_crossing" porque tienen la vista mas directa.
6. Compara tu conclusion con la deteccion del modelo on-device Y el clasificador temporal.
7. Si hay curvatura, ten en cuenta que las lineas pueden aparecer distorsionadas por perspectiva.

IMPORTANTE:
- Para DOBLE MIXTA, lo que importa es TU LADO (derecho). Si tu lado es segmentado = LEGAL.
- Si el auto retorna (movimiento a derecha), la legalidad ya se determino en el cruce inicial.
- El clasificador temporal (si esta disponible) es un PRIOR FUERTE — tiene informacion de 30 frames.

Responde SOLO en JSON con el formato indicado en las instrucciones del sistema.`,
		maneuverCtx, lineInstruction, onDevicePrior, roadClassPrior,
		sensorFacts, trajectoryCtx, temporalCtx, qualityCtx, frameCtx)
}

func ptrToFloat(p *float32) float32 {
	if p == nil {
		return 0.0
	}
	return *p
}

// ============================================================================
// RESPONSE PARSING
// ============================================================================

func (s *OpenAIService) parseResponse(response string) (*LaneAnalysisResult, error) {
	response = strings.TrimSpace(response)
	if idx := strings.Index(response, "{"); idx != -1 {
		response = response[idx:]
	}
	if idx := strings.LastIndex(response, "}"); idx != -1 {
		response = response[:idx+1]
	}

	var result LaneAnalysisResult
	if err := json.Unmarshal([]byte(response), &result); err != nil {
		log.Printf("Error parseando JSON: %v. Raw: %s", err, truncate(response, 200))
		return nil, err
	}

	if result.IllegalReasons == nil {
		result.IllegalReasons = []string{}
	}
	if result.ApplicablePattern == "" && result.NearLanePattern != "" {
		result.ApplicablePattern = result.NearLanePattern
	}
	result.IsInCurve = result.RoadCurvature.IsBlindCurve

	return &result, nil
}

// ============================================================================
// DISPLAY
// ============================================================================

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}

func printWrapped(text string, maxLen int, prefix string) {
	words := strings.Fields(text)
	line := ""
	for _, word := range words {
		if len(line)+len(word)+1 > maxLen {
			fmt.Printf("%s%-67s|\n", prefix, line)
			line = word
		} else {
			if line == "" {
				line = word
			} else {
				line += " " + word
			}
		}
	}
	if line != "" {
		fmt.Printf("%s%-67s|\n", prefix, line)
	}
}

func printDetailedResult(req LaneChangeLegalityRequest, result *LaneAnalysisResult, elapsed time.Duration) {
	meta := req.DetectionMetadata

	fmt.Println()
	fmt.Println("+-----------------------------------------------------------------------+")
	fmt.Println("|        RESULTADO DE ANALISIS v8.0 - Transito Peru                     |")
	fmt.Println("+-----------------------------------------------------------------------+")

	fmt.Printf("|  Event:     %-57s|\n", truncate(req.EventID, 57))

	dirDisplay := "-> DERECHA"
	if strings.ToUpper(req.Direction) == "LEFT" {
		dirDisplay = "<- IZQUIERDA"
	}
	fmt.Printf("|  Direccion: %-57s|\n", dirDisplay)

	fmt.Println("+-----------------------------------------------------------------------+")

	if result.IsLegal {
		fmt.Println("|           [OK] M A N I O B R A   L E G A L                           |")
	} else {
		fmt.Println("|          [!!] M A N I O B R A   I L E G A L                           |")
	}

	fmt.Println("+-----------------------------------------------------------------------+")
	fmt.Println("|  ANALISIS DE LINEAS (LLM Vision):                                     |")

	fmt.Printf("|     Color:        %-53s|\n", result.LineColor)
	fmt.Printf("|     Estructura:   %-53s|\n", result.LineStructure)

	if result.LineStructure == "DOBLE" || result.LineStructure == "DOBLE_MIXTA" {
		fmt.Println("|     -------------------------------------------------------------------|")
		fmt.Printf("|     Linea Der. (Tuya): %-48s|\n", result.NearLanePattern)
		fmt.Printf("|     Linea Izq. (Contra): %-46s|\n", result.FarLanePattern)
		fmt.Println("|     -------------------------------------------------------------------|")
		applicableStr := result.ApplicablePattern
		if strings.Contains(strings.ToUpper(result.ApplicablePattern), "SEGMENTADA") {
			applicableStr += " (OK Permite cruzar)"
		} else {
			applicableStr += " (!! Prohibe cruzar)"
		}
		fmt.Printf("|     APLICA:      %-53s|\n", applicableStr)
	} else {
		fmt.Printf("|     Patron:       %-53s|\n", result.LinePattern)
	}

	if result.OnDeviceLineType != "" {
		fmt.Println("|     -------------------------------------------------------------------|")
		agreeText := "COINCIDE"
		if !result.AgreesWithOnDeviceModel {
			agreeText = "DISCREPA"
		}
		fmt.Printf("|     Sensor: %-53s|\n", result.OnDeviceLineType)
		fmt.Printf("|     LLM %s con sensor on-device%-32s|\n", agreeText, "")
		if result.DisagreementReason != "" {
			fmt.Printf("|     Razon: %-54s|\n", truncate(result.DisagreementReason, 54))
		}
	}

	fmt.Println("+-----------------------------------------------------------------------+")
	fmt.Println("|  DATOS DE SENSORES (matematicos, confiables):                         |")

	if meta.CurveDetected {
		fmt.Printf("|     CURVA: %s %.1f grados (conf: %.0f%%)%-26s|\n",
			meta.CurveType, meta.CurveAngleDegrees, meta.CurveConfidence*100, "")
	} else {
		fmt.Printf("|     Curva: No detectada%-46s|\n", "")
	}

	if meta.ShadowOvertakeDetected {
		fmt.Printf("|     SHADOW: %d vehiculo(s), sev: %-35s|\n",
			meta.ShadowOvertakeCount, meta.ShadowWorstSeverity)
	}

	if meta.VehiclesOvertakenCount > 0 {
		types := strings.Join(meta.OvertakenVehicleTypes, ", ")
		fmt.Printf("|     Vehiculos: %d (%s)%-38s|\n",
			meta.VehiclesOvertakenCount, truncate(types, 30), "")
	}

	fmt.Printf("|     Conf: %.0f%% | Offset: %.0f%% | Carriles: %.0f%%%-22s|\n",
		meta.DetectionConfidence*100, meta.MaxOffsetNormalized*100, meta.BothLanesRatio*100, "")

	if len(result.IllegalReasons) > 0 {
		fmt.Println("+-----------------------------------------------------------------------+")
		fmt.Println("|  RAZONES ILEGALIDAD:                                                  |")
		for _, reason := range result.IllegalReasons {
			fmt.Printf("|     - %-64s|\n", truncate(reason, 64))
		}
	}

	fmt.Println("+-----------------------------------------------------------------------+")
	fmt.Println("|  RAZONAMIENTO:                                                        |")
	printWrapped(result.Reasoning, 67, "|     ")

	fmt.Println("+-----------------------------------------------------------------------+")

	status := "LEGAL"
	if !result.IsLegal {
		status = "ILEGAL"
	}
	log.Printf("RESULT: %s | Conf: %.0f%% | Agrees: %v | Curve: %v | Shadow: %v | Vehicles: %d | %s",
		status, result.Confidence*100, result.AgreesWithOnDeviceModel,
		meta.CurveDetected, meta.ShadowOvertakeDetected,
		meta.VehiclesOvertakenCount, req.EventID)
}

func printErrorResult(eventID, direction string, err error, elapsed time.Duration) {
	fmt.Println("\n+-----------------------------------------------------------------------+")
	fmt.Println("|                      ERROR EN ANALISIS                                |")
	fmt.Printf("|  Event ID: %-59s|\n", truncate(eventID, 59))
	fmt.Printf("|  Error:    %-59s|\n", truncate(err.Error(), 59))
	fmt.Println("+-----------------------------------------------------------------------+")
}

// ============================================================================
// HANDLERS
// ============================================================================

type Handler struct {
	openai *OpenAIService
}

func (h *Handler) AnalyzeLaneChange(c *fiber.Ctx) error {
	var req LaneChangeLegalityRequest
	if err := c.BodyParser(&req); err != nil {
		log.Printf("Error parseando request: %v", err)
		return c.JSON(LaneChangeLegalityResponse{
			EventID: "",
			Status:  "error",
			Message: "Request invalido",
		})
	}

	if req.EnhancementMode == "" {
		req.EnableImageEnhancement = true
		req.EnhancementMode = "auto"
	}

	enhancementStatus := "off"
	if req.EnableImageEnhancement {
		enhancementStatus = "on"
	}

	curveStatus := ""
	if req.DetectionMetadata.CurveDetected {
		curveStatus = fmt.Sprintf(" | CURVE %s %.1f deg",
			req.DetectionMetadata.CurveType, req.DetectionMetadata.CurveAngleDegrees)
	}
	shadowStatus := ""
	if req.DetectionMetadata.ShadowOvertakeDetected {
		shadowStatus = fmt.Sprintf(" | SHADOW %dx%s",
			req.DetectionMetadata.ShadowOvertakeCount, req.DetectionMetadata.ShadowWorstSeverity)
	}
	onDeviceStatus := "none"
	if req.DetectionMetadata.LineCrossingInfo != nil {
		onDeviceStatus = fmt.Sprintf("%s(%.0f%%)",
			req.DetectionMetadata.LineCrossingInfo.LineType,
			req.DetectionMetadata.LineCrossingInfo.LineDetectionConfidence*100)
	}
	vehicleStatus := ""
	if req.DetectionMetadata.VehiclesOvertakenCount > 0 {
		vehicleStatus = fmt.Sprintf(" | VEH %d", req.DetectionMetadata.VehiclesOvertakenCount)
	}
	roadClassStatus := ""
	if req.DetectionMetadata.RoadClassification != nil {
		rc := req.DetectionMetadata.RoadClassification
		roadClassStatus = fmt.Sprintf(" | ROAD %s/%s(%.0f%%)",
			rc.RoadType, rc.PassingLegality, rc.Confidence*100)
	}

	// v8.0: Log capture reasons for strategic frames
	captureReasons := []string{}
	for _, f := range req.Frames {
		if f.CaptureReason != nil {
			captureReasons = append(captureReasons, *f.CaptureReason)
		}
	}

	log.Printf("RECV %s | %s | Frames: %d [%s] | Enh: %s %s | Line: %s | Conf: %.0f%%%s%s%s%s",
		req.EventID, req.Direction, len(req.Frames),
		strings.Join(captureReasons, ","),
		enhancementStatus, req.EnhancementMode, onDeviceStatus,
		req.DetectionMetadata.DetectionConfidence*100,
		curveStatus, shadowStatus, vehicleStatus, roadClassStatus)

	if len(req.Frames) == 0 {
		return c.JSON(LaneChangeLegalityResponse{
			EventID: req.EventID,
			Status:  "error",
			Message: "No hay frames",
		})
	}

	go analyzeAsync(h.openai, req)

	return c.JSON(LaneChangeLegalityResponse{
		EventID: req.EventID,
		Status:  "processing",
		Message: fmt.Sprintf("Analizando %s con %d frames estrategicos.", req.Direction, len(req.Frames)),
	})
}

func (h *Handler) HealthCheck(c *fiber.Ctx) error {
	return c.JSON(fiber.Map{
		"status":  "ok",
		"model":   h.openai.model,
		"version": "8.0",
		"features": []string{
			"Strategic frame capture (pre/at/peak/return/post crossing)",
			"Full pipeline metadata ingestion",
			"On-device model fusion (YOLOv8-seg prior)",
			"Road classification temporal consensus prior",
			"Sensor-trusted: shadow, vehicles, curve",
			"LLM focused: line type confirmation with multi-perspective",
			"Per-frame marking context + capture reason labels",
			"Trajectory, velocity, positioning, temporal data",
		},
	})
}

// ============================================================================
// MAIN
// ============================================================================

func main() {
	port := getEnv("PORT", "3000")
	openaiKey := os.Getenv("OPENAI_API_KEY")
	openaiModel := getEnv("OPENAI_MODEL", "gpt-4o")

	if openaiKey == "" {
		log.Fatal("OPENAI_API_KEY required")
	}

	openaiService := NewOpenAIService(openaiKey, openaiModel)
	handler := &Handler{openai: openaiService}

	app := fiber.New(fiber.Config{
		AppName:   "Lane Legality Server v8.0",
		BodyLimit: 150 * 1024 * 1024,
	})

	app.Use(recover.New())
	app.Use(logger.New(logger.Config{
		Format:     "${time} | ${status} | ${latency} | ${method} ${path}\n",
		TimeFormat: "15:04:05",
	}))
	app.Use(cors.New())

	app.Get("/health", handler.HealthCheck)
	app.Post("/api/analyze", handler.AnalyzeLaneChange)

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-quit
		log.Println("Shutting down...")
		app.Shutdown()
	}()

	fmt.Println()
	fmt.Println("+-----------------------------------------------------------------------+")
	fmt.Println("|     Lane Legality Server v8.0 - Peru                                  |")
	fmt.Println("|     LLM Vision: Multi-Perspective Line Confirmation                   |")
	fmt.Println("+-----------------------------------------------------------------------+")
	fmt.Printf("|  Port:  %-62s|\n", port)
	fmt.Printf("|  Model: %-62s|\n", openaiModel)
	fmt.Println("|  Strategy:                                                            |")
	fmt.Println("|    Shadow/Vehicles/Curve -> Trusted from sensors                      |")
	fmt.Println("|    Lane line type -> LLM confirms YOLOv8-seg + RoadClassifier         |")
	fmt.Println("|    Frames -> Strategically captured at key maneuver moments            |")
	fmt.Println("+-----------------------------------------------------------------------+")
	fmt.Println()

	if err := app.Listen(":" + port); err != nil {
		log.Fatalf("Error: %v", err)
	}
}

func getEnv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// ============================================================================
// ASYNC ANALYSIS + LOGGING
// ============================================================================

func appendResultToJSONL(entry AnalysisLogEntry) {
	f, err := os.OpenFile("server_results.jsonl", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Printf("Failed to open log file: %v", err)
		return
	}
	defer f.Close()

	jsonData, err := json.Marshal(entry)
	if err == nil {
		if _, err := f.Write(jsonData); err != nil {
			log.Printf("Failed to write to log: %v", err)
		}
		f.WriteString("\n")
	} else {
		log.Printf("Failed to marshal log entry: %v", err)
	}
}

func analyzeAsync(openaiService *OpenAIService, req LaneChangeLegalityRequest) {
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()

	startTime := time.Now()
	meta := req.DetectionMetadata

	log.Printf("ANALYZING [%s] %s | Frames: %d | Conf: %.0f%% | Offset: %.0f%%",
		req.EventID, req.Direction, len(req.Frames),
		meta.DetectionConfidence*100, meta.MaxOffsetNormalized*100)

	result, err := openaiService.AnalyzeLaneChange(ctx, &req)
	elapsed := time.Since(startTime)

	if err != nil {
		log.Printf("ERROR [%s]: %v", req.EventID, err)
		printErrorResult(req.EventID, req.Direction, err, elapsed)
		return
	}

	logEntry := AnalysisLogEntry{
		EventID:           req.EventID,
		Timestamp:         time.Now(),
		ProcessingTimeSec: elapsed.Seconds(),
		ModelUsed:         openaiService.model,
		Direction:         req.Direction,

		LaneAnalysisResult: *result,

		OnDeviceDetection:  meta.LineCrossingInfo,
		SensorVehicleCount: meta.VehiclesOvertakenCount,
		SensorShadow:       meta.ShadowOvertakeDetected,
		SensorCurve:        meta.CurveDetected,
		SensorCurveAngle:   meta.CurveAngleDegrees,
		ManeuverTypeInput:  meta.ManeuverType,
		DurationMs:         req.DurationMs,
		MaxOffset:          meta.MaxOffsetNormalized,
	}

	appendResultToJSONL(logEntry)
	printDetailedResult(req, result, elapsed)
}
