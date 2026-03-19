"""
api/schemas.py
---------------
Pydantic schemas for the competition API.

Structure:
  - Generic schemas (flexible, works as-is for exploration)
  - Task-specific schemas (filled in on competition day when specs are released)

Convention:
  All competition tasks follow the pattern:
    Input:  { "id": str, ...task-specific fields... }
    Output: { "id": str, ...predictions + optional metadata... }

  The "id" field is mandatory — organisers use it to match predictions to cases.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Optional, Union


# ===========================================================================
# SHARED
# ===========================================================================

class HealthResponse(BaseModel):
    status: str
    message: str
    version: str = "1.0"


class PredictionMetadata(BaseModel):
    """Optional metadata returned alongside predictions. Useful for debugging."""
    model_name: Optional[str] = None
    latency_ms: Optional[float] = None
    confidence: Optional[float] = None


# ===========================================================================
# TASK 1 — Tabular / Classification
# ---------------------------------------------------------------------------
# AINM 2025 analogue: Race Car (numeric state vector → discrete action)
# Adapt feature names and prediction fields on competition day.
# ===========================================================================

class Task1Input(BaseModel):
    id: str = Field(..., description="Unique sample identifier")
    features: List[float] = Field(..., description="Numeric feature vector")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "sample_001",
            "features": [0.1, 0.5, -0.3, 1.2, 0.0]
        }
    })


class Task1Output(BaseModel):
    id: str
    prediction: Union[int, str, float] = Field(..., description="Predicted class or value")
    confidence: Optional[float] = None
    meta: Optional[PredictionMetadata] = None


# ===========================================================================
# TASK 2 — Language / RAG / Classification
# ---------------------------------------------------------------------------
# AINM 2025 analogue: Emergency Healthcare RAG
#   - is_correct: bool          (true/false statement classification)
#   - category: str             (one of 115 medical themes)
# ===========================================================================

class Task2Input(BaseModel):
    id: str = Field(..., description="Unique sample identifier")
    text: str = Field(..., description="Input text to classify or query")
    context: Optional[str] = Field(None, description="Optional supporting context for RAG queries")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "case_042",
            "text": "Severe chest pain radiating to left arm with diaphoresis.",
            "context": None
        }
    })


class Task2Output(BaseModel):
    id: str
    label: str = Field(..., description="Primary predicted label or category")
    is_correct: Optional[bool] = Field(None, description="For true/false classification tasks")
    answer: Optional[str] = Field(None, description="For RAG-style open-ended questions")
    meta: Optional[PredictionMetadata] = None


# ===========================================================================
# TASK 3 — Vision / Segmentation / Detection
# ---------------------------------------------------------------------------
# AINM 2025 analogue: Tumor Segmentation in MIP-PET images
#   - mask: list[list[int]]   (2D binary mask, 0=background 1=tumor)
# ===========================================================================

class Task3Input(BaseModel):
    id: str = Field(..., description="Unique sample identifier")
    image_b64: Optional[str] = Field(
        None,
        description="Base64-encoded image bytes (PNG or DICOM). Use this OR image_url."
    )
    image_url: Optional[str] = Field(
        None,
        description="URL to fetch the image from. Use this OR image_b64."
    )
    image_array: Optional[List[List[float]]] = Field(
        None,
        description="Flattened or 2D float array for medical images (e.g. MIP projections)."
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "scan_007",
            "image_b64": "<base64-encoded-PNG>",
            "image_url": None,
            "image_array": None
        }
    })


class Task3Output(BaseModel):
    id: str
    label: Optional[str] = Field(None, description="Top-level classification label (if applicable)")
    mask: Optional[List[List[int]]] = Field(
        None,
        description="2D binary segmentation mask (H x W), values 0 or 1"
    )
    bounding_boxes: Optional[List[Dict[str, float]]] = Field(
        None,
        description="List of {x, y, w, h} bounding boxes (detection tasks)"
    )
    meta: Optional[PredictionMetadata] = None


# ===========================================================================
# TRIPLETEX — AI Accounting Agent
# ===========================================================================

class TripletexFileAttachment(BaseModel):
    filename: str
    content_base64: str
    mime_type: str


class TripletexCredentials(BaseModel):
    base_url: str
    session_token: str


class TripletexSolveInput(BaseModel):
    prompt: str = Field(..., description="Accounting task prompt (any of 7 languages)")
    files: List[TripletexFileAttachment] = Field(default=[], description="Optional PDF/image attachments")
    tripletex_credentials: TripletexCredentials

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "prompt": "Opprett en ansatt med navn Ola Nordmann, ola@example.org.",
            "files": [],
            "tripletex_credentials": {
                "base_url": "https://tx-proxy.ainm.no/v2",
                "session_token": "abc123"
            }
        }
    })


class TripletexSolveOutput(BaseModel):
    status: str = "completed"
