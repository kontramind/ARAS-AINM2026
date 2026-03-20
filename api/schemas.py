"""
api/schemas.py
---------------
Pydantic schemas for the Tripletex accounting agent API.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional


# ===========================================================================
# SHARED
# ===========================================================================

class HealthResponse(BaseModel):
    status: str
    message: str
    version: str = "1.0"


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
