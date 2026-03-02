from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# --- DUMMY INPUT/OUTPUT SCHEMAS ---
# These will be replaced by the actual competition schemas on March 19

class Task1Input(BaseModel):
    id: str
    features: List[float]

class Task1Output(BaseModel):
    id: str
    prediction: int
    confidence: float

class Task2Input(BaseModel):
    id: str
    text: str

class Task2Output(BaseModel):
    id: str
    category: str
    is_urgent: bool  # Inspired by the 2025 Emergency RAG task

class HealthResponse(BaseModel):
    status: str
    message: str