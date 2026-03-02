from fastapi import FastAPI, HTTPException
import uvicorn
import time
import random

from api.schemas import (
    Task1Input, Task1Output, 
    Task2Input, Task2Output, 
    HealthResponse
)

app = FastAPI(title="NM i AI 2026 - Team DNV GRD - \"Name\" API", version="1.0")

@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Organizers usually require a root or health endpoint to verify your API is alive."""
    return HealthResponse(status="ok", message="API is locked and loaded.")

@app.post("/task1/predict", response_model=Task1Output)
async def predict_task1(payload: Task1Input):
    """
    Dummy endpoint for Task 1 (e.g., Tabular / Classification)
    """
    try:
        # TODO: Import your actual ML model inference here
        # prediction = my_model.predict(payload.features)
        
        # Dummy logic:
        dummy_pred = random.choice([0, 1])
        dummy_conf = random.uniform(0.5, 0.99)
        
        return Task1Output(
            id=payload.id,
            prediction=dummy_pred,
            confidence=round(dummy_conf, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/task2/predict", response_model=Task2Output)
async def predict_task2(payload: Task2Input):
    """
    Dummy endpoint for Task 2 (e.g., NLP / RAG)
    """
    try:
        # TODO: Route text to local LLM or RAG pipeline
        
        # Dummy logic:
        urgent = "urgent" in payload.text.lower()
        
        return Task2Output(
            id=payload.id,
            category="medical_evaluation",
            is_urgent=urgent
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run locally for testing
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)