# AINM 2026 — Norwegian AI Championship DNV GRD Base Repository

Competition date: **March 19–22, 2026** | Website: [ainm.no/en](https://ainm.no/en) | [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## READ THIS FIRST — Competition Day Playbook

> This repo was built **before** the 2026 tasks are known. Everything here is
> deliberately task-agnostic infrastructure. On March 19 the tasks are revealed
> and you have ~4 days to win. This section tells you exactly what to do and
> in what order.

### What this repo gives you (works regardless of the 2026 tasks)

| Component | What it does | Changes needed on day 1? |
|---|---|---|
| `Dockerfile` + `docker-compose.yml` | Build and run the API anywhere | ❌ None |
| `tasks/machine_learning/baseline.py` | Trains RF/GBM/XGB/LGB on **any** tabular data, picks the best | ❌ None |
| `tasks/machine_learning/feature_engineering.py` | Cleans/encodes **any** DataFrame | ❌ None |
| `tasks/language/factory.py` | Returns an LLM (Azure / Ollama / HF) — provider-agnostic | ❌ None |
| `tasks/language/rag.py` | RAG over **any** document corpus | ❌ None |
| `tasks/language/classifier.py` | Classifies **any** texts into **any** labels | ❌ None |
| `tasks/language/agent.py` | LangChain tool-calling agent with multi-turn history + batch mode | ❌ None |
| `tasks/vision/preprocessing.py` | Loads/resizes/normalizes **any** image (including DICOM, NIfTI) | ❌ None |
| `tasks/vision/segmentation.py` | Produces binary masks — useful if the vision task is segmentation | ⚠️ Swap if task is classification/detection |
| `scripts/evaluate.py` | Measures accuracy **and** latency locally before submitting | ❌ None |
| `scripts/eda_report.py` | Generates HTML EDA report on **any** CSV/Parquet dataset | ❌ None |
| `api/main.py` | FastAPI server skeleton, model registry, latency tracking | ⚠️ Wire in real models (marked clearly) |
| `api/schemas.py` | I/O schemas — **currently placeholders inspired by 2025 tasks** | 🔴 Rewrite with real field names |

---

### Competition Day Step-by-Step

**Hour 0–1 · Understand the tasks**
- Read all three task specs carefully
- Identify: input type, output type, evaluation metric, latency constraint
- Assign one owner per task

**Hour 1–2 · Adapt the two task-specific files**

*Step 1 — Rewrite `api/schemas.py`*

The current schemas are placeholders. Replace the field names with whatever
the real spec says. The structure (Input/Output per task, `id` always echoed)
stays the same:

```python
# Example: if Task 1 turns out to be a game action predictor
class Task1Input(BaseModel):
    id: str
    sensor_readings: List[float]   # ← rename from 'features'
    speed: float                   # ← add real fields

class Task1Output(BaseModel):
    id: str
    action: str                    # ← rename from 'prediction'
```

*Step 2 — Wire real models into `api/main.py`*

Each endpoint has a clearly marked swap block:
```python
# ── REPLACE THIS BLOCK ON COMPETITION DAY ──────────────────────────
# dummy logic here
# ───────────────────────────────────────────────────────────────────
```
Replace the dummy block with a call to the appropriate task module.
Also uncomment the relevant lines in the model registry at the top of the file.

**Hour 2–12 · Build and iterate**
- Drop competition data into `data/`
- Run EDA: `python scripts/eda_report.py data/train.csv`
- Fit a baseline using the task modules (see [Task Module Usage](#task-module-usage) below)
- Save the artifact to `models/`
- Wire it into the endpoint, run `pytest tests/ -v`, verify health check
- Benchmark: `python scripts/evaluate.py --mode api --task all`
- Iterate — improve model, re-save, re-verify latency

**Hour 12+ · Harden and deploy**
- `docker compose up` — confirm the container works end-to-end
- `bash scripts/start_ngrok.sh` — expose the API if judges need external access
- Keep `pytest tests/ -v` green before every submission

---

### Competition Day Checklist

- [ ] Tasks revealed — read all specs, assign owners
- [ ] Rewrite `api/schemas.py` with real field names per task
- [ ] EDA: `python scripts/eda_report.py data/train.csv`
- [ ] Fit baseline model per task (target: first working submission < 3 hours)
- [ ] Wire models into `api/main.py` (remove dummy blocks, uncomment registry)
- [ ] Run `pytest tests/ -v` — all green before any submission
- [ ] Benchmark latency: `python scripts/evaluate.py --mode api`
- [ ] Docker smoke test: `docker compose up && curl localhost:8000/health`
- [ ] Expose via ngrok if required: `bash scripts/start_ngrok.sh`
- [ ] Iterate: improve models → re-evaluate → re-deploy

---

## Setup (do this before March 19)

This repository is the shared foundation for the team. It provides a ready-to-run
skeleton for all three competition tracks, a FastAPI serving layer, evaluation scripts,
and tests — so on competition day, the focus is purely on model quality, not on setup.

---

## Repository Structure

```
AINM2026/
├── api/
│   ├── main.py              # FastAPI server — 3 task endpoints + health check
│   └── schemas.py           # Pydantic I/O schemas for all tasks
├── tasks/
│   ├── language/
│   │   ├── factory.py       # Provider-agnostic LLM + embeddings factory (Azure / Ollama / HF)
│   │   ├── rag.py           # RAG pipeline (Chroma vector DB + LangChain)
│   │   ├── classifier.py    # Text classifier (TF-IDF → embeddings → zero-shot)
│   │   └── agent.py         # AgentRunner: tool-calling agent, multi-turn, run_batch
│   ├── machine_learning/
│   │   ├── baseline.py      # Tabular pipeline: model race + auto-selection
│   │   └── feature_engineering.py  # Imputation, encoding, interaction features
│   └── vision/
│       ├── preprocessing.py # Image loading, resize, normalize, MIP projection
│       └── segmentation.py  # Otsu baseline → torchvision → SMP U-Net
├── scripts/
│   ├── setup.sh             # One-command bootstrap
│   ├── evaluate.py          # Local scoring harness (accuracy + latency)
│   ├── eda_report.py        # Sweetviz EDA report generator
│   └── start_ngrok.sh       # Expose local API via ngrok
├── tests/
│   ├── test_api.py          # Integration tests for all endpoints
│   ├── test_ml.py           # Unit tests for ML pipeline
│   ├── test_language.py     # Unit tests for RAG + classifier
│   └── test_vision.py       # Unit tests for preprocessing + segmentation
├── data/                    # Competition datasets (not committed)
│   └── chroma_db/           # Persisted vector store
├── models/                  # Trained model artifacts (not committed)
├── notebooks/
│   └── reports/             # Auto-generated EDA HTML reports
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## Quick Start

### 1. Bootstrap

```bash
git clone <repo-url> && cd AINM2026
chmod +x scripts/setup.sh

./scripts/setup.sh          # Core dependencies
./scripts/setup.sh --vision # + PyTorch, OpenCV (heavy)
./scripts/setup.sh --dev    # + pytest, httpx
./scripts/setup.sh --all    # Everything
```

This script:
- Checks Python ≥ 3.11
- Installs all packages via `uv`
- Creates `.env` from `.env.example` if missing
- Verifies all imports work

### 2. Configure credentials

```bash
cp .env.example .env
# edit .env — fill in API keys
```

### 3. Run the API

```bash
# Local development (hot-reload)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Docker — cloud LLM mode
docker compose up

# Docker — fully offline (Ollama runs locally)
docker compose --profile local up
```

### 4. Verify it's working

```bash
curl http://localhost:8000/health
# → {"status":"ok","message":"API is live.","version":"1.0.0"}
```

---

## LLM Provider Configuration

Set `LLM_PROVIDER` in `.env` to switch between backends:

| Provider | `LLM_PROVIDER` value | When to use |
|---|---|---|
| Azure OpenAI (GPT-4o, o3) | `azure_native` | Cloud, competition day |
| DeepSeek / Mistral via Azure AI Foundry | `openai_compatible` | Cloud, cost-effective |
| Ollama (local) | `ollama` | Fully offline, no API key |

Same logic for embeddings — `EMBEDDING_PROVIDER=azure_native` or `huggingface`.

---

## Task Module Usage

### Task 1 — Tabular / Classification

```python
from tasks.machine_learning import TabularPipeline, FeatureEngineer

fe = FeatureEngineer(create_interactions=True, clip_outliers=True)
X_clean = fe.fit_transform(X_train)

pipeline = TabularPipeline(task="classification")
pipeline.fit(X_clean, y_train)          # Runs model race (RF, GBM, XGB, LGB)
preds = pipeline.predict(X_test)
pipeline.save("models/task1_v1.pkl")    # Serialize for API use
```

### Task 2 — Language / RAG

```python
from tasks.language import TextClassifier, RAGPipeline

# Classification
clf = TextClassifier(strategy="embeddings")   # or "tfidf" or "zero_shot"
clf.fit(texts_train, labels_train)
clf.predict(texts_test)
clf.save("models/task2_clf.pkl")

# RAG
rag = RAGPipeline()
rag.ingest_texts(documents, metadatas)
answer = rag.query("What is the triage for cardiac arrest?")
```

### Task 2 (or any task) — Agent Orchestration

Use `AgentRunner` when a task benefits from tool-calling, multi-step reasoning,
or chaining multiple capabilities together.

```python
from tasks.language import AgentRunner, make_tool

# 1. Define tools as plain decorated functions
@make_tool
def lookup_patient_record(patient_id: str) -> str:
    """Fetch the clinical record for a patient by their ID."""
    return records_db[patient_id]

@make_tool
def compute_risk_score(age: int, systolic_bp: float) -> float:
    """Compute a simple cardiovascular risk score."""
    return round(age * 0.02 + systolic_bp * 0.001, 3)

# 2. Create the agent — uses the factory LLM (Azure / Ollama / OpenAI-compat)
agent = AgentRunner(
    tools=[lookup_patient_record, compute_risk_score],
    system_prompt="You are a clinical decision support assistant.",
)

# 3. Single query
result = agent.run("What is the risk score for patient P-001?")
print(result["output"])

# 4. Multi-turn conversation
history = []
r1 = agent.run("Look up patient P-001.", chat_history=history)
r2 = agent.run("Is their risk score high?", chat_history=r1["chat_history"])

# 5. Score a whole batch of competition rows (stateless per row)
answers = agent.run_batch([row["question"] for row in test_data])

# 6. Add a tool discovered on competition day without rebuilding from scratch
agent.add_tool(some_new_tool)
```

> `AgentRunner` reuses `factory.py` — no new dependencies, all three LLM
> providers work without any code changes.

### Task 3 — Vision / Segmentation

```python
from tasks.vision import ImagePreprocessor, SegmentationPipeline

prep = ImagePreprocessor(img_size=224, normalize=True)
arr  = prep.load_and_transform("scan.png")    # Works with DICOM via prep.load_dicom()
mip  = ImagePreprocessor.mip_projection(volume, axis=2)  # For 3D PET volumes

seg = SegmentationPipeline(backend="otsu")    # or "torchvision" or "smp"
seg.load_pretrained("models/task3_seg.pth")  # (for deep backends)
mask = seg.predict_single(arr)               # Returns uint8 binary mask
print(SegmentationPipeline.dice_score(mask, ground_truth))
```

---

## Plugging in Real Models on Competition Day

All task endpoints in `api/main.py` are clearly marked:

```python
# ── REPLACE THIS BLOCK ON COMPETITION DAY ──────────────────────────
# ... dummy logic ...
# ───────────────────────────────────────────────────────────────────
```

The model registry at the top of `main.py` loads once at startup:

```python
models["task1"] = TabularPipeline.load("models/task1.pkl")
models["task2_rag"] = RAGPipeline(persist_directory="./data/chroma_db")
models["task3"] = SegmentationPipeline(backend="torchvision")
models["task3"].load_pretrained("models/task3.pth")
```

---

## Running Tests

```bash
pytest tests/ -v                   # All tests (63 total)
pytest tests/test_api.py -v        # API only
pytest tests/test_ml.py -v         # ML only
pytest tests/test_language.py -v   # Language (classifier + RAG + agent)
pytest tests/test_vision.py -v     # Vision only
pytest tests/ --cov=tasks --cov=api  # With coverage
```

All tests are designed to run **without API keys or GPU** — they use mocks and
synthetic data. Run them immediately after cloning to verify the setup.

---

## Evaluating Solutions Locally

Before submitting, benchmark both accuracy and latency:

```bash
# Score tabular model on test CSV
python scripts/evaluate.py --task 1 --data data/task1_test.csv

# Score language model
python scripts/evaluate.py --task 2 --data data/task2_test.csv

# Score via live API (end-to-end test)
python scripts/evaluate.py --task all --mode api --api-url http://localhost:8000

# Evaluate all tasks if data/ contains named CSVs
python scripts/evaluate.py --task all --data data/
```

---

## EDA on a New Dataset

```bash
python scripts/eda_report.py data/your_data.csv
# → opens notebooks/reports/your_data_report.html
```

---

## Optional Dependencies

| Feature | Installation |
|---|---|
| Core (all tasks) | `pip install -e .` |
| Vision (PyTorch, OpenCV) | `pip install -e ".[vision]"` |
| Medical images (DICOM) | `pip install pydicom` |
| Medical images (NIfTI) | `pip install nibabel` |
| Advanced segmentation | `pip install segmentation-models-pytorch` |
| Tests | `pip install -e ".[dev]"` |

---

> The Competition Day Checklist and step-by-step playbook are at the **top of this file**.
