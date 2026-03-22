# Tripletex — AI Accounting Agent

**NM i AI 2026** | Team DNV GRD — ARAS

An autonomous AI agent that performs accounting tasks in [Tripletex ERP](https://www.tripletex.no/) via its v2 REST API, deployed on Google Cloud Run.

---

## Architecture

```
Competition validator
        │
        ▼
  POST /solve  (FastAPI)
        │
        ▼
   ┌─────────┐       ┌──────────────┐
   │  Router  │──────▶│  Sub-agent   │
   │ (LLM)   │       │  (focused    │
   └─────────┘       │   prompt)    │
                      └──────┬───────┘
                             │  LangChain tool calls
                             ▼
                      ┌──────────────┐
                      │  Tripletex   │
                      │  REST API    │
                      │  (proxy)     │
                      └──────────────┘
```

1. **Router** — A single LLM call classifies the incoming task into one of 16 categories
2. **Sub-agent** — A specialized prompt + tool-calling agent executes the task via API calls
3. **File extraction** — PDFs/images are extracted via `pymupdf4llm` (layout-preserving) with Gemini vision fallback

## Task Categories (16)

| Category | Description |
|---|---|
| `employee` | Create/update employees, employment, salary |
| `customer` | Create/update customers |
| `supplier` | Create/update suppliers |
| `product` | Create products |
| `department` | Create departments |
| `contact` | Create contacts for customers/suppliers |
| `order_invoice` | Orders, invoices, payments, credit notes, reminders, currency exchange |
| `travel_expense` | Travel expense reports |
| `supplier_invoice` | Record received supplier invoices |
| `payroll` | Run payroll, register wages |
| `receipt` | Book receipts from images/PDFs |
| `corrections` | Find and fix errors in existing vouchers |
| `bank_recon` | Bank reconciliation from CSV statements |
| `yearend` | Monthly/year-end closing (depreciation, prepaid, tax) |
| `ledger` | Generic ledger vouchers and journal entries |
| `project` | Project management, time registration, project invoicing |

## Key Files

| File | Purpose |
|---|---|
| [`solve.py`](solve.py) | Entry point, router, HTTP client, LangChain tool factory |
| [`prompts.py`](prompts.py) | 16 specialized system prompts + shared preamble + router prompt |
| [`files.py`](files.py) | PDF/image extraction (pymupdf4llm + Gemini vision fallback) |

## How It Works

### Request Flow

```
POST /solve
├── prompt: str              ← task description (multilingual)
├── files: [FileAttachment]  ← PDFs, images, CSVs
└── tripletex_credentials
    ├── base_url             ← competition proxy URL
    └── session_token        ← per-request auth token
```

1. **File extraction** — Attached PDFs are converted to Markdown via `pymupdf4llm`; scanned PDFs and images fall back to Gemini vision
2. **Routing** — The task prompt is classified into one of 16 categories by a fast LLM call
3. **Agent execution** — A LangChain tool-calling agent runs with the category-specific prompt, using 6 tools:
   - `search_resource` — GET with query filters
   - `get_by_id` — GET single resource
   - `create_resource` — POST
   - `update_resource` — PUT
   - `delete_resource` — DELETE
   - `action_endpoint` — PUT for Tripletex action endpoints (`:payment`, `:invoice`, `:createReminder`, etc.)
4. **Response** — Always returns `{"status": "completed"}`

### Authentication

- Basic Auth: `username="0"`, `password=<session_token>`
- Session tokens are per-request and expire after the validator receives the response

### LLM

- Gemini 2.5 Flash via `langchain-google-genai` (temperature=0.0)
- Provider-agnostic factory in [`tasks/language/factory.py`](../language/factory.py)

## Deployment

- **Platform**: Google Cloud Run
- **Service**: `ainm2026`
- **Region**: `europe-north1`
- **Container**: See [`Dockerfile`](../../Dockerfile) in repo root
