# 🫀 Heart Disease Prediction System

A full-stack, AI-powered clinical platform for heart disease risk assessment. Patients submit lab results through a React frontend; the Node.js gateway validates and routes the data to a Python/FastAPI AI service that runs an ensemble ML model, generates SHAP explainability charts, and produces an LLM-written PDF medical report — all within a single secure request.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Variables](#environment-variables)
  - [Running Locally](#running-locally)
- [API Overview](#-api-overview)
- [Database Schema](#-database-schema)
- [Postman Collections](#-postman-collections)
- [Deployment](#-deployment)
- [License](#-license)

---

## 🔍 Overview

This project is a **monorepo** containing three applications:

| App | Technology | Description |
|---|---|---|
| `apps/Backend` | Node.js + Express + Prisma | REST API gateway — auth, user/lab management, AI proxy |
| `apps/AI` | Python + FastAPI | Internal ML service — prediction, SHAP, ECG analysis, LLM reports |
| `apps/Frontend` | React 18 + Bootstrap | Patient-facing web UI |

The frontend **only talks to the Node.js backend** on port `5000`. The AI service on port `8000` is an internal service protected by a shared `INTERNAL_API_KEY` and is never called directly from the browser.

---

## 🏗️ Architecture

```
┌─────────────────┐        JWT         ┌──────────────────────────────┐
│                 │ ─────────────────► │                              │
│  React Frontend │                    │   Node.js Backend (port 5000)│
│   (port 3000)   │ ◄───────────────── │   Express · Prisma · Neon PG │
│                 │    JSON responses  │                              │
└─────────────────┘                    └──────────────┬───────────────┘
                                                      │ INTERNAL_API_KEY
                                                      │ (never public)
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │  Python AI Service (port 8000)│
                                       │  FastAPI · XGBoost · LightGBM │
                                       │  CatBoost · PyTorch · SHAP    │
                                       │  LangChain/Groq · WeasyPrint  │
                                       └──────────────────────────────┘
```

---

## ✨ Features

### Backend (Node.js)
- 🔐 **JWT Authentication** — register, login, refresh tokens, logout
- 👥 **User Management** — CRUD with national ID validation
- 🏥 **Hospital Directory** — searchable hospital list with Google Maps links
- 🧪 **Lab Management** — lab registration with unique lab codes
- 📋 **Lab Test Ingestion** — CSV bulk upload via lab portal (`x-lab-key`)
- 🫀 **ECG Upload** — WFDB `.dat`/`.hea` pair upload for ECG analysis
- 🤖 **AI Prediction Proxy** — securely forwards requests to the Python service
- 🔒 **Security** — Helmet headers, rate limiting (300 req/15 min), CORS, bcrypt
- 📜 **Logging** — Winston structured logging
- ⚡ **Optional HTTPS** — SSL cert support via environment variables

### AI Service (Python)
- 🧠 **Ensemble ML Model** — XGBoost, LightGBM, CatBoost stacked classifier
- 📊 **SHAP Explainability** — per-prediction SHAP waterfall chart (PNG)
- 🫀 **ECG Deep Learning** — PyTorch model on WFDB ECG signals
- 📄 **PDF Report Generation** — LLM (Groq/LangChain) writes a clinical report rendered via WeasyPrint/Playwright to PDF
- ⚠️ **Risk Stratification** — Low / Moderate / High risk levels with clinical decision text
- 🔑 **Internal Auth** — `X-INTERNAL-API-KEY` validates every request from Node

### Frontend (React)
- Patient registration & login
- Lab test result viewer
- Prediction result display with risk level and SHAP chart
- PDF report download
- Hospital directory browser
- Responsive Bootstrap 5 design

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, React Router v6, Bootstrap 5, Axios |
| Backend | Node.js, Express 5, Prisma ORM, PostgreSQL (Neon) |
| AI Service | Python 3.11, FastAPI, Uvicorn, SQLAlchemy |
| ML | XGBoost, LightGBM, CatBoost, scikit-learn, PyTorch, WFDB |
| Explainability | SHAP, Matplotlib |
| LLM | LangChain + Groq API |
| PDF | WeasyPrint (HTML→PDF), Playwright (Chromium fallback) |
| Auth | JWT (jsonwebtoken), bcrypt, Refresh Tokens |
| Database | PostgreSQL via Neon (serverless) |
| Validation | Zod (Node), Pydantic (Python) |
| Logging | Winston |
| Containerization | Docker (AI service) |

---

## 📁 Project Structure

```
heart-disease-prediction/
├── apps/
│   ├── Backend/               # Node.js API gateway
│   │   ├── src/
│   │   │   ├── api/           # External API integrations
│   │   │   ├── config/        # Prisma client, env validation
│   │   │   ├── controllers/   # Route handlers
│   │   │   ├── integrations/  # AI service HTTP client
│   │   │   ├── middlewares/   # Auth, error, upload
│   │   │   ├── routes/        # Express routers
│   │   │   ├── services/      # Business logic
│   │   │   ├── utils/         # Helpers, logger
│   │   │   ├── validators/    # Zod schemas
│   │   │   └── server.js      # Entry point
│   │   ├── prisma/
│   │   │   └── schema.prisma  # Database schema
│   │   └── .env.example
│   │
│   ├── AI/                    # Python FastAPI ML service
│   │   ├── app/
│   │   │   ├── api/           # FastAPI routers & endpoints
│   │   │   ├── core/          # Config, security
│   │   │   ├── data/          # Feature preprocessing
│   │   │   ├── db/            # SQLAlchemy connection
│   │   │   ├── schemas/       # Pydantic models
│   │   │   ├── services/      # ML inference, SHAP, LLM, ECG
│   │   │   ├── templates/     # Jinja2 HTML report templates
│   │   │   └── main.py        # FastAPI app entry point
│   │   ├── notebooks/         # Training notebooks
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   └── Frontend/              # React patient portal
│       ├── src/
│       └── public/
│
├── docs/
│   └── FRONTEND_API.md        # Full API reference for frontend devs
│
├── postman/                   # Postman collections & environments
│   ├── Heart_Disease_Prediction_Unified.postman_collection.json
│   ├── Heart_Disease_Prediction_Admin_Seed.postman_collection.json
│   └── fixtures/              # Sample payloads & lab CSVs
│
└── package.json               # Monorepo root scripts
```

---

## 🚀 Getting Started

### Prerequisites

- **Node.js** v18+
- **Python** 3.11+
- **PostgreSQL** database (or a [Neon](https://neon.tech) serverless account — free tier works)
- **Groq API Key** — free at [console.groq.com](https://console.groq.com) (for LLM reports)

---

### Environment Variables

#### Backend — `apps/Backend/.env`

Copy from the example and fill in your values:

```bash
cp apps/Backend/.env.example apps/Backend/.env
```

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string (e.g. Neon) |
| `JWT_SECRET` | Strong random string for signing JWT tokens |
| `JWT_EXPIRE` | Token expiry, e.g. `30d` |
| `AI_SERVICE_URL` | URL of the Python AI service (default `http://127.0.0.1:8000`) |
| `INTERNAL_API_KEY` | Shared secret between Node and FastAPI — **must match AI `.env`** |
| `AI_REQUEST_TIMEOUT_MS` | Timeout for AI calls in ms (default `120000`) |
| `ADMIN_API_KEY` | Key for admin-only routes (hospital seeding) |
| `LAB_API_KEY` | Key for lab CSV ingestion portal |
| `PORT` | Backend port (default `5000`) |

#### AI Service — `apps/AI/.env`

```bash
cp apps/AI/.env.example apps/AI/.env
```

| Variable | Description |
|---|---|
| `INTERNAL_API_KEY` | **Must match** the Backend `INTERNAL_API_KEY` |
| `DATABASE_URL` | Same PostgreSQL DB as the backend |
| `GROQ_API_KEY` | Groq API key for LLM report generation |

#### Frontend — `apps/Frontend/.env`

```bash
cp apps/Frontend/.env.example apps/Frontend/.env
```

| Variable | Description |
|---|---|
| `REACT_APP_API_URL` | Backend URL, e.g. `http://localhost:5000` |

---

### Running Locally

#### 1. Install dependencies

```bash
# From the monorepo root — installs Backend and Frontend node_modules
npm run install:all

# AI Python dependencies
cd apps/AI
pip install -r requirements.txt
```

#### 2. Set up the database

```bash
cd apps/Backend
npx prisma migrate deploy   # Apply migrations to your Neon/Postgres DB
npm run seed:labs            # (Optional) seed lab data
```

#### 3. Start all three services

Open three terminal windows:

```bash
# Terminal 1 — Backend API (port 5000)
npm run dev:backend

# Terminal 2 — AI Service (port 8000)
npm run dev:ai

# Terminal 3 — Frontend (port 3000)
npm run dev:frontend
```

Or run each service individually from its own directory.

---

## 📡 API Overview

> Full documentation is available in [`docs/FRONTEND_API.md`](./docs/FRONTEND_API.md).

All endpoints are prefixed with `/api`. The frontend always talks to the Node.js backend — **never directly to the AI service**.

### Authentication

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth/register` | Register a new patient account |
| `POST` | `/api/auth/login` | Login and receive JWT + refresh token |
| `POST` | `/api/auth/refresh` | Exchange refresh token for new access token |
| `POST` | `/api/auth/logout` | Invalidate refresh token |

### Users

| Method | Endpoint | Auth |
|---|---|---|
| `GET` | `/api/users/me` | JWT |
| `PUT` | `/api/users/me` | JWT |
| `DELETE` | `/api/users/me` | JWT |

### Predictions (Lab Tests)

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/api/predictions` | JWT | Get all predictions for logged-in user |
| `GET` | `/api/predictions/:id` | JWT | Get a single prediction + SHAP chart |
| `POST` | `/api/predictions/:labTestId/run` | JWT | Run ML prediction on a lab test |
| `GET` | `/api/predictions/:id/report` | JWT | Download PDF report |

### ECG

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/lab-portal/ecg` | `x-lab-key` | Upload WFDB `.dat`/`.hea` files (lab staff) |
| `POST` | `/api/ecg/:id/analyze` | JWT | Run ECG analysis pipeline |
| `GET` | `/api/ecg/:id/result` | JWT | Get ECG inference result |

### Lab Portal (internal — lab staff only)

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/lab-portal/upload` | `x-lab-key` | Bulk CSV upload of lab test results |

### Hospitals

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/hospitals` | List all hospitals |
| `GET` | `/api/hospitals?area=:area` | Filter by area |

---

## 🗄️ Database Schema

The PostgreSQL database is managed via **Prisma ORM**. Key models:

```
User          — patient accounts (national_id unique identifier)
Lab           — registered labs (lab_code)
LabTest       — raw clinical measurements (13 Cleveland Heart Disease features)
Prediction    — ML results: score, risk level, SHAP image, LLM report PDF
EcgTest       — WFDB ECG upload + PyTorch inference results
Hospital      — hospital directory with Google Maps links
RefreshToken  — JWT refresh token store
```

---

## 📮 Postman Collections

Two Postman collections are included in `postman/`:

| File | Purpose |
|---|---|
| `Heart_Disease_Prediction_Unified.postman_collection.json` | Full patient + lab workflow |
| `Heart_Disease_Prediction_Admin_Seed.postman_collection.json` | Admin seeding (hospitals, labs) |

Import the matching `.postman_environment.json` file alongside each collection and set your `baseUrl` and API keys in the environment variables.

Sample fixtures (user payloads, lab CSV files) are in `postman/fixtures/`.

---

## ☁️ Deployment

### AI Service (Docker / Railway)

The AI service includes a production-ready `Dockerfile`:

```bash
cd apps/AI
docker build -t heart-ai .
docker run -p 8000:8000 --env-file .env heart-ai
```

A `nixpacks.toml` is also provided for one-click deployment to [Railway](https://railway.app).

### Backend

Standard Node.js deployment — the `npm start` script runs `prisma generate` then starts the server. Recommended platforms: Railway, Render, Fly.io.

### Frontend

```bash
npm run build:frontend
```

Deploy the `apps/Frontend/build/` directory to Vercel, Netlify, or any static host.

---

## 📄 License

[ISC](./LICENSE)
