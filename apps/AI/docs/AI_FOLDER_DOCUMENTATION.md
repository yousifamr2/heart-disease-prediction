# Comprehensive AI Module Technical Documentation

This document serves as an exhaustive, in-depth technical guide to the `apps/AI` directory. This module acts as the artificial intelligence and machine learning powerhouse of the Heart Disease Prediction system. It is built around a robust **FastAPI** backend, integrating tabular machine learning models, deep learning ECG classifiers, and Large Language Models (LLM) for medical report generation.

---

## 📂 1. Root Level Configuration & Infrastructure

The root of the `apps/AI` folder contains essential files for environment configuration, dependency management, and containerization.

- **`Dockerfile` & `docker-compose.yml`**: Defines the containerized environment for the AI service. Ensures consistency across development, testing, and production environments by packaging the FastAPI application, its dependencies, and the required ML models.
- **`requirements.txt`**: The exhaustive list of Python dependencies, including `fastapi`, `uvicorn`, `scikit-learn`, `pandas`, `torch` (for ECG), `langchain-groq` (for LLMs), `shap`, and `pytest`.
- **`.env` & `.env.example`**: Environment variable files. They securely inject critical configurations such as `INTERNAL_API_KEY`, `GROQ_API_KEY`, `DATABASE_URL`, and `AI_ALLOWED_HOSTS`.
- **`heart disease prediction.postman_collection.json`**: A comprehensive Postman collection containing pre-configured requests to test all exposed endpoints of the AI API.
- **`test.db`**: A local SQLite database primarily used for automated testing, mocking, and rapid local development without requiring a full PostgreSQL instance.

---

## 📂 2. The FastAPI Backend (`app/`)

This directory is the core of the service. It handles HTTP requests, routes them to the appropriate services, and returns structured JSON responses.

### 2.1 Entry Point: `main.py`
The application entry point. It initializes the FastAPI instance, configures CORS/TrustedHost middleware, and registers all API routers. Notably, it includes a robust `/health` endpoint that checks the operational status of the Database, Local ML Model, LLM API key presence, and the Deep Learning ECG predictor. It gracefully falls back to "degraded" status if external dependencies (like Groq) are unavailable.

### 2.2 Routing Layer (`app/api/`)
Contains all the exposed RESTful endpoints, grouped logically:
- **`endpoints/predict.py`**: Handles incoming patient tabular data, requests predictions from the ML service, and generates SHAP explanations.
- **`endpoints/internal_ecg.py`**: Accepts raw ECG signal arrays (12-lead), processes them, and returns multi-label diagnostic predictions.
- **`endpoints/users.py`**: Manages user-related data interactions.
- **`endpoints/report.py` & `endpoints/shap.py`**: Endpoints dedicated to generating comprehensive medical reports (via LLM) and SHAP feature importance charts.

### 2.3 Core Configuration (`app/core/`)
- **`config.py`**: Centralized configuration management parsing `.env` files.
- **`logger.py`**: Configures standard Python logging to ensure all API requests, ML inferences, and errors are cleanly logged for observability.
- **`security.py`**: Manages API key validation (`INTERNAL_API_KEY`) and ensures that only authorized microservices (like the Node.js backend) can invoke the AI endpoints.

### 2.4 Data Transfer Objects (`app/schemas/`)
Defines Pydantic models (e.g., `patient_schema.py`) used for strict data validation. This ensures that incoming payloads for predictions or ECG arrays strictly adhere to the expected data types and bounds, preventing unexpected model crashes.

### 2.5 Business Logic & AI Services (`app/services/`)
This is where the actual intelligence resides:
- **`ml_service.py`**: A hybrid ML service. It attempts to call an external HuggingFace inference API first (`omarbm52-artemis-heart-api`). If unavailable, it seamlessly fails over to the local `.pkl` model (`local_ml_service.py`). It calculates risk probabilities and generates SHAP values.
- **`local_ml_service.py`**: Handles loading the `models/best.pkl` (e.g., CatBoost/Stacking) using `joblib` or `pickle` and executing local inference.
- **`ecg_service.py`**: The deep learning service for ECG. It loads a pre-trained `xresnet1d101` PyTorch model. It maps PTB-XL SCP statement codes to human-readable strings (e.g., "1AVB" -> "First-degree atrioventricular block"). It utilizes a `StandardScaler` and `MultiLabelBinarizer` to preprocess 12-lead signal arrays (shape: Tx12) into tensors for inference, returning the top 5 probabilities.
- **`llm_service.py`**: Integrates with Groq via LangChain (`llama-3.3-70b-versatile`). It features dynamic prompt engineering for both tabular data and ECG outputs. **Crucially, it implements a Medical Safety Layer** (`sanitize_llm_output`) via regex patterns to intercept and replace absolute medical claims (e.g., "you have heart disease" becomes "[medically reviewed]") before returning the report to the user.
- **`pdf_service.py` & `pdf_exporter.py`**: Generates highly formatted, professional PDF reports consolidating patient vitals, ML predictions, SHAP charts, and LLM-generated recommendations.
- **`risk_classifier.py`**: A deterministic layer that categorizes numerical probabilities into standard risk buckets (Low, Moderate, High).

---

## 📂 3. ECG Processing Ecosystem (`ECG/`)

A dedicated module for advanced Deep Learning on 1-D time-series data.
- **`Skeleton/`**: Contains the PyTorch neural network architectures, notably `xresnet1d.py` (1D ResNet).
- **`weights/`**: Stores the heavy `.pth` PyTorch model weights used by `ecg_service.py`.
- **`Data Preprocessing/`**: Contains `mlb.pkl` (MultiLabelBinarizer for decoding outputs) and `standard_scaler.pkl` (for normalizing the raw milliVolt signals to the distribution the model was trained on).

---

## 📂 4. Machine Learning Models (`models/`)

The persistent storage for trained Scikit-learn or similar ML tabular models.
- **`best.pkl`**: The artifact generated from the Jupyter Notebooks. It encapsulates the fully trained pipeline (Imputers, Scalers, Classifiers) ready for zero-latency local predictions.

---

## 📂 5. Dataset Repository (`dataset/`)

Contains the foundational data used for training and local integration testing:
- **`heart_statlog_cleveland_hungary_final.csv`**: The massive aggregated dataset (almost 40KB) used for training the tabular models.
- **`patients_with_heart_disease.csv` & `patients_no_heart_disease.csv`**: Stratified micro-datasets used strictly for unit testing the models to ensure they correctly predict edge cases.

---

## 📂 6. Quality Assurance & Testing (`tests/`)

A comprehensive Pytest suite guaranteeing the reliability of the AI module:
- **`conftest.py`**: Sets up global Pytest fixtures, including mocking the database and generating dummy client instances.
- **`test_predict.py`**: Validates the end-to-end flow of the tabular prediction API.
- **`test_ecg.py`**: Feeds dummy noise matrices `(N, 12)` into the ECG endpoints to verify shape handling, scaling, and Pytorch tensor execution.
- **`test_llm.py`**: Strictly tests the LLM generation and, most importantly, the safety sanitation layers to prevent rogue medical advice.
- **`test_security.py`**: Ensures the API rejects unauthorized requests missing the `INTERNAL_API_KEY`.
- **`test_pdf.py`**: Validates the byte-stream generation of the PDF engines.

---

## 📂 7. Research, Development & Documentation

- **`notebooks/`**: Contains Jupyter Notebooks (e.g., `1000heart.ipynb`) where Data Scientists perform Exploratory Data Analysis (EDA), feature selection, hyperparameter tuning, and threshold calculation (e.g., Youden's J statistic).
- **`docs/`**: Stores overarching project documentation, including technical handover scripts, presentation scripts, and this very architecture document.
- **`assets/`**: Static files, potentially including static placeholder SHAP charts or logos used within PDF generation.

---

### Summary
The `apps/AI` architecture is a state-of-the-art hybrid AI inference engine. It is strictly modular, ensuring clear separation of concerns between HTTP transport (`api/`), intelligent business logic (`services/`), strict data typing (`schemas/`), and deep learning execution (`ECG/`). It is fully equipped for secure, scalable, and safe healthcare operations.
