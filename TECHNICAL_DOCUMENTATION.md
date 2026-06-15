# Healthcare AI System: Technical Specification & Architecture

## Section 1: Microservices Architecture (FastAPI)

The Heart Disease Prediction platform employs a decoupled, highly-available microservices architecture. At its core is an ASGI-compliant **FastAPI Gateway**, chosen for its native asynchronous capabilities and high throughput capacity. 

### Request Routing & Integration
- **Node.js Frontend/API Consumer:** Client requests are forwarded to the FastAPI backend via asynchronous REST endpoints.
- **PostgreSQL (PGSQL):** Used for persistent, ACID-compliant storage of patient metadata, session histories, and clinical audit logs. The FastAPI service connects to PGSQL using asynchronous ORMs (e.g., SQLAlchemy/SQLModel with async drivers).
- **HuggingFace Inference:** For specific NLP and lightweight AI inferences, requests are dispatched asynchronously to HuggingFace endpoints, ensuring that main thread execution is not blocked.
- **Groq API:** Leveraged for ultra-low latency LLM inference (Llama-3.3-70b-versatile). The Gateway handles graceful degradation; if the Groq API times out, it employs a robust local fallback mechanism.

## Section 2: Tabular ML Engine (CatBoost)

While the Stacking Classifier achieved an impressive 93.70% accuracy, the **CatBoost Classifier** (92.44% Accuracy, 92.91% F1 Score) was strategically selected for production. This 1.26% sacrifice in raw accuracy enables deterministic TreeExplainer integration, satisfying strict medical Explainable AI (XAI) mandates.

### The 11 Clinical Features
The tabular engine processes 11 critical clinical features, normalized and engineered for optimal gradient boosting performance:
1. `age`: Patient's age in years.
2. `sex`: Biological sex.
3. `chest pain type`: Categorical indicator of angina type.
4. `resting bp s`: Resting systolic blood pressure.
5. `cholesterol`: Serum cholesterol levels.
6. `fasting blood sugar`: Fasting blood sugar > 120 mg/dl.
7. `resting ecg`: Resting electrocardiogram results.
8. `max heart rate`: Maximum heart rate achieved during stress test.
9. `exercise angina`: Exercise-induced angina.
10. `oldpeak`: ST depression induced by exercise relative to rest.
11. `ST slope`: The slope of the peak exercise ST segment.

### Explainable AI (XAI) Integration
CatBoost integrates seamlessly with SHAP (SHapley Additive exPlanations) TreeExplainer. This allows clinicians to see exactly how much each feature contributed to the final risk probability in real-time.

![Feature Importance / SHAP Plot](./visualizations/shap_summary_plot.png)

### Threshold Optimization (Youden's J Statistic)
In medical diagnostics, False Negatives (missing a diseased patient) are significantly more dangerous than False Positives. By applying **Youden's J statistic**, we optimized the classification threshold away from the default 0.50 to **0.41**.
- **Impact:** This 0.41 cutoff drastically maximizes Recall, ensuring the system aggressively flags potential cardiac events at the expense of a slight increase in False Positives.

![Confusion Matrix](./visualizations/confusion_matrix_0.41.png)

## Section 3: Deep Learning ECG Engine (xresnet1d101)

For raw signal processing, the system utilizes a state-of-the-art 1D Convolutional Neural Network: **xresnet1d101**.

### WFDB Pipeline & Preprocessing
- **Data Parsing:** The pipeline ingests raw physiological waveforms using the WFDB format, parsing the signal data (`.dat`) and header metadata (`.hea`).
- **Resampling:** Raw signals often arrive at 500Hz. The pipeline applies Scipy-based resampling to downsample the signals to **100Hz**, standardizing the input dimensions without losing diagnostic frequencies.
- **Amplitude Normalization:** A `StandardScaler` normalizes the voltages across the 12-lead ECG, ensuring that amplitude variations caused by sensor calibration or patient impedance do not bias the network weights.

### Architectural Superiority
- **xresnet1d101 vs. LSTMs:** Traditional Recurrent Neural Networks (RNNs) and LSTMs suffer from vanishing gradients when processing long 1D sequences like 10-second ECG strips (1000 time steps at 100Hz). The xresnet1d101 utilizes residual connections to allow gradients to flow directly through the network, capturing both local morphological features (QRS complexes) and long-term temporal relationships (arrhythmias).
- **Output Mechanism:** The final layer consists of **71 independent Sigmoid activations**, allowing for multi-label classification (a patient can have both Atrial Fibrillation and an Inferior Myocardial Infarction simultaneously).

## Section 4: Generative AI Explainer (Llama-3.3-70b-versatile)

To bridge the gap between raw statistical output and clinician comprehension, the system employs **Llama-3.3-70b-versatile** via the Groq API.

### Retrieval-Augmented Generation (RAG)
- **Knowledge Base Injection:** The RAG pipeline dynamically injects context from `ecg_diagnosis_kb.py`, providing the LLM with verified, textbook cardiology definitions and guidelines.
- **Prompt Engineering:** The LLM prompt is heavily engineered to include the patient's real-time vitals and the deterministic SHAP scores from the CatBoost model. This grounds the LLM's explanation in the exact mathematics of the prediction, mitigating hallucinations.

### Medical Safety & The Kill-Switch
Healthcare applications cannot risk unauthorized medical advice.
- **Safety Regex Layer:** A strict `_UNSAFE_PATTERNS` regex acts as an impenetrable middleware. If the LLM generates any prescriptive language (e.g., "take Aspirin", "prescribe", "dosage"), the regex intercepts the response and sanitizes the output to `[medically reviewed]`. 
- **Graceful Failure:** If the Groq API times out, the system automatically falls back to returning a structured, safe JSON response without the generative text, ensuring the core diagnostic pipeline never goes down.
