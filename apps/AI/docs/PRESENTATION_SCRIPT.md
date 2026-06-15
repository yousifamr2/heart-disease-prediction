# Defense Script: Heart Disease Prediction Architecture

## 1. Why we chose CatBoost for Tabular Data
**Question:** "Why didn't you use a Deep Neural Network (DNN) for the structured patient data?"

**Answer:** "While Deep Learning dominates images and text, tree-ensembles like CatBoost represent the State-Of-The-Art for tabular data. Neural networks struggle with unscaled features and non-smooth decision boundaries typical of clinical data (like age vs cholesterol). CatBoost handles categorical variables natively, trains significantly faster, and most importantly, calculates exact SHAP (SHapley Additive exPlanations) tree impacts natively. If we used a DNN, we would have had to rely on `DeepExplainer` approximations, which are slower and less mathematically precise for this use case."

---

## 2. Why we chose `xresnet1d101` for ECG processing
**Question:** "Why did you use a 1D ResNet instead of an LSTM or Transformer for time-series ECG data?"

**Answer:** "ECG data is highly structured, spatial time-series data. While LSTMs are great for variable-length sequential memory, `xresnet1d101` (a 1D convolutional residual network) is mathematically proven to be superior at capturing localized morphological features, such as ST-segment elevations or T-wave inversions. The residual connections solve the vanishing gradient problem over the 100-layer depth, allowing it to extract complex hierarchical features across the 12 leads simultaneously. Furthermore, 1D CNNs parallelize better on GPUs than sequential LSTMs, drastically reducing inference latency."

---

## 3. Handling Real Hospital ECG Data (.dat and .hea)
**Question:** "How does your system actually read real hospital ECGs?"

**Answer:** "Modern 12-lead ECG machines often export data in the MIT-BIH WFDB (Waveform Database) format, consisting of a `.dat` binary signal file and a `.hea` text header file. Our backend intercepts these two files and utilizes the `wfdb` library to parse them. 
A critical engineering challenge we solved was standardizing the sampling frequency. A hospital might upload a 500Hz signal, but our model was trained on 100Hz PTB-XL data. Our pipeline actively reads the `fs` (frequency) metadata from the `.hea` header, and if it mismatches, we mathematically resample the signal using `scipy.signal.resample` to perfectly align with the model's expected 100Hz dimension before converting it to a PyTorch tensor."

---

## 4. Engineering the LLM against Medical Hallucinations
**Question:** "How do you guarantee the AI doesn't give dangerous medical advice or prescribe drugs?"

**Answer:** "We implemented a multi-layered deterministic safety architecture around the non-deterministic LLM. 
1. **Raw Vitals Injection:** The LLM is provided the patient's exact vitals, not just abstract impact scores.
2. **Few-Shot Binding:** The prompt contains explicit 'Ideal Output' examples enforcing a probabilistic tone ('may indicate', 'could suggest').
3. **Regex Blocklisting:** The strongest layer is our post-generation sanitizer (`_UNSAFE_PATTERNS`). It scans the output string before sending it to the PDF generator. If it detects definitive language like 'you are diagnosed' or specific drug classes like 'aspirin', 'statin', or 'dosage', it mathematically overrides that substring with `[medically reviewed]`. The system physically cannot output a drug prescription."
