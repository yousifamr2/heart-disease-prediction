import React, { useState } from "react";
import "./Docs.css";

export default function Docs() {
  const [activeTab, setActiveTab] = useState("guide");

  const factors = [
    { name: "Age & Sex", desc: "Age and biological sex are fundamental demographic risk factors in cardiovascular health assessments." },
    { name: "Chest Pain Type", desc: "Categorized into Typical Angina, Atypical Angina, Non-anginal pain, and Asymptomatic. Typical/Atypical angina are strongly correlated with coronary artery disease." },
    { name: "Resting Blood Pressure", desc: "The blood pressure (systolic) in mmHg measured upon resting. Values higher than 120-130 mmHg suggest hypertension risks." },
    { name: "Cholesterol Level", desc: "Serum cholesterol in mg/dL. Desirable levels are below 200 mg/dL. Higher values indicate plaque build-up risks." },
    { name: "Fasting Blood Sugar", desc: "Indicates if fasting blood sugar level > 120 mg/dL, highlighting potential diabetes, a significant heart disease accelerator." },
    { name: "Resting ECG", desc: "Resting electrocardiogram patterns categorized as normal, ST-T wave abnormalities, or left ventricular hypertrophy." },
    { name: "Max Heart Rate (Thalach)", desc: "The maximum heart rate achieved during exercise. Lower values relative to age can suggest cardiac output limitations." },
    { name: "Exercise-Induced Angina", desc: "Indicates whether physical exercise induces chest pain/angina symptoms (Yes/No)." },
    { name: "ST Depression (Oldpeak)", desc: "ST depression induced by exercise relative to rest. Indicates potential myocardial ischemia." },
    { name: "ST Slope", desc: "The slope of the peak exercise ST segment (Upsloping, Flat, Downsloping). Flat or downsloping patterns suggest blockages." }
  ];

  return (
    <div className="docs-page-container">
      <div className="docs-header">
        <h1 className="docs-title">Documentation & Help</h1>
        <p className="docs-subtitle">Learn how Nabdak calculates and handles heart care predictions.</p>
      </div>

      <div className="docs-tabs-wrapper">
        <div className="docs-tabs">
          <button className={`doc-tab-btn ${activeTab === "guide" ? "active" : ""}`} onClick={() => setActiveTab("guide")}>Patient Guide</button>
          <button className={`doc-tab-btn ${activeTab === "factors" ? "active" : ""}`} onClick={() => setActiveTab("factors")}>Medical Factors</button>
          <button className={`doc-tab-btn ${activeTab === "ai" ? "active" : ""}`} onClick={() => setActiveTab("ai")}>AI & Thresholds</button>
          <button className={`doc-tab-btn ${activeTab === "ecg" ? "active" : ""}`} onClick={() => setActiveTab("ecg")}>ECG Guide</button>
        </div>
      </div>

      <div className="docs-content-card">
        {activeTab === "guide" && (
          <div className="tab-content animate-fade">
            <h2>Patient Guide</h2>
            <p className="lead-text">Follow these steps to generate and view your diagnostic reports:</p>
            <ol className="guide-steps">
              <li>
                <strong>Visit a Medical Lab:</strong> 
                Go to one of our trusted partner medical labs (e.g. Al Mokhtabar, Al Borg, Hassab, Royal, Al Shams, Al Nile) and complete a standard cardiovascular blood panel/ECG check.
              </li>
              <li>
                <strong>Wait for Lab Upload:</strong> 
                The lab will upload your medical test variables directly to Nabdak under your National ID.
              </li>
              <li>
                <strong>Run AI Prediction:</strong> 
                Login to your account, navigate to the <em>Heart Disease Prediction Tool</em>, and click <strong>Start Prediction</strong>.
              </li>
              <li>
                <strong>View Results & Download PDF:</strong> 
                If your risk score is above 50% (High Risk), you can view interactive SHAP charts explaining the primary contributors and download a complete, clinical PDF medical report.
              </li>
            </ol>
          </div>
        )}

        {activeTab === "factors" && (
          <div className="tab-content animate-fade">
            <h2>Medical Factors Explained</h2>
            <p className="lead-text">Nabdak utilizes the following 11 key clinical features for diagnosis:</p>
            <div className="factors-list">
              {factors.map((f, idx) => (
                <div key={idx} className="factor-item">
                  <h4>{f.name}</h4>
                  <p>{f.desc}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === "ecg" && (
          <div className="tab-content animate-fade">
            <h2>ECG Analysis Guide</h2>
            <p className="lead-text">How Nabdak visualizes and analyzes Electrocardiogram (ECG) data:</p>
            <div className="ecg-guide-box">
              <div className="ecg-step-card">
                <h3>1. Lab Upload</h3>
                <p>
                  Partner labs upload your raw ECG records using standard WFDB format (e.g. <code>.dat</code> and <code>.hea</code> files) via the secure lab portal.
                </p>
              </div>
              <div className="ecg-step-card">
                <h3>2. AI Processing</h3>
                <p>
                  Our internal AI pipeline analyzes the signal peaks, identifies intervals (like QRS complexes, P waves, and T waves), and checks for common arrhythmias or irregularities.
                </p>
              </div>
              <div className="ecg-step-card">
                <h3>3. Interactive Visualizations</h3>
                <p>
                  Once analyzed, a high-resolution signal chart is generated, showing the waveforms clearly. Patients and doctors can view this interactive graph directly on the dashboard.
                </p>
              </div>
              <div className="ecg-step-card">
                <h3>4. ECG PDF Reports</h3>
                <p>
                  A comprehensive PDF medical report with details on detected heart rate, anomalies, and clinical findings can be downloaded with a single click.
                </p>
              </div>
            </div>
          </div>
        )}

        {activeTab === "ai" && (
          <div className="tab-content animate-fade">
            <h2>AI & Decision Logic</h2>
            <p className="lead-text">How Nabdak interprets risk probability percentage output:</p>
            <div className="ai-explanation-box">
              <div className="boundary-card low-risk">
                <h3>Low Risk (&lt; 50%)</h3>
                <p>Represents a healthy cardiovascular assessment. Standard lifestyle tips and routine checkups are recommended.</p>
              </div>
              <div className="boundary-card high-risk">
                <h3>High Risk (&gt;= 50%)</h3>
                <p>A higher probability of coronary heart disease. Clinical attention is recommended, and specialist consult lists are displayed.</p>
              </div>
            </div>
            <h3 style={{ marginTop: "30px" }}>SHAP Explanations</h3>
            <p>
              Rather than presenting a black-box result, Nabdak runs SHAP (SHapley Additive exPlanations) 
              inference to output exactly how much each patient medical value shifted the model prediction 
              away from the baseline average. Elements pushing right (red/positive) increase risk, while elements 
              pushing left (green/negative) reduce risk.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
