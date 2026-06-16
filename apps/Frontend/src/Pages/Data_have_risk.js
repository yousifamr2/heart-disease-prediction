import React, { useEffect, useState } from "react";
import axios from "axios";
import API_BASE_URL from "../config";


import "../Pages/Data_have_risk.css";
import "@fortawesome/fontawesome-free/css/all.min.css";

import { useNavigate } from "react-router-dom";

import {
  FaMapMarkerAlt,
  FaDownload,
} from "react-icons/fa";

function Home() {

  const [prediction, setPrediction] =
    useState(null);

  const [hospitals, setHospitals] =
    useState([]);

  const [shapData, setShapData] =
    useState(null);

  const navigate = useNavigate();

  // ================= GET DATA =================
  useEffect(() => {

    const token =
      localStorage.getItem("token");

    // ================= PROTECT PAGE =================
    if (!token) {

      navigate("/login");

      return;
    }

    // ================= GET PREDICTION =================
    const savedPrediction =
      localStorage.getItem("prediction");

    // ================= NO PREDICTION =================
    if (!savedPrediction) {

      navigate("/prediction");

      return;
    }

    const parsedPrediction =
      JSON.parse(savedPrediction);

    setPrediction(parsedPrediction);

    // ================= REDIRECT IF LOW RISK =================
    if (
      parsedPrediction?.probability < 50
    ) {

      navigate("/have_no_risk");


      return;
    }

    // ================= GET FEATURE IMPORTANCE DATA =================
    if (parsedPrediction.show_shap) {

      fetchShapData(
        parsedPrediction.prediction_id
      );
    }

    // ================= GET HOSPITALS =================
    fetchHospitals();

  }, [navigate]);

  // ================= GET SHAP DATA =================
  const fetchShapData = async (
    predictionId
  ) => {

    try {

      const token =
        localStorage.getItem("token");

      const res = await axios.get(
        `${API_BASE_URL}/api/predictions/${predictionId}/shap/data`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (res.data && res.data.success) {
        setShapData(res.data.data);
      }

    } catch (err) {

      console.log("Error fetching SHAP data:", err);

    }
  };

  // ================= GET NEAR HOSPITALS =================
  const fetchHospitals = async () => {

    try {

      // ================= GET ALL HOSPITALS =================
      const hospitalsRes = await axios.get(
        `${API_BASE_URL}/api/hospitals?limit=100`
      );

      const allHospitals = hospitalsRes.data.data;

      // ================= GROUP BY AREA =================
      const grouped = allHospitals.reduce((acc, hospital) => {
        // Extract main city name (e.g., "Alexandria , Egypt" -> "Alexandria")
        let areaName = hospital.area.split(",")[0].trim();
        if (!acc[areaName]) {
          acc[areaName] = [];
        }
        acc[areaName].push(hospital);
        return acc;
      }, {});

      setHospitals(grouped);

    } catch (err) {

      console.log(err);

    }
  };

  // ================= DOWNLOAD REPORT =================
  const handleDownloadReport =
    async () => {

      try {

        const token =
          localStorage.getItem("token");

        const predictionId =
          prediction?.prediction_id;

        if (!predictionId) {

          alert("No Report Found");

          return;
        }

        const res = await axios.get(
          `${API_BASE_URL}/api/predictions/${predictionId}/report`,
          {
            headers: {
              Authorization:
                `Bearer ${token}`,
            },
            responseType: "blob",
          }
        );

        const url =
          window.URL.createObjectURL(
            new Blob([res.data])
          );

        const link =
          document.createElement("a");

        link.href = url;

        link.setAttribute(
          "download",
          "Heart_Report.pdf"
        );

        document.body.appendChild(link);

        link.click();

      } catch (err) {

        console.log(err);

        alert(
          "Failed To Download Report"
        );
      }
    };

  return (

    <div className="home-page">

      {/* RESULT SECTION */}
      <section className="result-section text-center" style={{ paddingTop: '80px' }}>

        <h3 className="result-title">

          The Percentage That You Have Heart Disease Or Not

        </h3>

        <p className="result-note">

          If the percentage is higher than 50%
          it means you have heart disease

        </p>

        <div className="result-card mx-auto">

          <p className="result-label">

            The Percentage Is :

          </p>

          <p className="result-value">

            {prediction?.probability
              ? `${prediction.probability}%`
              : "0%"}

          </p>

          <p className="result-status">

            {prediction?.decision_label ||
              "Heart Disease Risk"}

          </p>

        </div>

        {/* ================= INTERACTIVE FEATURE IMPORTANCE CHART ================= */}
        {prediction?.show_shap && (

          <div className="shap-container">

            <h3 className="shap-title">

              The Most Effected Factors In The Result

            </h3>

            {shapData ? (
              <InteractiveShapChart data={shapData} />
            ) : (
              <p className="no-data">Loading feature importance data...</p>
            )}

          </div>

        )}

        {/* REPORT */}
        {prediction?.show_report && (

          <div className="medical-report-container">

            <h3 className="medical-report-title">

              Your Medical Report:

            </h3>

            <button
              onClick={handleDownloadReport}
              className="download-report-btn"
            >

              <FaDownload className="download-icon" />

              Download Report

            </button>

          </div>

        )}

      </section>

      {/* HOSPITALS */}
      {prediction?.show_hospitals && (

        <section className="hospitals-section">

          <h3 className="cap-effect">

            You Should Go To One Of These Hospitals
            <br />
            That Specialize In Heart Diseases.

          </h3>

          <div className="hospitals-wrapper" style={{ textAlign: "left" }}>

            {Object.keys(hospitals).length > 0 ? (

              Object.entries(hospitals)
                .sort(([a], [b]) => a.localeCompare(b))
                .map(([area, areaHospitals]) => (

                  <div key={area} className="area-group">

                    <div className="city-box">
                      <h4 className="city-title">{area}</h4>
                    </div>

                    <div className="hospitals-container">

                      {areaHospitals.map((hospital) => (

                        <a
                          key={hospital.id}
                          href={hospital.google_maps_link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="hospital-card"
                        >

                          {/* HOSPITAL NAME */}
                          <p className="hospital-name">

                            {hospital.name}

                          </p>

                          {/* LOCATION */}
                          <div className="location">

                            <FaMapMarkerAlt className="location_icon" />

                            <span className="location_name">

                              {hospital.area}

                            </span>

                          </div>

                          {/* GOOGLE MAP */}
                          <iframe
                            title={hospital.name}
                            src={`https://maps.google.com/maps?q=${encodeURIComponent(
                              hospital.name + " " + hospital.area
                            )}&t=&z=13&ie=UTF8&iwloc=&output=embed`}
                            className="hospital-map"
                            loading="lazy"
                          ></iframe>

                        </a>

                      ))}

                    </div>

                  </div>

                ))

            ) : (

              <h3 className="text-center mt-5">

                No Nearby Hospitals Found

              </h3>
            )}

          </div>

        </section>

      )}

    </div>
  );
}

// ==================== INTERACTIVE SHAP CHART COMPONENT ====================
function InteractiveShapChart({ data }) {
  const [hoveredBar, setHoveredBar] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [animate, setAnimate] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  useEffect(() => {
    // Trigger slide-in animation shortly after mount
    const timer = setTimeout(() => setAnimate(true), 100);
    
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
    };
    window.addEventListener("resize", handleResize);
    
    return () => {
      clearTimeout(timer);
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  if (!data || !data.top_features || data.top_features.length === 0) {
    return <p className="no-data">No SHAP data available.</p>;
  }

  const features = data.top_features;
  const maxVal = Math.max(...features.map(f => Math.abs(f.impact)), 0.10);

  // Calculate ticks
  const tickCount = 6;
  const ticks = [];
  for (let i = 0; i < tickCount; i++) {
    ticks.push((maxVal / (tickCount - 1)) * i);
  }

  // Layout constants
  const width = 760;
  const height = 460;
  const margin = { left: 180, right: 40, top: 40, bottom: 50 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;
  const rowHeight = chartHeight / features.length;
  const barHeight = 18;

  const featureDescriptions = {
    "ST slope": "How your heart recovers after exercise. A flat or downward slope can mean the heart muscle isn't getting enough oxygen.",
    "oldpeak": "Stress-induced heart strain. Measures how much your heart struggled during physical activity compared to when resting.",
    "chest pain type": "The kind of chest discomfort you feel. Typical heart pain is a tight, squeezing pressure, while other types are less specific.",
    "exercise angina": "Chest pain triggered specifically by physical exertion. If yes, it suggests blood vessels might be narrowed.",
    "max heart rate": "The highest speed your heart reached during exercise. Lower maximum rates can indicate a weaker cardiac response.",
    "age": "Your age. Heart disease risk naturally increases as blood vessels age and stiffen over time.",
    "fasting blood sugar": "The amount of sugar in your blood after fasting. High blood sugar can damage blood vessels and speed up heart risks.",
    "cholesterol": "The level of fat in your blood. High cholesterol can build up inside your arteries, creating dangerous blockages.",
    "sex": "Biological sex. Men statistically tend to develop heart issues at an earlier age than women.",
    "resting bp s": "Your resting blood pressure. High pressure forces your heart to work much harder and damages arterial walls.",
    "resting ecg": "An electrical recording of your resting heart. It detects irregular rhythms or signs of previous heart strain.",
  };

  const formatFeatureValue = (feature, val) => {
    if (val === undefined || val === null || val === "N/A") return "N/A";
    const numVal = Number(val);
    switch (feature) {
      case "sex":
        return numVal === 1 ? "Male" : "Female";
      case "chest pain type":
        if (numVal === 1) return "Typical Angina (1)";
        if (numVal === 2) return "Atypical Angina (2)";
        if (numVal === 3) return "Non-anginal Pain (3)";
        if (numVal === 4) return "Asymptomatic (4)";
        return val;
      case "fasting blood sugar":
        return numVal === 1 ? "> 120 mg/dL (1)" : "≤ 120 mg/dL (0)";
      case "exercise angina":
        return numVal === 1 ? "Yes (1)" : "No (0)";
      case "ST slope":
        if (numVal === 1) return "Upsloping (1)";
        if (numVal === 2) return "Flat (2)";
        if (numVal === 3) return "Downsloping (3)";
        return val;
      case "resting ecg":
        if (numVal === 0) return "Normal (0)";
        if (numVal === 1) return "ST-T Wave Abnormality (1)";
        if (numVal === 2) return "Left Ventricular Hypertrophy (2)";
        return val;
      case "age":
        return `${val} years`;
      case "resting bp s":
        return `${val} mmHg`;
      case "cholesterol":
        return `${val} mg/dL`;
      case "max heart rate":
        return `${val} bpm`;
      case "oldpeak":
        return `${val} (ST depression)`;
      default:
        return String(val);
    }
  };

  const handleMouseMove = (e, feature) => {
    const card = e.currentTarget.closest(".shap-chart-card");
    if (!card) return;
    const rect = card.getBoundingClientRect();
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;
    
    // Bounds check to prevent tooltip from overflowing the card boundary
    const tooltipWidth = 260;
    if (x + tooltipWidth + 30 > rect.width) {
      x = x - tooltipWidth - 25;
    }
    
    setTooltipPos({ x, y });
    setHoveredBar(feature);
  };

  if (isMobile) {
    return (
      <div className="shap-mobile-container">
        <div className="shap-mobile-list">
          {features.map((item, idx) => {
            const absoluteImpact = Math.abs(item.impact);
            const percent = Math.min(100, (absoluteImpact / maxVal) * 100);
            const isIncrease = item.direction === "increase";
            
            return (
              <div key={idx} className="shap-mobile-item-card">
                <div className="shap-mobile-header">
                  <span className="shap-mobile-feature">{item.feature}</span>
                  <span className="shap-mobile-value">
                    Value: <strong>{formatFeatureValue(item.feature, item.value)}</strong>
                  </span>
                </div>
                
                <div className="shap-mobile-bar-wrapper">
                  <div 
                    className="shap-mobile-bar" 
                    style={{ 
                      width: animate ? `${percent}%` : "0%",
                      background: isIncrease 
                        ? "linear-gradient(90deg, #ef4444, #dc2626)" 
                        : "linear-gradient(90deg, #10b981, #059669)",
                      transition: "width 0.8s cubic-bezier(0.16, 1, 0.3, 1)"
                    }}
                  />
                </div>
                
                <div className="shap-mobile-meta">
                  <span className="shap-mobile-score">
                    Importance Score: <strong>{absoluteImpact.toFixed(4)}</strong>
                  </span>
                  <span className={`shap-mobile-direction ${item.direction}`}>
                    {isIncrease ? (
                      <>
                        <i className="fa-solid fa-triangle-exclamation" style={{ marginRight: "4px" }}></i>
                        Increases Risk
                      </>
                    ) : (
                      <>
                        <i className="fa-solid fa-circle-check" style={{ marginRight: "4px" }}></i>
                        Decreases Risk
                      </>
                    )}
                  </span>
                </div>
                
                <div className="shap-mobile-desc">
                  {featureDescriptions[item.feature] || "N/A"}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <div className="shap-chart-card">
      <div className="shap-chart-header">
        <h4 className="shap-chart-subtitle">Feature Importance</h4>
      </div>

      <div className="shap-svg-wrapper">
        <svg viewBox={`0 0 ${width} ${height}`} className="shap-svg">
          <defs>
            {/* Gradient for bars */}
            <linearGradient id="barGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#2563eb" />
              <stop offset="100%" stopColor="#3b82f6" />
            </linearGradient>
            <linearGradient id="barGradientHover" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#1d4ed8" />
              <stop offset="100%" stopColor="#2563eb" />
            </linearGradient>

            {/* Subtle drop shadow filter for bars */}
            <filter id="barShadow" x="-10%" y="-10%" width="120%" height="120%">
              <feDropShadow dx="1" dy="1" stdDeviation="1.5" floodOpacity="0.1" />
            </filter>
          </defs>

          {/* Border surrounding the entire plot */}
          <rect
            x={margin.left}
            y={margin.top}
            width={chartWidth}
            height={chartHeight}
            fill="none"
            stroke="#2c3e50"
            strokeWidth="1.2"
          />

          {/* Grid lines (vertical ticks) */}
          {ticks.map((tick, idx) => {
            const x = margin.left + (tick / maxVal) * chartWidth;
            return (
              <g key={idx}>
                {idx > 0 && idx < ticks.length - 1 && (
                  <line
                    x1={x}
                    y1={margin.top}
                    x2={x}
                    y2={margin.top + chartHeight}
                    stroke="#e2e8f0"
                    strokeWidth="0.8"
                    strokeDasharray="2,2"
                  />
                )}
                <line
                  x1={x}
                  y1={margin.top + chartHeight}
                  x2={x}
                  y2={margin.top + chartHeight + 5}
                  stroke="#2c3e50"
                  strokeWidth="1.2"
                />
                <text
                  x={x}
                  y={margin.top + chartHeight + 20}
                  textAnchor="middle"
                  className="shap-tick-text"
                  fontSize="11"
                  fill="#475569"
                >
                  {tick.toFixed(2)}
                </text>
              </g>
            );
          })}

          {/* Render Bars and Labels */}
          {features.map((item, idx) => {
            const rowY = margin.top + idx * rowHeight;
            const barY = rowY + (rowHeight - barHeight) / 2;
            const absoluteImpact = Math.abs(item.impact);
            const targetWidth = (absoluteImpact / maxVal) * chartWidth;
            const currentWidth = animate ? targetWidth : 0;

            const isHovered = hoveredBar && hoveredBar.feature === item.feature;

            return (
              <g key={idx} className="shap-row-group">
                {/* Row Hover Background */}
                <rect
                  x={margin.left - 170}
                  y={rowY + 1}
                  width={width - 20}
                  height={rowHeight - 2}
                  fill={isHovered ? "rgba(241, 245, 249, 0.6)" : "transparent"}
                  rx="4"
                  style={{ transition: "fill 0.2s ease" }}
                />

                {/* Y-axis Label */}
                <text
                  x={margin.left - 15}
                  y={rowY + rowHeight / 2 + 4}
                  textAnchor="end"
                  className="shap-y-label"
                  fontSize="12.5"
                  fontWeight={isHovered ? "600" : "500"}
                  fill={isHovered ? "#1e293b" : "#475569"}
                  style={{ transition: "all 0.2s ease" }}
                >
                  {item.feature}
                </text>

                {/* Bar */}
                <rect
                  x={margin.left}
                  y={barY}
                  width={currentWidth}
                  height={barHeight}
                  fill={isHovered ? "url(#barGradientHover)" : "url(#barGradient)"}
                  filter="url(#barShadow)"
                  rx="2"
                  className="shap-bar"
                  style={{
                    transition: "width 0.8s cubic-bezier(0.16, 1, 0.3, 1), fill 0.2s ease",
                    cursor: "pointer"
                  }}
                  onMouseMove={(e) => handleMouseMove(e, item)}
                  onMouseLeave={() => setHoveredBar(null)}
                />

                {/* Value overlay inside the bar */}
                {isHovered && absoluteImpact > 0.01 && (
                  <text
                    x={margin.left + currentWidth - 8}
                    y={barY + barHeight / 2 + 4}
                    textAnchor="end"
                    fontSize="9.5"
                    fontWeight="700"
                    fill="#ffffff"
                    pointerEvents="none"
                  >
                    {absoluteImpact.toFixed(3)}
                  </text>
                )}
              </g>
            );
          })}

          {/* Bottom X-axis Title (Removed) */}
        </svg>
      </div>

      {/* Floating Tooltip */}
      {hoveredBar && (
        <div
          className="shap-tooltip"
          style={{
            left: `${tooltipPos.x + 15}px`,
            top: `${tooltipPos.y + 15}px`,
          }}
        >
          <div className="tooltip-header">
            <span className="tooltip-feature-name">{hoveredBar.feature}</span>
          </div>
          <div className="tooltip-body">
            <div className="tooltip-row">
              <span className="tooltip-label">Description:</span>
              <span className="tooltip-value desc">{featureDescriptions[hoveredBar.feature] || "N/A"}</span>
            </div>
            <div className="tooltip-row highlight-row">
              <span className="tooltip-label">Your Measured Value:</span>
              <span className="tooltip-value user-val">{formatFeatureValue(hoveredBar.feature, hoveredBar.value)}</span>
            </div>
            <div className="tooltip-row">
              <span className="tooltip-label">Importance Score:</span>
              <span className="tooltip-value score">{Math.abs(hoveredBar.impact).toFixed(4)}</span>
            </div>
            <div className="tooltip-row">
              <span className="tooltip-label">Risk Direction:</span>
              <span className={`tooltip-value direction ${hoveredBar.direction}`}>
                {hoveredBar.direction === "increase" ? (
                  <>
                    <i className="fa-solid fa-triangle-exclamation" style={{ marginRight: "4px" }}></i>
                    Increases Risk
                  </>
                ) : (
                  <>
                    <i className="fa-solid fa-circle-check" style={{ marginRight: "4px" }}></i>
                    Decreases Risk
                  </>
                )}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Home;