import React, { useEffect, useState } from "react";

import "../Pages/Have_no_risk.css";

import { useNavigate } from "react-router-dom";

import {
  FaCheckCircle,
} from "react-icons/fa";

function Home() {

  const [prediction, setPrediction] =
    useState(null);

  const navigate = useNavigate();

  // ================= GET PREDICTION =================
  useEffect(() => {

    const token =
      localStorage.getItem("token");

    // ================= PROTECT PAGE =================
    if (!token) {

      navigate("/login");

      return;
    }

    const savedPrediction =
      localStorage.getItem("prediction");

    // ================= NO PREDICTION =================
    if (!savedPrediction) {

      navigate("/prediction");

      return;
    }

    // ================= GET SAVED DATA =================
    const parsedPrediction =
      JSON.parse(savedPrediction);

    setPrediction(parsedPrediction);

    // ================= REDIRECT IF RISK EXISTS =================
    if (
      parsedPrediction?.probability >= 50
    ) {

      navigate("/have_risk");

    }

  }, [navigate]);

  return (

    <div className="home-page">

      {/* ================= RESULT SECTION ================= */}
      <section className="result-section text-center" style={{ paddingTop: '80px' }}>

        <h3 className="title-result">

          Your Heart Health Result

        </h3>

        <p className="result-note_">

          Your heart condition appears healthy
          based on the AI prediction analysis.

        </p>

        <div className="result-card mx-auto">

          <p className="result-label">

            Your Heart Disease Risk :

          </p>

          {/* ================= PERCENTAGE ================= */}
          <h2 className="result-value_">

            {prediction?.probability != null
              ? `${prediction.probability}%`
              : "0%"}

          </h2>

          {/* ================= STATUS ================= */}
          <p className="result-status healthy-status">

            <FaCheckCircle />

            You Are Healthy ❤️

          </p>

        </div>

        {/* ================= HEALTH TIPS ================= */}
        <div className="tips-container">

          <h3 className="tips-title">

            Healthy Lifestyle Tips

          </h3>

          <div className="tips-grid">

            <div className="tip-card">

              🥗 Eat healthy food rich in vegetables and fruits

            </div>

            <div className="tip-card">

              🏃 Exercise regularly for at least 30 minutes daily

            </div>

            <div className="tip-card">

              💧 Drink enough water every day

            </div>

            <div className="tip-card">

              😴 Maintain good sleep habits and reduce stress

            </div>

          </div>

        </div>

      </section>

    </div>
  );
}

export default Home;