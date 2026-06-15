import React, { useState, useEffect } from "react";

import axios from "axios";
<<<<<<< HEAD

import "./Prediction.css";
import API_BASE_URL from "../../config";
=======
import API_BASE_URL from "../../config";


import "./Prediction.css";
>>>>>>> main

import {
  Link,
  useNavigate,
} from "react-router-dom";

import {
  BsGeoAltFill,
} from "react-icons/bs";

const Prediction = () => {

  // ================= STATE =================
  const [result, setResult] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [labs, setLabs] =
    useState([]);

  const [hasLabTests, setHasLabTests] =
    useState(false);

  const [userLocation, setUserLocation] =
    useState(null);

  const navigate = useNavigate();

  // ================= GET LABS + STATUS + LOCATION =================
  useEffect(() => {

    fetchLabs();

    checkLabStatus();

    navigator.geolocation.getCurrentPosition(

      (position) => {

        setUserLocation({
          lat: position.coords.latitude,
          lng: position.coords.longitude,
        });

      },

      (error) => {

        console.log(error);

      }

    );

  }, []);

  // ================= FETCH LABS =================
  const fetchLabs = async () => {

    try {

      const res = await axios.get(
        `${API_BASE_URL}/api/labs`
      );

      console.log(
        "LABS => ",
        res.data
      );

      setLabs(
        res.data.data
      );

    } catch (err) {

      console.log(err);

    }
  };

  // ================= CHECK LAB TEST STATUS =================
  const checkLabStatus = async () => {

    try {

      const token =
        localStorage.getItem("token");

      if (!token) return;

      const res = await axios.get(
<<<<<<< HEAD
=======

>>>>>>> main
        `${API_BASE_URL}/api/labtests/me/status`,

        {
          headers: {
            Authorization:
              `Bearer ${token}`,
          },
        }

      );

      console.log(
        "LAB STATUS => ",
        res.data
      );

      setHasLabTests(
        res.data.data.hasLabTests
      );

    } catch (err) {

      console.log(err);

    }
  };

  // ================= START PREDICTION =================
  const handleStartPrediction =
    async () => {

      try {

        setLoading(true);

        const token =
          localStorage.getItem("token");

        // ================= CHECK LOGIN =================
        if (!token) {

          alert(
            "Please Login First"
          );

          setLoading(false);

          return;
        }

        // ================= CHECK LAB TESTS =================
        if (!hasLabTests) {

          alert(
            "No lab test found. Please visit a trusted medical lab first."
          );

          setLoading(false);

          return;
        }

        // ================= START PREDICTION =================
        const res = await axios.post(
<<<<<<< HEAD
=======

>>>>>>> main
          `${API_BASE_URL}/api/predictions/start`,

          {},

          {
            headers: {
              Authorization:
                `Bearer ${token}`,
            },
          }

        );

        console.log(
          "FULL RESPONSE => ",
          res.data
        );

        const predictionData =
          res.data.data;

        console.log(
          "PREDICTION DATA => ",
          predictionData
        );

        // ================= SAVE =================
        localStorage.setItem(
          "prediction",
          JSON.stringify(
            predictionData
          )
        );

        localStorage.setItem(
          "prediction_id",
          predictionData.prediction_id
        );

        setResult(
          predictionData
        );

        // ================= NAVIGATE =================
        if (
          predictionData.probability < 70
        ) {


          navigate(
            "/have_no_risk"
          );


        } else {

          navigate(
            "/have_risk"
          );

        }

      } catch (err) {

        console.log(err);

        alert(

          err.response?.data?.message ||

          "Prediction Failed"

        );

      } finally {

        setLoading(false);

      }
    };

    

  // ================= LOADING =================
  if (loading) {

    return (

      <div className="prediction-page">

        <div className="prediction-card">

          <h2>
            Loading...
          </h2>

        </div>

      </div>
    );
  }

  // ================= UI =================
  return (

    <div className="prediction-page">

      <div className="prediction-card">

        <h1>
          Heart Disease Prediction Tool
        </h1>

        <p className="subtitle">

          Advanced AI Powered Analysis To Assess

          <br />

          <span>
            Your Heart Health Risk Factors
          </span>

        </p>

        {/* ================= BUTTONS ================= */}
        <div className="prediction-buttons">

          <button
            onClick={handleStartPrediction}
            className="btn start"
          >

            Start Prediction →

          </button>

          <Link
            to="/learnmore"
            className="btn learn"
          >

            Learn More →

          </Link>

        </div>

        {/* ================= REPORT ================= */}
        <p className="report-title">

          The Percentage That You Have Heart Diseases Or Not

          <br />

          <span className="highlight">

            If the percentage is higher than 70%
            it means you have Heart Diseases

          </span>

        </p>

        <div className="report-box">

          <h4>

            {result?.probability != null
              ? `${result.probability}%`
              : "No Prediction Yet"}

          </h4>

          <span>

            {result
              ? result.decision_label
              : hasLabTests
              ? "Ready To Start Prediction"
              : "Please Visit A Trusted Lab First"}

          </span>

        </div>

        <p className="info-text">

          {hasLabTests

            ? "Your Lab Results Are Ready For Prediction"

            : "You Should Go To Trusted Medical Labs So You Can Start Prediction"}

        </p>

        {/* ================= LABS SECTION ================= */}
        {!hasLabTests && (
          <div className="labs-section">

            <div className="labs-top">

              <div>

                <h3 className="labs-title">
                  Trusted Medical Labs
                </h3>

                <p className="labs-sub">

                  There Is Thousands Of Trusted Medical Labs

                </p>

              </div>

            </div>

            {/* ================= DYNAMIC LABS ================= */}
            <div className="labs-wrapper">

              {labs.map((lab) => (

                <a
                  key={lab.id}

                  href={
                    userLocation

                      ? `https://www.google.com/maps/dir/${userLocation.lat},${userLocation.lng}/${encodeURIComponent(lab.address)}`

                      : `https://www.google.com/maps/search/${encodeURIComponent(lab.address)}`
                  }

                  target="_blank"

                  rel="noopener noreferrer"

                  className="lab-card"
                >

                  <div className="lab-content">

                    <div className="lab-title-row">

                      <h4>
                        {lab.name}
                      </h4>

                      <span className="rating-badge">
                        Lab
                      </span>

                    </div>

                    <div className="lab-info">

                      <p>

                        <BsGeoAltFill />

                        {lab.address}

                      </p>

                    </div>

                  </div>

                </a>

              ))}

            </div>

          </div>
        )}

      </div>

    </div>
  );
};

export default Prediction;