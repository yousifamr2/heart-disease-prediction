import React, { useState, useEffect } from "react";
import axios from "axios";
import API_BASE_URL from "../../config";

import "./Prediction.css";

import { Link, useNavigate } from "react-router-dom";
import { BsGeoAltFill } from "react-icons/bs";
import { getLatestLabTest, startPrediction } from "../../services/api";
import mokhtabarImg from "../../assets/mokhtabar.png";
import borgImg from "../../assets/borg.png";
import hassabImg from "../../assets/hassab.png";
import royalImg from "../../assets/royal.png";
import shamsImg from "../../assets/shams.png";
import nileImg from "../../assets/nile.png";

const labImages = {
  "Al Mokhtabar labs": mokhtabarImg,
  "AL Borg Labs": borgImg,
  "Hassab Labs": hassabImg,
  "Royal Labs": royalImg,
  "Al Shams Labs": shamsImg,
  "Al Nile Labs": nileImg,
};

const labLinks = {
  "Al Mokhtabar labs":
    "https://almokhtabar.com/ar/%d8%a7%d9%84%d9%81%d8%b1%d9%88%d8%b9/",

  "AL Borg Labs":
    "https://alborglab.com/branches/",

  "Hassab Labs":
    "https://hassab.com/site/ar/%D9%81%D8%B1%D9%88%D8%B9%D9%86%D8%A7/",

  "Royal Labs":
    "https://royal-lab.net/ar/%D9%81%D8%B1%D9%88%D8%B9%D9%86%D8%A7/",

  "Al Shams Labs":
    "http://alshamslabs.com/branches.aspx",

  "Al Nile Labs":
    "https://nilescanandlabs.net/%D9%81%D8%B1%D9%88%D8%B9%D9%86%D8%A7/",
};

const Prediction = () => {
  // ================= STATE =================
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [labs, setLabs] = useState([]);
  const [hasLabTests, setHasLabTests] = useState(false);
  // eslint-disable-next-line no-unused-vars
  const [latestLabTest, setLatestLabTest] = useState(null);
  // eslint-disable-next-line no-unused-vars
  const [userLocation, setUserLocation] = useState(null);

  const navigate = useNavigate();

  const getStoredNationalId = () => {
    const storedUser = localStorage.getItem("user");

    if (!storedUser) return null;

    try {
      return JSON.parse(storedUser).national_id;
    } catch {
      return null;
    }
  };

  // ================= GET LABS + STATUS + LOCATION =================
  useEffect(() => {
    fetchLabs();
    fetchLatestLabTest();

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
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ================= FETCH LABS =================
  const fetchLabs = async () => {
    try {
      const res = await axios.get(
        `${API_BASE_URL}/api/labs`
      );

      console.log("LABS => ", res.data);

      setLabs(res.data.data);

    } catch (err) {
      console.log(err);
    }
  };

  // ================= CHECK LATEST LAB TEST =================
  const fetchLatestLabTest = async () => {
    try {
      const token = localStorage.getItem("token");
      const nationalId = getStoredNationalId();

      if (!token || !nationalId) return null;

      const res = await getLatestLabTest(nationalId);

      if (res?.status === 401) {
        localStorage.removeItem("token");
        localStorage.removeItem("user");

        navigate("/login");

        return null;
      }

      if (res?.success && res.data) {
        setHasLabTests(true);

        setLatestLabTest(res.data);

        setResult({
          probability: res.data.prediction_percentage,

          decision_label:
            res.data.prediction_result ||
            (res.data.prediction_percentage >= 50
              ? "High Risk"
              : "Low Risk"),
        });

        return res.data;
      }

      setHasLabTests(false);

      return null;

    } catch (err) {
      console.log(err);

      return null;
    }
  };

  // ================= START PREDICTION =================
  const handleStartPrediction = async () => {
    try {
      setLoading(true);

      const token = localStorage.getItem("token");

      if (!token) {
        alert("Please Login First");

        navigate("/login");

        return;
      }

      const response = await startPrediction();

      if (response?.status === 401) {
        alert(
          "Session expired or invalid token. Please log in again."
        );

        localStorage.removeItem("token");
        localStorage.removeItem("user");

        navigate("/login");

        return;
      }

      if (!response?.success) {
        if (
          response?.message
            ?.toLowerCase()
            .includes("no lab test")
        ) {
          alert(
            "No lab test found. Please visit a trusted medical lab first."
          );
        } else {
          alert(
            response?.message ||
            "Prediction failed"
          );
        }

        return;
      }

      const predictionData = response.data;

      localStorage.setItem(
        "prediction",
        JSON.stringify(predictionData)
      );

      localStorage.setItem(
        "prediction_id",
        predictionData.prediction_id
      );

      setResult(predictionData);

      const normalizedPrediction =
        (
          predictionData.decision ||
          predictionData.decision_label ||
          ""
        ).toLowerCase();

      if (
        normalizedPrediction.includes("low") ||
        (
          predictionData.probability != null &&
          predictionData.probability < 50
        )
      ) {
        navigate("/have_no_risk");

      } else if (
        normalizedPrediction.includes("high") ||
        (
          predictionData.probability != null &&
          predictionData.probability >= 50
        )
      ) {
        navigate("/have_risk");

      } else {
        alert(
          "Prediction information is not available."
        );
      }

    } catch (err) {
      console.log(err);

      alert(
        err?.response?.data?.message ||
        err?.message ||
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
          <h2>Loading...</h2>
        </div>
      </div>
    );
  }
  console.log(borgImg);

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

        <p className="report-title">
          The Percentage That You Have Heart Diseases Or Not
          <br />

          <span className="highlight">
            If the percentage is higher than 50%
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
                  href={labLinks[lab.name]}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="lab-card"
                >

                  <img
                    src={labImages[lab.name]}
                    alt={lab.name}
                    className="lab-image"
                  />


                  <div className="lab-details">
                    <div className="lab-header">
                      <h4>{lab.name}</h4>

                      <span className="lab-rating">
                        ⭐ {lab.rating}
                      </span>
                    </div>

                    <p className="lab-address">
                      <BsGeoAltFill />
                      {lab.address}
                    </p>
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