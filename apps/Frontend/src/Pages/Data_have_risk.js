import React, { useEffect, useState } from "react";
import axios from "axios";
<<<<<<< HEAD

import "../Pages/Data_have_risk.css";
import "@fortawesome/fontawesome-free/css/all.min.css";
import API_BASE_URL from "../config";
=======
import API_BASE_URL from "../config";


import "../Pages/Data_have_risk.css";
import "@fortawesome/fontawesome-free/css/all.min.css";
>>>>>>> main

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

  const [shapImage, setShapImage] =
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
      parsedPrediction?.probability < 70
    ) {

      navigate("/have_no_risk");


      return;
    }

    // ================= GET SHAP IMAGE =================
    if (parsedPrediction.show_shap) {

      fetchShapImage(
        parsedPrediction.prediction_id
      );
    }

    // ================= GET HOSPITALS =================
    fetchHospitals();

  }, [navigate]);

  // ================= FETCH SHAP IMAGE =================
  const fetchShapImage = async (
    predictionId
  ) => {

    try {

      const token =
        localStorage.getItem("token");

      const res = await axios.get(
        `${API_BASE_URL}/api/predictions/${predictionId}/shap`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
          responseType: "blob",
        }
      );

      const imageUrl =
        URL.createObjectURL(res.data);

      setShapImage(imageUrl);

    } catch (err) {

      console.log(err);

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

          If the percentage is higher than 70%
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

        {/* ================= SHAP IMAGE ================= */}
        {prediction?.show_shap &&
          shapImage && (

            <div className="shap-container">

              <h3 className="shap-title">

                The Most Effected Factor In The Result

              </h3>

              <img
                src={shapImage}
                alt="SHAP Explanation"
                className="shap-image"
              />

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

export default Home;