import React from "react";
import "../Pages/Have_no_risk.css";
import "../bootstrap.min.css";
import profile from "../Image/profile.png";
import Navbar from "../Components/Navbar/Navbar";
import logo from "../Image/logo.png";
import { Link } from "react-router-dom";
function Home() {
  return (
    <div className="home-page">
      {" "}
      
      {/* Hero Section */}{" "}
      <section className="hero text-center">
        {" "}
        <h2 className="hero-title">Heart Disease Prediction Tool</h2>{" "}
        <p className="hero-subtitle">Advanced AI-Powered Analysis To Assess </p>{" "}
        <p className="hero-subtitle">Your Heart Health Risk Factors</p>{" "}
        <div className="hero-buttons">
          {" "}
          <Link to="/prediction">
            <button className="btn custom-btn px-4 py-2 rounded-pill me-3">
               Start Prediction →
            </button>
          </Link>
          <Link to="/learnmore" className="btn learn btn-outline-dark rounded-pill">
            Learn More →
          </Link>
        </div>{" "}
      </section>{" "}
      {/* Result Section */}
      <section className="result-section text-center">
        <h3 className="title-result">
          The Percentage That You Have Heart Diseases Or Not
        </h3>

        <p className="result-note_">
          If The Percentage Is Higher Than 70% It Means You Have Heart Diseases
        </p>

        <div className="result-card mx-auto">
          <p className="result-label">The Percentage Is :</p>
          <h2 className="result-value_">55 %</h2>
          <p className="result-status">You Are Ok</p>
        </div>
      </section>
      
    </div>
  );
}
export default Home;
