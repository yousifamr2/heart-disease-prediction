import React from "react";
import "../Pages/Data_have_risk.css";
import "../bootstrap.min.css";
import "../fontawesome-free-7.0.0-web/css/all.min.css";
import profile from "../Image/profile.png";
import effect from "../Image/The_most_factor.jpg";
import logo from "../Image/logo.png";
import location_icon from "../Image/Loction_icon.png";
import { Link } from "react-router-dom";
import { Navbar } from "react-bootstrap";

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
      {/* Result Section */}{" "}
      <section className="result-section text-center">
        {" "}
        <h3 className="result-title">
          {" "}
          The Percentage That You Have Heart Diseases Or Not{" "}
        </h3>{" "}
        <p className="result-note">
          {" "}
          If The Percentage Is Higher Than 70% It Means You Have Heart
          Diseases{" "}
        </p>{" "}
        <div className="result-card mx-auto">
          {" "}
          <p className="result-label">The Percentage Is :</p>{" "}
          <p className="result-value">70 %</p>{" "}
          <p className="result-status">you Have A problem</p>{" "}
        </div>{" "}
        <div className="most-effect">
          {" "}
          <h3 className="effect-title">
            {" "}
            The Most Effected Factor In The Result{" "}
          </h3>{" "}
          <img className="effect-img" src={effect} />{" "}
          <h3 className="cap-effect">
            {" "}
            You should Go To One Of These Hospitals <br /> That Specialize In
            Heart Diseases.{" "}
          </h3>{" "}
        </div>{" "}
      </section>{" "}
      {/* Hospitals Section */}
      <section className="hospitals-section">
        {/* Alexandria */}
        <div className="city-title position-relative">
          <h5 className="city-tittle">Alexandria</h5>
        </div>
        <div className="hospitals-container">
          <div>
            <a
              href="https://www.google.com/maps?q=31.1716947,29.9435354"
              target="_blank"
              rel="noopener noreferrer"
              className="hospital-card me-2 "
            >
              <p className="hospital-name text-center">Elite Hospital</p>
              <div className="d-flex align-items-center justify-content-center gap-2  mb-2 location">
                <img src={location_icon} className="location_icon" />
                <p className="location_name text-center"> Alexandria ,Egypt</p>
              </div>
              <iframe
                className="hospital-map"
                src="https://maps.google.com/maps?q=31.1716947,29.9435354&z=15&output=embed"
                loading="lazy"
              ></iframe>
            </a>
          </div>
          <div>
            <a
              href="https://www.google.com/maps?q=31.2034328,29.9097238"
              target="_blank"
              rel="noopener noreferrer"
              className="hospital-card me-2"
            >
              <p className="hospital-name text-center">Andalusia Hospitals</p>
              <div className="d-flex align-items-center justify-content-center gap-2  mb-2 location">
                <img src={location_icon} className="location_icon" />
                <p className="location_name text-center"> Alexandria ,Egypt</p>
              </div>
              <iframe
                className="hospital-map"
                src="https://maps.google.com/maps?q=31.2034328,29.9097238&z=15&output=embed"
                loading="lazy"
              ></iframe>
            </a>
          </div>
          <div>
            <a
              href="https://www.google.com/maps?q=31.2158436,29.9419761"
              target="_blank"
              rel="noopener noreferrer"
              className="hospital-card me-2"
            >
              <p className="hospital-name text-center">
                Alexandria International Hospital
              </p>
              <div className="d-flex align-items-center justify-content-center gap-2  mb-2 location">
                <img src={location_icon} className="location_icon" />
                <p className="location_name text-center"> Alexandria ,Egypt</p>
              </div>

              <iframe
                className="hospital-map"
                src="https://maps.google.com/maps?q=31.2158436,29.9419761&z=15&output=embed"
                loading="lazy"
              ></iframe>
            </a>
          </div>
        </div>

        {/* Cairo */}
        <div className="city-title position-relative px-5">
          <h5 className="city-tittle">Cairo</h5>
        </div>

        <div className="hospitals-container">
          <div>
            <a
              href="https://www.google.com/maps?q=29.9849277,31.2303078"
              target="_blank"
              rel="noopener noreferrer"
              className="hospital-card me-2"
            >
              <p className="hospital-name text-center">Al Salam Hospitals</p>
              <div className="d-flex align-items-center justify-content-center gap-2 mb-2 location">
                <img src={location_icon} className="location_icon" />
                <p className="location_name text-center"> Cairo ,Egypt</p>
              </div>
              <iframe
                className="hospital-map"
                src="https://maps.google.com/maps?q=29.9849277,31.2303078&z=15&output=embed"
                loading="lazy"
              ></iframe>
            </a>
          </div>
          <div>
            <a
              href="https://www.google.com/maps?q=30.132158,31.3847382"
              target="_blank"
              rel="noopener noreferrer"
              className="hospital-card me-2"
            >
              <p className="hospital-name text-center">Saudi German Hospital</p>
              <div className="d-flex align-items-center justify-content-center gap-2  mb-2 location">
                <img src={location_icon} className="location_icon" />
                <p className="location_name text-center"> Cairo ,Egypt</p>
              </div>
              <iframe
                className="hospital-map"
                src="https://maps.google.com/maps?q=30.132158,31.3847382&z=15&output=embed"
                loading="lazy"
              ></iframe>
            </a>
          </div>
          <div>
            <a
              href="https://www.google.com/maps?q=30.0541899,31.2965291"
              target="_blank"
              rel="noopener noreferrer"
              className="hospital-card me-2"
            >
              <p className="hospital-name text-center">
                Arab Contractors Hospital
              </p>
              <div className="d-flex align-items-center justify-content-center gap-2 mb-2 location">
                <img src={location_icon} className="location_icon" />
                <p className="location_name text-center"> Cairo ,Egypt</p>
              </div>
              <iframe
                className="hospital-map"
                src="https://maps.google.com/maps?q=30.0541899,31.2965291&z=15&output=embed"
                loading="lazy"
              ></iframe>
            </a>
          </div>
        </div>
      </section>
     
    </div>
  );
}
export default Home;
