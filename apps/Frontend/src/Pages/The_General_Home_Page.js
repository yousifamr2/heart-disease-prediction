import React from "react";
import "../Pages/The_General_Home_Page.css";
import "../fontawesome-free-7.0.0-web/css/all.min.css";
import "../bootstrap.min.css";
import profile from "../Image/profile.png";
import logo from "../Image/logo.png";
import nabd from "../Image/nabd.png";
import heart_icons from "../Image/heart_icons.png";
import Box from "../Image/Box.png";
import { Link } from "react-router-dom";
import Navbar from "../Components/Navbar/Navbar";
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
      {/* Features Section */}
      <section className="container  my-5">
        <div className="row align-items-center justify-content-center text-center g-4">
          {/* Card 1 */}
          <div className="col-md-3">
            <div className="feature-card">
              <div className="icon">
                <img src={nabd} className="image_iconn" />
              </div>
              <h5 className="Title_card">Accurate Analysis</h5>
              <p className="cap_Card">
                Advanced Machine Learning Models Trained On Extensive Medical
                Data
              </p>
            </div>
          </div>

          {/* Card 2 */}
          <div className="col-md-3">
            <div className="feature-card">
              <div className="icon">
                <img src={heart_icons} className="image_iconn" />
              </div>
              <h5 className="Title_card">Health Insights</h5>
              <p className="cap_Card">Detailed Risk Factor Analysis</p>{" "}
              <p className="cap_Card">
                And Personalized Health Recommendations
              </p>
            </div>
          </div>

          {/* Card 3 */}
          <div className="col-md-3">
            <div className="feature-card">
              <div className="icon">
                <img src={Box} className="image_iconn" />
              </div>
              <h5 className="Title_card">Early Detection</h5>
              <p className="cap_Card">
                Identify Potential Heart Health Concerns Before They Become
                Serious
              </p>
            </div>
          </div>
        </div>
      </section>
      
    </div>
  );
}
export default Home;
