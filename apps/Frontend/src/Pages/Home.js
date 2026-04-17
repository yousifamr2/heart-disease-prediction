import React from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import "../Pages/Home.css";

export default function App() {
  return (
    <div className="page">

      <div className="main container-fluid p-0">

        {/* Hero Section */}
        <div className="content">

          {/* Text */}
          <div className="text">
            <h2>Heart Disease Prediction Tool</h2>

            <p>
              Advanced AI-Powered Analysis <br />
              To Assess Your Heart Health <br />
              Risk Factors
            </p>
          </div>

          {/* Corner Shape */}
          <div className="corner">
            <div className="info-box">
              <p>Your Heart Is Your Life</p>
              <button className="know-btn">Know More →</button>
            </div>
          </div>

        </div>
      </div>

    </div>
  );
}