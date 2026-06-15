import React from "react";
import { useNavigate } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";
import "../Pages/Home.css";

export default function Home() {
  const navigate = useNavigate();

  return (
    <div className="page">
      <div className="main container-fluid p-0">
        <div className="content">
          {/* ================= TEXT ================= */}
          <div className="text">
<<<<<<< Updated upstream
            <h2>
              Heart Disease Prediction Tool
            </h2>

            <p>
              Advanced AI-Powered Analysis <br />
              To Assess Your Heart Health <br />
              Risk Factors
            </p>
          </div>

          {/* ================= CORNER ================= */}
          <div className="corner">

            {/* BLUE BOX */}
            <div className="corner-info-box">
              <p>Your Heart Is Your Life</p>

              <button
                className="know-btn"
                onClick={() => navigate("/learnmore")}
              >
                Know More →
              </button>
            </div>

          </div>

=======
            <h2>Heart Disease Prediction Tool</h2>
            <div className="subtitles">
              <p>Advanced AI-Powered Analysis</p>
              <p>To Assess Your Heart Health</p>
              <p>Risk Factors</p>
            </div>
          </div>

          {/* ================= BADGE ================= */}
          <div className="badge-container" onClick={() => navigate("/learnmore")}>
            <div className="badge-content">
              <p className="badge-title">
                Your Heart Is <br />
                Your Life
              </p>
              <div className="badge-btn">
                <span>Know More</span>
                <span className="arrow">→</span>
              </div>
            </div>
          </div>
>>>>>>> Stashed changes
        </div>
      </div>
    </div>
  );
}