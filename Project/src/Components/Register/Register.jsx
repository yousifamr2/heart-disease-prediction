import React from "react";
import "./Register.css";
import heartImg from "../../assets/heart.png";
import logo from "../../assets/Logo.png";
import { FaUser, FaEnvelope, FaLock, FaIdCard } from "react-icons/fa";
import { Link } from "react-router-dom";

const Register = () => {
  return (
    <div className="container-fluid register-container p-0">
  <div className="row  min-vh-100">

   
        {/* LEFT */}
        <div
           className="col-lg-5 col-md-5 col-12 left-side pe-lg-5"
          style={{
            backgroundImage: `linear-gradient(rgba(0,0,0,0.2), rgba(0,0,0,0.2)), url(${heartImg})`,
          }}
        >
          <div className="logo-title-wrapper">
            <img src={logo} alt="logo" className="logo" />
            <div className="heart-title">
              <h1>Heart Diseases</h1>
            </div>
          </div>
        </div>


        {/* RIGHT */}
  <div className="col-lg-7 col-md-7 col-12 right-side ps-lg-5 d-flex align-items-start justify-content-center position-relative min-vh-100"
  >

          <div className="register-content">
            <h2>Register Page</h2>

            <div className="input-group mb-3 position-relative">
              <input type="email" placeholder="Email" />
              <FaEnvelope className="input-icon" />
            </div>

            <div className="input-group mb-3 position-relative">
              <input type="text" placeholder="National ID" />
              <FaIdCard className="input-icon" />
            </div>

            <div className="input-group mb-3 position-relative">
              <input type="text" placeholder="Username" />
              <FaUser className="input-icon" />
            </div>

            <div className="input-group mb-4 position-relative">
              <input type="password" placeholder="Password" />
              <FaLock className="input-icon" />
            </div>

            <button className="btn-gradient mb-3">
              Create Account
            </button>

            <div className="login-link">
              Already have an account? <Link to="/login">Login</Link>
            </div>
          </div>

          {/* Shadows */}
          <div className="position-absolute top-0 end-0 w-100 h-100" style={{ pointerEvents: "none" }}>
            <div className="shadow-top-end"></div>
            <div className="shadow-bottom-start"></div>
          </div>

        </div>

      </div>
    </div>
  );
};

export default Register;
