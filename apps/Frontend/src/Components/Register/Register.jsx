import React from "react";
import "./Register.css";
import heartImg from "../../assets/heart.png";
import logo from "../../assets/Logo.png";
import { FaUser, FaEnvelope, FaLock, FaIdCard } from "react-icons/fa";
import { Link } from "react-router-dom";

const Register = () => {
  return (
    <div className="container-fluid register-container p-0">

      <div className="row register-card w-100  ming-1">

        {/* LEFT SIDE */}
        <div
          className="col-lg-5 col-12 left-side"
          style={{
            backgroundImage: `linear-gradient(rgba(0,0,0,0.2), rgba(0,0,0,0.2)), url(${heartImg})`,
            backgroundSize: "cover",
            backgroundPosition: "center",
          }}
        >
          <div className="logo-title-wrapper">
            <img src={logo} alt="logo" className="logo" />
            <h1>Heart Diseases</h1>
          </div>
        </div>

        {/* RIGHT SIDE */}
        <div className="col-lg-7 col-12 right-side">

          <div className="register-content">

            <h2>Register Page</h2>

            <div className="input-group">
              <input type="email" placeholder="Email" />
              <FaEnvelope className="input-icon" />
            </div>

            <div className="input-group">
              <input type="text" placeholder="National ID" />
              <FaIdCard className="input-icon" />
            </div>

            <div className="input-group">
              <input type="text" placeholder="Username" />
              <FaUser className="input-icon" />
            </div>

            <div className="input-group">
              <input type="password" placeholder="Password" />
              <FaLock className="input-icon" />
            </div>

            <button className="btn-gradient">
              Create Account
            </button>

            <div className="login-link">
              Already have an account?{" "}
              <Link to="/login">Login</Link>
            </div>

          </div>

        </div>

      </div>
    </div>
  );
};

export default Register;