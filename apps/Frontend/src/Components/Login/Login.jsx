import React from "react";
import "./Login.css";
import heartImg from "../../assets/heartLog.png";
import logo from "../../assets/Logo.png";
import { FaUser, FaLock } from "react-icons/fa";
import { Link } from "react-router-dom";

const Login = () => {
  return (
    <div className="login-container">

      <div className="login-card">

        {/* LEFT SIDE */}
        <div className="login-left">
          <div className="login-content">
            <h2>Login Page</h2>

            <div className="input-group">
              <input type="text" placeholder="Username" />
              <FaUser className="input-icon" />
            </div>

            <div className="input-group">
              <input type="password" placeholder="Password" />
              <FaLock className="input-icon" />
            </div>

            <button className="btn-gradient">Log In</button>

            <div className="register-link">
              Don't have an account? <Link to="/register">Register Now</Link>
            </div>
          </div>
        </div>

        {/* RIGHT SIDE */}
        <div
          className="login-right"
          style={{
            backgroundImage: `linear-gradient(rgba(0,0,0,0.25), rgba(0,0,0,0.25)), url(${heartImg})`,
          }}
        >
          <div className="logo-title-wrapper">
            <img src={logo} className="logo" alt="logo" />
            <h1>Heart Diseases</h1>
          </div>
        </div>

      </div>
    </div>
  );
};

export default Login;