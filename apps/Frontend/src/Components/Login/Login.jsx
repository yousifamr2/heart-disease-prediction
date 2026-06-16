import React, { useState } from "react";
import "./Login.css";

import heartImg from "../../assets/heartLog.png";
import logo from "../../assets/Logo.png";

import { FaUser, FaHospital, FaEye, FaEyeSlash } from "react-icons/fa";
import { Link, useNavigate } from "react-router-dom";

import axios from "axios";
import API_BASE_URL from "../../config";

const Login = () => {

  const [activeTab, setActiveTab] = useState("user"); // "user" or "lab"

  const [form, setForm] = useState({
    username: "",
    password: "",
    labName: "",
    labCode: "",
  });

  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  
  // Toggle states
  const [showPassword, setShowPassword] = useState(false);
  const [showLabCode, setShowLabCode] = useState(false);

  const navigate = useNavigate();

  // ================= VALIDATION =================
  const validate = (name, value) => {
    let error = "";

    if (activeTab === "user") {
      if (name === "username") {
        if (!value.trim()) error = "Username is required";
        else if (value.length < 3) error = "Username must be at least 3 characters";
      }
      if (name === "password") {
        if (!value.trim()) error = "Password is required";
        else if (value.length < 6) error = "Password must be at least 6 characters";
      }
    } else {
      if (name === "labName") {
        if (!value.trim()) error = "Lab Name is required";
      }
      if (name === "labCode") {
        if (!value.trim()) error = "Lab Code is required";
      }
    }

    return error;
  };

  // ================= HANDLE CHANGE =================
  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
    const error = validate(name, value);
    setErrors((prev) => ({ ...prev, [name]: error }));
  };

  // ================= HANDLE BLUR =================
  const handleBlur = (e) => {
    const { name, value } = e.target;
    const error = validate(name, value);
    setErrors((prev) => ({ ...prev, [name]: error }));
  };

  // ================= LOGIN =================
  const handleLogin = async () => {
    if (activeTab === "user") {
      const usernameError = validate("username", form.username);
      const passwordError = validate("password", form.password);

      if (usernameError || passwordError) {
        setErrors({ username: usernameError, password: passwordError });
        return;
      }

      try {
        setLoading(true);
        const res = await axios.post(`${API_BASE_URL}/api/auth/login`, {
          username: form.username,
          password: form.password,
        });

        const token = res.data.token || res.data.data?.token;
        if (token) localStorage.setItem("token", token);
        localStorage.setItem("user", JSON.stringify(res.data.data || res.data.user));
        setErrors({});
        navigate("/the_general");
      } catch (err) {
        setErrors({ password: "Invalid username or password" });
      } finally {
        setLoading(false);
      }
    } else {
      // LAB LOGIN
      const labNameError = validate("labName", form.labName);
      const labCodeError = validate("labCode", form.labCode);

      if (labNameError || labCodeError) {
        setErrors({ labName: labNameError, labCode: labCodeError });
        return;
      }

      try {
        setLoading(true);
        const res = await axios.get(`${API_BASE_URL}/api/labs`);
        const labs = res.data.data;
        
        const matchedLab = labs.find(
          (lab) => lab.name.toLowerCase() === form.labName.toLowerCase() && lab.lab_code === form.labCode
        );

        if (matchedLab) {
          localStorage.setItem("lab", JSON.stringify(matchedLab));
          setErrors({});
          navigate("/lab-portal");
        } else {
          setErrors({ labCode: "Invalid Lab Name or Lab Code" });
        }
      } catch (err) {
        setErrors({ labCode: "Failed to connect to server" });
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
        {/* ================= LEFT SIDE ================= */}
        <div className="login-left">
          <div className="login-content">
            <h2>Login Page</h2>

            {/* ================= TABS ================= */}
            <div className="login-tabs">
              <button 
                className={`tab-btn ${activeTab === "user" ? "active" : ""}`}
                onClick={() => { setActiveTab("user"); setErrors({}); }}
              >
                User
              </button>
              <button 
                className={`tab-btn ${activeTab === "lab" ? "active" : ""}`}
                onClick={() => { setActiveTab("lab"); setErrors({}); }}
              >
                Lab
              </button>
            </div>

            {activeTab === "user" ? (
              <>
                {/* ================= USERNAME ================= */}
                <div className="input-group">
                  <input
                    type="text"
                    name="username"
                    placeholder="Username"
                    value={form.username}
                    onChange={handleChange}
                    onBlur={handleBlur}
                  />
                  <FaUser className="input-icon" />
                  {errors.username && <span className="error">{errors.username}</span>}
                </div>

                {/* ================= PASSWORD ================= */}
                <div className="input-group">
                  <input
                    type={showPassword ? "text" : "password"}
                    name="password"
                    placeholder="Password"
                    value={form.password}
                    onChange={handleChange}
                    onBlur={handleBlur}
                  />
                  <button 
                    type="button" 
                    className="toggle-visibility-btn"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? <FaEyeSlash /> : <FaEye />}
                  </button>
                  {errors.password && <span className="error">{errors.password}</span>}
                </div>
              </>
            ) : (
              <>
                {/* ================= LAB NAME ================= */}
                <div className="input-group">
                  <input
                    type="text"
                    name="labName"
                    placeholder="Lab Name"
                    value={form.labName}
                    onChange={handleChange}
                    onBlur={handleBlur}
                  />
                  <FaHospital className="input-icon" />
                  {errors.labName && <span className="error">{errors.labName}</span>}
                </div>

                {/* ================= LAB CODE ================= */}
                <div className="input-group">
                  <input
                    type={showLabCode ? "text" : "password"}
                    name="labCode"
                    placeholder="Lab Code"
                    value={form.labCode}
                    onChange={handleChange}
                    onBlur={handleBlur}
                  />
                  <button 
                    type="button" 
                    className="toggle-visibility-btn"
                    onClick={() => setShowLabCode(!showLabCode)}
                  >
                    {showLabCode ? <FaEyeSlash /> : <FaEye />}
                  </button>
                  {errors.labCode && <span className="error">{errors.labCode}</span>}
                </div>
              </>
            )}

            {/* ================= BUTTON ================= */}
            <button className="btn-gradient" onClick={handleLogin} disabled={loading}>
              {loading ? "Logging in..." : "Log In"}
            </button>

            {/* ================= REGISTER ================= */}
            {activeTab === "user" && (
              <div className="register-link">
                Don't have an account? <Link to="/register">Register Now</Link>
              </div>
            )}
          </div>
        </div>

        {/* ================= RIGHT SIDE ================= */}
        <div
          className="login-right"
          style={{
            backgroundImage: `linear-gradient(rgba(0,0,0,0.25), rgba(0,0,0,0.25)), url(${heartImg})`,
          }}
        >
          <div className="logo-title-wrapper">
            <img src={logo} className="logo" alt="logo" />
            <h1>Nabdak</h1>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;