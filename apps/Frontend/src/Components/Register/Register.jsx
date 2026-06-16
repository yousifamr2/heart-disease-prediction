import React, { useState } from "react";
import axios from "axios";
import "./Register.css";
import API_BASE_URL from "../../config";


import heartImg from "../../assets/heart.png";
import logo from "../../assets/Logo.png";

import { FaUser, FaEnvelope, FaLock, FaIdCard, FaEye, FaEyeSlash } from "react-icons/fa";
import { Link, useNavigate } from "react-router-dom";

// ===== INPUT COMPONENT =====
const Input = ({
  icon: Icon,
  name,
  type,
  placeholder,
  value,
  onChange,
  error,
  isPassword,
  showPassword,
  togglePassword,
}) => (
  <div className="input-wrapper">
    <div className={`input-group ${error ? "error-border" : ""}`}>
      <input
        name={name}
        type={type}
        placeholder={placeholder}
        value={value}
        onChange={onChange}
      />
      {isPassword ? (
        <button 
          type="button" 
          className="toggle-visibility-btn"
          onClick={togglePassword}
        >
          {showPassword ? <FaEyeSlash /> : <FaEye />}
        </button>
      ) : (
        <Icon className="input-icon" />
      )}
    </div>
    {error && <span className="error-text">{error}</span>}
  </div>
);

const Register = () => {
  const [form, setForm] = useState({
    email: "",
    nationalId: "",
    username: "",
    password: "",
  });

  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const navigate = useNavigate();

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const validate = () => {
    const err = {};

    if (!form.email) err.email = "Email is required";
    else if (!/\S+@\S+\.\S+/.test(form.email))
      err.email = "Invalid email format";

    if (!form.nationalId) err.nationalId = "National ID is required";
    else if (!/^\d{14}$/.test(form.nationalId))
      err.nationalId = "Must be 14 digits";

    if (!form.username) err.username = "Username is required";
    else if (form.username.length < 3)
      err.username = "Min 3 characters";

    if (!form.password) err.password = "Password is required";
    else if (form.password.length < 6)
      err.password = "Min 6 characters";

    setErrors(err);
    return Object.keys(err).length === 0;
  };

  const handleRegister = async () => {
    if (!validate()) return;

    try {
      setLoading(true);

      const res = await axios.post(
        `${API_BASE_URL}/api/auth/register`,
        {
          email: form.email,
          national_id: form.nationalId,
          username: form.username,
          password: form.password,
        }
      );

      if (res.data.token) {
        localStorage.setItem("token", res.data.token);
      }

      setErrors({});
      navigate("/login");
    } catch (error) {
      setErrors({
        api:
          error.response?.data?.message ||
          "Something went wrong",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="register-page">
      <div className="register-card">

        {/* LEFT SIDE */}
        <div
          className="register-image-side"
          style={{
            backgroundImage: `
              linear-gradient(rgba(0,0,0,0.12), rgba(0,0,0,0.12)),
              url(${heartImg})
            `,
          }}
        >
          <div className="brand">
            <img src={logo} alt="logo" className="brand-logo" />
            <h1>Nabdak</h1>
          </div>
        </div>

        {/* RIGHT SIDE */}
        <div className="register-form-side">
          <div className="form-content">

            <h2>Create Account</h2>

            {errors.api && (
              <div className="api-error">{errors.api}</div>
            )}

            <Input
              icon={FaUser}
              name="username"
              type="text"
              placeholder="Username"
              value={form.username}
              onChange={handleChange}
              error={errors.username}
            />

            <Input
              icon={FaIdCard}
              name="nationalId"
              type="text"
              placeholder="National ID"
              value={form.nationalId}
              onChange={handleChange}
              error={errors.nationalId}
            />

            <Input
              icon={FaEnvelope}
              name="email"
              type="email"
              placeholder="Email"
              value={form.email}
              onChange={handleChange}
              error={errors.email}
            />

            <Input
              icon={FaLock}
              name="password"
              type={showPassword ? "text" : "password"}
              placeholder="Password"
              value={form.password}
              onChange={handleChange}
              error={errors.password}
              isPassword={true}
              showPassword={showPassword}
              togglePassword={() => setShowPassword(!showPassword)}
            />

            <button
              className="create-btn"
              onClick={handleRegister}
              disabled={loading}
            >
              {loading ? "Creating Account..." : "Create Account"}
            </button>

            <p className="login-text">
              Already Have An Account?
              <Link to="/login"> Log In</Link>
            </p>

          </div>
        </div>

      </div>
    </div>
  );
};

export default Register;