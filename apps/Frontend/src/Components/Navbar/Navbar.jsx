import React, { useState } from "react";
import { Link } from "react-router-dom";
import logo from "../../Image/logo.png";
import profileImg from "../../Image/profile.png"; // عدلي المسار حسب عندك

const Navbar = () => {
  // مؤقت لحد ما الباك يجهز
  const [isLoggedIn] = useState(false);

  return (
    <nav className="navbar px-5 py-2 mx-auto">

      {/* Logo + Brand */}
      <div className="d-flex align-items-center gap-2">
        <img src={logo} className="logo" alt="logo" />
        <span className="brand">Heart Diseases</span>
      </div>

      {/* Links */}
      <ul className="nav mx-auto gap-4">
        <li>HOME</li>
        <li>DOCS</li>
        <li>HEART</li>
        <li>ABOUT</li>
      </ul>

      {/* Buttons */}
      <div className="hero-buttons d-flex gap-2">

        {!isLoggedIn ? (
          <>
            <Link to="/login" className="btn custom-btn-outline">
              Login
            </Link>

            <Link to="/register" className="btn custom-btn">
              Register
            </Link>
          </>
        ) : (
          <Link
            to="/profile"
            className="btn learn btn-outline-dark rounded-pill custom-btn d-flex align-items-center gap-2"
          >
            My Profile
            <img src={profileImg} className="profile" alt="profile" />
          </Link>
        )}

      </div>
    </nav>
  );
};

export default Navbar;