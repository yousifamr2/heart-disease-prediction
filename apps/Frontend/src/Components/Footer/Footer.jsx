import React from "react";
import { Link } from "react-router-dom";
import logo from "../../Image/logo.png"; // 👈 نفس مسار اللوجو عندك
import "../Footer/Footer.css";

export default function Footer({ isDashboard }) {
  const colClass = isDashboard ? "col-6 col-md-3 mb-3" : "col-md-3 mb-3";

  return (
    <footer className="footer">
      <div className="container">
        <div className="row text-start">
          {/* Column 1 */}
          <div className={colClass}>
            <div className="d-flex align-items-center gap-2 mb-2">
              <img src={logo} className="logo" alt="logo" />
              <span className="brand">Nabdak</span>
            </div>

            <p className="footer-text">
              Check your heart care and make your life better.
            </p>

            <div className="social-icons d-flex gap-3">
              <a href="https://github.com/yousifamr2/heart-disease-prediction" target="_blank" rel="noopener noreferrer" style={{ color: "inherit" }}>
                <i className="fa-brands fa-github"></i>
              </a>
              <a href="https://x.com" target="_blank" rel="noopener noreferrer" style={{ color: "inherit" }}>
                <i className="fa-brands fa-x-twitter"></i>
              </a>
              <a href="https://facebook.com" target="_blank" rel="noopener noreferrer" style={{ color: "inherit" }}>
                <i className="fa-brands fa-facebook"></i>
              </a>
              <a href="https://instagram.com" target="_blank" rel="noopener noreferrer" style={{ color: "inherit" }}>
                <i className="fa-brands fa-instagram"></i>
              </a>
              <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer" style={{ color: "inherit" }}>
                <i className="fa-brands fa-linkedin"></i>
              </a>
            </div>
          </div>

          {/* Column 2 */}
          <div className={colClass}>
            <h6 className="footer-title">Heart Disease</h6>
            <ul className="footer-list">
              <li><Link to="/prediction" style={{ color: "inherit", textDecoration: "none" }}>Heart Care</Link></li>
              <li><Link to="/ecg" style={{ color: "inherit", textDecoration: "none" }}>Health Care</Link></li>
              <li><Link to="/about" style={{ color: "inherit", textDecoration: "none" }}>About Us</Link></li>
              <li><Link to="/contact" style={{ color: "inherit", textDecoration: "none" }}>Contact Us</Link></li>
            </ul>
          </div>

          {/* Column 3 */}
          <div className={colClass}>
            <h6 className="footer-title">Labs</h6>
            <ul className="footer-list">
              <li><a href="https://almokhtabar.com/ar/%d8%a7%d9%84%d9%81%d8%b1%d9%88%d8%b9/" target="_blank" rel="noopener noreferrer" style={{ color: "inherit", textDecoration: "none" }}>AlMokhtabar</a></li>
              <li><a href="https://alborglab.com/branches/" target="_blank" rel="noopener noreferrer" style={{ color: "inherit", textDecoration: "none" }}>Al Borg</a></li>
              <li><a href="https://hassab.com/site/ar/%D9%81%D8%B1%D9%88%D8%B9%D9%86%D8%A7/" target="_blank" rel="noopener noreferrer" style={{ color: "inherit", textDecoration: "none" }}>Hassab</a></li>
              <li><a href="http://alshamslabs.com/branches.aspx" target="_blank" rel="noopener noreferrer" style={{ color: "inherit", textDecoration: "none" }}>Al Shams</a></li>
            </ul>
          </div>

          {/* Column 4 */}
          <div className={colClass}>
            <h6 className="footer-title">Resources</h6>
            <ul className="footer-list">
              <li><a href="https://www.heart.org/" target="_blank" rel="noopener noreferrer" style={{ color: "inherit", textDecoration: "none" }}>AHA</a></li>
              <li><a href="https://www.cdc.gov/heartdisease/" target="_blank" rel="noopener noreferrer" style={{ color: "inherit", textDecoration: "none" }}>CDC</a></li>
              <li><a href="https://www.nhlbi.nih.gov/" target="_blank" rel="noopener noreferrer" style={{ color: "inherit", textDecoration: "none" }}>NHLBI</a></li>
              <li><a href="https://hfsa.org/" target="_blank" rel="noopener noreferrer" style={{ color: "inherit", textDecoration: "none" }}>HFSA</a></li>
            </ul>
          </div>
        </div>

        <hr className="footer-line" />

        <p className="text-center copyright">
          © 2026 Nabdak. All Rights Reserved.
        </p>
      </div>
    </footer>
  );
}
