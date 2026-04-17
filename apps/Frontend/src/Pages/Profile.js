import React from "react";
import "../Pages/The_General_Home_Page.css";
import "../fontawesome-free-7.0.0-web/css/all.min.css";
import "../bootstrap.min.css";
import "../Pages/Profile.css";
import profile from "../Image/prof.png";
import logo from "../Image/logo.png";
import { Link } from "react-router-dom";
export function Home() {
  return (
    <div className="profile-page">
      

      {/* Profile Card */}
      <div className="profile-container justify-content-center m-auto">
        <h2 className="title">My Profile</h2>

        <div className="card-box">
          <div className="user-info d-flex justify-content-between align-items-center">
            <div className="d-flex align-items-center gap-3">
              <div className="avatar" />
              <img src={profile} className="prof" />
              <div>
                <h4>George Anwar</h4>
                <div className="icons">
                  <span className="heart">❤</span>
                  <span className="plus">#</span>
                </div>
              </div>
            </div>
            <button className="edit-btn">Edit User Profile</button>
          </div>

          <div className="info-list">
            <div className="info-item">
              <div>
                <p>National Id</p>
                <span>12345678901</span>
              </div>
              <button>Edit</button>
            </div>

            <div className="info-item">
              <div>
                <p>Username</p>
                <span>George Anwar</span>
              </div>
              <button>Edit</button>
            </div>

            <div className="info-item">
              <div>
                <p>Password</p>
                <span>************</span>
              </div>
              <button>Edit</button>
            </div>

            <div className="info-item">
              <div>
                <p>Email</p>
                <span>george10@gmail.com</span>
              </div>
              <button>Edit</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
export default Home;
