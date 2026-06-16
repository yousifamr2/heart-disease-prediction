import React from "react";
import "./About.css";

export default function About() {
  const row1 = [
    { name: "Omar Ahmed Mohamed", role: "AI Engineer" },
    { name: "Youssef Amr Saeed", role: "Backend Developer" }
  ];

  const row2 = [
    { name: "Ranim Mohamed Elsayed", role: "Frontend Developer" },
    { name: "George Anwar Maksadallah", role: "UI/UX Designer" },
    { name: "Samar Hamza Mbry", role: "Frontend Developer" }
  ];

  const row3 = [
    { name: "Abdelrahman Essam Sheredah", role: "Backend Developer" },
    { name: "Marwan Yaser Elserafy", role: "AI Engineer" },
    { name: "Youssef Mohamed Bassiony", role: "Backend Developer" }
  ];

  const renderMember = (member, idx) => (
    <div key={idx} className="team-member-card">
      <div className="member-avatar">
        {member.name.split(" ").map(n => n[0]).slice(0, 2).join("")}
      </div>
      <h4 className="member-name">{member.name}</h4>
      <p className="member-role">{member.role}</p>
    </div>
  );

  return (
    <div className="about-page-container">
      <div className="about-hero">
        <h1 className="about-title">About Nabdak</h1>
        <p className="about-subtitle">Graduation Project 2026</p>
      </div>

      <div className="about-card max-width-card">
        <h2 className="section-title">Our Graduation Project</h2>
        <p className="project-description">
          We are IT students from <strong>EELU Alexandria University</strong>, class of 2026. 
          Nabdak is our Graduation Project, aiming to leverage advanced artificial intelligence 
          and machine learning models to help predict cardiovascular disease risks. 
          By integrating seamless lab uploads, dynamic patient dashboards, ECG analysis, and instant AI-generated 
          medical PDF reports, Nabdak provides a hybrid support tool for early heart care detection.
        </p>
        <p className="project-description" style={{ marginTop: "15px" }}>
          Explore our open-source codebase and repository on <a href="https://github.com/yousifamr2/heart-disease-prediction" target="_blank" rel="noopener noreferrer" className="github-link">GitHub</a>.
        </p>

        <h3 className="section-title" style={{ marginTop: "40px" }}>The Team</h3>
        <div className="team-container">
          <div className="team-row row-two">
            {row1.map(renderMember)}
          </div>
          <div className="team-row row-three">
            {row2.map(renderMember)}
          </div>
          <div className="team-row row-three">
            {row3.map(renderMember)}
          </div>
        </div>
      </div>
    </div>
  );
}
