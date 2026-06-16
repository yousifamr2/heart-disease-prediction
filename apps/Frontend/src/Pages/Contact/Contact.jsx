import React, { useState } from "react";
import { FaPhoneAlt, FaEnvelope, FaMapMarkerAlt, FaHeadset } from "react-icons/fa";
import "./Contact.css";

export default function Contact() {
  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    phoneNumber: "",
    email: "",
    message: ""
  });
  const [submitted, setSubmitted] = useState(false);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Submitting contact form:", formData);
    setSubmitted(true);
    // Reset form after submission animation
    setTimeout(() => {
      setSubmitted(false);
      setFormData({
        firstName: "",
        lastName: "",
        phoneNumber: "",
        email: "",
        message: ""
      });
    }, 4000);
  };

  return (
    <div className="contact-page-container">
      <div className="contact-hero">
        <h1 className="contact-title">Contact Us</h1>
        <p className="contact-subtitle">We would love to hear from you</p>
      </div>

      <div className="contact-main-card">
        {/* Left Column: Contact Details */}
        <div className="contact-info-panel">
          <div className="support-icon-wrapper">
            <FaHeadset className="support-icon" />
          </div>
          <h2 className="info-title">CONTACT US</h2>
          <p className="info-desc">
            Welcome to Nabdak! Whether you are a patient looking for heart disease risk predictions, a lab portal user, or a guest, our team is here to support you. Browse guided sections, check your heart health report, and start reaching your health goals today!
          </p>

          <div className="contact-details-list">
            <div className="contact-detail-item">
              <FaPhoneAlt className="detail-icon" />
              <span>+20 111 222 3333</span>
            </div>
            <div className="contact-detail-item">
              <FaEnvelope className="detail-icon" />
              <span>info.nabdak@gmail.com</span>
            </div>
            <div className="contact-detail-item">
              <FaMapMarkerAlt className="detail-icon" />
              <span>Alexandria, Egypt</span>
            </div>
          </div>
        </div>

        {/* Right Column: Contact Form */}
        <div className="contact-form-panel">
          {submitted ? (
            <div className="success-message-container">
              <div className="success-checkmark">✓</div>
              <h3>Message Sent Successfully!</h3>
              <p>Thank you for contacting us. We will get back to you as soon as possible.</p>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="contact-form">
              <div className="form-row-grid">
                <div className="form-group-custom">
                  <label htmlFor="firstName">FIRST NAME</label>
                  <input
                    type="text"
                    id="firstName"
                    name="firstName"
                    value={formData.firstName}
                    onChange={handleChange}
                    placeholder="Enter your first name"
                    required
                  />
                </div>
                <div className="form-group-custom">
                  <label htmlFor="lastName">LAST NAME</label>
                  <input
                    type="text"
                    id="lastName"
                    name="lastName"
                    value={formData.lastName}
                    onChange={handleChange}
                    placeholder="Enter your last name"
                    required
                  />
                </div>
              </div>

              <div className="form-group-custom">
                <label htmlFor="phoneNumber">PHONE NUMBER</label>
                <input
                  type="tel"
                  id="phoneNumber"
                  name="phoneNumber"
                  value={formData.phoneNumber}
                  onChange={handleChange}
                  placeholder="Enter your phone number"
                  required
                />
              </div>

              <div className="form-group-custom">
                <label htmlFor="email">EMAIL ADDRESS</label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  placeholder="Enter your email address"
                  required
                />
              </div>

              <div className="form-group-custom">
                <label htmlFor="message">MESSAGE</label>
                <textarea
                  id="message"
                  name="message"
                  value={formData.message}
                  onChange={handleChange}
                  placeholder="How can we help you?"
                  rows="4"
                  required
                />
              </div>

              <div className="form-submit-container">
                <button type="submit" className="submit-btn-custom">
                  SUBMIT
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
