import React from "react";
import { Routes, Route, Navigate, useLocation } from "react-router-dom";

import Navbar from "./Components/Navbar/Navbar";
import Footer from "./Components/Footer/Footer";

import Heart from "./Pages/Home";
import TheGeneralHome from "./Pages/The_General_Home_Page";
import HaveRisk from "./Pages/Data_have_risk";
import HaveNoRisk from "./Pages/Have_no_risk";
import Profile from "./Pages/Profile";
import Login from "./Components/Login/Login";
import Register from "./Components/Register/Register";
import Prediction from "./Components/Prediction/Prediction";
import EcgPrediction from "./Components/Ecg/EcgPrediction";
import Learnmore from "./Components/Learnmore/Learnmore";
import ProtectedRoute from "./Components/ProtectedRoute";
import LabPortal from "./Pages/LabPortal";
import Docs from "./Pages/Docs/Docs";
import About from "./Pages/About/About";
import Contact from "./Pages/Contact/Contact";

import { AuthProvider } from "./Context/AuthContext";

export default function App() {
  const location = useLocation();

  const hideNavbar =
    location.pathname === "/login" ||
    location.pathname === "/register" ||
    location.pathname === "/lab-portal";

  const hideFooter =
    location.pathname === "/login" ||
    location.pathname === "/register" ||
    location.pathname === "/heart" ||
    location.pathname === "/lab-portal";

  return (
    <AuthProvider>
      <div className="app-container">

        {/* NAVBAR */}
        {!hideNavbar && <Navbar />}

        {/* PAGE CONTENT */}
        <div className="page-content">
          <Routes>

  <Route path="/" element={<Navigate to="/heart" />} />

  {/* PUBLIC ROUTES */}
  <Route path="/heart" element={<Heart />} />
  <Route path="/home" element={<Navigate to="/the_general" />} />
  <Route path="/login" element={<Login />} />
  <Route path="/register" element={<Register />} />
  <Route path="/lab-portal" element={<LabPortal />} />
  <Route path="/learnmore" element={<Learnmore />} />
  <Route path="/docs" element={<Docs />} />
  <Route path="/about" element={<About />} />
  <Route path="/contact" element={<Contact />} />

  {/* PROTECTED ROUTES */}
  <Route
    path="/the_general"
    element={
      <ProtectedRoute>
        <TheGeneralHome />
      </ProtectedRoute>
    }
  />

  <Route
    path="/have_risk"
    element={
      <ProtectedRoute>
        <HaveRisk />
      </ProtectedRoute>
    }
  />

  <Route
    path="/have_no_risk"
    element={
      <ProtectedRoute>
        <HaveNoRisk />
      </ProtectedRoute>
    }
  />

  <Route
    path="/prediction"
    element={
      <ProtectedRoute>
        <Prediction />
      </ProtectedRoute>
    }
  />

  <Route
    path="/ecg"
    element={
      <ProtectedRoute>
        <EcgPrediction />
      </ProtectedRoute>
    }
  />

  <Route
    path="/profile"
    element={
      <ProtectedRoute>
        <Profile />
      </ProtectedRoute>
    }
  />

</Routes>
        </div>

        {/* FOOTER */}
        {!hideFooter && <Footer isDashboard={location.pathname === "/profile"} />}

      </div>
    </AuthProvider>
  );
}