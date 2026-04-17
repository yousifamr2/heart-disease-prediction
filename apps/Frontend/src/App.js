import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";

import Navbar from "./Components/Navbar/Navbar";
import Home from "./Pages/Home";
import TheGeneralHome from "./Pages/The_General_Home_Page"; 
import HaveRisk from "./Pages/Data_have_risk";
import HaveNoRisk from "./Pages/Have_no_risk";
import Profile from "./Pages/Profile";
import Login from "./Components/Login/Login"; 
import Register from "./Components/Register/Register";
import Prediction from "./Components/Prediction/Prediction";
import Learnmore from "./Components/Learnmore/Learnmore";

export default function App() {
  

  return (
    <div>

      

      <Routes>
        <Route path="/" element={<Navigate to="/home" />} />
        <Route path="/home" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/the_general" element={<TheGeneralHome />} />
        <Route path="/have_risk" element={<HaveRisk />} /> 
        <Route path="/have_no_risk" element={<HaveNoRisk />} /> 
        <Route path="/prediction" element={<Prediction />} /> 
        <Route path="/learnmore" element={<Learnmore />} />
        <Route path="/profile" element={<Profile />} />
      </Routes>

    </div>
  );
}