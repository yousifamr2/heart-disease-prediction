import React from "react";
import { Routes, Route, Navigate,  } from "react-router-dom";

// استيراد الصفحات حسب هيكل مشروعك
import Login from "./Components/Login/Login";
import Register from "./Components/Register/Register";
import Prediction from "./Components/Prediction/Prediction"
import Learnmore from "./Components/Learnmore/Learnmore"


export default function App() {
  return (
    <Routes>
       <Route path="/register" element={<Register />} />
      {/* تحويل تلقائي للرابط الرئيسي للـ Login */}
      <Route path="/" element={<Navigate to="/login" />} />

      {/* صفحة تسجيل الدخول */}
      <Route path="/login" element={<Login />} />

      {/* صفحة التسجيل */}
      
      <Route path="/prediction" element={<Prediction />} />
      <Route path="/learnmore" element={<Learnmore />} />
    </Routes>
  );
}
