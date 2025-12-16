const express = require("express");
const Patient = require("../models/patients");

const router = express.Router();

// Create new patient
router.post("/", async (req, res, next) => {
  try {
    const patient = await Patient.create(req.body);
    res.status(201).json({ success: true, data: patient });
  } catch (err) {
    next(err);
  }
});

// Get all patients
router.get("/", async (req, res, next) => {
  try {
    const patients = await Patient.find();
    res.json({ success: true, data: patients });
  } catch (err) {
    next(err);
  }
});

// Get single patient by id
router.get("/:id", async (req, res, next) => {
  try {
    const patient = await Patient.findById(req.params.id);
    if (!patient) {
      return res.status(404).json({ success: false, message: "Patient not found" });
    }
    res.json({ success: true, data: patient });
  } catch (err) {
    next(err);
  }
});

// Update patient
router.put("/:id", async (req, res, next) => {
  try {
    const patient = await Patient.findByIdAndUpdate(req.params.id, req.body, {
      new: true,
      runValidators: true,
    });
    if (!patient) {
      return res.status(404).json({ success: false, message: "Patient not found" });
    }
    res.json({ success: true, data: patient });
  } catch (err) {
    next(err);
  }
});

// Delete patient
router.delete("/:id", async (req, res, next) => {
  try {
    const patient = await Patient.findByIdAndDelete(req.params.id);
    if (!patient) {
      return res.status(404).json({ success: false, message: "Patient not found" });
    }
    res.json({ success: true, message: "Patient deleted" });
  } catch (err) {
    next(err);
  }
});

module.exports = router;


