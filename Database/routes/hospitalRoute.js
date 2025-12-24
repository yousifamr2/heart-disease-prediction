const express = require("express");
const Hospital = require("../models/hospital");

const router = express.Router();

// Create new hospital
router.post("/", async (req, res, next) => {
  try {
    const hospital = await Hospital.create(req.body);
    res.status(201).json({ success: true, data: hospital });
  } catch (err) {
    next(err);
  }
});

// Get all hospitals
router.get("/", async (req, res, next) => {
  try {
    const hospitals = await Hospital.find();
    res.json({ success: true, data: hospitals });
  } catch (err) {
    next(err);
  }
});

// Get single hospital by id
router.get("/:id", async (req, res, next) => {
  try {
    const hospital = await Hospital.findById(req.params.id);
    if (!hospital) {
      return res.status(404).json({ success: false, message: "Hospital not found" });
    }
    res.json({ success: true, data: hospital });
  } catch (err) {
    next(err);
  }
});

// Get hospitals by area
router.get("/area/:area", async (req, res, next) => {
  try {
    const hospitals = await Hospital.find({ area: req.params.area });
    res.json({ success: true, data: hospitals });
  } catch (err) {
    next(err);
  }
});

// Update hospital
router.put("/:id", async (req, res, next) => {
  try {
    const hospital = await Hospital.findByIdAndUpdate(req.params.id, req.body, {
      new: true,
      runValidators: true,
    });
    if (!hospital) {
      return res.status(404).json({ success: false, message: "Hospital not found" });
    }
    res.json({ success: true, data: hospital });
  } catch (err) {
    next(err);
  }
});

// Delete hospital
router.delete("/:id", async (req, res, next) => {
  try {
    const hospital = await Hospital.findByIdAndDelete(req.params.id);
    if (!hospital) {
      return res.status(404).json({ success: false, message: "Hospital not found" });
    }
    res.json({ success: true, message: "Hospital deleted" });
  } catch (err) {
    next(err);
  }
});

module.exports = router;

