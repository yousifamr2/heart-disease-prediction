const express = require("express");
const Hospital = require("../models/hospital");
const { authenticate } = require("../middleware/auth");

const router = express.Router();

// Create new hospital (protected)
router.post("/", authenticate, async (req, res, next) => {
  try {
    const hospital = await Hospital.create(req.body);
    res.status(201).json({ success: true, data: hospital });
  } catch (err) {
    next(err);
  }
});

// Get all hospitals (with pagination)
router.get("/", async (req, res, next) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10;
    const skip = (page - 1) * limit;

    const total = await Hospital.countDocuments();
    const hospitals = await Hospital.find()
      .skip(skip)
      .limit(limit)
      .sort({ createdAt: -1 });

    res.json({
      success: true,
      data: hospitals,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit)
      }
    });
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

// Update hospital (protected)
router.put("/:id", authenticate, async (req, res, next) => {
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

// Delete hospital (protected)
router.delete("/:id", authenticate, async (req, res, next) => {
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

