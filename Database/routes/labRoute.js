const express = require("express");
const Lab = require("../models/lab");
const { authenticate } = require("../middleware/auth");

const router = express.Router();

// Create new lab (protected)
router.post("/", authenticate, async (req, res, next) => {
  try {
    const lab = await Lab.create(req.body);
    res.status(201).json({ success: true, data: lab });
  } catch (err) {
    next(err);
  }
});

// Get all labs (with pagination)
router.get("/", async (req, res, next) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10;
    const skip = (page - 1) * limit;

    const total = await Lab.countDocuments();
    const labs = await Lab.find()
      .skip(skip)
      .limit(limit)
      .sort({ createdAt: -1 });

    res.json({
      success: true,
      data: labs,
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

// Get single lab by id
router.get("/:id", async (req, res, next) => {
  try {
    const lab = await Lab.findById(req.params.id);
    if (!lab) {
      return res.status(404).json({ success: false, message: "Lab not found" });
    }
    res.json({ success: true, data: lab });
  } catch (err) {
    next(err);
  }
});

// Update lab (protected)
router.put("/:id", authenticate, async (req, res, next) => {
  try {
    const lab = await Lab.findByIdAndUpdate(req.params.id, req.body, {
      new: true,
      runValidators: true,
    });
    if (!lab) {
      return res.status(404).json({ success: false, message: "Lab not found" });
    }
    res.json({ success: true, data: lab });
  } catch (err) {
    next(err);
  }
});

// Delete lab (protected)
router.delete("/:id", authenticate, async (req, res, next) => {
  try {
    const lab = await Lab.findByIdAndDelete(req.params.id);
    if (!lab) {
      return res.status(404).json({ success: false, message: "Lab not found" });
    }
    res.json({ success: true, message: "Lab deleted" });
  } catch (err) {
    next(err);
  }
});

module.exports = router;

