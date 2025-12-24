const express = require("express");
const Lab = require("../models/lab");

const router = express.Router();

// Create new lab
router.post("/", async (req, res, next) => {
  try {
    const lab = await Lab.create(req.body);
    res.status(201).json({ success: true, data: lab });
  } catch (err) {
    next(err);
  }
});

// Get all labs
router.get("/", async (req, res, next) => {
  try {
    const labs = await Lab.find();
    res.json({ success: true, data: labs });
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

// Update lab
router.put("/:id", async (req, res, next) => {
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

// Delete lab
router.delete("/:id", async (req, res, next) => {
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

