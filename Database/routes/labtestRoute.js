const express = require("express");
const LabTest = require("../models/labtest");

const router = express.Router();

// Create new lab test
router.post("/", async (req, res, next) => {
  try {
    const labTest = await LabTest.create(req.body);
    res.status(201).json({ success: true, data: labTest });
  } catch (err) {
    next(err);
  }
});

// Get all lab tests
router.get("/", async (req, res, next) => {
  try {
    const labTests = await LabTest.find().populate("lab_id");
    res.json({ success: true, data: labTests });
  } catch (err) {
    next(err);
  }
});

// Get single lab test by id
router.get("/:id", async (req, res, next) => {
  try {
    const labTest = await LabTest.findById(req.params.id).populate("lab_id");
    if (!labTest) {
      return res.status(404).json({ success: false, message: "Lab test not found" });
    }
    res.json({ success: true, data: labTest });
  } catch (err) {
    next(err);
  }
});

// Get lab tests by national_id
router.get("/patient/:national_id", async (req, res, next) => {
  try {
    const labTests = await LabTest.find({ national_id: req.params.national_id }).populate("lab_id");
    res.json({ success: true, data: labTests });
  } catch (err) {
    next(err);
  }
});

// Get lab tests by lab_id
router.get("/lab/:lab_id", async (req, res, next) => {
  try {
    const labTests = await LabTest.find({ lab_id: req.params.lab_id }).populate("lab_id");
    res.json({ success: true, data: labTests });
  } catch (err) {
    next(err);
  }
});

// Update lab test
router.put("/:id", async (req, res, next) => {
  try {
    const labTest = await LabTest.findByIdAndUpdate(req.params.id, req.body, {
      new: true,
      runValidators: true,
    }).populate("lab_id");
    if (!labTest) {
      return res.status(404).json({ success: false, message: "Lab test not found" });
    }
    res.json({ success: true, data: labTest });
  } catch (err) {
    next(err);
  }
});

// Update prediction result only
router.patch("/:id/prediction", async (req, res, next) => {
  try {
    const { prediction_result, prediction_percentage } = req.body;
    const labTest = await LabTest.findByIdAndUpdate(
      req.params.id,
      { prediction_result, prediction_percentage },
      {
        new: true,
        runValidators: true,
      }
    ).populate("lab_id");
    if (!labTest) {
      return res.status(404).json({ success: false, message: "Lab test not found" });
    }
    res.json({ success: true, data: labTest });
  } catch (err) {
    next(err);
  }
});

// Delete lab test
router.delete("/:id", async (req, res, next) => {
  try {
    const labTest = await LabTest.findByIdAndDelete(req.params.id);
    if (!labTest) {
      return res.status(404).json({ success: false, message: "Lab test not found" });
    }
    res.json({ success: true, message: "Lab test deleted" });
  } catch (err) {
    next(err);
  }
});

module.exports = router;

