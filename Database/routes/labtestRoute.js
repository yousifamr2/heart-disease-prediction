const express = require("express");
const LabTest = require("../models/labtest");
const { validateLabTest } = require("../middleware/validation");
const { authenticate } = require("../middleware/auth");

const router = express.Router();

// Create new lab test (protected)
router.post("/", authenticate, validateLabTest, async (req, res, next) => {
  try {
    const labTest = await LabTest.create(req.body);
    res.status(201).json({ success: true, data: labTest });
  } catch (err) {
    next(err);
  }
});

// Get all lab tests (with pagination)
router.get("/", async (req, res, next) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10;
    const skip = (page - 1) * limit;

    const total = await LabTest.countDocuments();
    const labTests = await LabTest.find()
      .populate("lab_id")
      .skip(skip)
      .limit(limit)
      .sort({ createdAt: -1 });

    res.json({
      success: true,
      data: labTests,
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

// Get latest lab test by national_id (أحدث تحليل للمريض)
router.get("/patient/:national_id/latest", async (req, res, next) => {
  try {
    const latestLabTest = await LabTest.findOne({ national_id: req.params.national_id })
      .sort({ createdAt: -1 })  // ترتيب تنازلي (الأحدث أولاً)
      .populate("lab_id");
    
    if (!latestLabTest) {
      return res.status(404).json({ success: false, message: "No lab tests found for this patient" });
    }
    
    res.json({ success: true, data: latestLabTest });
  } catch (err) {
    next(err);
  }
});

// Get lab tests by national_id (كل التحاليل للمريض)
router.get("/patient/:national_id", async (req, res, next) => {
  try {
    const labTests = await LabTest.find({ national_id: req.params.national_id })
      .sort({ createdAt: -1 })  // ترتيب تنازلي (الأحدث أولاً)
      .populate("lab_id");
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

// Update lab test (protected)
router.put("/:id", authenticate, async (req, res, next) => {
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

// Update prediction result only (protected)
router.patch("/:id/prediction", authenticate, async (req, res, next) => {
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

// Delete lab test (protected)
router.delete("/:id", authenticate, async (req, res, next) => {
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

