const express = require("express");
const prisma = require("../config/prisma");
const { validate } = require("../middleware/validate");
const { handlePrismaError } = require("../middleware/prismaErrors");
const {
  labTestCreateSchema,
  labTestUpdateSchema,
  predictionUpdateSchema,
} = require("../schemas/labtest.schema");
const { authenticate } = require("../middleware/auth");

const router = express.Router();

// Helper: flatten features object into top-level Prisma fields
const flattenFeatures = (body) => {
  const { features, ...rest } = body;
  return { ...rest, ...(features || {}) };
};

// Helper: nest flat Prisma fields back into features object for response
const shapeLabTest = (labTest) => {
  if (!labTest) return null;
  const {
    age, sex, chest_pain_type, resting_bp_s, cholesterol,
    fasting_blood_sugar, resting_ecg, max_heart_rate,
    exercise_angina, oldpeak, st_slope, ...rest
  } = labTest;
  return {
    ...rest,
    features: {
      age, sex, chest_pain_type, resting_bp_s, cholesterol,
      fasting_blood_sugar, resting_ecg, max_heart_rate,
      exercise_angina, oldpeak, st_slope,
    },
  };
};

const labTestInclude = { lab: true };

// Create new lab test (protected)
router.post("/", authenticate, validate(labTestCreateSchema), async (req, res, next) => {
  try {
    const data = flattenFeatures(req.body);
    const labTest = await prisma.labTest.create({
      data,
      include: labTestInclude,
    });
    res.status(201).json({ success: true, data: shapeLabTest(labTest) });
  } catch (err) {
    if (handlePrismaError(err, res)) return;
    next(err);
  }
});

// Get all lab tests (with pagination)
router.get("/", async (req, res, next) => {
  try {
    const page = Math.max(1, parseInt(req.query.page) || 1);
    const limit = Math.min(100, Math.max(1, parseInt(req.query.limit) || 10));
    const skip = (page - 1) * limit;

    const [total, labTests] = await Promise.all([
      prisma.labTest.count(),
      prisma.labTest.findMany({
        skip, take: limit,
        orderBy: { createdAt: "desc" },
        include: labTestInclude,
      }),
    ]);

    res.json({
      success: true,
      data: labTests.map(shapeLabTest),
      pagination: { page, limit, total, totalPages: Math.ceil(total / limit) },
    });
  } catch (err) {
    next(err);
  }
});

// Get single lab test by id
router.get("/:id", async (req, res, next) => {
  try {
    const labTest = await prisma.labTest.findUnique({
      where: { id: req.params.id },
      include: labTestInclude,
    });
    if (!labTest) return res.status(404).json({ success: false, message: "Lab test not found" });
    res.json({ success: true, data: shapeLabTest(labTest) });
  } catch (err) {
    next(err);
  }
});

// Get all lab tests by national_id
router.get("/patient/:national_id", async (req, res, next) => {
  try {
    const labTests = await prisma.labTest.findMany({
      where: { national_id: req.params.national_id },
      orderBy: { createdAt: "desc" },
      include: labTestInclude,
    });
    res.json({ success: true, data: labTests.map(shapeLabTest) });
  } catch (err) {
    next(err);
  }
});

// Get latest lab test by national_id
router.get("/patient/:national_id/latest", async (req, res, next) => {
  try {
    const labTest = await prisma.labTest.findFirst({
      where: { national_id: req.params.national_id },
      orderBy: { createdAt: "desc" },
      include: labTestInclude,
    });
    if (!labTest) return res.status(404).json({ success: false, message: "No lab tests found for this patient" });
    res.json({ success: true, data: shapeLabTest(labTest) });
  } catch (err) {
    next(err);
  }
});

// Get lab tests by lab_id
router.get("/lab/:lab_id", async (req, res, next) => {
  try {
    const labTests = await prisma.labTest.findMany({
      where: { lab_id: req.params.lab_id },
      orderBy: { createdAt: "desc" },
      include: labTestInclude,
    });
    res.json({ success: true, data: labTests.map(shapeLabTest) });
  } catch (err) {
    next(err);
  }
});

// Update lab test (protected)
router.put("/:id", authenticate, validate(labTestUpdateSchema), async (req, res, next) => {
  try {
    const data = flattenFeatures(req.body);
    const labTest = await prisma.labTest.update({
      where: { id: req.params.id },
      data,
      include: labTestInclude,
    });
    res.json({ success: true, data: shapeLabTest(labTest) });
  } catch (err) {
    if (handlePrismaError(err, res)) return;
    next(err);
  }
});

// Update prediction result only (protected)
router.patch("/:id/prediction", authenticate, validate(predictionUpdateSchema), async (req, res, next) => {
  try {
    const { prediction_result, prediction_percentage } = req.body;
    const labTest = await prisma.labTest.update({
      where: { id: req.params.id },
      data: { prediction_result, prediction_percentage },
      include: labTestInclude,
    });
    res.json({ success: true, data: shapeLabTest(labTest) });
  } catch (err) {
    if (handlePrismaError(err, res)) return;
    next(err);
  }
});

// Delete lab test (protected)
router.delete("/:id", authenticate, async (req, res, next) => {
  try {
    await prisma.labTest.delete({ where: { id: req.params.id } });
    res.json({ success: true, message: "Lab test deleted successfully" });
  } catch (err) {
    if (handlePrismaError(err, res)) return;
    next(err);
  }
});

module.exports = router;
