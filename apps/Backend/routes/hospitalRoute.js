const express = require("express");
const prisma = require("../config/prisma");
const { validate } = require("../middleware/validate");
const { handlePrismaError } = require("../middleware/prismaErrors");
const { hospitalCreateSchema, hospitalUpdateSchema } = require("../schemas/hospital.schema");
const { authenticate } = require("../middleware/auth");

const router = express.Router();

// Create new hospital (protected)
router.post("/", authenticate, validate(hospitalCreateSchema), async (req, res, next) => {
  try {
    const hospital = await prisma.hospital.create({ data: req.body });
    res.status(201).json({ success: true, data: hospital });
  } catch (err) {
    if (handlePrismaError(err, res)) return;
    next(err);
  }
});

// Get all hospitals (with pagination)
router.get("/", async (req, res, next) => {
  try {
    const page = Math.max(1, parseInt(req.query.page) || 1);
    const limit = Math.min(100, Math.max(1, parseInt(req.query.limit) || 10));
    const skip = (page - 1) * limit;

    const [total, hospitals] = await Promise.all([
      prisma.hospital.count(),
      prisma.hospital.findMany({
        skip, take: limit, orderBy: { createdAt: "desc" },
      }),
    ]);

    res.json({
      success: true,
      data: hospitals,
      pagination: { page, limit, total, totalPages: Math.ceil(total / limit) },
    });
  } catch (err) {
    next(err);
  }
});

// Get single hospital by id
router.get("/:id", async (req, res, next) => {
  try {
    const hospital = await prisma.hospital.findUnique({ where: { id: req.params.id } });
    if (!hospital) return res.status(404).json({ success: false, message: "Hospital not found" });
    res.json({ success: true, data: hospital });
  } catch (err) {
    next(err);
  }
});

// Get hospitals by area (case-insensitive)
router.get("/area/:area", async (req, res, next) => {
  try {
    const hospitals = await prisma.hospital.findMany({
      where: { area: { equals: req.params.area, mode: "insensitive" } },
    });
    res.json({ success: true, data: hospitals });
  } catch (err) {
    next(err);
  }
});

// Update hospital (protected)
router.put("/:id", authenticate, validate(hospitalUpdateSchema), async (req, res, next) => {
  try {
    const hospital = await prisma.hospital.update({
      where: { id: req.params.id },
      data: req.body,
    });
    res.json({ success: true, data: hospital });
  } catch (err) {
    if (handlePrismaError(err, res)) return;
    next(err);
  }
});

// Delete hospital (protected)
router.delete("/:id", authenticate, async (req, res, next) => {
  try {
    await prisma.hospital.delete({ where: { id: req.params.id } });
    res.json({ success: true, message: "Hospital deleted successfully" });
  } catch (err) {
    if (handlePrismaError(err, res)) return;
    next(err);
  }
});

module.exports = router;
