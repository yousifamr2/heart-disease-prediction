const express = require("express");
const prisma = require("../config/prisma");
const { validate } = require("../middleware/validate");
const { handlePrismaError } = require("../middleware/prismaErrors");
const { labCreateSchema, labUpdateSchema } = require("../schemas/lab.schema");
const { authenticate } = require("../middleware/auth");

const router = express.Router();

// Create new lab (protected)
router.post("/", authenticate, validate(labCreateSchema), async (req, res, next) => {
  try {
    const lab = await prisma.lab.create({ data: req.body });
    res.status(201).json({ success: true, data: lab });
  } catch (err) {
    if (handlePrismaError(err, res)) return;
    next(err);
  }
});

// Get all labs (with pagination)
router.get("/", async (req, res, next) => {
  try {
    const page = Math.max(1, parseInt(req.query.page) || 1);
    const limit = Math.min(100, Math.max(1, parseInt(req.query.limit) || 10));
    const skip = (page - 1) * limit;

    const [total, labs] = await Promise.all([
      prisma.lab.count(),
      prisma.lab.findMany({
        skip, take: limit, orderBy: { createdAt: "desc" },
      }),
    ]);

    res.json({
      success: true,
      data: labs,
      pagination: { page, limit, total, totalPages: Math.ceil(total / limit) },
    });
  } catch (err) {
    next(err);
  }
});

// Get single lab by id
router.get("/:id", async (req, res, next) => {
  try {
    const lab = await prisma.lab.findUnique({ where: { id: req.params.id } });
    if (!lab) return res.status(404).json({ success: false, message: "Lab not found" });
    res.json({ success: true, data: lab });
  } catch (err) {
    next(err);
  }
});

// Update lab (protected)
router.put("/:id", authenticate, validate(labUpdateSchema), async (req, res, next) => {
  try {
    const lab = await prisma.lab.update({
      where: { id: req.params.id },
      data: req.body,
    });
    res.json({ success: true, data: lab });
  } catch (err) {
    if (handlePrismaError(err, res)) return;
    next(err);
  }
});

// Delete lab (protected)
router.delete("/:id", authenticate, async (req, res, next) => {
  try {
    await prisma.lab.delete({ where: { id: req.params.id } });
    res.json({ success: true, message: "Lab deleted successfully" });
  } catch (err) {
    if (handlePrismaError(err, res)) return;
    next(err);
  }
});

module.exports = router;
