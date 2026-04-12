const express = require("express");
const bcrypt = require("bcrypt");
const prisma = require("../config/prisma");
const { validate } = require("../middleware/validate");
const { handlePrismaError } = require("../middleware/prismaErrors");
const { userCreateSchema, userUpdateSchema } = require("../schemas/user.schema");
const { authenticate } = require("../middleware/auth");

const router = express.Router();

// Create new user (protected — admin only)
router.post("/", authenticate, validate(userCreateSchema), async (req, res, next) => {
  try {
    const { national_id, username, email, password } = req.body;
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    const user = await prisma.user.create({
      data: { national_id, username, email: email.toLowerCase(), password: hashedPassword },
    });

    const { password: _, ...userWithoutPassword } = user;
    res.status(201).json({ success: true, data: userWithoutPassword });
  } catch (err) {
    if (handlePrismaError(err, res)) return;
    next(err);
  }
});

// Get all users (with pagination)
router.get("/", authenticate, async (req, res, next) => {
  try {
    const page = Math.max(1, parseInt(req.query.page) || 1);
    const limit = Math.min(100, Math.max(1, parseInt(req.query.limit) || 10));
    const skip = (page - 1) * limit;

    const [total, users] = await Promise.all([
      prisma.user.count(),
      prisma.user.findMany({
        skip,
        take: limit,
        orderBy: { createdAt: "desc" },
        select: {
          id: true, national_id: true, username: true,
          email: true, createdAt: true, updatedAt: true,
        },
      }),
    ]);

    res.json({
      success: true,
      data: users,
      pagination: { page, limit, total, totalPages: Math.ceil(total / limit) },
    });
  } catch (err) {
    next(err);
  }
});

// Get single user by id
router.get("/:id", authenticate, async (req, res, next) => {
  try {
    const user = await prisma.user.findUnique({
      where: { id: req.params.id },
      select: {
        id: true, national_id: true, username: true,
        email: true, createdAt: true, updatedAt: true,
      },
    });
    if (!user) return res.status(404).json({ success: false, message: "User not found" });
    res.json({ success: true, data: user });
  } catch (err) {
    next(err);
  }
});

// Update user (protected)
router.put("/:id", authenticate, validate(userUpdateSchema), async (req, res, next) => {
  try {
    const updateData = { ...req.body };
    if (updateData.password) {
      const salt = await bcrypt.genSalt(10);
      updateData.password = await bcrypt.hash(updateData.password, salt);
    }
    if (updateData.email) updateData.email = updateData.email.toLowerCase();

    const user = await prisma.user.update({
      where: { id: req.params.id },
      data: updateData,
      select: {
        id: true, national_id: true, username: true,
        email: true, createdAt: true, updatedAt: true,
      },
    });
    res.json({ success: true, data: user });
  } catch (err) {
    if (handlePrismaError(err, res)) return;
    next(err);
  }
});

// Delete user (protected)
router.delete("/:id", authenticate, async (req, res, next) => {
  try {
    await prisma.user.delete({ where: { id: req.params.id } });
    res.json({ success: true, message: "User deleted successfully" });
  } catch (err) {
    if (handlePrismaError(err, res)) return;
    next(err);
  }
});

module.exports = router;
