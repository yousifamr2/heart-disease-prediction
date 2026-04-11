const express = require("express");
const jwt = require("jsonwebtoken");
const bcrypt = require("bcrypt");
const User = require("../models/user");

const router = express.Router();

// Generate JWT Token (requires JWT_SECRET in .env)
const generateToken = (userId) => {
  const secret = process.env.JWT_SECRET;
  if (!secret || secret.trim() === "") {
    throw new Error("Server misconfiguration: JWT_SECRET is not set in .env");
  }
  return jwt.sign(
    { userId },
    secret,
    { expiresIn: process.env.JWT_EXPIRE || "30d" }
  );
};

// Register new user
router.post("/register", async (req, res, next) => {
  try {
    const { national_id, username, email, password } = req.body;

    // Validation
    if (!national_id || !username || !email || !password) {
      return res.status(400).json({
        success: false,
        message: "Please provide all required fields"
      });
    }

    // Check if user already exists
    const existingUser = await User.findOne({
      $or: [{ email }, { national_id }]
    });

    if (existingUser) {
      return res.status(400).json({
        success: false,
        message: "User with this email or national ID already exists"
      });
    }

    // Hash password then create user
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);
    const user = await User.create({ national_id, username, email, password: hashedPassword });

    // Generate token
    const token = generateToken(user._id);

    res.status(201).json({
      success: true,
      message: "User registered successfully",
      data: user,
      token
    });
  } catch (err) {
    next(err);
  }
});

// Login user
router.post("/login", async (req, res, next) => {
  try {
    const { email, password } = req.body;

    // Validation
    if (!email || !password) {
      return res.status(400).json({
        success: false,
        message: "Please provide email and password"
      });
    }

    // Find user by email
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(401).json({
        success: false,
        message: "Invalid email or password"
      });
    }

    // Check password
    const isPasswordValid = await user.comparePassword(password);
    if (!isPasswordValid) {
      return res.status(401).json({
        success: false,
        message: "Invalid email or password"
      });
    }

    // Generate token
    const token = generateToken(user._id);

    res.json({
      success: true,
      message: "Login successful",
      data: user,
      token
    });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
