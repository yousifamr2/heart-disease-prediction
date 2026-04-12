const jwt = require("jsonwebtoken");
const prisma = require("../config/prisma");

// Middleware to verify JWT token
const authenticate = async (req, res, next) => {
  try {
    // Get token from header
    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      return res.status(401).json({
        success: false,
        message: "No token provided. Please provide a valid token."
      });
    }

    // Extract token
    const token = authHeader.substring(7); // Remove "Bearer " prefix

    if (!token) {
      return res.status(401).json({
        success: false,
        message: "No token provided"
      });
    }

    // Verify token
    const decoded = jwt.verify(token, process.env.JWT_SECRET);

    const user = await prisma.user.findUnique({
      where: { id: decoded.userId },
      select: { id: true, national_id: true, username: true, email: true },
    });

    if (!user) {
      return res.status(401).json({
        success: false,
        message: "User not found"
      });
    }

    req.user = user;
    next();
  } catch (err) {
    if (err.name === "JsonWebTokenError") {
      return res.status(401).json({
        success: false,
        message: "Invalid token"
      });
    }
    if (err.name === "TokenExpiredError") {
      return res.status(401).json({
        success: false,
        message: "Token expired"
      });
    }
    next(err);
  }
};

// Optional: Middleware to check if user is admin (if you add role field later)
const authorize = (...roles) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({
        success: false,
        message: "Unauthorized"
      });
    }

    // If roles array is empty, allow all authenticated users
    if (roles.length === 0) {
      return next();
    }

    // Check if user role is in allowed roles
    // This is optional - you can add role field to User model later
    // if (roles.includes(req.user.role)) {
    //   return next();
    // }

    // For now, allow all authenticated users
    next();
  };
};

module.exports = {
  authenticate,
  authorize
};
