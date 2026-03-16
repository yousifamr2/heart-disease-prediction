const path = require("path");
const dotenv = require("dotenv");

// تحميل .env من نفس مجلد المشروع (حتى لو السيرفر شغّل من مجلد آخر)
dotenv.config({ path: path.join(__dirname, ".env") });

const express = require("express");
const cors = require("cors"); 
const helmet = require("helmet");
const connectDB = require("./config/db");

// التأكد من وجود JWT_SECRET قبل تشغيل السيرفر
if (!process.env.JWT_SECRET || String(process.env.JWT_SECRET).trim() === "") {
  console.error("ERROR: JWT_SECRET is missing or empty in .env");
  process.exit(1);
}

const app = express();

// Security middleware
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

// Connect to database
connectDB();

// Request logging middleware
app.use((req, res, next) => {
    console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
    next();
});

// Routes
app.use("/api/auth", require("./routes/authRoute"));
app.use("/api/users", require("./routes/userRoute"));
app.use("/api/labs", require("./routes/labRoute"));
app.use("/api/labtests", require("./routes/labtestRoute"));
app.use("/api/hospitals", require("./routes/hospitalRoute"));
// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(err.status || 500).json({
        success: false,
        message: err.message || "Internal Server Error",
        ...(process.env.NODE_ENV === 'development' && { stack: err.stack })
    });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({
        success: false,
        message: "Route not found"
    });
});

// Start server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
