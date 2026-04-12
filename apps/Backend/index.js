const path = require("path");
const dotenv = require("dotenv");

// يجب أن يكون dotenv أول سطر قبل أي require آخر
dotenv.config({ path: path.join(__dirname, ".env") });

// تأكيد تحميل DATABASE_URL قبل إنشاء Prisma Client
if (!process.env.DATABASE_URL) {
  console.error("ERROR: DATABASE_URL is missing in .env");
  process.exit(1);
}

const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const prisma = require("./config/prisma");

// التأكد من وجود المتغيرات المطلوبة
if (!process.env.JWT_SECRET || String(process.env.JWT_SECRET).trim() === "") {
  console.error("ERROR: JWT_SECRET is missing or empty in .env");
  process.exit(1);
}

const app = express();

app.use(helmet());
app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

// Request logging
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

// 404 handler — must come before error handler
app.use((req, res) => {
  res.status(404).json({ success: false, message: "Route not found" });
});

// Global error handler
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(err.status || 500).json({
    success: false,
    message: err.message || "Internal Server Error",
    ...(process.env.NODE_ENV === "development" && { stack: err.stack }),
  });
});

// Start server + connect to Neon PostgreSQL
const PORT = process.env.PORT || 5000;

async function startServer() {
  try {
    await prisma.$connect();
    console.log("PostgreSQL (Neon) Connected Successfully!");
    app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
  } catch (err) {
    console.error("Database connection failed:", err.message);
    process.exit(1);
  }
}

startServer();
