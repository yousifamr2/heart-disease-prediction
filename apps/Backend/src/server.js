const path = require("path");
const dotenv = require("dotenv");

// يجب أن يكون dotenv أول سطر قبل أي require آخر
dotenv.config({ path: path.join(__dirname, "..", ".env") });

// تأكيد تحميل DATABASE_URL قبل إنشاء Prisma Client
if (!process.env.DATABASE_URL) {
  console.error("ERROR: DATABASE_URL is missing in .env");
  process.exit(1);
}

// Trigger nodemon reload
const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const rateLimit = require("express-rate-limit");
const prisma = require("./config/prisma");

// التأكد من وجود المتغيرات المطلوبة
if (!process.env.JWT_SECRET || String(process.env.JWT_SECRET).trim() === "") {
  console.error("ERROR: JWT_SECRET is missing or empty in .env");
  process.exit(1);
}

if (!process.env.INTERNAL_API_KEY || String(process.env.INTERNAL_API_KEY).trim() === "") {
  console.warn("WARN: INTERNAL_API_KEY is missing — prediction routes will fail until set.");
}

const app = express();

app.use(helmet());
app.use(
  cors({
    origin: process.env.CORS_ORIGIN || true,
    credentials: true,
  })
);
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: Number(process.env.RATE_LIMIT_MAX) || 300,
  standardHeaders: true,
  legacyHeaders: false,
});
app.use("/api", apiLimiter);

// Request logging
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
  next();
});

// Routes
app.use("/api/auth", require("./routes/authRoute"));
app.use("/api/users", require("./routes/userRoute"));
app.use("/api/labs", require("./routes/labRoute"));
app.use("/api/lab-portal", require("./routes/labPortalRoute"));
app.use("/api/labtests", require("./routes/labtestRoute"));
app.use("/api/predictions", require("./routes/predictionRoute"));
app.use("/api/ecg", require("./routes/ecgRoute"));
app.use("/api/hospitals", require("./routes/hospitalRoute"));

const { notFoundHandler, globalErrorHandler } = require("./middlewares/errorMiddleware");


// 404 handler — must come before error handler
app.use(notFoundHandler);

// Global error handler
app.use(globalErrorHandler);

const fs = require("fs");
const https = require("https");

// Start server + connect to Neon PostgreSQL
const PORT = process.env.PORT || 5000;

async function startServer() {
  try {
    await prisma.$connect();
    console.log("PostgreSQL (Neon) Connected Successfully!");

    const sslKey = process.env.SSL_KEY_PATH;
    const sslCert = process.env.SSL_CERT_PATH;

    if (sslKey && sslCert && fs.existsSync(sslKey) && fs.existsSync(sslCert)) {
      const options = {
        key: fs.readFileSync(sslKey),
        cert: fs.readFileSync(sslCert),
      };
      https.createServer(options, app).listen(PORT, () => {
        console.log(`Secure Server running over HTTPS on port ${PORT}`);
      });
    } else {
      app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
    }
  } catch (err) {
    console.error("Database connection failed:", err.message);
    process.exit(1);
  }
}

startServer();
