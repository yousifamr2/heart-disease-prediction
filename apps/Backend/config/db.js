const mongoose = require("mongoose");

const connectDB = async () => {
  try {
    // mongodb+srv:// يتعامل مع SSL/TLS تلقائياً — لا داعي لتحديدها يدوياً
    await mongoose.connect(process.env.MONGO_URI, {
      serverSelectionTimeoutMS: 10000, // انتظر 10 ثوان قبل إظهار الخطأ
      socketTimeoutMS: 45000,
    });
    console.log("MongoDB Connected Successfully!");
  } catch (err) {
    console.error("MongoDB Connection Error:", err.message);
    process.exit(1);
  }
};

module.exports = connectDB;
