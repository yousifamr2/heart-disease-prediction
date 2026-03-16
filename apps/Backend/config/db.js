const mongoose = require("mongoose");

const connectDB = async () => {
  try {
    const options = {
      // SSL/TLS options for MongoDB Atlas
      ssl: true,
      tls: true,
      // Remove deprecated options and use modern connection
    };
    
    await mongoose.connect(process.env.MONGO_URI, options);
    console.log("MongoDB Connected Successfully!");
  } catch (err) {
    console.error("MongoDB Connection Error: ", err);
    process.exit(1);
  }
};

module.exports = connectDB;
