const mongoose = require("mongoose");

const doctorSchema = mongoose.Schema({
   
   _id: String,
   name: String,
   address: String,
   specialization: String,
   rating: Number,
   latitude: Number,
   longitude: Number,
   
});

module.exports = mongoose.model("Doctor", doctorSchema);