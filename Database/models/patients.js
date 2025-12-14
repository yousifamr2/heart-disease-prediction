const mongoose = require("mongoose");

const patientSchema = mongoose.Schema({
    
    _id : String,
    email : String,
    address : String,
    username : String,
    password : String,
    
});

module.exports = mongoose.model("Patient", patientSchema);