const mongoose = require("mongoose");

const predictionresultSchema = mongoose.Schema({
    
    patient_id: String,
    prediction: String,
    percentage: Number,
    recommenddoctor: String,
    
});

module.exports = mongoose.model("Prediction Result", predictionresultSchema);