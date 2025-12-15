const mongoose = require("mongoose");

const predictionresultSchema = mongoose.Schema({
    
    patient_id: {
        type: String,
        required: true
    },
    prediction: {
        type: String,
        required: true,
        enum: ['Positive', 'Negative', 'High Risk', 'Low Risk']
    },
    percentage: {
        type: Number,
        required: true,
        min: 0,
        max: 100
    },
    recommenddoctor: {
        type: String,
        required: false
    },
    
}, {
    timestamps: true
});

module.exports = mongoose.model("Prediction Result", predictionresultSchema);