const mongoose = require("mongoose");

const datapredictionSchema = mongoose.Schema({
    
    patient_id: String,
    age: Number,
    sex: String,
    cp: Number,
    trestbps: Number,
    chol: Number,
    fbs: Number,
    restecg: Number,
    thalach: Number,
    exang: Number,
    oldpeak: String,
    slope: Number,
    ca: Number,
    thal: Number,
    
});

module.exports = mongoose.model("Data Prediction", datapredictionSchema);