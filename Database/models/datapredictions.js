const mongoose = require("mongoose");

const datapredictionSchema = mongoose.Schema({
    
    patient_id: {
        type: String,
        required: true
    },
    age: {
        type: Number,
        required: true,
        min: 0
    },
    sex: {
        type: String,
        required: true,
        enum: ['M', 'F', 'Male', 'Female']
    },
    cp: {
        type: Number,
        required: true
    },
    trestbps: {
        type: Number,
        required: true,
        min: 0
    },
    chol: {
        type: Number,
        required: true,
        min: 0
    },
    fbs: {
        type: Number,
        required: true
    },
    restecg: {
        type: Number,
        required: true
    },
    thalach: {
        type: Number,
        required: true,
        min: 0
    },
    exang: {
        type: Number,
        required: true
    },
    oldpeak: {
        type: String,
        required: true
    },
    slope: {
        type: Number,
        required: true
    },
    ca: {
        type: Number,
        required: true
    },
    thal: {
        type: Number,
        required: true
    },
    
}, {
    timestamps: true
});

module.exports = mongoose.model("Data Prediction", datapredictionSchema);