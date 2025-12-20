const mongoose = require("mongoose");

const labTestsSchema = mongoose.Schema({
    _id: {
        type: Number,
        required: true
    },
    lab_id: {
        type: Number,
        required: true,
        ref: "Lab"
    },
    test_token: {
        type: String,
        required: true,
        unique: true
    },

    age: { type: Number, required: true },
    sex: { type: Number, required: true },
    chest_pain_type: { type: Number, required: true },
    resting_blood_pressure: { type: Number, required: true },
    cholesterol: { type: Number, required: true },
    fasting_blood_sugar: { type: Number, required: true },
    resting_ecg: { type: Number, required: true },
    max_heart_rate: { type: Number, required: true },
    exercise_angina: { type: Number, required: true },
    oldpeak: { type: Number, required: true },
    slope: { type: Number, required: true },
    ca: { type: Number, required: true },
    thal: { type: Number, required: true },

    is_claimed: {
        type: Boolean,
        default: false
    }
}, {
    timestamps: true
});

module.exports = mongoose.model("LabTest", labTestsSchema);
