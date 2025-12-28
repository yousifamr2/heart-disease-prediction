const mongoose = require("mongoose");

const labTestSchema = mongoose.Schema({

    // ربط التحليل بالمعمل
    lab_id: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "Lab",
        required: true
    },

    // ربط التحليل بالمريض
    national_id: {
        type: String,
        required: true,
        length: 14,
        index: true
    },

    // Features (الميزات الموجودة في التحليل بتاعه الـ ML)
    features: { 
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
        thal: { type: Number, required: true }
    },

    // نتيجة التنبؤ (بتتخزن بعد Start Prediction)
    prediction_result: {
        type: String,
        enum: ["High Risk", "Low Risk"],
        default: null
    },

    prediction_percentage: { // النسبة المئوية للتنبؤ
        type: Number,
        min: 0,
        max: 100,
        default: null
    }

}, {
    timestamps: true 
});

module.exports = mongoose.model("LabTest", labTestSchema);
