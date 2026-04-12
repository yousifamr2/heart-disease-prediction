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
        minlength: 14,
        maxlength: 14,
        index: true,
        validate: {
            validator: function(v) {
                return /^\d{14}$/.test(v); // يجب أن يكون 14 رقم فقط
            },
            message: "National ID must be exactly 14 digits"
        }
    },

    // Features (الميزات الموجودة في التحليل بتاعه الـ ML)
    features: { 
        age: { type: Number, required: true },
        sex: { type: Number, required: true },
        chest_pain_type: { type: Number, required: true },
        resting_bp_s: { type: Number, required: true },
        cholesterol: { type: Number, required: true },
        fasting_blood_sugar: { type: Number, required: true },
        resting_ecg: { type: Number, required: true },
        max_heart_rate: { type: Number, required: true },
        exercise_angina: { type: Number, required: true },
        oldpeak: { type: Number, required: true },
        st_slope: { type: Number, required: true }
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
