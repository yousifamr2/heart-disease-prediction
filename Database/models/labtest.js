const mongoose = require("mongoose");

const labTestsSchema = mongoose.Schema({

    //  المعمل (pre-registered)
    lab_id: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "Lab",
        required: true
    },

    //  التوكن اللي المريض بياخده من المعمل
    test_token: {
        type: String,
        required: true,
        unique: true
    },

    //  بيتربط بالمستخدم بعد ما يدخل التوكن
    user_id: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "User",
        default: null
    },

    //  Features بتاعة الـ ML Model
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

    //  هل التوكن اتستخدم ولا لسه
    is_claimed: {
        type: Boolean,
        default: false
    },

    //  نتيجة التنبؤ (محفوظة هنا بدل table تانية)
    prediction_result: {
        type: String,
        enum: ["High Risk", "Low Risk"],
        default: null
    },

    //  نسبة الخطورة
    prediction_percentage: {
        type: Number,
        min: 0,
        max: 100,
        default: null
    }

}, {
    timestamps: true
});

module.exports = mongoose.model("LabTest", labTestsSchema);
