const mongoose = require("mongoose");

const predictionsSchema = mongoose.Schema({
    _id: {
        type: Number,
        required: true
    },
    user_id: {
        type: Number,
        required: true,
        ref: "User"
    },
    result: {
        type: Number, // 0 = No Disease, 1 = Disease
        required: true
    },
    probability: {   // 0 → 1
        type: Number,
        required: true
    }
}, {
    timestamps: true
});

module.exports = mongoose.model("Prediction", predictionsSchema);
