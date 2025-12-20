const mongoose = require("mongoose");

const usersSchema = mongoose.Schema({
    _id: {
        type: Number,
        required: true
    },
    username: {
        type: String,
        required: true,
        unique: true,
        trim: true
    },
    email: {
        type: String,
        required: true,
        unique: true,
        lowercase: true,
        trim: true
    },
    password: {
        type: String,
        required: true,
        minlength: 6
    },
    address: {
        type: String,
        required: true
    },

    // 👇 مهم جدًا للسيناريو
    is_verified: {
        type: Boolean,
        default: false
    },

    // 👇 توضيحي للمناقشة (اختياري)
    registered_via_lab: {
        type: Boolean,
        default: false
    }

}, {
    timestamps: true
});

module.exports = mongoose.model("User", usersSchema);
