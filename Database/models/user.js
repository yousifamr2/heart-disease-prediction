const mongoose = require("mongoose");

const userSchema = mongoose.Schema({

    national_id: {
        type: String,
        required: true,
        unique: true,
        length: 14,
        trim: true
    },

    username: {
        type: String,
        required: true,
        trim: true
    },

    email: {
        type: String,
        required: true,
        unique: true,
        trim: true
    },

    password: {
        type: String,
        required: true,
        minlength: 6
    }
}, {
    timestamps: true
});

module.exports = mongoose.model("User", userSchema);

