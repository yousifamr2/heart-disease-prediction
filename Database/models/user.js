const mongoose = require("mongoose");

const usersSchema = mongoose.Schema({
    _id: {
        type: Number,
        required: true,
        unique: true
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
    },

}, {
    timestamps: true
});

module.exports = mongoose.model("User", usersSchema);
