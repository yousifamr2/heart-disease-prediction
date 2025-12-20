const mongoose = require("mongoose");

const labsSchema = mongoose.Schema({
    _id: {
        type: Number,
        required: true
    },
    name: {
        type: String,
        required: true,
        trim: true
    },
    lab_code: {
        type: String,
        required: true,
        unique: true,
        uppercase: true,
        trim: true
    },
    address: {
        type: String,
        required: true
    }
}, {
    timestamps: true
});

module.exports = mongoose.model("Lab", labsSchema);
