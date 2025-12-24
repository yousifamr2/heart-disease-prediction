const mongoose = require("mongoose");

const labsSchema = mongoose.Schema({

    name: {
        type: String,
        required: true,
        trim: true
    },
    lab_code: {
        type: String,
        required: true,
        unique: true,
        trim: true
    },
    address: {
        type: String,
        required: true
    },
    google_maps_link: {
        type: String,
        required: true
    },
}, {
    timestamps: true
});

module.exports = mongoose.model("Lab", labsSchema);
