const mongoose = require("mongoose");

const hospitalsSchema = mongoose.Schema({

    name: {
        type: String,
        required: true,
        trim: true
    },

    area: {
        type: String, // Smouha, Sidi Gaber, Loran...
        required: true,
        trim: true
    },

    google_maps_link: {
        type: String,
        required: true
    },

    address: {
        type: String,
        trim: true
    }

}, {
    timestamps: true
});

module.exports = mongoose.model("Hospital", hospitalsSchema);
