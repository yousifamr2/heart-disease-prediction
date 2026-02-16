const mongoose = require("mongoose");

const hospitalsSchema = mongoose.Schema({

    name: { // الاسم الموجود في جوجل مابس للمستشفى
        type: String,
        required: true,
        trim: true
    },

    area: {
        type: String, // Smouha, Sidi Gaber, Loran...
        required: true,
        trim: true
    },

    google_maps_link: { // الرابط الموجود في جوجل مابس للمستشفى
        type: String,
        required: true
    }

}, {
    timestamps: true
});

module.exports = mongoose.model("Hospital", hospitalsSchema);
