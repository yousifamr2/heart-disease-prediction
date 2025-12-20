const mongoose = require("mongoose");

const hospitalsSchema = mongoose.Schema({
    _id: {
        type: Number,
        required: true
    },
    name: {
        type: String,
        required: true,
        trim: true
    },
    location_link: {
        type: String,
        required: true
    }
}, {
    timestamps: true
});

module.exports = mongoose.model("Hospital", hospitalsSchema);