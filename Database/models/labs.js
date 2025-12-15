const mongoose = require("mongoose");

const labsSchema = mongoose.Schema({
    
    email : {
        type: String,
        required: true,
        unique: true,
        lowercase: true,
        trim: true
    },
    username : {
        type: String,
        required: true,
        unique: true,
        trim: true
    },
    password : {
        type: String,
        required: true,
        minlength: 6
    },
    
}, {
    timestamps: true
});

module.exports = mongoose.model("Lab", labsSchema);