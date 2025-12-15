const mongoose = require("mongoose");

const patientSchema = mongoose.Schema({
    
    _id : {
        type: String,
        required: true,
        unique: true
    },
    email : {
        type: String,
        required: true,
        unique: true,
        lowercase: true,
        trim: true
    },
    address : {
        type: String,
        required: true
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

module.exports = mongoose.model("Patient", patientSchema);