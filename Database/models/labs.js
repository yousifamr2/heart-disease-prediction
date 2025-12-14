const mongoose = require("mongoose");

const labsSchema = mongoose.Schema({
    
    _id : String,
    email : String,
    username : String,
    password : String,
    
});

module.exports = mongoose.model("Lab", labsSchema);