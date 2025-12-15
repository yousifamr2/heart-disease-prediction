const mongoose = require("mongoose");

const labsSchema = mongoose.Schema({
    
    email : String,
    username : String,
    password : String,
    
});

module.exports = mongoose.model("Lab", labsSchema);