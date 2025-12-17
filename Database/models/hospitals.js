const mongoose = require("mongoose");

const hospitalsSchema = mongoose.Schema({
   
   _id: {
       type: String,
       required: true,
       unique: true
   },
   name: {
       type: String,
       required: true,
       trim: true
   },
   address: {
       type: String,
       required: true
   },
   specialization: {
       type: String,
       required: true,
       trim: true
   },
   rating: {
       type: Number,
       min: 0,
       max: 5,
       default: 0
   },
   latitude: {
       type: Number,
       required: true
   },
   longitude: {
       type: Number,
       required: true
   },
   
}, {
    timestamps: true
});

module.exports = mongoose.model("Hospitals", hospitalsSchema);