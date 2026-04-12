const mongoose = require("mongoose");

const labsSchema = mongoose.Schema({

    name: { // الاسم الموجود في جوجل مابس للمعمل
        type: String,
        required: true,
        trim: true
    },
    lab_code: { // الكود الموجود في جوجل مابس للمعمل
        type: String,
        required: true,
        unique: true,
        trim: true
    },
    address: { // العنوان الموجود في جوجل مابس للمعمل
        type: String,
        required: true
    }
 
}, {
    timestamps: true
});

module.exports = mongoose.model("Lab", labsSchema);
