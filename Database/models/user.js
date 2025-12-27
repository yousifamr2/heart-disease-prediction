const mongoose = require("mongoose");

const userSchema = mongoose.Schema({

    national_id: { // الرقم القومي للمريض
        type: String,
        required: true,
        unique: true,
        length: 14,
        trim: true
    },

    username: { // الاسم المستخدم للمريض
        type: String,
        required: true,
        trim: true
    },

    email: { // البريد الإلكتروني للمريض
        type: String,
        required: true,
        unique: true,
        trim: true
    },

    password: { // كلمة المرور للمريض
        type: String,
        required: true,
        minlength: 6
    }
}, {
    timestamps: true
});

module.exports = mongoose.model("User", userSchema);

