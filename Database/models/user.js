const mongoose = require("mongoose");
const bcrypt = require("bcrypt");

const userSchema = mongoose.Schema({

    national_id: { // الرقم القومي للمريض
        type: String,
        required: true,
        unique: true,
        minlength: 14,
        maxlength: 14,
        trim: true,
        validate: {
            validator: function(v) {
                return /^\d{14}$/.test(v); // يجب أن يكون 14 رقم فقط
            },
            message: "National ID must be exactly 14 digits"
        }
    },

    username: { // الاسم المستخدم للمريض
        type: String,
        required: true,
        trim: true,
        minlength: 2,
        maxlength: 50
    },

    email: { // البريد الإلكتروني للمريض
        type: String,
        required: true,
        unique: true, 
        trim: true,
        lowercase: true,
        validate: {
            validator: function(v) {
                return /^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$/.test(v);
            },
            message: "Please enter a valid email address"
        }
    },

    password: { // كلمة المرور للمريض
        type: String,
        required: true,
        minlength: 6
    }
}, {
    timestamps: true
});

// تشفير الباسورد يتم في الـ routes قبل الحفظ (لتجنب مشكلة next مع Mongoose 9)

// دالة للتحقق من كلمة المرور
userSchema.methods.comparePassword = async function(candidatePassword) {
    return await bcrypt.compare(candidatePassword, this.password);
};

// إخفاء كلمة المرور عند إرجاع البيانات
userSchema.methods.toJSON = function() {
    const user = this.toObject();
    delete user.password;
    return user;
};

module.exports = mongoose.model("User", userSchema);

