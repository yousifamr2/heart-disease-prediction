// Input validation middleware

const validateLabTest = (req, res, next) => {
  const { lab_id, national_id, features } = req.body;

  if (!lab_id || !national_id || !features) {
    return res.status(400).json({
      success: false,
      message: "lab_id, national_id, and features are required"
    });
  }

  const requiredFeatures = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_bp_s",
    "cholesterol",
    "fasting_blood_sugar",
    "resting_ecg",
    "max_heart_rate",
    "exercise_angina",
    "oldpeak",
    "st_slope"
  ];

  const missingFeatures = requiredFeatures.filter(
    (feature) => features[feature] === undefined
  );

  if (missingFeatures.length > 0) {
    return res.status(400).json({
      success: false,
      message: `Missing required features: ${missingFeatures.join(", ")}`
    });
  }

  next();
};

const validateUser = (req, res, next) => {
  const { national_id, username, email, password } = req.body;

  if (!national_id || !username || !email || !password) {
    return res.status(400).json({
      success: false,
      message: "national_id, username, email, and password are required"
    });
  }

  if (national_id.length !== 14 || !/^\d{14}$/.test(national_id)) {
    return res.status(400).json({
      success: false,
      message: "National ID must be exactly 14 digits"
    });
  }

  if (password.length < 6) {
    return res.status(400).json({
      success: false,
      message: "Password must be at least 6 characters"
    });
  }

  next();
};

module.exports = {
  validateLabTest,
  validateUser
};
