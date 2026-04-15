const { z } = require("zod");

// z.coerce.number() يقبل أرقام مرسلة كـ string أو number من JSON
const featuresSchema = z.object({
  age: z.coerce
    .number({ invalid_type_error: "Age must be a number" })
    .int("Age must be an integer")
    .min(1, "Age must be at least 1")
    .max(120, "Age must not exceed 120"),

  sex: z.coerce
    .number({ invalid_type_error: "Sex must be a number" })
    .int("Sex must be 0 or 1")
    .min(0, "Sex must be 0 or 1")
    .max(1, "Sex must be 0 or 1"),

  chest_pain_type: z.coerce
    .number({ invalid_type_error: "Chest pain type must be a number" })
    .int("Chest pain type must be an integer"),

  resting_bp_s: z.coerce
    .number({ invalid_type_error: "Resting blood pressure must be a number" })
    .positive("Resting blood pressure must be positive"),

  cholesterol: z.coerce
    .number({ invalid_type_error: "Cholesterol must be a number" })
    .nonnegative("Cholesterol must be non-negative"),

  fasting_blood_sugar: z.coerce
    .number({ invalid_type_error: "Fasting blood sugar must be a number" })
    .int("Fasting blood sugar must be 0 or 1")
    .min(0, "Fasting blood sugar must be 0 or 1")
    .max(1, "Fasting blood sugar must be 0 or 1"),

  resting_ecg: z.coerce
    .number({ invalid_type_error: "Resting ECG must be a number" })
    .int("Resting ECG must be an integer"),

  max_heart_rate: z.coerce
    .number({ invalid_type_error: "Max heart rate must be a number" })
    .int("Max heart rate must be an integer")
    .min(40, "Max heart rate must be at least 40")
    .max(250, "Max heart rate must not exceed 250"),

  exercise_angina: z.coerce
    .number({ invalid_type_error: "Exercise angina must be a number" })
    .int("Exercise angina must be 0 or 1")
    .min(0, "Exercise angina must be 0 or 1")
    .max(1, "Exercise angina must be 0 or 1"),

  oldpeak: z.coerce
    .number({ invalid_type_error: "Oldpeak must be a number" }),

  st_slope: z.coerce
    .number({ invalid_type_error: "ST slope must be a number" })
    .int("ST slope must be an integer"),
});

// Lab Test Create Schema — lab_id هو cuid (string) مش MongoDB ObjectId
const labTestCreateSchema = z.object({
  body: z.object({
    lab_id: z
      .string({ required_error: "Lab ID is required" })
      .min(1, "Lab ID is required"),

    national_id: z
      .string({ required_error: "National ID is required" })
      .length(14, "National ID must be exactly 14 digits")
      .regex(/^\d{14}$/, "National ID must contain only digits"),

    features: featuresSchema,
  }).strict(),
});

// Lab Test Update Schema (all fields optional)
const labTestUpdateSchema = z.object({
  body: z.object({
    lab_id: z.string().min(1, "Lab ID is required").optional(),

    national_id: z
      .string()
      .length(14, "National ID must be exactly 14 digits")
      .regex(/^\d{14}$/, "National ID must contain only digits")
      .optional(),

    features: featuresSchema.partial().optional(),
  }).strict(),
});

// Prediction Update Schema
const predictionUpdateSchema = z.object({
  body: z.object({
    prediction_result: z.enum(["High Risk", "Low Risk"], {
      required_error: "Prediction result is required",
      invalid_type_error: "Prediction result must be 'High Risk' or 'Low Risk'",
    }),

    prediction_percentage: z.coerce
      .number({
        required_error: "Prediction percentage is required",
        invalid_type_error: "Prediction percentage must be a number",
      })
      .min(0, "Prediction percentage must be between 0 and 100")
      .max(100, "Prediction percentage must be between 0 and 100"),
  }).strict(),
});

module.exports = {
  labTestCreateSchema,
  labTestUpdateSchema,
  predictionUpdateSchema,
};
