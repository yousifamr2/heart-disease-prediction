const { z } = require("zod");

// User Registration/Creation Schema
const userCreateSchema = z.object({
  body: z.object({
    national_id: z
      .string({
        required_error: "National ID is required",
      })
      .length(14, "National ID must be exactly 14 digits")
      .regex(/^\d{14}$/, "National ID must contain only digits"),
    
    username: z
      .string({
        required_error: "Username is required",
      })
      .min(2, "Username must be at least 2 characters")
      .max(50, "Username must not exceed 50 characters")
      .trim(),
    
    email: z
      .string({
        required_error: "Email is required",
      })
      .email("Please enter a valid email address")
      .toLowerCase()
      .trim(),
    
    password: z
      .string({
        required_error: "Password is required",
      })
      .min(6, "Password must be at least 6 characters"),
  }).strict(),
});

// User Login Schema
const userLoginSchema = z.object({
  body: z.object({
    email: z
      .string({
        required_error: "Email is required",
      })
      .email("Please enter a valid email address"),

    password: z
      .string({
        required_error: "Password is required",
      })
      .min(1, "Password is required"),
  }).strict(),
});

// User Update Schema (all fields optional)
const userUpdateSchema = z.object({
  body: z.object({
    username: z
      .string()
      .min(2, "Username must be at least 2 characters")
      .max(50, "Username must not exceed 50 characters")
      .trim()
      .optional(),

    email: z
      .string()
      .email("Please enter a valid email address")
      .toLowerCase()
      .trim()
      .optional(),

    password: z
      .string()
      .min(6, "Password must be at least 6 characters")
      .optional(),
  }).strict(),
});

module.exports = {
  userCreateSchema,
  userLoginSchema,
  userUpdateSchema,
};
