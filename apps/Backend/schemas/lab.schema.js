const { z } = require("zod");

// Lab Create Schema
const labCreateSchema = z.object({
  body: z.object({
    name: z
      .string({
        required_error: "Lab name is required",
      })
      .min(2, "Lab name must be at least 2 characters")
      .max(100, "Lab name must not exceed 100 characters")
      .trim(),
    
    lab_code: z
      .string({
        required_error: "Lab code is required",
      })
      .min(1, "Lab code is required")
      .trim(),
    
    address: z
      .string({
        required_error: "Lab address is required",
      })
      .min(5, "Lab address must be at least 5 characters")
      .trim(),
  }).strict(),
});

// Lab Update Schema (all fields optional)
const labUpdateSchema = z.object({
  body: z.object({
    name: z
      .string()
      .min(2, "Lab name must be at least 2 characters")
      .max(100, "Lab name must not exceed 100 characters")
      .trim()
      .optional(),

    lab_code: z
      .string()
      .min(1, "Lab code is required")
      .trim()
      .optional(),

    address: z
      .string()
      .min(5, "Lab address must be at least 5 characters")
      .trim()
      .optional(),
  }).strict(),
});

module.exports = {
  labCreateSchema,
  labUpdateSchema,
};
