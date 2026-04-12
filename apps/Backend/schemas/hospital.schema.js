const { z } = require("zod");

// Hospital Create Schema
const hospitalCreateSchema = z.object({
  body: z.object({
    name: z
      .string({
        required_error: "Hospital name is required",
      })
      .min(2, "Hospital name must be at least 2 characters")
      .max(100, "Hospital name must not exceed 100 characters")
      .trim(),
    
    area: z
      .string({
        required_error: "Area is required",
      })
      .min(2, "Area must be at least 2 characters")
      .trim(),
    
    google_maps_link: z
      .string({
        required_error: "Google Maps link is required",
      })
      .url("Please provide a valid URL for Google Maps link")
      .trim(),
  }).strict(),
});

// Hospital Update Schema (all fields optional)
const hospitalUpdateSchema = z.object({
  body: z.object({
    name: z
      .string()
      .min(2, "Hospital name must be at least 2 characters")
      .max(100, "Hospital name must not exceed 100 characters")
      .trim()
      .optional(),

    area: z
      .string()
      .min(2, "Area must be at least 2 characters")
      .trim()
      .optional(),

    google_maps_link: z
      .string()
      .url("Please provide a valid URL for Google Maps link")
      .trim()
      .optional(),
  }).strict(),
});

module.exports = {
  hospitalCreateSchema,
  hospitalUpdateSchema,
};
