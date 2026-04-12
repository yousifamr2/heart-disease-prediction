# Zod Validation Schemas

This folder contains all Zod validation schemas used for request validation in the Heart Disease Prediction API.

## Overview

Zod is a TypeScript-first schema validation library that provides:
- Type-safe validation
- Clear error messages
- Easy-to-read schema definitions
- Automatic type inference

## Available Schemas

### User Schemas (`user.schema.js`)

- **`userCreateSchema`**: Validates user registration/creation
  - `national_id`: Exactly 14 digits
  - `username`: 2-50 characters
  - `email`: Valid email format
  - `password`: Minimum 6 characters

- **`userLoginSchema`**: Validates user login
  - `email`: Valid email format
  - `password`: Required

- **`userUpdateSchema`**: Validates user updates (all fields optional)
  - `username`: 2-50 characters (optional)
  - `email`: Valid email format (optional)
  - `password`: Minimum 6 characters (optional)

### Lab Test Schemas (`labtest.schema.js`)

- **`labTestCreateSchema`**: Validates lab test creation
  - `lab_id`: Valid MongoDB ObjectId
  - `national_id`: Exactly 14 digits
  - `features`: Object with 11 required medical features
    - `age`: 1-120
    - `sex`: 0 or 1
    - `chest_pain_type`: Integer
    - `resting_bp_s`: Positive number
    - `cholesterol`: Non-negative number
    - `fasting_blood_sugar`: 0 or 1
    - `resting_ecg`: Integer
    - `max_heart_rate`: 40-250
    - `exercise_angina`: 0 or 1
    - `oldpeak`: Number
    - `st_slope`: Integer

- **`labTestUpdateSchema`**: Validates lab test updates (all fields optional)

- **`predictionUpdateSchema`**: Validates prediction result updates
  - `prediction_result`: "High Risk" or "Low Risk"
  - `prediction_percentage`: 0-100

### Lab Schemas (`lab.schema.js`)

- **`labCreateSchema`**: Validates lab creation
  - `name`: 2-100 characters
  - `lab_code`: Required string
  - `address`: Minimum 5 characters

- **`labUpdateSchema`**: Validates lab updates (all fields optional)

### Hospital Schemas (`hospital.schema.js`)

- **`hospitalCreateSchema`**: Validates hospital creation
  - `name`: 2-100 characters
  - `area`: Minimum 2 characters
  - `google_maps_link`: Valid URL

- **`hospitalUpdateSchema`**: Validates hospital updates (all fields optional)

## Usage

### In Routes

```javascript
const { validate } = require("../middleware/validate");
const { userCreateSchema } = require("../schemas/user.schema");

router.post("/register", validate(userCreateSchema), async (req, res, next) => {
  // Request body is already validated
  // Access validated data from req.body
});
```

### Validation Middleware

The `validate` middleware (`middleware/validate.js`) automatically:
1. Validates incoming requests against the provided schema
2. Returns formatted error messages if validation fails
3. Passes control to the next middleware if validation succeeds

### Error Response Format

When validation fails, the API returns:

```json
{
  "success": false,
  "message": "Validation failed",
  "errors": [
    {
      "field": "body.email",
      "message": "Please enter a valid email address"
    },
    {
      "field": "body.password",
      "message": "Password must be at least 6 characters"
    }
  ]
}
```

## Benefits of Zod

1. **Type Safety**: Schemas are type-safe and provide IntelliSense support
2. **Clear Errors**: Validation errors are descriptive and user-friendly
3. **Maintainability**: Schemas are centralized and easy to update
4. **Reusability**: Schemas can be composed and reused
5. **Performance**: Zod is lightweight and fast

## Adding New Schemas

To add a new validation schema:

1. Create a new file in the `schemas` folder (e.g., `resource.schema.js`)
2. Define your schema using Zod:

```javascript
const { z } = require("zod");

const resourceCreateSchema = z.object({
  body: z.object({
    name: z.string().min(2),
    // ... other fields
  }),
});

module.exports = { resourceCreateSchema };
```

3. Import and use in your routes:

```javascript
const { validate } = require("../middleware/validate");
const { resourceCreateSchema } = require("../schemas/resource.schema");

router.post("/", validate(resourceCreateSchema), handler);
```

## Migration from Old Validation

The old validation middleware (`middleware/validation.js`) has been replaced with Zod schemas. All routes have been updated to use the new validation system.

### Key Changes:

- ✅ Removed manual validation checks in routes
- ✅ Centralized validation logic in schema files
- ✅ Improved error messages
- ✅ Added type safety
- ✅ Consistent validation across all endpoints
