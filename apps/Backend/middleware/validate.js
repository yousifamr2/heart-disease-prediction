const { ZodError } = require("zod");

/**
 * Zod validation middleware
 * - Validates req.body / req.query / req.params against the schema
 * - Replaces req.body with the parsed (coerced + stripped) output
 */
const validate = (schema) => {
  return (req, res, next) => {
    try {
      const parsed = schema.parse({
        body: req.body,
        query: req.query,
        params: req.params,
      });

      // استبدل req.body بالقيم المُنقّحة من Zod (بعد coerce + strip)
      req.body = parsed.body ?? req.body;

      next();
    } catch (error) {
      if (error instanceof ZodError) {
        const errors = error.errors.map((err) => ({
          field: err.path.slice(1).join("."), // إزالة "body." من أول الـ path
          message: err.message,
        }));

        return res.status(400).json({
          success: false,
          message: "Validation failed",
          errors,
        });
      }

      next(error);
    }
  };
};

module.exports = { validate };
