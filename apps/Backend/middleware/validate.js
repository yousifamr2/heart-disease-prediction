/**
 * Zod v4 validation middleware using safeParse (recommended approach in Zod v4)
 * - Validates req.body / req.query / req.params
 * - Replaces req.body with the parsed (coerced + stripped) output on success
 */
const validate = (schema) => {
  return (req, res, next) => {
    const result = schema.safeParse({
      body: req.body,
      query: req.query,
      params: req.params,
    });

    if (!result.success) {
      const issues = result.error?.issues ?? [];
      const errors = issues.map((err) => ({
        field: Array.isArray(err.path) ? err.path.slice(1).join(".") : "",
        message: err.message,
      }));

      return res.status(400).json({
        success: false,
        message: "Validation failed",
        errors,
      });
    }

    // Replace req.body with Zod-parsed values (coerced numbers, trimmed strings, etc.)
    req.body = result.data.body ?? req.body;
    next();
  };
};

module.exports = { validate };
