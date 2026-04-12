/**
 * Handles common Prisma error codes and returns user-friendly responses.
 * Usage: call at the top of each catch block before next(err).
 * Returns true if the error was handled (response sent), false otherwise.
 */
const handlePrismaError = (err, res) => {
  if (!err.code) return false;

  switch (err.code) {
    // Unique constraint violation (e.g. duplicate email, national_id, lab_code)
    case "P2002": {
      const fields = err.meta?.target ?? ["field"];
      return res.status(400).json({
        success: false,
        message: `A record with this ${fields.join(", ")} already exists`,
      });
    }

    // Record not found (update/delete on non-existent id)
    case "P2025":
      return res.status(404).json({
        success: false,
        message: err.meta?.cause ?? "Record not found",
      });

    // Foreign key constraint failed (e.g. invalid lab_id)
    case "P2003":
      return res.status(400).json({
        success: false,
        message: `Invalid reference: the related record does not exist (field: ${err.meta?.field_name ?? "unknown"})`,
      });

    default:
      return false;
  }
};

module.exports = { handlePrismaError };
