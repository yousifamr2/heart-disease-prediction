// Pagination middleware

const paginate = (model) => {
  return async (req, res, next) => {
    try {
      const page = parseInt(req.query.page) || 1;
      const limit = parseInt(req.query.limit) || 10;
      const skip = (page - 1) * limit;

      // Get total count
      const total = await model.countDocuments();

      // Apply pagination to query
      req.pagination = {
        page,
        limit,
        skip,
        total,
        totalPages: Math.ceil(total / limit)
      };

      next();
    } catch (err) {
      next(err);
    }
  };
};

module.exports = { paginate };
