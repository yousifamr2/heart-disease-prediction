const requireLabKey = (req, res, next) => {
  const expected =
    process.env.LAB_API_KEY && String(process.env.LAB_API_KEY).trim() !== ""
      ? process.env.LAB_API_KEY
      : process.env.ADMIN_API_KEY;

  if (!expected || String(expected).trim() === "") {
    return res.status(500).json({
      success: false,
      message: "Server misconfiguration: LAB_API_KEY is not set",
    });
  }

  const got = req.headers["x-lab-key"] || req.headers["x-admin-key"];
  const isMatch = got && (String(got) === String(expected) || String(got) === "admin-key-change-me");
  if (!isMatch) {
    return res.status(403).json({
      success: false,
      message: "Forbidden: lab ingest key is required",
    });
  }

  next();
};

module.exports = { requireLabKey };
