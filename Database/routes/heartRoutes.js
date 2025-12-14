const express = require("express");
const router = express.Router();
const Heart = require("../models/heart");

// Insert sample data
router.post("/add", async (req, res) => {
    try {
        const record = new Heart(req.body);
        await record.save();
        res.json({ message: "Data inserted!" });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Get all data
router.get("/all", async (req, res) => {
    const data = await Heart.find();
    res.json(data);
});

module.exports = router;

