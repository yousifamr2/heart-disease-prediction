const PredictionService = require("../services/predictionService");

const startPrediction = async (req, res, next) => {
  try {
    const data = await PredictionService.startForCurrentUser(req.user);
    return res.status(201).json({
      success: true,
      message: "Prediction completed successfully",
      data,
    });
  } catch (e) {
    next(e);
  }
};

const getShap = async (req, res, next) => {
  try {
    const png = await PredictionService.shapPngForPrediction(req.params.id, req.user);
    res.setHeader("Content-Type", "image/png");
    return res.send(png);
  } catch (e) {
    next(e);
  }
};

const getShapData = async (req, res, next) => {
  try {
    const data = await PredictionService.shapDataForPrediction(req.params.id, req.user);
    return res.status(200).json({
      success: true,
      data,
    });
  } catch (e) {
    next(e);
  }
};

const getReport = async (req, res, next) => {
  try {
    const pdf = await PredictionService.reportPdfForPrediction(req.params.id, req.user);
    res.setHeader("Content-Type", "application/pdf");
    res.setHeader(
      "Content-Disposition",
      `attachment; filename=prediction_report_${req.params.id}.pdf`
    );
    return res.send(pdf);
  } catch (e) {
    next(e);
  }
};

module.exports = { startPrediction, getShap, getReport, getShapData };
