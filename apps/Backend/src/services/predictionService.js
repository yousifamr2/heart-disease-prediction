const prisma = require("../config/prisma");
const {
  internalPredict,
  internalShapPng,
  internalShapData,
  internalReportPdf,
} = require("../integrations/ai.service");

class PredictionService {
  /**
   * Legacy rows may lack user_id; allow access only if lab test national_id matches.
   */
  static async assertPredictionOwnedByUser(predictionId, user) {
    const prediction = await prisma.prediction.findUnique({
      where: { id: predictionId },
      include: { labTest: true },
    });
    if (!prediction) {
      const err = new Error("Prediction not found");
      err.statusCode = 404;
      throw err;
    }
    if (prediction.user_id) {
      if (prediction.user_id !== user.id) {
        const err = new Error("Forbidden");
        err.statusCode = 403;
        throw err;
      }
      return prediction;
    }
    if (!prediction.labTest || prediction.labTest.national_id !== user.national_id) {
      const err = new Error("Forbidden");
      err.statusCode = 403;
      throw err;
    }
    return prediction;
  }

  static async startForCurrentUser(user) {
    const labTest = await prisma.labTest.findFirst({
      where: { national_id: user.national_id },
      orderBy: { createdAt: "desc" },
    });
    if (!labTest) {
      const err = new Error(
        "No lab test found for this user. Upload results via your lab or contact support."
      );
      err.statusCode = 404;
      throw err;
    }

    // Check if prediction already exists for this lab test
    let existingPrediction = await prisma.prediction.findUnique({
      where: { lab_test_id: labTest.id }
    });

    let ai;
    if (existingPrediction) {
      // Use existing prediction
      const isHigh = String(existingPrediction.decision || "").toLowerCase() === "high";
      ai = {
        id: existingPrediction.id,
        lab_test_id: existingPrediction.lab_test_id,
        decision: existingPrediction.decision,
        probability: existingPrediction.prediction_percentage,
        risk_level: existingPrediction.risk_level,
        risk_color: isHigh ? "#ef4444" : "#22c55e",
        decision_label: isHigh ? "High Heart Disease Risk Detected" : "Low Heart Disease Risk",
      };
      
      // Ensure user_id is set
      if (existingPrediction.user_id !== user.id) {
        await prisma.prediction.update({
          where: { id: existingPrediction.id },
          data: { user_id: user.id }
        });
      }
    } else {
      // Call AI service
      ai = await internalPredict(labTest.id, user.id);

      await prisma.prediction.updateMany({
        where: { lab_test_id: labTest.id },
        data: { user_id: user.id },
      });
    }

    const isHigh = String(ai.decision || "").toLowerCase() === "high";
    const probability =
      typeof ai.probability === "number" ? ai.probability : Number(ai.probability) || 0;

    return {
      prediction_id: ai.id,
      lab_test_id: ai.lab_test_id,
      decision: ai.decision,
      probability,
      risk_level: ai.risk_level,
      risk_color: ai.risk_color,
      decision_label: ai.decision_label,
      show_shap: isHigh,
      show_report: isHigh,
      show_hospitals: isHigh,
    };
  }

  static async shapPngForPrediction(predictionId, user) {
    const prediction = await this.assertPredictionOwnedByUser(predictionId, user);
    if (prediction.decision === "low") {
      const err = new Error("SHAP image is not available for low risk predictions.");
      err.statusCode = 400;
      throw err;
    }

    // ── Try database first ────────────────────────────────────────────
    if (prediction.shap_image) {
      return Buffer.from(prediction.shap_image);
    }

    // ── Fallback: AI service ──────────────────────────────────────────
    try {
      return await internalShapPng(prediction.lab_test_id);
    } catch (aiErr) {
      const err = new Error("SHAP image is not available yet. Please run Start Prediction again.");
      err.statusCode = 503;
      throw err;
    }
  }

  static async shapDataForPrediction(predictionId, user) {
    const prediction = await this.assertPredictionOwnedByUser(predictionId, user);
    if (prediction.decision === "low") {
      const err = new Error("SHAP data is not available for low risk predictions.");
      err.statusCode = 400;
      throw err;
    }

    try {
      return await internalShapData(prediction.lab_test_id);
    } catch (aiErr) {
      if (prediction.shap_values_json) {
        const shap_data = prediction.shap_values_json;
        const sorted_features = Object.entries(shap_data)
          .map(([feature, val]) => ({
            feature,
            impact: Math.abs(Number(val) || 0)
          }))
          .sort((a, b) => b.impact - a.impact);

        const labels = sorted_features.map(f => f.feature);
        const values = sorted_features.map(f => f.impact);

        return {
          prediction_probability: prediction.prediction_percentage,
          risk_level: prediction.risk_level,
          top_features: sorted_features.map(f => ({
            feature: f.feature,
            value: "N/A",
            impact: f.impact,
            direction: "increase"
          })),
          chart_data: { labels, values },
          explanation: "Feature importance calculated from cached prediction values."
        };
      }
      const err = new Error("SHAP data is not available yet. Please run Start Prediction again.");
      err.statusCode = 503;
      throw err;
    }
  }

  static async reportPdfForPrediction(predictionId, user) {
    const prediction = await this.assertPredictionOwnedByUser(predictionId, user);
    if (prediction.decision === "low") {
      const err = new Error("Report PDF is not available for low risk predictions.");
      err.statusCode = 400;
      throw err;
    }

    // ── Try database first (pdf_binary stored during prediction) ──────
    if (prediction.pdf_binary) {
      return Buffer.from(prediction.pdf_binary);
    }

    // ── Fallback: ask AI service to generate it (only works when AI is reachable) ──
    try {
      return await internalReportPdf(prediction.lab_test_id);
    } catch (aiErr) {
      const err = new Error(
        "PDF report is not available yet. Please run Start Prediction again to regenerate it."
      );
      err.statusCode = 503;
      throw err;
    }
  }
}

module.exports = PredictionService;
