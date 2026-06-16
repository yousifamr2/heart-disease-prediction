/**
 * Internal AI service client (FastAPI).
 * Injects X-INTERNAL-API-KEY on every request — never call from browser code.
 */

const axios = require("axios");
const { logger } = require("../utils/logger");

const baseURL = process.env.AI_SERVICE_URL || "http://127.0.0.1:8000";
const internalKey = process.env.INTERNAL_API_KEY || "";

const aiClient = axios.create({
  baseURL,
  timeout: Number(process.env.AI_REQUEST_TIMEOUT_MS) || 120000,
  headers: {
    "Content-Type": "application/json",
  },
  validateStatus: () => true,
});

aiClient.interceptors.request.use((config) => {
  if (!internalKey) {
    throw new Error("INTERNAL_API_KEY is not set on the Node gateway");
  }
  config.headers["X-INTERNAL-API-KEY"] = internalKey;
  config.metadata = { start: Date.now() };
  return config;
});

aiClient.interceptors.response.use(
  (response) => {
    const ms = Date.now() - (response.config.metadata?.start || Date.now());
    logger.info("ai_service_response", {
      event: "ai_service_response",
      method: response.config.method,
      path: response.config.url,
      status: response.status,
      duration_ms: ms,
    });
    return response;
  },
  (error) => {
    const cfg = error.config || {};
    const ms = Date.now() - (cfg.metadata?.start || Date.now());
    logger.error("ai_service_error", {
      event: "ai_service_error",
      method: cfg.method,
      path: cfg.url,
      message: error.message,
      duration_ms: ms,
    });
    return Promise.reject(error);
  }
);

function parseErrorPayload(data) {
  if (data == null) return null;
  if (Buffer.isBuffer(data)) {
    try {
      const j = JSON.parse(data.toString("utf8"));
      return j.message || j.detail || j.error || null;
    } catch {
      return data.toString("utf8").slice(0, 300);
    }
  }
  if (typeof data === "object") {
    if (Array.isArray(data.detail)) {
      return data.detail
        .map((d) => (typeof d === "string" ? d : d.msg || d.loc?.join(".") || JSON.stringify(d)))
        .join("; ");
    }
    return data.message || data.detail || data.error || null;
  }
  return String(data);
}

function assertOk(response, context) {
  if (response.status === 401) {
    const err = new Error("Unauthorized");
    err.statusCode = 502;
    throw err;
  }
  if (response.status >= 400) {
    const detail = parseErrorPayload(response.data) || response.statusText;
    const err = new Error(`${context} failed: ${detail}`);
    err.statusCode = response.status >= 500 ? 502 : response.status;
    throw err;
  }
}

async function internalPredict(labTestId, userId) {
  const res = await aiClient.post("/internal/predict", {
    target_id: labTestId,
    user_id: userId,
  });
  assertOk(res, "internal predict");
  return res.data;
}

async function internalShapPng(labTestId) {
  const res = await aiClient.post(
    "/internal/shap",
    { target_id: labTestId, user_id: null },
    { responseType: "arraybuffer" }
  );
  assertOk(res, "internal shap");
  return Buffer.from(res.data);
}

async function internalShapData(labTestId) {
  const res = await aiClient.post(
    "/internal/shap/data",
    { target_id: labTestId, user_id: null }
  );
  assertOk(res, "internal shap data");
  return res.data;
}

async function internalReportPdf(labTestId) {
  const res = await aiClient.post(
    "/internal/report",
    { target_id: labTestId, user_id: null },
    { responseType: "arraybuffer" }
  );
  assertOk(res, "internal report");
  return Buffer.from(res.data);
}

/**
 * PTB-XL-style ECG inference + LLM (multipart .dat + .hea → FastAPI /internal/ecg/pipeline).
 */
async function internalEcgPipeline({ ecgTestId, datBuffer, heaBuffer }) {
  if (!Buffer.isBuffer(datBuffer)) datBuffer = Buffer.from(datBuffer);
  if (!Buffer.isBuffer(heaBuffer)) heaBuffer = Buffer.from(heaBuffer);
  if (!ecgTestId || String(ecgTestId).trim() === "") {
    throw new Error("ecg_test_id is required for ECG pipeline");
  }

  const fd = new FormData();
  fd.append("ecg_test_id", String(ecgTestId).trim());
  fd.append("dat_file", new Blob([datBuffer], { type: "application/octet-stream" }), "record.dat");
  fd.append("hea_file", new Blob([heaBuffer], { type: "text/plain" }), "record.hea");

  const res = await axios.post(`${baseURL}/internal/ecg/pipeline`, fd, {
    headers: {
      "X-INTERNAL-API-KEY": internalKey,
    },
    timeout: Number(process.env.AI_REQUEST_TIMEOUT_MS) || 120000,
    validateStatus: () => true,
  });
  assertOk(res, "internal ecg pipeline");
  return res.data;
}

/** PNG bar chart from top_5 JSON (FastAPI POST /internal/ecg/chart). */
async function internalEcgChartFromTop5(top5, opts = {}) {
  const res = await aiClient.post(
    "/internal/ecg/chart",
    { top_5: top5, compact: !!opts.compact },
    {
      responseType: "arraybuffer",
      timeout: Number(process.env.AI_REQUEST_TIMEOUT_MS) || 120000,
      validateStatus: () => true,
    }
  );
  assertOk(res, "internal ecg chart");
  return Buffer.from(res.data);
}

/** ECG medical PDF (FastAPI POST /internal/ecg/report). */
async function internalEcgReportPdf(payload) {
  const res = await aiClient.post("/internal/ecg/report", payload, {
    responseType: "arraybuffer",
    timeout: Number(process.env.AI_REQUEST_TIMEOUT_MS) || 120000,
    validateStatus: () => true,
  });
  assertOk(res, "internal ecg report");
  return Buffer.from(res.data);
}

module.exports = {
  aiClient,
  internalPredict,
  internalShapPng,
  internalShapData,
  internalReportPdf,
  internalEcgPipeline,
  internalEcgChartFromTop5,
  internalEcgReportPdf,
};
