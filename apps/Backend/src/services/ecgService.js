const prisma = require("../config/prisma");
const { saveWfdbPair, readWfdbPair } = require("./ecgUploadStorage");
const {
  internalEcgPipeline,
  internalEcgChartFromTop5,
  internalEcgReportPdf,
} = require("../integrations/ai.service");

function shapeEcgPublic(row) {
  if (!row) return null;
  const top5 = row.detailed_results_json?.top_5 ?? row.detailed_results_json ?? null;
  return {
    ecg_test_id: row.id,
    lab_id: row.lab_id,
    national_id: row.national_id,
    createdAt: row.createdAt,
    inference_status: row.inference_status,
    primary_diagnosis: row.primary_diagnosis,
    primary_probability: row.primary_probability,
    top_5: Array.isArray(top5) ? top5 : null,
    llm_ecg_json: row.llm_ecg_json,
    model_name: row.model_name,
    model_version: row.model_version,
  };
}

class EcgService {
  /**
   * Lab staff upload — creates row, writes disk, never runs inference.
   */
  static async createLabPortalUpload({ expectedLabId, national_id, datFile, heaFile, client_request_id }) {
    if (!expectedLabId || String(expectedLabId).trim() === "") {
      const err = new Error("x-lab-id header is required");
      err.statusCode = 400;
      throw err;
    }
    const nid = String(national_id || "").trim();
    if (!nid) {
      const err = new Error("national_id is required");
      err.statusCode = 400;
      throw err;
    }
    if (!datFile?.buffer?.length) {
      const err = new Error("Missing or empty dat_file upload");
      err.statusCode = 400;
      throw err;
    }
    if (!heaFile?.buffer?.length) {
      const err = new Error("Missing or empty hea_file upload");
      err.statusCode = 400;
      throw err;
    }

    const lab = await prisma.lab.findUnique({ where: { id: String(expectedLabId).trim() } });
    if (!lab) {
      const err = new Error("Lab not found for x-lab-id");
      err.statusCode = 404;
      throw err;
    }

    const patient = await prisma.user.findUnique({ where: { national_id: nid } });
    if (!patient) {
      const err = new Error(
        "No registered patient found for this national_id. Ask the patient to register before ECG upload."
      );
      err.statusCode = 400;
      throw err;
    }

    const row = await prisma.ecgTest.create({
      data: {
        lab_id: lab.id,
        national_id: nid,
        user_id: patient.id,
        inference_status: "pending",
        uploaded_by: "lab_portal",
        client_request_id: client_request_id ? String(client_request_id).slice(0, 200) : null,
      },
    });

    try {
      const saved = await saveWfdbPair(row.id, datFile.buffer, heaFile.buffer);

      // Run ECG AI pipeline immediately in memory using uploaded buffers
      let aiResult = null;
      let aiError = null;
      let inferenceStatus = "pending";
      try {
        aiResult = await internalEcgPipeline({
          ecgTestId: row.id,
          datBuffer: datFile.buffer,
          heaBuffer: heaFile.buffer,
        });
        inferenceStatus = "ok";
      } catch (err) {
        console.error("Immediate ECG inference failed on upload:", err);
        aiError = String(err.message || err).slice(0, 2000);
        inferenceStatus = "failed";
      }

      let updateData = {
        dat_file_path: saved.relativeDat,
        hea_file_path: saved.relativeHea,
        original_dat_name: datFile.originalname || null,
        original_hea_name: heaFile.originalname || null,
        file_size_bytes: saved.file_size_bytes,
        checksum_dat: saved.checksum_dat,
        checksum_hea: saved.checksum_hea,
      };

      if (inferenceStatus === "ok" && aiResult) {
        const primary = Array.isArray(aiResult.top_5) && aiResult.top_5[0] ? aiResult.top_5[0] : null;
        const primaryLabel = primary?.label ?? null;
        const primaryProb = primary != null ? Number(primary.probability) : null;
        const detailedPayload = {
          type: "ecg_inference",
          top_5: aiResult.top_5,
          primary_code: primary?.code ?? null,
          primary_label: primaryLabel,
        };

        updateData = {
          ...updateData,
          primary_diagnosis: primaryLabel,
          primary_probability: primaryProb,
          detailed_results_json: detailedPayload,
          llm_ecg_json: aiResult.llm_ecg_json ?? null,
          model_name: aiResult.model_name ?? null,
          model_version: aiResult.model_version ?? null,
          llm_model: aiResult.llm_model ?? null,
          llm_prompt_version: aiResult.llm_prompt_version ?? null,
          inference_status: "ok",
          inference_error: null,
          inferred_at: new Date(),
          prediction_completed_at: new Date(),
        };
      } else {
        updateData = {
          ...updateData,
          inference_status: inferenceStatus,
          inference_error: aiError,
        };
      }

      const updated = await prisma.ecgTest.update({
        where: { id: row.id },
        data: updateData,
        include: { lab: { select: { id: true, name: true, lab_code: true, address: true } } },
      });
      return shapeEcgPublic(updated);
    } catch (e) {
      await prisma.ecgTest.update({
        where: { id: row.id },
        data: {
          inference_status: "failed",
          inference_error: String(e.message || e).slice(0, 2000),
        },
      });
      throw e;
    }
  }

  static async assertEcgOwnedByUser(ecgTestId, user) {
    const row = await prisma.ecgTest.findUnique({
      where: { id: String(ecgTestId).trim() },
      include: { lab: { select: { id: true, name: true, lab_code: true, address: true } } },
    });
    if (!row) {
      const err = new Error("ECG test not found");
      err.statusCode = 404;
      throw err;
    }
    if (row.national_id !== user.national_id) {
      const err = new Error("Forbidden");
      err.statusCode = 403;
      throw err;
    }
    return row;
  }

  static async getMyStatus(user) {
    const latest = await prisma.ecgTest.findFirst({
      where: {
        national_id: user.national_id,
        inference_status: { not: "failed" },
      },
      orderBy: { createdAt: "desc" },
      select: {
        id: true,
        createdAt: true,
        inference_status: true,
        primary_diagnosis: true,
        primary_probability: true,
      },
    });
    return {
      hasEcgTests: !!latest,
      latestEcgTestId: latest?.id ?? null,
      latestInferenceStatus: latest?.inference_status ?? null,
      latestSummary: latest
        ? {
            ecg_test_id: latest.id,
            createdAt: latest.createdAt,
            primary_diagnosis: latest.primary_diagnosis,
            primary_probability: latest.primary_probability,
            inference_status: latest.inference_status,
          }
        : null,
    };
  }

  static async listForCurrentUser(user, { page = 1, limit = 10 } = {}) {
    const p = Math.max(1, parseInt(page, 10) || 1);
    const l = Math.min(50, Math.max(1, parseInt(limit, 10) || 10));
    const skip = (p - 1) * l;
    const where = { national_id: user.national_id };
    const [total, rows] = await Promise.all([
      prisma.ecgTest.count({ where }),
      prisma.ecgTest.findMany({
        where,
        orderBy: { createdAt: "desc" },
        skip,
        take: l,
        include: { lab: { select: { name: true, lab_code: true } } },
      }),
    ]);
    return {
      data: rows.map((r) => ({
        id: r.id,
        createdAt: r.createdAt,
        inference_status: r.inference_status,
        primary_diagnosis: r.primary_diagnosis,
        primary_probability: r.primary_probability,
        lab: r.lab,
      })),
      pagination: { page: p, limit: l, total, totalPages: Math.ceil(total / l) || 1 },
    };
  }

  static async getDetailForUser(ecgTestId, user) {
    const row = await this.assertEcgOwnedByUser(ecgTestId, user);
    return shapeEcgPublic(row);
  }

  /** Latest ECG for patient — cache hit skips AI, falls back to next valid test if corrupted. */
  static async startForCurrentUser(user) {
    const tests = await prisma.ecgTest.findMany({
      where: { national_id: user.national_id },
      orderBy: { createdAt: "desc" },
      include: { lab: { select: { id: true, name: true, lab_code: true, address: true } } },
    });

    if (!tests || tests.length === 0) {
      const err = new Error("No ECG data");
      err.statusCode = 404;
      err.code = "NO_ECG";
      throw err;
    }

    for (const test of tests) {
      // Skip previously failed tests
      if (test.inference_status === "failed") {
        continue;
      }

      // If already ok, return cached results
      if (test.inference_status === "ok" && test.detailed_results_json) {
        if (test.user_id !== user.id) {
          await prisma.ecgTest.update({
            where: { id: test.id },
            data: { user_id: user.id },
          });
        }
        const top5 = test.detailed_results_json?.top_5 ?? [];
        return {
          cached: true,
          ecg_test_id: test.id,
          primary_diagnosis: test.primary_diagnosis,
          primary_probability: test.primary_probability,
          top_5: top5,
          llm_ecg_json: test.llm_ecg_json,
          model_name: test.model_name,
          model_version: test.model_version,
          createdAt: test.createdAt,
        };
      }

      // If pending, try to read files and analyze
      if (!test.dat_file_path || !test.hea_file_path) {
        // Mark as failed since paths are invalid/missing
        await prisma.ecgTest.update({
          where: { id: test.id },
          data: {
            inference_status: "failed",
            inference_error: "ECG record is missing file paths in database."
          }
        });
        continue;
      }

      try {
        const { datBuffer, heaBuffer } = await readWfdbPair(test.dat_file_path, test.hea_file_path);

        const ai = await internalEcgPipeline({
          ecgTestId: test.id,
          datBuffer,
          heaBuffer,
        });

        const primary = Array.isArray(ai.top_5) && ai.top_5[0] ? ai.top_5[0] : null;
        const primaryLabel = primary?.label ?? null;
        const primaryProb = primary != null ? Number(primary.probability) : null;

        const detailedPayload = {
          type: "ecg_inference",
          top_5: ai.top_5,
          primary_code: primary?.code ?? null,
          primary_label: primaryLabel,
        };

        await prisma.ecgTest.update({
          where: { id: test.id },
          data: {
            user_id: user.id,
            primary_diagnosis: primaryLabel,
            primary_probability: primaryProb,
            detailed_results_json: detailedPayload,
            llm_ecg_json: ai.llm_ecg_json ?? null,
            model_name: ai.model_name ?? null,
            model_version: ai.model_version ?? null,
            llm_model: ai.llm_model ?? null,
            llm_prompt_version: ai.llm_prompt_version ?? null,
            inference_status: "ok",
            inference_error: null,
            inferred_at: new Date(),
            prediction_completed_at: new Date(),
          },
        });

        return {
          cached: false,
          ecg_test_id: test.id,
          primary_diagnosis: primaryLabel,
          primary_probability: primaryProb,
          top_5: ai.top_5,
          llm_ecg_json: ai.llm_ecg_json,
          model_name: ai.model_name,
          model_version: ai.model_version,
          createdAt: test.createdAt,
        };
      } catch (err) {
        if (err.statusCode === 400 || err.code === "ENOENT" || String(err.message).includes("missing on local disk")) {
          // Mark as failed in DB on-the-fly and warn
          await prisma.ecgTest.update({
            where: { id: test.id },
            data: {
              inference_status: "failed",
              inference_error: "ECG recording files were lost due to server restart."
            }
          });
          console.warn(`Skipped and marked corrupted ECG test ${test.id} as failed.`);
          continue;
        }
        throw err;
      }
    }

    // If loop finishes and all tests were skipped
    const err = new Error("All ECG recording files are missing on local disk. Please upload the ECG again.");
    err.statusCode = 400;
    throw err;
  }

  static async chartPngForUser(ecgTestId, user) {
    const row = await this.assertEcgOwnedByUser(ecgTestId, user);
    const top5 = row.detailed_results_json?.top_5;
    if (!Array.isArray(top5) || top5.length === 0) {
      const err = new Error("No ECG prediction chart available for this test yet. Run Start ECG first.");
      err.statusCode = 404;
      throw err;
    }
    return internalEcgChartFromTop5(top5, { compact: true });
  }

  static async reportPdfForUser(ecgTestId, user) {
    const row = await this.assertEcgOwnedByUser(ecgTestId, user);
    if (row.inference_status !== "ok" || !row.detailed_results_json) {
      const err = new Error("ECG report is not available until prediction completes.");
      err.statusCode = 400;
      throw err;
    }
    const top5 = row.detailed_results_json?.top_5 ?? [];
    const patient = await prisma.user.findUnique({
      where: { id: user.id },
      select: { username: true, national_id: true, email: true },
    });
    const payload = {
      ecg_test: {
        id: row.id,
        createdAt: row.createdAt?.toISOString?.() ?? String(row.createdAt),
      },
      patient: {
        name: patient?.username ?? "Patient",
        national_id: patient?.national_id ?? row.national_id,
        email: patient?.email ?? "",
      },
      lab: row.lab
        ? {
            name: row.lab.name,
            address: row.lab.address,
            lab_code: row.lab.lab_code,
          }
        : { name: "N/A", address: "N/A", lab_code: "N/A" },
      top_5: top5,
      llm_ecg_json: row.llm_ecg_json,
      primary_diagnosis: row.primary_diagnosis,
      primary_probability: row.primary_probability,
    };
    return internalEcgReportPdf(payload);
  }
}

module.exports = EcgService;
