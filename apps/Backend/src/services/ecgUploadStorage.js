const crypto = require("crypto");
const fs = require("fs").promises;
const path = require("path");

function uploadRoot() {
  const fromEnv = process.env.ECG_UPLOAD_DIR;
  if (fromEnv && String(fromEnv).trim()) {
    return path.resolve(String(fromEnv).trim());
  }
  return path.join(__dirname, "..", "..", "uploads", "ecg");
}

function sha256Hex(buf) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

/**
 * Persist WFDB pair under {root}/{ecgTestId}/record.dat|.hea
 * @returns {{ relativeDat: string, relativeHea: string, checksum_dat: string, checksum_hea: string, file_size_bytes: number }}
 */
async function saveWfdbPair(ecgTestId, datBuffer, heaBuffer) {
  const root = uploadRoot();
  const dir = path.join(root, ecgTestId);
  await fs.mkdir(dir, { recursive: true });
  const datPath = path.join(dir, "record.dat");
  const heaPath = path.join(dir, "record.hea");
  await fs.writeFile(datPath, datBuffer);
  await fs.writeFile(heaPath, heaBuffer);
  const relDat = path.posix.join(ecgTestId, "record.dat");
  const relHea = path.posix.join(ecgTestId, "record.hea");
  return {
    relativeDat: relDat,
    relativeHea: relHea,
    checksum_dat: sha256Hex(datBuffer),
    checksum_hea: sha256Hex(heaBuffer),
    file_size_bytes: datBuffer.length + heaBuffer.length,
  };
}

function absolutePathForStored(relativePosixPath) {
  if (!relativePosixPath) return null;
  const root = uploadRoot();
  const normalized = String(relativePosixPath).replace(/\\/g, path.sep);
  const full = path.join(root, normalized);
  const resolvedRoot = path.resolve(root);
  const resolvedFull = path.resolve(full);
  if (!resolvedFull.startsWith(resolvedRoot + path.sep) && resolvedFull !== resolvedRoot) {
    throw new Error("Invalid stored path");
  }
  return resolvedFull;
}

async function readWfdbPair(relativeDat, relativeHea) {
  const datAbs = absolutePathForStored(relativeDat);
  const heaAbs = absolutePathForStored(relativeHea);
  if (!datAbs || !heaAbs) {
    const err = new Error("ECG files are not available on disk.");
    err.statusCode = 404;
    throw err;
  }
  try {
    const [datBuffer, heaBuffer] = await Promise.all([fs.readFile(datAbs), fs.readFile(heaAbs)]);
    return { datBuffer, heaBuffer };
  } catch (e) {
    if (e.code === "ENOENT") {
      const err = new Error("ECG recording files are missing on local disk. Please upload the ECG again.");
      err.statusCode = 400;
      throw err;
    }
    throw e;
  }
}


module.exports = {
  uploadRoot,
  saveWfdbPair,
  absolutePathForStored,
  readWfdbPair,
  sha256Hex,
};
