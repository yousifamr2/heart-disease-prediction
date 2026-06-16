import API_BASE_URL from "../config";
const BASE_URL = API_BASE_URL;

// Helper: Get token from localStorage
const getToken = () => localStorage.getItem("token");

// Helper: Headers for authenticated requests
const authHeaders = () => ({
  "Content-Type": "application/json",
  Authorization: `Bearer ${getToken()}`,
});

const handleJsonResponse = async (response) => {
  const data = await response.json().catch(() => null);
  if (response.status === 401) {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    return {
      success: false,
      status: 401,
      message: data?.message || "Unauthorized",
    };
  }
  return data;
};

// ==================== AUTH ====================
export const login = async (credentials) => {
  const res = await fetch(`${BASE_URL}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(credentials),
  });
  return res.json();
};

export const register = async (data) => {
  const res = await fetch(`${BASE_URL}/api/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return res.json();
};

// ==================== LAB TESTS ====================
export const getLabStatus = async () => {
  const res = await fetch(`${BASE_URL}/api/labtests/me/status`, {
    headers: authHeaders(),
  });
  return handleJsonResponse(res);
};

export const uploadCSV = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE_URL}/api/labtests/upload-csv`, {
    method: "POST",
    headers: { Authorization: `Bearer ${getToken()}` }, // No Content-Type for FormData
    body: formData,
  });
  return res.json();
};

// Get latest lab test for a patient
export const getLatestLabTest = async (nationalId) => {
  const res = await fetch(`${BASE_URL}/api/labtests/patient/${nationalId}/latest`, {
    headers: authHeaders(),
  });
  return handleJsonResponse(res);
};

// ==================== PREDICTIONS ====================
export const startPrediction = async () => {
  const res = await fetch(`${BASE_URL}/api/predictions/start`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({}), // Empty body as per docs
  });
  return handleJsonResponse(res);
};

// Get SHAP Image as Blob (للعرض في <img>)
export const getShapImage = async (predictionId) => {
  const res = await fetch(`${BASE_URL}/api/predictions/${predictionId}/shap`, {
    headers: { Authorization: `Bearer ${getToken()}` },
  });
  
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.message || "Failed to fetch SHAP image");
  }
  
  return res.blob(); // Return Blob for image display
};

// Get SHAP numeric data for interactive visualization
export const getShapData = async (predictionId) => {
  const res = await fetch(`${BASE_URL}/api/predictions/${predictionId}/shap/data`, {
    headers: authHeaders(),
  });
  
  if (!res.ok) {
    const error = await res.json().catch(() => ({}));
    throw new Error(error.message || "Failed to fetch SHAP data");
  }
  
  return handleJsonResponse(res);
};

// Get PDF Report as Blob (للـ Download)
export const getReportPDF = async (predictionId) => {
  const res = await fetch(`${BASE_URL}/api/predictions/${predictionId}/report`, {
    headers: { Authorization: `Bearer ${getToken()}` },
  });
  
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.message || "Failed to fetch report");
  }
  
  return res.blob(); // Return Blob for download
};

// ==================== HOSPITALS ====================
export const getHospitals = async (page = 1, limit = 10) => {
  const res = await fetch(
    `${BASE_URL}/api/hospitals?page=${page}&limit=${limit}`
  );
  return res.json();
};

export const getHospitalsByArea = async (area) => {
  const res = await fetch(
    `${BASE_URL}/api/hospitals/area/${encodeURIComponent(area)}`
  );
  return res.json();
};
