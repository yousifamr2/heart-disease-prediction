import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import {
  FaArrowLeft,
  FaCamera,
  FaEdit,
  FaSave,
  FaTimes,
  FaEye,
  FaEyeSlash,
  FaDownload,
  FaSearch,
  FaFlask,
  FaHeartbeat,
  FaExclamationTriangle,
  FaChartLine,
  FaHospital,
} from "react-icons/fa";

import "./PatientDashboard.css";
import defaultProfile from "../../Image/prof.png";
import API_BASE_URL from "../../config";

const API = `${API_BASE_URL}/api`;
const PAGE_SIZE = 5;
const CONTACT_KEY = "patientProfile_contact";

function pctFromRow(row) {
  const p = row?.prediction?.prediction_percentage;
  if (p == null || Number.isNaN(Number(p))) return null;
  return Math.round(Number(p) * 10) / 10;
}

function riskTierFromPct(p) {
  if (p == null) return "none";
  if (p >= 50) return "high";
  return "low";
}

/** Risk tier for UI when percentage missing but decision exists */
function tierFromRow(row) {
  const p = pctFromRow(row);
  if (p != null) return riskTierFromPct(p);
  const d = String(row?.prediction?.decision || "").toLowerCase();
  if (d === "high") return "high";
  if (d === "low") return "low";
  return "none";
}

function riskLabel(tier) {
  if (tier === "high") return "High Risk";
  if (tier === "low") return "Low Risk";
  return "No prediction";
}

function barColor(tier) {
  if (tier === "high") return "var(--pd-risk-high)";
  if (tier === "low") return "var(--pd-risk-low)";
  return "#cbd5e1";
}

function Toast({ toasts, remove }) {
  return (
    <div className="pd-toast-wrap" aria-live="polite">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`pd-toast ${t.type === "error" ? "pd-toast-error" : "pd-toast-success"}`}
          role="status"
        >
          {t.message}
          <button
            type="button"
            onClick={() => remove(t.id)}
            style={{
              marginLeft: 12,
              border: "none",
              background: "transparent",
              cursor: "pointer",
              fontWeight: 700,
            }}
            aria-label="Dismiss"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}

function StatCard({ label, value, sub, icon }) {
  return (
    <div className="pd-stat-card">
      <div className="label">{label}</div>
      <div className="value" style={{ display: "flex", alignItems: "center", gap: 10 }}>
        {icon}
        {value}
      </div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Interactive Prediction Trend Chart                                          */
/* -------------------------------------------------------------------------- */
function PredictionTrendChart({ points, onPointClick }) {
  const svgRef = useRef(null);
  const [hoverIdx, setHoverIdx] = useState(null);
  const [animateIn, setAnimateIn] = useState(false);

  useEffect(() => {
    const id = window.requestAnimationFrame(() => setAnimateIn(true));
    return () => window.cancelAnimationFrame(id);
  }, [points]);

  const W = 800;
  const H = 260;
  const PAD = { l: 48, r: 28, t: 28, b: 48 };
  const innerW = W - PAD.l - PAD.r;
  const innerH = H - PAD.t - PAD.b;

  const xAt = (i) =>
    points.length <= 1 ? PAD.l + innerW / 2 : PAD.l + (i / (points.length - 1)) * innerW;
  const yAt = (pct) => PAD.t + innerH - (Math.max(0, Math.min(100, pct)) / 100) * innerH;

  const dotColor = (pct) => {
    if (pct >= 50) return "#ef4444";
    return "#22c55e";
  };

  const linePath = useMemo(() => {
    if (!points.length) return "";
    return points
      .map((p, i) => `${i === 0 ? "M" : "L"} ${xAt(i).toFixed(2)} ${yAt(p.pct).toFixed(2)}`)
      .join(" ");
  }, [points]); // eslint-disable-line react-hooks/exhaustive-deps

  const areaPath = useMemo(() => {
    if (!points.length) return "";
    const top = points
      .map((p, i) => `${i === 0 ? "M" : "L"} ${xAt(i).toFixed(2)} ${yAt(p.pct).toFixed(2)}`)
      .join(" ");
    const last = xAt(points.length - 1).toFixed(2);
    const first = xAt(0).toFixed(2);
    const bottom = (PAD.t + innerH).toFixed(2);
    return `${top} L ${last} ${bottom} L ${first} ${bottom} Z`;
  }, [points]); // eslint-disable-line react-hooks/exhaustive-deps

  const y0 = yAt(0);
  const y50 = yAt(50);
  const y100 = yAt(100);

  const handleMouseMove = (e) => {
    if (!svgRef.current || !points.length) return;
    const rect = svgRef.current.getBoundingClientRect();
    const ratio = W / rect.width;
    const xInSvg = (e.clientX - rect.left) * ratio;
    if (points.length === 1) {
      setHoverIdx(0);
      return;
    }
    let best = 0;
    let bestDist = Infinity;
    for (let i = 0; i < points.length; i += 1) {
      const d = Math.abs(xAt(i) - xInSvg);
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    }
    setHoverIdx(best);
  };

  const handleMouseLeave = () => setHoverIdx(null);

  const active = hoverIdx != null ? points[hoverIdx] : null;
  const activeX = active ? xAt(hoverIdx) : 0;
  const activeY = active ? yAt(active.pct) : 0;

  const tooltipW = 150;
  const tooltipH = 64;
  let tipX = activeX + 12;
  if (tipX + tooltipW > W - PAD.r) tipX = activeX - tooltipW - 12;
  let tipY = activeY - tooltipH - 12;
  if (tipY < PAD.t) tipY = activeY + 12;

  if (!points.length) {
    return (
      <div className="pd-empty">
        <strong>No chart data yet</strong>
        Complete a prediction to see your trend over time.
      </div>
    );
  }

  return (
    <div className="pd-chart-wrap">
      <svg
        ref={svgRef}
        className="pd-chart"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="Prediction trend over time"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        <defs>
          <linearGradient id="pdLineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#0f7f99" />
            <stop offset="100%" stopColor="#1198b7" />
          </linearGradient>
          <linearGradient id="pdAreaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#1198b7" stopOpacity="0.32" />
            <stop offset="100%" stopColor="#1198b7" stopOpacity="0.02" />
          </linearGradient>
          <filter id="pdDotShadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="2" stdDeviation="2" floodColor="#0f3d4c" floodOpacity="0.25" />
          </filter>
        </defs>

        <rect
          x={PAD.l}
          y={y100}
          width={innerW}
          height={y50 - y100}
          fill="rgba(239,68,68,0.10)"
        />
        <rect
          x={PAD.l}
          y={y50}
          width={innerW}
          height={y0 - y50}
          fill="rgba(34,197,94,0.12)"
        />

        {[0, 25, 50, 75, 100].map((v) => (
          <g key={v}>
            <line
              x1={PAD.l}
              x2={PAD.l + innerW}
              y1={yAt(v)}
              y2={yAt(v)}
              stroke="#e0eef3"
              strokeDasharray="3 4"
              strokeWidth="1"
            />
            <text x={PAD.l - 8} y={yAt(v) + 4} fontSize="11" fill="#64748b" textAnchor="end">
              {v}%
            </text>
          </g>
        ))}

        <text x={PAD.l + innerW - 4} y={y50 - 6} fontSize="10" fill="#ef4444" textAnchor="end">
          High risk
        </text>
        <text x={PAD.l + innerW - 4} y={y0 - 6} fontSize="10" fill="#22c55e" textAnchor="end">
          Low risk
        </text>

        <path
          d={areaPath}
          fill="url(#pdAreaGrad)"
          style={{
            opacity: animateIn ? 1 : 0,
            transition: "opacity 600ms ease 200ms",
          }}
        />

        <path
          className="pd-chart-line"
          d={linePath}
          fill="none"
          stroke="url(#pdLineGrad)"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ opacity: animateIn ? 1 : 0 }}
        />

        {hoverIdx != null && (
          <line
            x1={activeX}
            x2={activeX}
            y1={PAD.t}
            y2={PAD.t + innerH}
            stroke="#1198b7"
            strokeDasharray="3 3"
            strokeWidth="1"
            opacity="0.6"
          />
        )}

        {points.map((p, i) => {
          const cx = xAt(i);
          const cy = yAt(p.pct);
          const color = dotColor(p.pct);
          const isActive = hoverIdx === i;
          return (
            <g
              key={p.id}
              className="pd-chart-dot"
              style={{ cursor: onPointClick ? "pointer" : "default" }}
              onClick={() => onPointClick && onPointClick(p)}
            >
              {isActive && (
                <circle cx={cx} cy={cy} r="14" fill={color} opacity="0.18" />
              )}
              <circle
                cx={cx}
                cy={cy}
                r={isActive ? 7.5 : 5.5}
                fill="#fff"
                stroke={color}
                strokeWidth={isActive ? 3 : 2.5}
                filter="url(#pdDotShadow)"
              />
              <title>
                {p.date}: {p.pct}%
              </title>
            </g>
          );
        })}

        {points.map((p, i) => (
          <text
            key={`lbl-${p.id}`}
            x={xAt(i)}
            y={H - 16}
            fontSize="11"
            textAnchor="middle"
            fill="#64748b"
          >
            {p.shortDate}
          </text>
        ))}

        {active && (
          <g pointerEvents="none">
            <rect
              x={tipX}
              y={tipY}
              width={tooltipW}
              height={tooltipH}
              rx="10"
              ry="10"
              fill="#0f3d4c"
              opacity="0.96"
            />
            <text x={tipX + 12} y={tipY + 22} fontSize="12" fill="#9bd4e3" fontWeight="600">
              {active.date}
            </text>
            <text x={tipX + 12} y={tipY + 44} fontSize="18" fill="#fff" fontWeight="700">
              {active.pct}%
            </text>
            <circle
              cx={tipX + tooltipW - 18}
              cy={tipY + 38}
              r="7"
              fill={dotColor(active.pct)}
            />
          </g>
        )}
      </svg>
    </div>
  );
}

function RiskBadge({ tier }) {
  const cls =
    tier === "high"
      ? "pd-badge pd-badge-high"
      : tier === "medium"
        ? "pd-badge pd-badge-mid"
        : tier === "low"
          ? "pd-badge pd-badge-low"
          : "pd-badge pd-badge-none";
  return <span className={cls}>{riskLabel(tier)}</span>;
}

export default function PatientDashboard() {
  const navigate = useNavigate();

  const [user, setUser] = useState(null);
  const [labTests, setLabTests] = useState([]);
  const [loading, setLoading] = useState(true);
  const [profilePic, setProfilePic] = useState(() => localStorage.getItem("profilePic") || "");

  const [contact, setContact] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(CONTACT_KEY) || "{}");
    } catch {
      return {};
    }
  });

  const [isEditing, setIsEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [editForm, setEditForm] = useState({ username: "", email: "", password: "" });
  const [editErrors, setEditErrors] = useState({});

  const [search, setSearch] = useState("");
  const [riskFilter, setRiskFilter] = useState("all");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [sortKey, setSortKey] = useState("date");
  const [sortDir, setSortDir] = useState("desc");
  const [page, setPage] = useState(1);

  const [detailRow, setDetailRow] = useState(null);
  const [dashboardTab, setDashboardTab] = useState("lab");
  const [ecgTests, setEcgTests] = useState([]);
  const [ecgStatus, setEcgStatus] = useState(null);
  const [ecgDetailRow, setEcgDetailRow] = useState(null);
  const [hospitals, setHospitals] = useState([]);

  const [toasts, setToasts] = useState([]);
  const pushToast = useCallback((message, type = "success") => {
    const id = `${Date.now()}-${Math.random()}`;
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 5000);
  }, []);

  const removeToast = (id) => setToasts((prev) => prev.filter((t) => t.id !== id));

  useEffect(() => {
    let meta = document.querySelector('meta[name="viewport"]');
    if (!meta) {
      meta = document.createElement('meta');
      meta.name = 'viewport';
      document.head.appendChild(meta);
    }
    const originalContent = meta.getAttribute('content') || 'width=device-width, initial-scale=1';
    meta.setAttribute('content', 'width=1280');
    return () => {
      meta.setAttribute('content', originalContent);
    };
  }, []);

  useEffect(() => {
    const saved = localStorage.getItem("user");
    if (!saved) {
      navigate("/login");
      return;
    }
    const u = JSON.parse(saved);
    setUser(u);
    setEditForm({ username: u.username || "", email: u.email || "", password: "" });
    try {
      setContact(JSON.parse(localStorage.getItem(CONTACT_KEY) || "{}"));
    } catch {
      setContact({});
    }
  }, [navigate]);

  const fetchLabTests = useCallback(
    async (national_id) => {
      try {
        const res = await axios.get(`${API}/labtests/patient/${national_id}`);
        setLabTests(res.data.data || []);
      } catch {
        pushToast("Could not load lab tests.", "error");
      } finally {
        setLoading(false);
      }
    },
    [pushToast]
  );

  const fetchEcgDashboard = useCallback(async () => {
    const token = localStorage.getItem("token");
    if (!user?.national_id || !token) return;
    try {
      const [st, list] = await Promise.all([
        axios.get(`${API}/ecg/me/status`, { headers: { Authorization: `Bearer ${token}` } }),
        axios.get(`${API}/ecg/me`, { params: { page: 1, limit: 50 }, headers: { Authorization: `Bearer ${token}` } }),
      ]);
      setEcgStatus(st.data.data);
      setEcgTests(list.data.data || []);
    } catch {
      /* non-fatal */
    }
  }, [user?.national_id]);

  useEffect(() => {
    if (user?.national_id) fetchLabTests(user.national_id);
  }, [user, fetchLabTests]);

  useEffect(() => {
    if (user?.national_id) fetchEcgDashboard();
  }, [user, fetchEcgDashboard]);

  const latestFeatures = labTests[0]?.features;
  const ageFromLab = latestFeatures?.age != null ? String(latestFeatures.age) : "—";
  const genderFromLab =
    latestFeatures?.sex === 1 ? "Male" : latestFeatures?.sex === 0 ? "Female" : "—";

  const overallTier = useMemo(() => {
    const sorted = [...labTests].sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
    for (const row of sorted) {
      const t = tierFromRow(row);
      if (t !== "none") return t;
    }
    try {
      const pred = JSON.parse(localStorage.getItem("prediction") || "null");
      if (pred?.probability != null) return riskTierFromPct(Number(pred.probability));
    } catch {
      /* ignore */
    }
    return "none";
  }, [labTests]);

  const lastPredictionAt = useMemo(() => {
    let max = null;
    for (const row of labTests) {
      const d = row?.prediction?.createdAt || row?.prediction?.updatedAt;
      if (d) {
        const t = new Date(d).getTime();
        if (!max || t > max) max = t;
      }
    }
    if (max) return new Date(max).toLocaleString();
    try {
      const pred = JSON.parse(localStorage.getItem("prediction") || "null");
      if (pred?.prediction_id) return "See latest session";
    } catch {
      /* ignore */
    }
    return "—";
  }, [labTests]);

  const stats = useMemo(() => {
    const withPct = labTests
      .map((r) => ({ row: r, pct: pctFromRow(r) }))
      .filter((x) => x.pct != null)
      .sort((a, b) => new Date(b.row.createdAt) - new Date(a.row.createdAt));

    const total = labTests.length;
    const avg =
      withPct.length > 0
        ? Math.round((withPct.reduce((s, x) => s + x.pct, 0) / withPct.length) * 10) / 10
        : null;
    const highCount = withPct.filter((x) => x.pct >= 50).length;
    const maxPct = withPct.length > 0 ? Math.max(...withPct.map((x) => x.pct)) : null;
    const latestPct = withPct.length ? withPct[0].pct : null;

    let improvement = null;
    if (withPct.length >= 2) {
      const latest = withPct[0].pct;
      const prev = withPct[1].pct;
      improvement = Math.round((prev - latest) * 10) / 10;
    }

    return { total, avg, highCount, maxPct, latestPct, improvement };
  }, [labTests]);

  const chartPoints = useMemo(() => {
    return labTests
      .filter((r) => pctFromRow(r) != null)
      .map((r) => ({
        id: r.id,
        date: new Date(r.createdAt).toLocaleDateString(),
        shortDate: new Date(r.createdAt).toLocaleDateString(undefined, {
          month: "short",
          day: "numeric",
        }),
        pct: pctFromRow(r),
        t: new Date(r.createdAt).getTime(),
      }))
      .sort((a, b) => a.t - b.t);
  }, [labTests]);

  const insights = useMemo(() => {
    const out = [];
    const sorted = [...labTests].sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt));
    if (sorted.length >= 2) {
      const last = sorted[sorted.length - 1];
      const prev = sorted[sorted.length - 2];
      const c1 = last?.features?.cholesterol;
      const c0 = prev?.features?.cholesterol;
      if (c1 != null && c0 != null) {
        if (c1 > c0 + 5) out.push("Cholesterol has increased compared to your previous test.");
        else if (c1 < c0 - 5) out.push("Cholesterol has improved compared to your previous test.");
        else out.push("Cholesterol is relatively stable across your last two tests.");
      }
      const bp1 = last?.features?.resting_bp_s;
      const bp0 = prev?.features?.resting_bp_s;
      if (bp1 != null && bp0 != null) {
        if (Math.abs(bp1 - bp0) < 5) out.push("Resting blood pressure is stable.");
        else if (bp1 > bp0) out.push("Resting blood pressure increased on your latest test.");
        else out.push("Resting blood pressure decreased on your latest test.");
      }
    }
    const pcts = [...labTests]
      .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
      .map(pctFromRow)
      .filter((p) => p != null);
    if (pcts.length >= 3) {
      const recent = pcts.slice(0, 3);
      const improving = recent[0] < recent[1] && recent[1] < recent[2];
      const worsening = recent[0] > recent[1] && recent[1] > recent[2];
      if (improving) out.push("Risk percentage has improved over your last three recorded predictions.");
      if (worsening) out.push("Risk percentage has risen over your last three recorded predictions.");
    }
    if (!out.length) out.push("Upload lab results and run predictions to unlock personalized insights.");
    return out;
  }, [labTests]);

  useEffect(() => {
    if (overallTier !== "high") return;
    let cancelled = false;
    (async () => {
      try {
        const res = await axios.get(`${API}/hospitals?limit=4`);
        if (!cancelled) setHospitals(res.data.data || []);
      } catch {
        /* ignore */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [overallTier]);

  const filteredSorted = useMemo(() => {
    let rows = [...labTests];

    if (search.trim()) {
      const q = search.toLowerCase();
      rows = rows.filter((r) => {
        const labName = r.lab?.name?.toLowerCase() || "";
        const id = r.id?.toLowerCase() || "";
        const d = new Date(r.createdAt).toLocaleDateString().toLowerCase();
        return labName.includes(q) || id.includes(q) || d.includes(q);
      });
    }

    if (riskFilter !== "all") {
      rows = rows.filter((r) => tierFromRow(r) === riskFilter);
    }

    if (dateFrom) {
      const from = new Date(dateFrom).setHours(0, 0, 0, 0);
      rows = rows.filter((r) => new Date(r.createdAt).getTime() >= from);
    }
    if (dateTo) {
      const to = new Date(dateTo).setHours(23, 59, 59, 999);
      rows = rows.filter((r) => new Date(r.createdAt).getTime() <= to);
    }

    const dir = sortDir === "asc" ? 1 : -1;
    rows.sort((a, b) => {
      if (sortKey === "date") {
        return (new Date(a.createdAt) - new Date(b.createdAt)) * dir;
      }
      if (sortKey === "pct") {
        const pa = pctFromRow(a) ?? -1;
        const pb = pctFromRow(b) ?? -1;
        return (pa - pb) * dir;
      }
      if (sortKey === "chol") {
        return ((a.features?.cholesterol || 0) - (b.features?.cholesterol || 0)) * dir;
      }
      return 0;
    });

    return rows;
  }, [labTests, search, riskFilter, dateFrom, dateTo, sortKey, sortDir]);

  useEffect(() => {
    setPage(1);
  }, [search, riskFilter, dateFrom, dateTo, sortKey, sortDir]);

  const totalPages = Math.max(1, Math.ceil(filteredSorted.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages);
  const pageRows = filteredSorted.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE);

  const toggleSort = (key) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const handleImageUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onloadend = () => {
      const url = reader.result;
      setProfilePic(url);
      localStorage.setItem("profilePic", url);
      pushToast("Profile photo updated.");
    };
    reader.readAsDataURL(file);
  };

  const handleContactChange = (e) => {
    const { name, value } = e.target;
    setContact((prev) => ({ ...prev, [name]: value }));
  };

  const saveContactLocal = () => {
    localStorage.setItem(CONTACT_KEY, JSON.stringify(contact));
    pushToast("Phone & address saved on this device.");
  };

  const handleEditChange = (e) => {
    const { name, value } = e.target;
    setEditForm((prev) => ({ ...prev, [name]: value }));
    setEditErrors((prev) => ({ ...prev, [name]: "" }));
  };

  const validateEdit = () => {
    const errors = {};
    if (!editForm.username.trim()) errors.username = "Username is required";
    else if (editForm.username.length < 3) errors.username = "At least 3 characters";
    if (!editForm.email.trim()) errors.email = "Email is required";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(editForm.email)) errors.email = "Invalid email";
    if (editForm.password && editForm.password.length < 6) errors.password = "Min 6 characters";
    return errors;
  };

  const handleSaveProfile = async () => {
    const errors = validateEdit();
    if (Object.keys(errors).length) {
      setEditErrors(errors);
      return;
    }
    const token = localStorage.getItem("token");
    if (!token) {
      pushToast("Please log in again.", "error");
      return;
    }
    try {
      setSaving(true);
      const payload = { username: editForm.username, email: editForm.email };
      if (editForm.password.trim()) payload.password = editForm.password;
      const res = await axios.put(`${API}/users/${user.id}`, payload, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const updated = { ...user, ...res.data.data };
      localStorage.setItem("user", JSON.stringify(updated));
      setUser(updated);
      setEditForm({ username: updated.username, email: updated.email, password: "" });
      setIsEditing(false);
      setEditErrors({});
      pushToast("Profile saved successfully.");
    } catch (err) {
      pushToast(err.response?.data?.message || "Update failed", "error");
    } finally {
      setSaving(false);
    }
  };

  const downloadEcgReport = async (row) => {
    const id = row?.id;
    if (!id) {
      pushToast("No ECG test id.", "error");
      return;
    }
    if (row.inference_status !== "ok") {
      pushToast("ECG report is available after analysis completes.", "error");
      return;
    }
    const token = localStorage.getItem("token");
    try {
      const res = await axios.get(`${API}/ecg/${id}/report`, {
        headers: { Authorization: `Bearer ${token}` },
        responseType: "blob",
      });
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const a = document.createElement("a");
      a.href = url;
      a.setAttribute("download", `ECG_Report_${id}.pdf`);
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      pushToast("ECG report download started.");
    } catch {
      pushToast("Could not download ECG report.", "error");
    }
  };

  const downloadReport = async (row) => {
    const predId = row?.prediction?.id;
    const pct = pctFromRow(row);
    if (!predId) {
      pushToast("No prediction report for this row.", "error");
      return;
    }
    if (pct != null && pct < 50) {
      pushToast("PDF report is only available for high-risk predictions.", "error");
      return;
    }
    const token = localStorage.getItem("token");
    try {
      const res = await axios.get(`${API}/predictions/${predId}/report`, {
        headers: { Authorization: `Bearer ${token}` },
        responseType: "blob",
      });
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const a = document.createElement("a");
      a.href = url;
      a.setAttribute("download", `Heart_Report_${row.id}.pdf`);
      document.body.appendChild(a);
      a.click();
      a.remove();
      pushToast("Report download started.");
    } catch {
      pushToast("Could not download report.", "error");
    }
  };

  if (loading && !user) {
    return (
      <div className="pd-page">
        <div className="pd-card" style={{ maxWidth: 400, margin: "80px auto" }}>
          <div className="pd-card-body">
            <div className="pd-skeleton" style={{ height: 24, width: "60%", marginBottom: 16 }} />
            <div className="pd-skeleton" style={{ height: 14, width: "100%" }} />
          </div>
        </div>
      </div>
    );
  }

  if (!user) return null;

  return (
    <div className="pd-page">
      <Toast toasts={toasts} remove={removeToast} />

      <header className="pd-header">
        <button type="button" className="pd-back" onClick={() => navigate("/the_general")}>
          <FaArrowLeft aria-hidden /> Back
        </button>
        <div className="pd-title-block">
          <h1>Patient Dashboard</h1>
          <p>Medical history, predictions, and profile — Heart Disease Prediction System</p>
        </div>
      </header>

      <section className="pd-grid-stats" aria-label="Dashboard statistics">
        <StatCard
          label="Total lab tests"
          value={stats.total}
          sub="All uploads on file"
          icon={<FaFlask color="#1198b7" />}
        />
        <StatCard
          label="Avg prediction risk"
          value={stats.avg != null ? `${stats.avg}%` : "—"}
          sub="Across tests with predictions"
          icon={<FaChartLine color="#1198b7" />}
        />
        <StatCard
          label="Highest risk"
          value={stats.maxPct != null ? `${stats.maxPct}%` : "—"}
          sub="Peak recorded"
          icon={<FaExclamationTriangle color="#f59e0b" />}
        />
        <StatCard
          label="Latest prediction"
          value={stats.latestPct != null ? `${stats.latestPct}%` : "—"}
          sub="Most recent test"
          icon={<FaHeartbeat color="#ef4444" />}
        />
        <StatCard
          label="High-risk cases"
          value={stats.highCount}
          sub="Tests at or above 50%"
          icon={<FaHospital color="#1198b7" />}
        />
        <StatCard
          label="Last vs previous"
          value={stats.improvement != null ? `${stats.improvement > 0 ? "+" : ""}${stats.improvement}%` : "—"}
          sub="Change between last two predictions"
          icon={<FaChartLine color="#22c55e" />}
        />
      </section>

      <div className="pd-layout">
        <div>
          <div className="pd-card">
            <div className="pd-card-header">
              <h2>Patient profile</h2>
              {!isEditing ? (
                <button type="button" className="pd-btn pd-btn-primary" onClick={() => setIsEditing(true)}>
                  <FaEdit /> Edit
                </button>
              ) : (
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  <button
                    type="button"
                    className="pd-btn pd-btn-primary"
                    onClick={handleSaveProfile}
                    disabled={saving}
                  >
                    <FaSave /> {saving ? "Saving…" : "Save"}
                  </button>
                  <button
                    type="button"
                    className="pd-btn pd-btn-ghost"
                    onClick={() => {
                      setIsEditing(false);
                      setEditForm({
                        username: user.username,
                        email: user.email,
                        password: "",
                      });
                      setEditErrors({});
                    }}
                  >
                    <FaTimes /> Cancel
                  </button>
                </div>
              )}
            </div>
            <div className="pd-card-body">
              <div className="pd-profile-top">
                <div className="pd-avatar-wrap">
                  <img src={profilePic || defaultProfile} alt="" className="pd-avatar" />
                  <label className="pd-avatar-btn">
                    <FaCamera />
                    <input type="file" accept="image/*" hidden onChange={handleImageUpload} />
                  </label>
                </div>
                <div style={{ flex: 1, minWidth: 200 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
                    <h3 style={{ margin: 0, fontSize: "1.25rem" }}>{user.username}</h3>
                    <RiskBadge tier={overallTier} />
                  </div>
                  <p style={{ margin: "8px 0 0", color: "var(--pd-muted)", fontSize: 14 }}>
                    National ID: <strong style={{ color: "var(--pd-text)" }}>{user.national_id}</strong>
                  </p>
                  <p style={{ margin: "4px 0 0", color: "var(--pd-muted)", fontSize: 13 }}>
                    Registration: {user.createdAt ? new Date(user.createdAt).toLocaleDateString() : "—"}
                    {" · "}
                    Last prediction: {lastPredictionAt}
                  </p>
                </div>
              </div>

              <div className="pd-fields">
                <div className="pd-field">
                  <label>Full name (username)</label>
                  <input
                    name="username"
                    value={isEditing ? editForm.username : user.username}
                    onChange={handleEditChange}
                    disabled={!isEditing}
                  />
                  {editErrors.username && <div className="pd-field-error">{editErrors.username}</div>}
                </div>
                <div className="pd-field">
                  <label>National ID</label>
                  <input value={user.national_id} disabled />
                </div>
                <div className="pd-field">
                  <label>Age (from latest lab test)</label>
                  <input value={ageFromLab} disabled />
                </div>
                <div className="pd-field">
                  <label>Gender (from latest lab test)</label>
                  <input value={genderFromLab} disabled />
                </div>
                <div className="pd-field">
                  <label>Email</label>
                  <input
                    name="email"
                    type="email"
                    value={isEditing ? editForm.email : user.email}
                    onChange={handleEditChange}
                    disabled={!isEditing}
                  />
                  {editErrors.email && <div className="pd-field-error">{editErrors.email}</div>}
                </div>
                <div className="pd-field">
                  <label>Password</label>
                  {isEditing ? (
                    <div className="pw-wrap">
                      <input
                        name="password"
                        type={showPassword ? "text" : "password"}
                        placeholder="Leave blank to keep current"
                        value={editForm.password}
                        onChange={handleEditChange}
                      />
                      <button
                        type="button"
                        className="pw-toggle"
                        onClick={() => setShowPassword((v) => !v)}
                        aria-label={showPassword ? "Hide password" : "Show password"}
                      >
                        {showPassword ? <FaEyeSlash /> : <FaEye />}
                      </button>
                    </div>
                  ) : (
                    <input value="••••••••" disabled />
                  )}
                  {editErrors.password && <div className="pd-field-error">{editErrors.password}</div>}
                </div>
                <div className="pd-field">
                  <label>Phone number</label>
                  <input
                    name="phone"
                    value={contact.phone || ""}
                    onChange={handleContactChange}
                    placeholder="Optional"
                  />
                  <div className="hint">Stored on this device (server profile has no phone field yet).</div>
                </div>
                <div className="pd-field">
                  <label>Address</label>
                  <input
                    name="address"
                    value={contact.address || ""}
                    onChange={handleContactChange}
                    placeholder="Optional"
                  />
                  <div className="hint">Stored on this device until backend fields exist.</div>
                </div>
              </div>
              <div className="pd-actions">
                <button type="button" className="pd-btn pd-btn-ghost" onClick={saveContactLocal}>
                  Save phone & address locally
                </button>
              </div>
            </div>
          </div>

          <div className="pd-card pd-insights" style={{ marginTop: 24 }}>
            <div className="pd-card-header">
              <h2>Health insights</h2>
            </div>
            <div className="pd-card-body">
              <ul className="pd-insight-list">
                {insights.map((t, i) => (
                  <li key={i}>{t}</li>
                ))}
              </ul>
              {overallTier === "high" && hospitals.length > 0 && (
                <div className="pd-hospitals-mini">
                  <strong>Recommended hospitals</strong>
                  {hospitals.slice(0, 4).map((h) => (
                    <a key={h.id} href={h.google_maps_link} target="_blank" rel="noopener noreferrer">
                      {h.name} — {h.area}
                    </a>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="pd-card pd-chart-card">
            <div className="pd-card-header pd-chart-header">
              <h2>
                <FaChartLine style={{ marginRight: 8, color: "var(--pd-primary)" }} />
                Prediction trend
              </h2>
              <div className="pd-chart-legend">
                <span><i style={{ background: "#22c55e" }} />Low</span>
                <span><i style={{ background: "#f59e0b" }} />Medium</span>
                <span><i style={{ background: "#ef4444" }} />High</span>
              </div>
            </div>
            <div className="pd-card-body">
              <PredictionTrendChart
                points={chartPoints}
                onPointClick={(p) => {
                  const row = labTests.find((r) => r.id === p.id);
                  if (row) setDetailRow(row);
                }}
              />
            </div>
          </div>
        </div>

        <div>
          <div className="pd-card">
            <div className="pd-card-header" style={{ flexWrap: "wrap", gap: 12 }}>
              <h2 style={{ margin: 0 }}>Tests & predictions</h2>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                <button
                  type="button"
                  className={`pd-btn ${dashboardTab === "lab" ? "pd-btn-primary" : "pd-btn-ghost"}`}
                  style={{ padding: "6px 14px", fontSize: 13 }}
                  onClick={() => setDashboardTab("lab")}
                >
                  Lab tests
                </button>
                <button
                  type="button"
                  className={`pd-btn ${dashboardTab === "ecg" ? "pd-btn-primary" : "pd-btn-ghost"}`}
                  style={{ padding: "6px 14px", fontSize: 13 }}
                  onClick={() => setDashboardTab("ecg")}
                >
                  ECG tests
                </button>
              </div>
            </div>
            <div className="pd-card-body">
              {dashboardTab === "lab" ? (
              <>
              <div className="pd-toolbar">
                <div>
                  <label htmlFor="pd-search">Search</label>
                  <div style={{ position: "relative" }}>
                    <FaSearch
                      style={{ position: "absolute", left: 10, top: 10, color: "#94a3b8", fontSize: 12 }}
                    />
                    <input
                      id="pd-search"
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      placeholder="Lab, date, id…"
                      style={{ paddingLeft: 30 }}
                    />
                  </div>
                </div>
                <div>
                  <label>Risk</label>
                  <select value={riskFilter} onChange={(e) => setRiskFilter(e.target.value)}>
                    <option value="all">All</option>
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                    <option value="none">No prediction</option>
                  </select>
                </div>
                <div>
                  <label>From</label>
                  <input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} />
                </div>
                <div>
                  <label>To</label>
                  <input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} />
                </div>
              </div>

              {loading ? (
                <div className="pd-table-wrap">
                  <div className="pd-card-body">
                    <div className="pd-skeleton" style={{ height: 40, marginBottom: 8 }} />
                    <div className="pd-skeleton" style={{ height: 40, marginBottom: 8 }} />
                    <div className="pd-skeleton" style={{ height: 40 }} />
                  </div>
                </div>
              ) : filteredSorted.length === 0 ? (
                <div className="pd-empty">
                  <strong>No rows match your filters</strong>
                  Clear search or change risk / date filters.
                </div>
              ) : (
                <>
                  <div className="pd-table-wrap">
                    <table className="pd-table">
                      <thead>
                        <tr>
                          <th onClick={() => toggleSort("date")}>
                            Test date{" "}
                            <span className="pd-sort-ind">{sortKey === "date" ? (sortDir === "asc" ? "▲" : "▼") : ""}</span>
                          </th>
                          <th onClick={() => toggleSort("chol")}>
                            Cholesterol{" "}
                            <span className="pd-sort-ind">{sortKey === "chol" ? (sortDir === "asc" ? "▲" : "▼") : ""}</span>
                          </th>
                          <th>BP</th>
                          <th>Heart rate</th>
                          <th>Blood sugar</th>
                          <th onClick={() => toggleSort("pct")}>
                            Prediction %{" "}
                            <span className="pd-sort-ind">{sortKey === "pct" ? (sortDir === "asc" ? "▲" : "▼") : ""}</span>
                          </th>
                          <th>Risk</th>
                          <th>Progress</th>
                          <th>Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {pageRows.map((row) => {
                          const pct = pctFromRow(row);
                          const tier = tierFromRow(row);
                          const f = row.features || {};
                          return (
                            <tr key={row.id}>
                              <td>{new Date(row.createdAt).toLocaleString()}</td>
                              <td>{f.cholesterol ?? "—"}</td>
                              <td>{f.resting_bp_s ?? "—"}</td>
                              <td>{f.max_heart_rate ?? "—"}</td>
                              <td>{f.fasting_blood_sugar ?? "—"}</td>
                              <td>{pct != null ? `${pct}%` : row.prediction?.decision ? String(row.prediction.decision) : "—"}</td>
                              <td>
                                <span className="pd-dot" style={{ background: barColor(tier) }} title={riskLabel(tier)} />{" "}
                                {riskLabel(tier)}
                              </td>
                              <td>
                                <div className="pd-progress">
                                  <div className="pd-progress-track" title={pct != null ? `${pct}%` : riskLabel(tier)}>
                                    <div
                                      className="pd-progress-fill"
                                      style={{
                                        width: pct != null ? `${Math.min(100, Math.max(0, pct))}%` : "0%",
                                        background: barColor(tier),
                                      }}
                                    />
                                  </div>
                                  <div className="pd-progress-label">{pct != null ? `${pct}%` : "N/A"}</div>
                                </div>
                              </td>
                              <td>
                                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                                  <button
                                    type="button"
                                    className="pd-btn pd-btn-ghost"
                                    style={{ padding: "6px 10px", fontSize: 12 }}
                                    onClick={() => setDetailRow(row)}
                                  >
                                    Details
                                  </button>
                                  <button
                                    type="button"
                                    className="pd-btn pd-btn-primary"
                                    style={{ padding: "6px 10px", fontSize: 12 }}
                                    onClick={() => downloadReport(row)}
                                  >
                                    <FaDownload /> PDF
                                  </button>
                                </div>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <div className="pd-pagination">
                    <span>
                      Page {safePage} of {totalPages} · {filteredSorted.length} row(s)
                    </span>
                    <div style={{ display: "flex", gap: 8 }}>
                      <button type="button" disabled={safePage <= 1} onClick={() => setPage((p) => Math.max(1, p - 1))}>
                        Previous
                      </button>
                      <button
                        type="button"
                        disabled={safePage >= totalPages}
                        onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                      >
                        Next
                      </button>
                    </div>
                  </div>
                </>
              )}
              </>
              ) : (
                <div className="pd-table-wrap">
                  {ecgTests.length === 0 ? (
                    <div className="pd-empty">
                      <strong>No ECG tests on file</strong>
                      Ask your lab to upload WFDB .dat + .hea for your national ID.
                    </div>
                  ) : (
                    <table className="pd-table">
                      <thead>
                        <tr>
                          <th>Date</th>
                          <th>ECG test</th>
                          <th>Status</th>
                          <th>Diagnosis</th>
                          <th>Probability</th>
                          <th>Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {ecgTests.map((row) => (
                          <tr key={row.id}>
                            <td>{new Date(row.createdAt).toLocaleString()}</td>
                            <td style={{ fontFamily: "monospace", fontSize: 11 }}>{row.id}</td>
                            <td>{row.inference_status}</td>
                            <td>{row.primary_diagnosis || "—"}</td>
                            <td>
                              {row.primary_probability != null
                                ? `${Number(row.primary_probability).toFixed(2)}%`
                                : "—"}
                            </td>
                            <td>
                              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                                <button
                                  type="button"
                                  className="pd-btn pd-btn-ghost"
                                  style={{ padding: "6px 10px", fontSize: 12 }}
                                  onClick={() => setEcgDetailRow(row)}
                                >
                                  Details
                                </button>
                                <button
                                  type="button"
                                  className="pd-btn pd-btn-primary"
                                  style={{ padding: "6px 10px", fontSize: 12 }}
                                  onClick={() => downloadEcgReport(row)}
                                >
                                  <FaDownload /> PDF
                                </button>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              )}
            </div>
          </div>

          <div className="pd-card" style={{ marginTop: 24 }}>
            <div className="pd-card-header">
              <h2>Notifications & reminders</h2>
            </div>
            <div className="pd-card-body">
              <ul className="pd-insight-list">
                {labTests.length === 0 && <li>No lab tests on file — upload via your lab or patient CSV flow.</li>}
                {overallTier === "high" && <li>High risk detected — consider cardiologist follow-up and lifestyle review.</li>}
                {overallTier === "medium" && <li>Medium risk — retest after a few weeks or if symptoms change.</li>}
                {overallTier === "low" && labTests.length > 0 && <li>Latest prediction is low risk — keep healthy habits.</li>}
                <li>Retake a lab test periodically as advised by your physician.</li>
              </ul>
            </div>
          </div>

          <div className="pd-card pd-chart-card" style={{ marginTop: 24 }}>
            <div className="pd-card-header pd-chart-header">
              <h2>
                <FaHeartbeat style={{ marginRight: 8, color: "#cb2323" }} />
                Latest ECG
              </h2>
            </div>
            <div className="pd-card-body">
              {!ecgStatus?.hasEcgTests ? (
                <div className="pd-empty">
                  <strong>You don&apos;t have any ECG test</strong>
                </div>
              ) : (
                <>
                  <p>
                    <strong>Diagnosis:</strong> {ecgStatus.latestSummary?.primary_diagnosis || "—"}
                  </p>
                  <p>
                    <strong>Probability:</strong>{" "}
                    {ecgStatus.latestSummary?.primary_probability != null
                      ? `${Number(ecgStatus.latestSummary.primary_probability).toFixed(2)}%`
                      : "—"}
                  </p>
                  <p>
                    <strong>Date:</strong>{" "}
                    {ecgStatus.latestSummary?.createdAt
                      ? new Date(ecgStatus.latestSummary.createdAt).toLocaleString()
                      : "—"}
                  </p>
                  <p className="hint" style={{ marginTop: 8 }}>
                    Status: {ecgStatus.latestSummary?.inference_status || "—"}
                  </p>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {detailRow && (
        <div
          className="pd-modal-overlay"
          role="dialog"
          aria-modal="true"
          aria-labelledby="pd-modal-title"
          onClick={(e) => e.target === e.currentTarget && setDetailRow(null)}
        >
          <div className="pd-modal">
            <div className="pd-modal-header">
              <h3 id="pd-modal-title">Test details</h3>
              <button type="button" className="pd-modal-close" onClick={() => setDetailRow(null)} aria-label="Close">
                ×
              </button>
            </div>
            <div className="pd-modal-body">
              <p>
                <strong>Lab:</strong> {detailRow.lab?.name || "—"}
              </p>
              <p>
                <strong>Date:</strong> {new Date(detailRow.createdAt).toLocaleString()}
              </p>
              <p>
                <strong>Prediction ID:</strong> {detailRow.prediction?.id || "—"}
              </p>
              <p>
                <strong>Features</strong>
              </p>
              <pre style={{ background: "#f8fafc", padding: 12, borderRadius: 8, fontSize: 12, overflow: "auto" }}>
                {JSON.stringify(detailRow.features, null, 2)}
              </pre>
              <p style={{ marginTop: 12, fontSize: 13, color: "var(--pd-muted)" }}>
                SHAP explainability: open your latest <strong>high risk</strong> result page after running prediction to
                view the full SHAP image from the server.
              </p>
            </div>
          </div>
        </div>
      )}

      {ecgDetailRow && (
        <div
          className="pd-modal-overlay"
          role="dialog"
          aria-modal="true"
          aria-labelledby="pd-ecg-modal-title"
          onClick={(e) => e.target === e.currentTarget && setEcgDetailRow(null)}
        >
          <div className="pd-modal">
            <div className="pd-modal-header">
              <h3 id="pd-ecg-modal-title">ECG test details</h3>
              <button type="button" className="pd-modal-close" onClick={() => setEcgDetailRow(null)} aria-label="Close">
                ×
              </button>
            </div>
            <div className="pd-modal-body">
              <p>
                <strong>Lab:</strong> {ecgDetailRow.lab?.name || "—"}
              </p>
              <p>
                <strong>Date:</strong> {new Date(ecgDetailRow.createdAt).toLocaleString()}
              </p>
              <p>
                <strong>ECG test ID:</strong> {ecgDetailRow.id}
              </p>
              <p>
                <strong>Status:</strong> {ecgDetailRow.inference_status}
              </p>
              <p>
                <strong>Primary diagnosis:</strong> {ecgDetailRow.primary_diagnosis || "—"}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
