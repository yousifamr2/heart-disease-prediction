import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# ECG Configuration
PTBXL_DATASET_PATH = os.getenv("PTBXL_DATASET_PATH", "")
try:
    EVAL_SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "500"))
except ValueError:
    EVAL_SAMPLE_SIZE = 500

if EVAL_SAMPLE_SIZE not in [100, 500, 1000]:
    EVAL_SAMPLE_SIZE = 500

# Base paths
EVAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_ROOT.parent

# Output paths
REPORTS_DIR = EVAL_ROOT / "reports"
FIGURES_ECG_DIR = EVAL_ROOT / "figures" / "ecg"
FIGURES_LLM_DIR = EVAL_ROOT / "figures" / "llm"
SCORECARDS_DIR = EVAL_ROOT / "scorecards"
BENCHMARK_DIR = EVAL_ROOT / "benchmark"

# Ensure output directories exist
for d in [REPORTS_DIR, FIGURES_ECG_DIR, FIGURES_LLM_DIR, SCORECARDS_DIR, BENCHMARK_DIR]:
    d.mkdir(parents=True, exist_ok=True)
