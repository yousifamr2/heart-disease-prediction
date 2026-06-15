import sys
import subprocess
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent

def run_script(script_path):
    print(f"\n[{script_path.name}] Running...")
    result = subprocess.run([sys.executable, str(script_path)])
    if result.returncode != 0:
        print(f"[{script_path.name}] FAILED with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"[{script_path.name}] SUCCESS")

def main():
    print("="*60)
    print("Executing Complete Evaluation Framework")
    print("="*60)
    
    scripts = [
        EVAL_ROOT / "ecg" / "evaluate_ecg.py",
        EVAL_ROOT / "llm" / "evaluate_llm.py",
        EVAL_ROOT / "benchmark" / "evaluate_benchmark.py",
        EVAL_ROOT / "reports" / "generate_final_report.py"
    ]
    
    for script in scripts:
        if script.exists():
            run_script(script)
        else:
            print(f"WARNING: Script not found - {script}")
            
    print("\n" + "="*60)
    print("EVALUATION FRAMEWORK COMPLETED SUCCESSFULLY")
    print("To view the interactive dashboard, run:")
    print(f"streamlit run {EVAL_ROOT / 'dashboard.py'}")
    print("="*60)

if __name__ == "__main__":
    main()
