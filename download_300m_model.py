"""
Backward-compatible wrapper for download_weights.py (--model 300m)
"""
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
cmd = [sys.executable, str(BASE_DIR / "download_weights.py"), "--model", "300m"]
subprocess.run(cmd)
