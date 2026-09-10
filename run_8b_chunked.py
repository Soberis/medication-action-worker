"""Convenience root entrypoint for tools/run_8b_chunked.py"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "tools" / "run_8b_chunked.py"
    sys.exit(subprocess.call([sys.executable, str(target)] + sys.argv[1:]))
