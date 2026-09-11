"""Compatibility command: recompute current temporal-model explanations."""
import runpy
import sys
from pathlib import Path

if __name__ == '__main__':
    if '--explain' not in sys.argv:
        sys.argv.append('--explain')
    runpy.run_path(str(Path(__file__).resolve().parent/'current/verify.py'),run_name='__main__')
