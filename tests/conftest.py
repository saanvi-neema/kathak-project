"""
Shared pytest setup: makes scripts/ importable from the test files, same way
app/pipeline.py does it at runtime.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "app"))
