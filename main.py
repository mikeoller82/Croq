#!/usr/bin/env python3
"""
Croq Optimized - Main Entry Point

High-performance AI code assistant competing with Claude Code and Gemini CLI
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cli import app

if __name__ == "__main__":
    app()
