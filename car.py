"""
Legacy entry point wrapper for Car_detection_Project.
Redirects to the modular Traffic Vision pipeline while preserving full backward compatibility.
"""

from __future__ import annotations

import sys
from main import main

if __name__ == "__main__":
    # If no arguments were passed, provide standard highway configuration
    if len(sys.argv) == 1:
        sys.argv.extend(["--config", "configs/default_highway.yaml", "--show"])
    sys.exit(main())
