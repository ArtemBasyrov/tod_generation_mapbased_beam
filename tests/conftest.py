"""
Pytest configuration shared across all test modules.

Sets up sys.path so that codebase modules can be imported from the project
root, and stubs out heavy third-party dependencies that are not under test.
"""

import os
import sys
from unittest.mock import MagicMock

# Make the project root (parent of this tests/ directory) importable.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Stub out unavailable third-party modules imported at module level by
# codebase modules we are NOT testing (pixell, etc.).
_STUB_MODULES = ["pixell", "pixell.enmap"]
for _mod_name in _STUB_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

# tod_calibrate imports tod_io (which needs pixell) at module level.
if "tod_io" not in sys.modules:
    sys.modules["tod_io"] = MagicMock()
