"""Make the repository root importable regardless of where pytest is started from.

Without this, `from models...` only resolves when the tests run from the checkout
root, and a bare `pytest tests/...` from elsewhere fails at collection.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
