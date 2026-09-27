"""Hold ES_SYSTEM_REQUIRED until a sentinel file appears, so a long run is not suspended.

The request is per-process and Windows drops it when this exits, so the user's power plan
is never modified -- same contract as run_exploration_remeasure.py's set_keep_awake.

Usage: py -3.10 keep_awake.py <sentinel-path> [max-hours, default 24]
"""
import ctypes
import os
import sys
import time

sentinel = sys.argv[1]
deadline = time.time() + float(sys.argv[2] if len(sys.argv) > 2 else 24.0) * 3600
set_state = ctypes.windll.kernel32.SetThreadExecutionState
set_state(0x80000000 | 0x00000001)
try:
    while True:
        if os.path.exists(sentinel):
            break
        if deadline and time.time() > deadline:
            break
        time.sleep(60)
finally:
    set_state(0x80000000)
