"""Hold ES_SYSTEM_REQUIRED until a sentinel file appears, so a long run is not suspended.

The request is per-process and Windows drops it when this exits, so the user's power plan is
never modified -- the same contract run_exploration_remeasure.py's set_keep_awake uses, factored
out so a WSL-side sweep started from another shell is covered too.

It also refuses to hold the flag forever: without --max-hours a crashed producer whose sentinel
never arrives would keep the machine awake indefinitely.

Usage:
    py -3.10 keep_awake.py <sentinel-path> [max-hours, default 24]
"""
import argparse
import ctypes
import os
import time


def hold(sentinel, max_hours, poll=60, set_state=None):
    """Set the execution state, then return once the sentinel exists or the deadline passes.

    set_state is injectable so a test can assert the request is released on every exit path;
    it defaults to the Windows API call.
    """
    CONTINUOUS, SYSTEM_REQUIRED = 0x80000000, 0x00000001
    if set_state is None:
        set_state = ctypes.windll.kernel32.SetThreadExecutionState
    set_state(CONTINUOUS | SYSTEM_REQUIRED)
    deadline = time.time() + max_hours * 3600
    reason = "deadline"
    try:
        while True:
            if os.path.exists(sentinel):
                reason = "sentinel"
                break
            if time.time() > deadline:
                break
            time.sleep(poll)
    finally:
        set_state(CONTINUOUS)
    return reason


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sentinel")
    ap.add_argument("max_hours", nargs="?", type=float, default=24.0)
    args = ap.parse_args(argv)
    return hold(args.sentinel, args.max_hours)


if __name__ == "__main__":
    main()
