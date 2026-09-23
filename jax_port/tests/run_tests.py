"""Aggregate test runner for the JAX port: python -m jax_port.tests.run_tests [--list]

Every case in every module runs: modules that expose run_all() contribute all of
their cases, the rest one named function. A case SKIPs only when its imports are
absent from this interpreter (the Windows study venv has no flax); any other
exception is a FAIL and the process exits non-zero.

Before 23/09/2026 this runner listed five hard-coded functions, which meant
test_temporal.py never ran at all and only test_losses() of the five MARL cases did
— while README section 19 cited the unrun ones as validation evidence.
"""

import argparse
import importlib
import traceback

# (label, module, entry point, needs the port venv)
CASES = [
    ("stats", "jax_port.tests.test_stats", "test_stats", False),
    ("parity", "jax_port.tests.test_parity", "test_env_parity", True),
    ("zoo", "jax_port.tests.test_zoo", "test_zoo", True),
    ("smoke", "jax_port.tests.test_smoke", "test_smoke_train", True),
    ("temporal", "jax_port.tests.test_temporal", "run_all", True),
    ("exploration", "jax_port.tests.test_exploration", "run_all", True),
    ("marl", "jax_port.tests.test_marl", "run_all", True),
]


def run_one(label, module_name, fn_name, needs_env, results):
    try:
        module = importlib.import_module(module_name)
        out = getattr(module, fn_name)()
    except ImportError as e:
        if needs_env:
            results.setdefault("skip", []).append(f"{label} ({e})")
            print(f"[{label}] SKIP — dependency missing: {e}", flush=True)
            return
        results.setdefault("fail", []).append(f"{label}: {e}")
        print(f"[{label}] FAIL — core test must not need optional deps", flush=True)
        return
    except Exception:
        traceback.print_exc()
        results.setdefault("fail", []).append(label)
        print(f"[{label}] FAIL", flush=True)
        return
    if isinstance(out, dict):
        for case, ok in out.items():
            print(f"[{label}:{case}] {ok}", flush=True)
        results.setdefault("pass", []).extend(f"{label}:{c}" for c in out)
    else:
        print(f"[{label}] {out}", flush=True)
        results.setdefault("pass", []).append(label)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print the registered cases and exit")
    args = parser.parse_args()

    if args.list:
        for label, module_name, fn_name, needs_env in CASES:
            print(f"{label:10s} {module_name}.{fn_name}{'  (needs port venv)' if needs_env else ''}")
        return

    results = {}
    for label, module_name, fn_name, needs_env in CASES:
        run_one(label, module_name, fn_name, needs_env, results)

    passed = results.get("pass", [])
    skipped = results.get("skip", [])
    failed = results.get("fail", [])
    print(f"\nPASS={len(passed)} SKIP={len(skipped)} FAIL={len(failed)}")
    if skipped:
        print("skipped: " + ", ".join(skipped))
    if failed:
        print("failed:  " + ", ".join(failed))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
