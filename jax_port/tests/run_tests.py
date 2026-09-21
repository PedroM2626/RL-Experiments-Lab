"""Test suite runner (without pytest): python -m jax_port.tests.run_tests.

test_stats always runs; test_parity/test_zoo/test_smoke skip (SKIP) if
venv lacks dependencies — never fail due to environment absence.
"""

import traceback

CASES = [
    ("stats", "jax_port.tests.test_stats", "test_stats", False),
    ("parity", "jax_port.tests.test_parity", "test_env_parity", True),
    ("zoo", "jax_port.tests.test_zoo", "test_zoo", True),
    ("smoke", "jax_port.tests.test_smoke", "test_smoke_train", True),
    ("marl", "jax_port.tests.test_marl", "test_losses", True),
]


def main():
    import sys
    fails, skips = [], []
    for name, mod, fn, needs_env in CASES:
        try:
            f = getattr(__import__(mod, fromlist=[fn]), fn)
            print(f"[{name}] {f()}", flush=True)
            print(f"[{name}] PASS", flush=True)
        except ImportError as e:
            print(f"[{name}] SKIP ({e})", flush=True)
            skips.append(name)
        except Exception:
            traceback.print_exc()
            fails.append(name)
    print(f"PASS={len(CASES)-len(fails)-len(skips)} "
          f"SKIP={len(skips)} FAIL={fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
