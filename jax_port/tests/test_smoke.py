"""End-to-end training smoke test (MLP+coinrun, ~1k steps, fast, non-conv).

Exercises the grid runner pipeline (train.train in process). Requires port
venv + GPU. Usage: python -m jax_port.tests.test_smoke
"""

import math
import os
import tempfile
import types


def test_smoke_train():
    from jax_port import train as T
    out = os.path.join(tempfile.mkdtemp(), "smoke.json")
    # Args come from train.py's own parser: a hand-built SimpleNamespace went stale
    # twice (missing stack/distribution/aux) and the failure looked like an
    # ImportError-skipped test rather than a broken contract.
    args = T.build_parser().parse_args(
        ["--game", "coinrun", "--algo", "ppo", "--extractor", "mlp",
         "--timesteps", "1024", "--seed", "3", "--num-envs", "8",
         "--rollout", "32", "--minibatch", "256", "--eval-eps", "2",
         "--out", out])
    r = T.train(args)
    assert r["timesteps"] >= 1024 and r["sps"] > 100, r
    assert r["curve"], "AUC curve missing"
    assert math.isfinite(r["train_ret_mean20"])
    assert r["eval_unseen"]["eps"] == 2
    return {"sps": r["sps"], "ret": r["train_ret_mean20"]}


if __name__ == "__main__":
    print(test_smoke_train())
    print("SMOKE_OK")
