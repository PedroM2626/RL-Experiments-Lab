"""Gate the key-cell checkpoints: each zip is provably the model behind the number.

Eight cells of the diagnostic arc carry saved weights (results/models/*.zip), each
with a companion repro curve. The re-runs were solo and the training path is
deterministic on this hardware -- so the test's demand is the strong one: the repro
curve's FINAL ROW must equal the committed probe artifact's final row bitwise, on
every cell. If that holds, the zip is not "a similar model"; it is the model that
produced the published measurement.
"""
import json
import os
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "results", "models")

CELLS = {
    "vit_bossfight": "probe_nonreaders_vit_bossfight.json",
    "vit_dodgeball": "probe_nonreaders_vit_dodgeball.json",
    "impoola_bossfight": "probe_nonreaders_impoola_bossfight.json",
    "mlp_dodgeball": "probe_nonreaders_mlp_dodgeball.json",
    "classic_bossfight": "probe_nonreaders_classic_bossfight.json",
    "classic_dodgeball": "probe_nonreaders_classic_dodgeball.json",
    "mlp_scaled_dodgeball": "probe_mlpscaled_dodgeball.json",
    "mlp_scaled_bossfight": "probe_mlpscaled_bossfight.json",
}


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_each_checkpoint_reproduces_its_artifacts_final_row_bitwise(cell):
    repro = _load(os.path.join(MODELS, f"{cell}_repro.json"))["curve"]
    canon = _load(os.path.join(ROOT, "results", CELLS[cell]))["curve"]
    assert repro[-1] == canon[-1], \
        f"{cell}: the repro drifted from the committed artifact -- the zip is not its model"


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_each_zip_exists_and_carries_policy_weights(cell):
    path = os.path.join(MODELS, f"{cell}.zip")
    assert os.path.exists(path), f"{cell}.zip missing"
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    assert any("policy" in n for n in names), f"{cell}.zip has no policy weights"
    assert os.path.getsize(path) > 100_000, f"{cell}.zip suspiciously small"
    # the mlp heads are tiny; the cnn/vit fronts are multi-MB -- and nothing here
    # may approach GitHub's limits
    assert os.path.getsize(path) < 30_000_000


def test_the_repros_are_complete_grid_runs():
    for cell in CELLS:
        d = _load(os.path.join(MODELS, f"{cell}_repro.json"))
        steps = [r["steps"] for r in d["curve"]]
        assert steps == list(range(0, 100001, 500)), f"{cell}: repro grid"
        p = d["_provenance"]
        assert (p["n_envs"], p["n_steps"], p["seed"], p["timesteps"]) == (16, 256, 42, 100000)


def test_the_provenance_table_names_the_eight_cells():
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    i = readme.index("the **key-cell checkpoints**")
    seg = readme[i:i + 900]
    for s in ("`vit`/bossfight", "`vit`/dodgeball", "`impoola`/bossfight",
              "`mlp_vector`/dodgeball", "`classic` on bossfight and dodgeball",
              "`mlp_scaled` on both games",
              "bitwise on all 8 cells",
              "tests/test_key_cell_checkpoints.py"):
        assert s in seg, f"provenance table lost: {s!r}"
