"""Gate the producers' default training recipe (30/10/2026).

README 3.6.4-3.6.7 established that the published single-stream recipe (1 env x 256 steps)
trains policies that do not read the screen, and that 16 envs x 256 steps does not. Every
compare_*.py that trains a Procgen policy now defaults to the second, keeps the first one
flag away (--published-recipe) so every archived cell stays reproducible, runs the
response guard unless told not to, and never writes a new-recipe result into a published
artifact.
"""
import argparse
import ast
import glob
import inspect
import os

import pytest

import response_guard as RG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Procgen producers on the shared recipe. Excluded on purpose: compare_fast.py (CartPole,
# not Procgen -- the cross-game response has no meaning there), compare_offline_rl.py (no
# online rollout), compare_hrl_learned.py (its own two-level trainer and response analogue).
PRODUCERS = [
    "compare_action_repeat.py", "compare_algo_families.py", "compare_augment_contrastive.py",
    "compare_bossfight_hard.py", "compare_budget_scaling.py", "compare_combined.py",
    "compare_hrl.py", "compare_maze_heist.py", "compare_new_archs.py", "compare_procgen.py",
    "compare_suite.py", "compare_suite_retrain.py", "compare_world_models.py",
]
EXCLUDED = {"compare_fast.py", "compare_offline_rl.py", "compare_hrl_learned.py"}


def _parse(*argv):
    p = RG.add_recipe_args(argparse.ArgumentParser())
    return p.parse_args(list(argv))


def test_defaults_are_the_responsive_recipe_with_the_guard_on():
    a = _parse()
    assert (a.n_envs, a.n_steps, a.published_recipe, a.response_guard) == (16, 256, False, True)
    assert RG.resolve_recipe(a) == (16, 256)


def test_published_recipe_flag_restores_one_by_256(capsys):
    a = _parse("--published-recipe", "--n-envs", "8")
    assert RG.resolve_recipe(a) == (1, 256), "--published-recipe must win over --n-envs"
    assert "WARNING" in capsys.readouterr().out, "the thin recipe still warns"


def test_the_guard_can_be_switched_off():
    assert _parse("--no-response-guard").response_guard is False
    assert RG.guard(object(), "x", enabled=False) is None


@pytest.mark.parametrize("recipe,tag", [((1, 256), ""), ((16, 256), "_nenv16"),
                                        ((16, 64), "_nenv16_nsteps64"), ((4, 256), "_nenv4")])
def test_new_recipes_never_write_into_the_published_artifact(recipe, tag):
    assert RG.recipe_tag(*recipe) == tag
    assert RG.recipe_out_path("results/x.json", *recipe) == f"results/x{tag}.json"


def test_every_procgen_producer_is_on_the_shared_recipe():
    found = {os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "compare_*.py"))}
    assert found == set(PRODUCERS) | EXCLUDED, "a new compare_*.py must be classified here"
    for name in PRODUCERS:
        with open(os.path.join(ROOT, name), encoding="utf-8") as f:
            src = f.read()
        ast.parse(src)
        assert "add_recipe_args(parser)" in src, f"{name}: no --n-envs/--published-recipe CLI"
        assert "resolve_recipe(args)" in src, f"{name}: recipe not resolved"
        assert "guard(model," in src, f"{name}: no response guard"
        assert "warn_thin_recipe()" not in src, f"{name}: still announces the single-stream default"
        assert "for rank in range(" in src, f"{name}: no per-rank level streams"


def test_train_functions_default_to_sixteen_envs():
    import compare_suite as CS
    import compare_new_archs as CNA
    import compare_budget_scaling as CBS
    import compare_suite_retrain as CSR
    for fn in (CS.train_one, CNA.train_one, CBS.train_one, CSR.train_one):
        p = inspect.signature(fn).parameters
        assert (p["n_envs"].default, p["n_steps"].default) == (16, 256), fn
