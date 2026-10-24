"""Gate the occlusion attribution artifact (README 3.6.12, section 4 figure).

The response measure says WHETHER a policy reads; the occlusion map says WHERE.
The committed summary must carry the three-rung separation the README paragraph
publishes: structured readers, a flat blind floor, and the within-game reader
strictly between -- so the paragraph's numbers cannot drift from the data.
"""
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(BASE, "results", "occlusion_summary.json")


def _rows():
    with open(ART, encoding="utf-8") as f:
        return json.load(f)["cells"]


def _by_cell():
    return {r["cell"]: r for r in _rows()}


def test_eight_key_cells_present():
    cells = _by_cell()
    for name in ("classic_bossfight", "classic_dodgeball", "mlp_scaled_bossfight",
                 "mlp_scaled_dodgeball", "vit_bossfight", "vit_dodgeball",
                 "impoola_bossfight", "mlp_dodgeball"):
        assert name in cells, name
        assert len(cells[name]["patch_map"]) == 4
        assert all(len(row) == 4 for row in cells[name]["patch_map"])


def test_readers_are_structured_blind_cells_are_flat():
    c = _by_cell()
    readers = [c["classic_bossfight"], c["classic_dodgeball"], c["mlp_scaled_bossfight"]]
    blind = [c["vit_bossfight"], c["vit_dodgeball"], c["impoola_bossfight"],
             c["mlp_dodgeball"]]
    assert all(r["occlusion_sensitivity"] > 0.01 for r in readers)
    assert all(b["occlusion_sensitivity"] < 0.001 for b in blind)
    # structure, not just magnitude: the reader's strongest site well above its mean
    for r in readers:
        assert r["occlusion_max_site"] > 2 * r["occlusion_sensitivity"], r["cell"]
    # every blind cell is at least an order of magnitude below every reader
    assert max(b["occlusion_sensitivity"] for b in blind) * 10 < \
        min(r["occlusion_sensitivity"] for r in readers)


def test_the_within_game_reader_sits_between_the_rungs():
    """3.6.10's category, made quantitative: mlp_scaled/dodgeball reads its screen
    (two orders of magnitude above the flat vit/impoola cells, and still above the
    least-flat blind cell) without generalizing across games (20-35x below readers)."""
    c = _by_cell()
    wg = c["mlp_scaled_dodgeball"]
    flat_min = min(c[n]["occlusion_sensitivity"] for n in
                   ("vit_bossfight", "vit_dodgeball", "impoola_bossfight"))
    blind_max = max(c[n]["occlusion_sensitivity"] for n in
                    ("vit_bossfight", "vit_dodgeball", "impoola_bossfight", "mlp_dodgeball"))
    reader_min = min(c[n]["occlusion_sensitivity"] for n in
                     ("classic_bossfight", "classic_dodgeball", "mlp_scaled_bossfight"))
    assert wg["occlusion_sensitivity"] > 50 * flat_min
    assert wg["occlusion_sensitivity"] > 5 * blind_max
    assert wg["occlusion_sensitivity"] < reader_min / 10
    # and its cross-game TV stays below the scorecard threshold, unlike the readers'
    assert wg["cross_game_tv"] < 0.1
    assert c["classic_bossfight"]["cross_game_tv"] >= 0.1


def test_readme_embeds_the_figure_and_the_paragraph():
    with open(os.path.join(BASE, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    assert "![occlusion attribution over the key cells](results/occlusion_maps.png)" in readme
    assert "two orders of magnitude above the flat" in readme
