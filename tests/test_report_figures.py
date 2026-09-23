"""Guards the figure provenance map in report_figures.py against the README.

A README that embeds a figure nobody can produce is how the section 4 images ended up
with no known producer; these assertions keep that from recurring silently.
"""

import os

from report_figures import PROVENANCE, REBUILDABLE, load_legacy, readme_figures

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_every_embedded_figure_exists_and_is_mapped():
    embedded = readme_figures()
    assert embedded, "no figures found in README section 4"
    for ref in embedded:
        assert os.path.exists(os.path.join(ROOT, ref)), f"README embeds a missing file: {ref}"
        assert os.path.basename(ref) in PROVENANCE, f"{ref} has no recorded producer"


def test_provenance_map_does_not_advertise_stale_figures():
    embedded = {os.path.basename(r) for r in readme_figures()}
    assert set(PROVENANCE) == embedded, (
        f"map-only: {sorted(set(PROVENANCE) - embedded)}; "
        f"README-only: {sorted(embedded - set(PROVENANCE))}")


def test_each_entry_names_a_real_script_and_data_source():
    for name, (script, produced, source) in PROVENANCE.items():
        assert script.endswith(".py") and name.endswith(".png"), name
        assert produced and source, f"{name} has no producer or data source"
        assert os.path.exists(os.path.join(ROOT, script)), f"{name}: {script} no longer exists"


def test_rebuildable_figures_have_their_per_seed_data():
    per_seed = load_legacy()
    for name, (_, _, keys) in REBUILDABLE.items():
        for k in keys:
            assert k in per_seed, f"{name} claims to rebuild but {k} is missing"
            assert len(per_seed[k]) == 5, f"{k}: expected 5 seeds"
