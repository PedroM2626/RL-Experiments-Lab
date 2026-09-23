"""Provenance tests for results/legacy_records.json.

The runs behind README sections 3.7 and 7 no longer have raw logs, so the file is
their only machine-readable record. These tests fail if it drifts away from the
published artifacts that were computed from it.
"""

import json
import os

import numpy as np

T_CRIT_95_DF4 = 2.776  # must match scorecard_analysis.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(*rel):
    with open(os.path.join(ROOT, *rel), encoding='utf-8') as f:
        return json.load(f)


def test_legacy_records_declare_provenance():
    prov = load('results', 'legacy_records.json')['_provenance']
    assert prov['retrainable_from_source'] is False
    assert 'section 7' in prov['note'].lower()


def test_per_seed_arrays_reproduce_published_scorecard():
    per_seed = load('results', 'legacy_records.json')['per_seed']
    published = load('results', 'scorecard.json')['ci_effect_size']
    assert per_seed, "no legacy per-seed records"
    for key, vals in per_seed.items():
        arr = np.asarray(vals, dtype=float)
        assert arr.size == 5, f"{key}: expected 5 seeds, got {arr.size}"
        half = T_CRIT_95_DF4 * arr.std(ddof=1) / np.sqrt(arr.size)
        expected = published[key]
        assert round(float(arr.mean()), 3) == expected['mean'], key
        assert round(float(arr.std(ddof=1)), 3) == expected['std'], key
        assert [round(float(arr.mean() - half), 3), round(float(arr.mean() + half), 3)] == expected['ci95'], key


def test_global_ranking_matches_retrain_analysis():
    legacy = load('results', 'legacy_records.json')['global_ranking_top10_3_7']
    assert legacy == load('results', 'retrain_analysis.json')['global_old']
