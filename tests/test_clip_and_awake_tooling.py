"""Tests for the small tooling written on 11/10/2026: the clip label band and keep_awake.

The clip renderer's job is that a viewer can tell what they are watching, so these test the
label rather than the pixels. The first version shipped a frame whose "NOT a trained agent"
disclaimer was cut off at the canvas edge -- the one sentence the clip exists to make sure a
reader sees. These tests are what catches that class of regression.

keep_awake is tested through an injected set_state, because the claim worth checking is not
"it waits" but "it never leaves a request behind": every exit path must clear the flag, and
without the seam there is no way to observe it.

The env-dependent helpers (probe_frames, cross_game_response) are not covered here -- they need
a live Procgen binary and a trained policy, and they were exercised for real when they produced
results/scope_probe_*.json.
"""
import os
import sys
import time

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from keep_awake import hold  # noqa: E402
from record_game_clips import GAMES, LINE_H, render, wrap  # noqa: E402

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
WIDTH = 256 - 12  # the renderer's own text width budget


def _draw():
    return ImageDraw.Draw(Image.new("RGB", (10, 10)))


def test_long_label_wraps_instead_of_being_cut_off():
    text = ("policy: uniform random (reference clip, NOT a trained agent)  -  "
            "frame 30/60  -  episode 2")
    d = _draw()
    lines = wrap(d, text, WIDTH)
    assert len(lines) > 1, "a label this wide must not stay on one line"
    for line in lines:
        assert d.textlength(line) <= WIDTH, f"wrapped line still too wide: {line!r}"
    assert " ".join(lines).split() == text.split(), "wrapping lost, gained or reordered words"


def test_short_label_stays_on_one_line():
    assert wrap(_draw(), "coinrun", WIDTH) == ["coinrun"]


def test_band_grows_to_fit_the_wrapped_label():
    obs = np.zeros((3, 64, 64), dtype=np.uint8)
    one = render(obs, ["a"])
    many = render(obs, ["x" * 400])
    assert one.width == 256 and many.width == 256
    assert one.height == 256 + 6 + LINE_H, "single-line band is not image + one text row"
    assert many.height > one.height, "the band did not grow for a longer label"
    # the disclaimer must survive: count how many rows the long label actually needed
    rows = (many.height - 256 - 6) // LINE_H
    assert rows >= 3


def test_every_game_the_study_names_has_a_clip_entry():
    assert set(GAMES) == {"coinrun", "starpilot", "dodgeball", "bossfight", "jumper",
                          "plunder", "maze", "heist"}
    for game, (mode, frames, fps) in GAMES.items():
        assert mode == "easy", f"{game}: clips must use the mode the study trains on"
        assert frames >= 40, f"{game}: {frames} frames is too short to show the game"
        assert fps >= 8, f"{game}: {fps} fps would be a slideshow"


def test_keep_awake_returns_as_soon_as_the_sentinel_exists(tmp_path):
    calls = []
    sentinel = tmp_path / "DONE"
    sentinel.write_text("")
    t0 = time.time()
    reason = hold(str(sentinel), max_hours=1, poll=1, set_state=calls.append)
    assert reason == "sentinel"
    assert time.time() - t0 < 5
    assert calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED, ES_CONTINUOUS]


def test_keep_awake_gives_up_at_its_deadline(tmp_path):
    calls = []
    reason = hold(str(tmp_path / "never"), max_hours=0, poll=1, set_state=calls.append)
    assert reason == "deadline"
    assert calls[-1] == ES_CONTINUOUS, "it left a system-required request behind"


def test_keep_awake_releases_the_flag_even_if_the_wait_raises(tmp_path, monkeypatch):
    """The finally clause is the only thing standing between a crash and a laptop that never sleeps."""
    calls = []
    real_exists = os.path.exists

    def exploding_exists(_path):
        raise RuntimeError("simulated failure while waiting")

    monkeypatch.setattr("keep_awake.os.path.exists", exploding_exists)
    with pytest.raises(RuntimeError):
        hold("whatever", max_hours=1, poll=1, set_state=calls.append)
    monkeypatch.undo()
    assert calls[-1] == ES_CONTINUOUS
    assert real_exists(__file__)  # the patch was scoped to the module, not the process
