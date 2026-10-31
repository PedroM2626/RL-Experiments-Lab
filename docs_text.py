"""The study's documentation as one text: README.md followed by the docs/ pages.

The README was split into docs/ on 30/10/2026. The tests that gate prose on committed
artifacts (and report_figures.py --check) read the whole documentation through here,
so a sentence that moves between pages stays verified, and one that is deleted fails.
Pages are concatenated in section order.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DOC_FILES = [
    "README.md",
    "docs/results-archive.md",
    "docs/diagnostic-arc.md",
    "docs/figures.md",
    "docs/offline-rl-and-roadmap.md",
    "docs/independent-benchmarks.md",
    "docs/jax-port.md",
    "docs/marl-and-rliable.md",
    "docs/verification.md",
    "docs/symbolic-regression.md",
]


def read_docs():
    parts = []
    for rel in DOC_FILES:
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            parts.append(f.read())
    return "\n".join(parts)


def open_docs():
    """A drop-in for open(README, encoding="utf-8"): a readable file over every page."""
    return io.StringIO(read_docs())
