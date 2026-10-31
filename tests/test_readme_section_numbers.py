"""Gate the §15.4 subsection numbering.

The port section drifted into two headings numbered 15.4.3, listed out of order
(15.4.1, .3, .2, .3, .4...), and cross-references that could only be resolved by guessing
which 15.4.3 the author meant. Fixed on 11/10/2026 by renumbering to document order and
updating each reference by its content, not its number.

These two assertions are what keep it fixed: headings must be unique and ascending in the
order they appear, and every reference to 15.4.<n> anywhere in the tracked source must name
a section that exists. A dangling "§15.4.7" is worse than an ugly number, because it reads
as authoritative.
"""
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")
HEADING = re.compile(r"^#### (15\.4\.(\d+))\.", re.M)
REFERENCE = re.compile(r"\b15\.4\.(\d+)\b")


def _headings():
    with open(README, encoding="utf-8") as f:
        return [(int(m.group(2)), m.start()) for m in HEADING.finditer(f.read())]


def _tracked_files():
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv", "node_modules"}]
        for name in files:
            if name.endswith((".py", ".md")):
                yield os.path.join(base, name)


def test_subsections_are_unique_and_in_document_order():
    nums = [n for n, _ in _headings()]
    assert nums, "no 15.4.x headings found -- the pattern and the README have drifted apart"
    dupes = {n for n in nums if nums.count(n) > 1}
    assert not dupes, f"duplicated 15.4.x subsection numbers: {sorted(dupes)}"
    assert nums == sorted(nums), f"15.4.x headings are out of order in the document: {nums}"
    assert nums == list(range(1, len(nums) + 1)), f"15.4.x numbering has a gap: {nums}"


@pytest.mark.parametrize("suffix", [".py", ".md"])
def test_every_reference_names_an_existing_subsection(suffix):
    known = {n for n, _ in _headings()}
    dangling = []
    for path in _tracked_files():
        if not path.endswith(suffix):
            continue
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        for m in REFERENCE.finditer(text):
            if int(m.group(1)) not in known:
                line = text[:m.start()].count("\n") + 1
                dangling.append(f"{rel}:{line} -> 15.4.{m.group(1)}")
    assert not dangling, f"references to non-existent subsections: {dangling}"


@pytest.mark.parametrize("pattern", [r"^### 3\.6\.(\d+)\.", r"^### 15\.(\d+)\."])
def test_subsections_appear_in_numeric_order(pattern):
    """3.6.8 once sat after 3.6.12, and 15.3/15.2 after 15.4 -- fixed 30/10/2026."""
    with open(README, encoding="utf-8") as f:
        nums = [int(m.group(1)) for m in re.finditer(pattern, f.read(), re.M)]
    assert nums and nums == sorted(nums), f"subsections out of document order: {nums}"


def test_readme_has_no_tab_characters():
    """A macro eaten by a shell turns \text into <TAB>ext, and KaTeX renders the gap silently."""
    with open(README, encoding="utf-8") as f:
        text = f.read()
    bad = [f"line {i}" for i, line in enumerate(text.splitlines(), 1) if "\t" in line]
    assert not bad, f"README contains TAB characters (collapsed LaTeX macros?): {bad[:5]}"


def test_readme_math_delimiters_are_balanced():
    """An unpaired $ does not fail a test -- it silently breaks the rendered page.

    Twice during 11/10/2026 a code span was closed with a dollar instead of a backtick
    (`16$ ...), which GitHub's tokenizer renders as "Unable to render expression." or, worse,
    as italic prose. Code spans are stripped first: a shell prompt or a currency sign inside
    backticks is not a math delimiter. Display math ($$) is even, so it passes untouched.
    """
    with open(README, encoding="utf-8") as f:
        lines = f.read().splitlines()
    bad = []
    for i, line in enumerate(lines, 1):
        outside = re.sub(r"`[^`]*`", "", line)
        if outside.count("$") % 2:
            bad.append(f"line {i}: {line[:70]}")
    assert not bad, f"lines with an odd number of $ outside code spans: {bad[:5]}"


def test_no_math_has_parentheses_inside_and_outside():
    """GitHub's tokenizer skips exactly this shape; the page then shows the raw source.

    Measured on the rendered README (11/10/2026): 267 of 270 inline formulas were marked as
    math, and the two that were not -- ($V(s)$) and ($h_t = f(h_{t-1}, x_t)$) -- are the only
    two in the file whose content contains a parenthesis while also being wrapped in one.
    The other 31 ($...$) spans render, so the outer parens alone are not the problem.
    """
    with open(README, encoding="utf-8") as f:
        text = f.read()
    offenders = [m.group(0) for m in re.finditer(r"\(\$[^$\n]*[()][^$\n]*\$\)", text)]
    assert not offenders, f"($...$) with inner parens render as literal text: {offenders[:5]}"
