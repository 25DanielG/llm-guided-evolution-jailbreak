"""Behavior descriptors used to place prompt strategies on a MAP-Elites grid.

Generated prompt modules are treated as data: this module parses their source
with :mod:`ast` and never imports or executes them.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Final, NamedTuple


FORMAT_STRATEGIES: Final = (
    "freeform",
    "base64",
    "rot13",
    "homoglyph",
    "json_steps",
    "transcript",
)
TURN_DEPTHS: Final = ("zero", "one_or_two", "three_plus")

_TRAIT_NAMES: Final = {"trait_format", "trait_turns"}
_FORMAT_TYPE_RE: Final = re.compile(
    r"<format\b[^>]*\btype\s*=\s*(['\"])(.*?)\1",
    re.IGNORECASE | re.DOTALL,
)
_TURN_RE: Final = re.compile(r"<turn(?=\s|>)", re.IGNORECASE)


class PromptDescriptor(NamedTuple):
    """A prompt's stable location in the initial 6 x 3 behavior map."""

    format_strategy: str
    conversation_depth: str


MAP_CELLS: Final = tuple(
    PromptDescriptor(format_strategy, conversation_depth)
    for format_strategy in FORMAT_STRATEGIES
    for conversation_depth in TURN_DEPTHS
)


def extract_trait_assignments(source: str) -> dict[str, str]:
    """Extract the MAP-Elites trait strings without executing prompt code."""
    tree = ast.parse(source)
    traits: dict[str, str] = {}

    for node in tree.body:
        name = None
        value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                name, value = target.id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, value = node.target.id, node.value

        if name not in _TRAIT_NAMES or value is None:
            continue
        try:
            literal = ast.literal_eval(value)
        except (ValueError, TypeError):
            continue
        if isinstance(literal, str):
            traits[name] = literal

    return traits


def classify_format(trait_format: str) -> str:
    """Return a renderer-compatible format bin.

    Missing and unknown format types behave like freeform in the prompt
    renderer, so they share its archive cell.
    """
    match = _FORMAT_TYPE_RE.search(trait_format or "")
    format_type = match.group(2).strip().lower() if match else "freeform"
    return format_type if format_type in FORMAT_STRATEGIES else "freeform"


def classify_turn_depth(trait_turns: str) -> str:
    """Bucket the number of prior conversation turns into three stable bins."""
    turn_count = len(_TURN_RE.findall(trait_turns or ""))
    if turn_count == 0:
        return "zero"
    if turn_count <= 2:
        return "one_or_two"
    return "three_plus"


def descriptor_from_source(source: str) -> PromptDescriptor:
    """Calculate a MAP-Elites descriptor from a prompt module's source."""
    traits = extract_trait_assignments(source)
    return PromptDescriptor(
        classify_format(traits.get("trait_format", "")),
        classify_turn_depth(traits.get("trait_turns", "")),
    )


def descriptor_from_file(path: str | Path) -> PromptDescriptor:
    """Calculate a MAP-Elites descriptor from a UTF-8 prompt module."""
    return descriptor_from_source(Path(path).read_text(encoding="utf-8"))
