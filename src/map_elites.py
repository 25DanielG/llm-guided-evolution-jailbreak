"""Behavior descriptors used to place prompt strategies on a MAP-Elites grid.

Generated prompt modules are treated as data: this module parses their source
with :mod:`ast` and never imports or executes them.
"""

from __future__ import annotations

import ast
import copy
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Iterable, NamedTuple


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


@dataclass(frozen=True)
class Elite:
    """One evaluated prompt retained by the archive."""

    gene_id: str
    descriptor: PromptDescriptor
    quality: tuple[float, ...]
    fitness: tuple[float, ...] = ()
    individual: Any = field(default=None, compare=False, repr=False)


class ArchiveDecision(str, Enum):
    """Outcome of attempting to add a candidate to the archive."""

    INSERTED = "inserted"
    REPLACED = "replaced"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ArchiveUpdate:
    """Result metadata used by later reporting and metrics code."""

    decision: ArchiveDecision
    descriptor: PromptDescriptor
    gene_id: str
    previous_gene_id: str | None = None


class MapElitesArchive:
    """Keep the highest-quality evaluated prompt in every behavior cell.

    Quality is compared lexicographically, so callers control the objective
    order. For example, ``(judge_score, hard_score, -token_cost)`` maximizes
    success first and uses lower token cost as a later tie-breaker.
    """

    def __init__(self, cells: Iterable[PromptDescriptor] = MAP_CELLS):
        ordered_cells = tuple(cells)
        if not ordered_cells:
            raise ValueError("a MAP-Elites archive needs at least one cell")
        if len(set(ordered_cells)) != len(ordered_cells):
            raise ValueError("archive cells must be unique")
        self._cells = ordered_cells
        self._cell_set = frozenset(ordered_cells)
        self._elites: dict[PromptDescriptor, Elite] = {}
        self._quality_dimensions: int | None = None

    @property
    def capacity(self) -> int:
        return len(self._cells)

    @property
    def coverage(self) -> float:
        return len(self) / self.capacity

    @property
    def occupied_cells(self) -> tuple[PromptDescriptor, ...]:
        return tuple(cell for cell in self._cells if cell in self._elites)

    def __len__(self) -> int:
        return len(self._elites)

    def __contains__(self, descriptor: object) -> bool:
        return descriptor in self._elites

    def get(self, descriptor: PromptDescriptor) -> Elite | None:
        """Return an isolated copy so callers cannot mutate the archive."""
        elite = self._elites.get(descriptor)
        return copy.deepcopy(elite) if elite is not None else None

    def elites(self) -> tuple[Elite, ...]:
        """Return isolated elite copies in deterministic cell order."""
        return tuple(
            copy.deepcopy(self._elites[cell])
            for cell in self._cells
            if cell in self._elites
        )

    def add(
        self,
        gene_id: str,
        descriptor: PromptDescriptor,
        quality: Iterable[float],
        *,
        fitness: Iterable[float] = (),
        individual: Any = None,
    ) -> ArchiveUpdate:
        """Insert, replace, or reject one evaluated candidate."""
        if descriptor not in self._cell_set:
            raise ValueError(f"descriptor is not part of this archive: {descriptor!r}")
        if not gene_id:
            raise ValueError("gene_id must not be empty")

        quality_tuple = tuple(float(value) for value in quality)
        if not quality_tuple:
            raise ValueError("quality must contain at least one objective")
        if not all(math.isfinite(value) for value in quality_tuple):
            raise ValueError("quality values must be finite")
        if self._quality_dimensions is None:
            self._quality_dimensions = len(quality_tuple)
        elif len(quality_tuple) != self._quality_dimensions:
            raise ValueError(
                f"quality has {len(quality_tuple)} objectives; "
                f"archive expects {self._quality_dimensions}"
            )

        incumbent = self._elites.get(descriptor)
        if incumbent is not None and quality_tuple <= incumbent.quality:
            return ArchiveUpdate(
                ArchiveDecision.REJECTED,
                descriptor,
                gene_id,
                previous_gene_id=incumbent.gene_id,
            )

        elite = Elite(
            gene_id=gene_id,
            descriptor=descriptor,
            quality=quality_tuple,
            fitness=tuple(float(value) for value in fitness),
            individual=copy.deepcopy(individual),
        )
        self._elites[descriptor] = elite
        decision = (
            ArchiveDecision.REPLACED
            if incumbent is not None
            else ArchiveDecision.INSERTED
        )
        return ArchiveUpdate(
            decision,
            descriptor,
            gene_id,
            previous_gene_id=incumbent.gene_id if incumbent is not None else None,
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
