import pickle

import pytest

from src.map_elites import (
    ArchiveDecision,
    MAP_CELLS,
    MapElitesArchive,
    PromptDescriptor,
)


TRANSCRIPT_ZERO = PromptDescriptor("transcript", "zero")
BASE64_DEEP = PromptDescriptor("base64", "three_plus")


def test_empty_archive_has_zero_coverage():
    archive = MapElitesArchive()

    assert len(archive) == 0
    assert archive.capacity == 18
    assert archive.coverage == 0.0
    assert archive.occupied_cells == ()


def test_candidate_is_inserted_into_empty_cell():
    archive = MapElitesArchive()

    update = archive.add(
        "gene-a",
        TRANSCRIPT_ZERO,
        quality=(0.7, -200),
        fitness=(0.7, 0.5, 200),
        individual=["gene-a"],
    )

    assert update.decision is ArchiveDecision.INSERTED
    assert update.previous_gene_id is None
    assert archive.get(TRANSCRIPT_ZERO).gene_id == "gene-a"
    assert archive.coverage == pytest.approx(1 / len(MAP_CELLS))


def test_better_candidate_replaces_cell_elite():
    archive = MapElitesArchive()
    archive.add("gene-a", TRANSCRIPT_ZERO, quality=(0.7, 0.3, -200))

    update = archive.add("gene-b", TRANSCRIPT_ZERO, quality=(0.7, 0.4, -300))

    assert update.decision is ArchiveDecision.REPLACED
    assert update.previous_gene_id == "gene-a"
    assert archive.get(TRANSCRIPT_ZERO).gene_id == "gene-b"


@pytest.mark.parametrize("quality", [(0.6, -100), (0.7, -200)])
def test_worse_or_equal_candidate_is_rejected(quality):
    archive = MapElitesArchive()
    archive.add("gene-a", TRANSCRIPT_ZERO, quality=(0.7, -200))

    update = archive.add("gene-b", TRANSCRIPT_ZERO, quality=quality)

    assert update.decision is ArchiveDecision.REJECTED
    assert update.previous_gene_id == "gene-a"
    assert archive.get(TRANSCRIPT_ZERO).gene_id == "gene-a"


def test_different_behavior_cells_keep_separate_elites():
    archive = MapElitesArchive()
    archive.add("gene-a", TRANSCRIPT_ZERO, quality=(0.7, -200))
    archive.add("gene-b", BASE64_DEEP, quality=(0.4, -100))

    assert len(archive) == 2
    assert archive.occupied_cells == (BASE64_DEEP, TRANSCRIPT_ZERO)
    assert {elite.gene_id for elite in archive.elites()} == {"gene-a", "gene-b"}


def test_archive_copies_individual_on_insert_and_read():
    archive = MapElitesArchive()
    individual = ["gene-a", {"mutable": True}]
    archive.add("gene-a", TRANSCRIPT_ZERO, quality=(0.7,), individual=individual)

    individual[0] = "changed-outside"
    returned = archive.get(TRANSCRIPT_ZERO)
    returned.individual[0] = "changed-copy"

    assert archive.get(TRANSCRIPT_ZERO).individual[0] == "gene-a"


def test_invalid_cell_or_quality_is_rejected():
    archive = MapElitesArchive()

    with pytest.raises(ValueError, match="not part"):
        archive.add("gene-a", PromptDescriptor("unknown", "zero"), quality=(0.5,))
    with pytest.raises(ValueError, match="finite"):
        archive.add("gene-a", TRANSCRIPT_ZERO, quality=(float("nan"),))
    with pytest.raises(ValueError, match="at least one"):
        archive.add("gene-a", TRANSCRIPT_ZERO, quality=())


def test_all_candidates_use_same_number_of_quality_objectives():
    archive = MapElitesArchive()
    archive.add("gene-a", TRANSCRIPT_ZERO, quality=(0.7, -200))

    with pytest.raises(ValueError, match="expects 2"):
        archive.add("gene-b", BASE64_DEEP, quality=(0.8,))


def test_archive_can_be_checkpointed_with_pickle():
    archive = MapElitesArchive()
    archive.add("gene-a", TRANSCRIPT_ZERO, quality=(0.7, -200), individual=["gene-a"])

    restored = pickle.loads(pickle.dumps(archive))

    assert restored.capacity == 18
    assert restored.get(TRANSCRIPT_ZERO).gene_id == "gene-a"
    assert restored.get(TRANSCRIPT_ZERO).individual == ["gene-a"]
