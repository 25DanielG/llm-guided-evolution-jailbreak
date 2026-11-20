from src import llm_mutation
from src.map_elites import classify_turn_depth, extract_trait_assignments


def test_structural_turn_mutation_moves_to_selected_depth(monkeypatch):
    source = (
        "trait_turns = '<turns>"
        '<turn role="user">first</turn>'
        '<turn role="assistant">second</turn>'
        "</turns>'"
    )
    monkeypatch.setattr(
        llm_mutation.np.random,
        "choice",
        lambda choices: "zero",
    )

    mutated = llm_mutation._mutate_turn_depth_structurally(source)

    trait = extract_trait_assignments(mutated)["trait_turns"]
    assert classify_turn_depth(trait) == "zero"
