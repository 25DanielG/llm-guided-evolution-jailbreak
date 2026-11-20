from pathlib import Path

import pytest

from src.map_elites import (
    FORMAT_STRATEGIES,
    MAP_CELLS,
    TURN_DEPTHS,
    PromptDescriptor,
    classify_format,
    classify_turn_depth,
    coerce_turn_depth,
    descriptor_from_file,
    descriptor_from_source,
    extract_trait_assignments,
    rewrite_turn_depth_source,
)


def test_initial_map_contains_18_unique_cells():
    assert len(MAP_CELLS) == 18
    assert len(set(MAP_CELLS)) == 18


@pytest.mark.parametrize("format_type", FORMAT_STRATEGIES)
def test_classify_each_supported_format(format_type):
    trait = f'<format type="{format_type}">{{ASK}}</format>'
    assert classify_format(trait) == format_type


@pytest.mark.parametrize(
    ("turns", "expected"),
    [
        ([], "zero"),
        ([('user', 'dangling')], "zero"),
        ([('assistant', 'leading')], "zero"),
        ([('user', 'first'), ('assistant', 'second')], "one_or_two"),
        (
            [
                ('user', 'first'),
                ('assistant', 'second'),
                ('user', 'third'),
                ('assistant', 'fourth'),
            ],
            "three_plus",
        ),
    ],
)
def test_classify_turn_depth_matches_rendered_prior_messages(turns, expected):
    trait = "<turns>" + "".join(
        f'<turn role="{role}">{text}</turn>' for role, text in turns
    ) + "</turns>"
    assert classify_turn_depth(trait) == expected


@pytest.mark.parametrize("target_depth", TURN_DEPTHS)
def test_coerce_turn_depth_makes_every_bin_reachable(target_depth):
    seed = (
        '<turns><turn role="user">first</turn>'
        '<turn role="assistant">second</turn></turns>'
    )

    coerced = coerce_turn_depth(seed, target_depth)

    assert classify_turn_depth(coerced) == target_depth


def test_rewrite_turn_depth_source_preserves_other_traits():
    source = '''
trait_format = """<format type="base64">{ASK}</format>"""
trait_turns = """<turns>
<turn role="user">first</turn>
<turn role="assistant">second</turn>
</turns>"""
trait_style = "<style>unchanged</style>"
'''

    rewritten = rewrite_turn_depth_source(source, "three_plus")

    assert descriptor_from_source(rewritten) == PromptDescriptor(
        "base64", "three_plus"
    )
    assert 'trait_style = "<style>unchanged</style>"' in rewritten


def test_descriptor_extracts_literal_traits_without_executing_code():
    source = '''
raise RuntimeError("this module must not be executed")
trait_format = """<format type="base64">{ASK}</format>"""
trait_turns = """<turns>
<turn role="user">first</turn>
<turn role="assistant">second</turn>
<turn role="user">third</turn>
<turn role="assistant">fourth</turn>
</turns>"""
'''

    assert extract_trait_assignments(source) == {
        "trait_format": '<format type="base64">{ASK}</format>',
        "trait_turns": '<turns>\n<turn role="user">first</turn>\n'
        '<turn role="assistant">second</turn>\n<turn role="user">third</turn>\n'
        '<turn role="assistant">fourth</turn>\n</turns>',
    }
    assert descriptor_from_source(source) == PromptDescriptor("base64", "three_plus")


def test_missing_or_unknown_traits_use_renderer_compatible_defaults():
    source = '''
trait_format = "<format type='unknown'>{ASK}</format>"
trait_turns = ""
'''
    descriptor = descriptor_from_source(source)

    assert descriptor == PromptDescriptor("freeform", "zero")
    assert descriptor.format_strategy in FORMAT_STRATEGIES
    assert descriptor.conversation_depth in TURN_DEPTHS


def test_descriptor_from_seed_prompt_file():
    root = Path(__file__).resolve().parents[1]
    descriptor = descriptor_from_file(root / "sota" / "Jailbreak" / "prompt.py")

    assert descriptor == PromptDescriptor("transcript", "one_or_two")
