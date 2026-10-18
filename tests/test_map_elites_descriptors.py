from pathlib import Path

import pytest

from src.map_elites import (
    FORMAT_STRATEGIES,
    MAP_CELLS,
    TURN_DEPTHS,
    PromptDescriptor,
    classify_format,
    classify_turn_depth,
    descriptor_from_file,
    descriptor_from_source,
    extract_trait_assignments,
)


def test_initial_map_contains_18_unique_cells():
    assert len(MAP_CELLS) == 18
    assert len(set(MAP_CELLS)) == 18


@pytest.mark.parametrize("format_type", FORMAT_STRATEGIES)
def test_classify_each_supported_format(format_type):
    trait = f'<format type="{format_type}">{{ASK}}</format>'
    assert classify_format(trait) == format_type


@pytest.mark.parametrize(
    ("turn_count", "expected"),
    [(0, "zero"), (1, "one_or_two"), (2, "one_or_two"), (3, "three_plus"), (6, "three_plus")],
)
def test_classify_turn_depth(turn_count, expected):
    turns = "<turns>" + '<turn role="user">text</turn>' * turn_count + "</turns>"
    assert classify_turn_depth(turns) == expected


def test_descriptor_extracts_literal_traits_without_executing_code():
    source = '''
raise RuntimeError("this module must not be executed")
trait_format = """<format type="base64">{ASK}</format>"""
trait_turns = """<turns>
<turn role="user">first</turn>
<turn role="assistant">second</turn>
<turn role="user">third</turn>
</turns>"""
'''

    assert extract_trait_assignments(source) == {
        "trait_format": '<format type="base64">{ASK}</format>',
        "trait_turns": '<turns>\n<turn role="user">first</turn>\n'
        '<turn role="assistant">second</turn>\n<turn role="user">third</turn>\n</turns>',
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
