"""Deterministic inference fixtures; actual mutation/crossover logic stays under test."""
import ast

import pytest

pytestmark = pytest.mark.mock_inference


@pytest.mark.parametrize('reply, expected', [
    ("trait_persona = '<persona>new</persona>'", '<persona>new</persona>'),
    ("trait_style = '<style>wrong slot</style>'", '<persona>old</persona>'),
    ('not valid python!', '<persona>old</persona>'),
    (None, '<persona>old</persona>'),
])
def test_mutation(reply, expected, tmp_path, operators, mock_codegen):
    mutation, _ = operators
    parent = tmp_path / 'parent.py'
    parent.write_text("HEADER = True\n# --OPTION--\ntrait_persona = '<persona>old</persona>'\n")
    child = tmp_path / 'new-directory' / 'child.py'
    generator = mock_codegen(reply)
    mutation.augment_network(str(parent), str(child), 'tests/operators/test_prompt_1.txt', llm_model='llama3')
    parts = mutation.split_file(child)
    assert parts[0] == 'HEADER = True\n'
    assert mutation._trait_assign_name(parts[1].strip()) == ('trait_persona', expected)
    generator.assert_called_once()
    ast.parse(child.read_text())


@pytest.mark.parametrize('reply, expected', [
    ("trait_persona = '<persona>combined</persona>'", '<persona>combined</persona>'),
    ("trait_style = '<style>wrong slot</style>'", '<persona>left</persona>'),
    (None, '<persona>left</persona>'),
])
def test_trait_crossover(reply, expected, tmp_path, operators, mock_codegen):
    mutation, crossover = operators
    x, y = tmp_path / 'x.py', tmp_path / 'y.py'
    x.write_text("# --OPTION--\ntrait_persona = '<persona>left</persona>'\n")
    y.write_text("# --OPTION--\ntrait_persona = '<persona>right</persona>'\n")
    out = tmp_path / 'crossed' / 'child.py'
    out.parent.mkdir()
    generator = mock_codegen(reply)
    crossover.augment_network(str(x), str(y), str(out), llm_model='llama3')
    assert mutation._trait_assign_name(crossover.split_file(out)[1].strip()) == ('trait_persona', expected)
    generator.assert_called_once()


def test_numeric_mutation_and_crossover_do_not_call_inference(tmp_path, operators, mock_codegen, monkeypatch):
    mutation, crossover = operators
    generator = mock_codegen('should never be used')
    monkeypatch.setattr(mutation.np.random, 'uniform', lambda *args: 0.0)
    x, y = tmp_path / 'x.py', tmp_path / 'y.py'
    x.write_text('''# --OPTION--
trait_decoding = '<decoding temperature="0.4" top_p="0.6"/>'
''')
    y.write_text('''# --OPTION--
trait_decoding = '<decoding temperature="0.8" top_p="1.0"/>'
''')
    out = tmp_path / 'out.py'
    mutation.augment_network(str(x), str(out), 'unused.txt')
    assert 'temperature="0.40"' in out.read_text()
    crossover.augment_network(str(x), str(y), str(out))
    assert 'temperature="0.60"' in out.read_text()
    assert 'top_p="0.80"' in out.read_text()
    generator.assert_not_called()
