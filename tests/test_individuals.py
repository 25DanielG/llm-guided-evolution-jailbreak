"""
LLMGE Individual Evaluation Tests

Tests eval.py on 12 frozen individuals from a real LLMGE run.
Each individual is evaluated and checked for valid output.

No LLM server needed — just evaluates pre-generated models.

Run:
    uv run pytest tests/test_individuals.py -v
"""
import os
import ast
import subprocess
from pathlib import Path
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOTA = os.path.join(ROOT, 'sota', 'Titanic')
EVAL_SCRIPT = os.path.join(SOTA, 'eval.py')
CROSSOVER_SCRIPT = os.path.join(ROOT, 'src', 'llm_crossover.py')
INDIVIDUALS_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'individuals')
RESULTS_DIR = os.path.join(SOTA, 'test_results')

# Total samples in the validation set (used for upper bound check)
MAX_SAMPLES = 179  # based on processed_train.csv 80/20 split

# Individuals that should pass eval.py
VALID_INDIVIDUALS = [
    'xXx04jaaxCjDADdZfQejz3gbNFD',
    'xXx05DUkH2xBIGogZRAFJ8nnJHE',
    'xXx07Gk5iNRGngQLpVQwbMDAAS1',
    'xXx09ejihbvx7yUX9JYxtIVPGDs',
    'xXx0LFCBTNE4Wx7KUjfsG1nZc4I',
    'xXx0MtgG8LTm4mSVWscg4s97g0W',
    'xXx0OCDKQ3u88o9xhlAyHpEcfhM',
    'xXx0T6XRn6JDQPVLadZ4zBFlTd0',
    'xXx0UVvEW16rb3wrfuDSAemYblI',
]

# Individuals expected to fail eval.py (kept to verify eval still catches them)
INVALID_INDIVIDUALS = {
    'xXx061oJQctwH2mBMMPeIwd7sCk': 'SyntaxError',   # missing bracket
    'xXx04PqApRQMPw55qx6cZsUfVSi': 'TypeError',       # keywords must be strings
    'xXx0WcxFNILJIkWzWO9MsEGL0Jx': 'TypeError',      # int instead of list in GridSearchCV
}

ALL_INDIVIDUALS = VALID_INDIVIDUALS + list(INVALID_INDIVIDUALS.keys())

# Individuals that produce valid Python after mutation (subset of VALID_INDIVIDUALS)
VALID_MUTATE = [
    'xXx04jaaxCjDADdZfQejz3gbNFD',
    'xXx07Gk5iNRGngQLpVQwbMDAAS1',
    'xXx09ejihbvx7yUX9JYxtIVPGDs',
    'xXx0LFCBTNE4Wx7KUjfsG1nZc4I',
    'xXx0OCDKQ3u88o9xhlAyHpEcfhM',
]


# ── 1. Each individual is valid Python with a Model class. Ensures models haven't changed and all can be evaluated. ──────────────────

@pytest.mark.parametrize('gene_id', VALID_INDIVIDUALS)
def test_individual_is_valid_python(gene_id):
    """Frozen individual parses as Python with Model(fit, predict)."""
    path = os.path.join(INDIVIDUALS_DIR, f'model_{gene_id}.py')
    with open(path) as f:
        tree = ast.parse(f.read())
    classes = [n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == 'Model']
    assert len(classes) >= 1, f"{gene_id}: no Model class"
    methods = {n.name for n in ast.walk(classes[0])
               if isinstance(n, ast.FunctionDef)}
    assert 'fit' in methods, f"{gene_id}: Model missing fit()"
    assert 'predict' in methods, f"{gene_id}: Model missing predict()"


# ── 2. Valid individuals evaluate and produce valid results ────────────────

@pytest.mark.parametrize('gene_id', VALID_INDIVIDUALS)
def test_individual_evaluates(gene_id, titanic_eval):
    """eval.py runs on the individual and produces bounded FP,FN."""
    model_name = f'model_{gene_id}'

    sandbox, evaluate = titanic_eval
    result = evaluate(model_name, INDIVIDUALS_DIR)
    assert result.returncode == 0, result.stderr

    assert 'job done' in result.stdout.lower(), \
        f"{gene_id}: 'job done' not in output"

    # Check results file was written
    results_file = sandbox / 'results' / f'{gene_id}_results.csv'
    assert os.path.isfile(results_file), \
        f"{gene_id}: results file not created"

    # Validate format and bounds
    with open(results_file) as f:
        lines = f.readlines()
    assert lines[0].strip() == 'fp_count,fn_count'
    assert len(lines) == 2
    content = lines[-1].strip() 
    parts = content.split(',')
    assert len(parts) == 2, f"{gene_id}: expected 'FP,FN', got '{content}'"

    fp, fn = float(parts[0].strip()), float(parts[1].strip())
    assert 0 <= fp <= MAX_SAMPLES, f"{gene_id}: FP={fp} out of bounds [0, {MAX_SAMPLES}]"
    assert 0 <= fn <= MAX_SAMPLES, f"{gene_id}: FN={fn} out of bounds [0, {MAX_SAMPLES}]"


# ── 3. Known-bad individuals should fail eval.py ──────────────────────────

@pytest.mark.parametrize('gene_id', INVALID_INDIVIDUALS.keys())
def test_invalid_individual_fails_eval(gene_id, titanic_eval):
    """eval.py should fail on known-bad individuals with the expected error."""
    model_name = f'model_{gene_id}'
    expected_error = INVALID_INDIVIDUALS[gene_id]

    _, evaluate = titanic_eval
    result = evaluate(model_name, INDIVIDUALS_DIR)

    assert result.returncode != 0, \
        f"{gene_id}: expected failure but eval.py succeeded"
    assert expected_error in result.stderr, \
        f"{gene_id}: expected '{expected_error}' in stderr, got:\n{result.stderr[-500:]}"


# ── 4. Mocked inference; real mutation assembly ──────────────────

@pytest.mark.mock_inference
@pytest.mark.parametrize('gene_id', VALID_MUTATE)
def test_mutation_produces_valid_model(gene_id, tmp_path, monkeypatch, operators, mock_codegen):
    mutation, _ = operators
    monkeypatch.setattr(mutation, "TRAIT_VALIDATION", None)
    monkeypatch.setattr(mutation.np.random, "randint", lambda *args: 1)
    parent = Path(INDIVIDUALS_DIR) / f'model_{gene_id}.py'
    parts = mutation.split_file(parent)
    changed = parts[1].strip() + '\nCI_MUTATION_FIXTURE = True'
    generator = mock_codegen(changed)
    child = tmp_path / 'nested' / f'mutated_{gene_id}.py'
    mutation.augment_network(str(parent), str(child), 'tests/operators/test_prompt_1.txt', llm_model='llama3')
    actual = mutation.split_file(child)
    assert actual[1].strip() == changed
    assert actual[0] == parts[0] and actual[2:] == parts[2:]
    ast.parse(child.read_text())
    generator.assert_called_once()


# ── 5. Mocked inference; real crossover assembly ─────────────────

# Verified crossover pairs from partner testing
CROSSOVER_PAIRS = [
    ('xXx0MtgG8LTm4mSVWscg4s97g0W', 'xXx09ejihbvx7yUX9JYxtIVPGDs'),
    ('xXx0MtgG8LTm4mSVWscg4s97g0W', 'xXx0UVvEW16rb3wrfuDSAemYblI'),
    ('xXx0LFCBTNE4Wx7KUjfsG1nZc4I', 'xXx07Gk5iNRGngQLpVQwbMDAAS1'),
    ('xXx0LFCBTNE4Wx7KUjfsG1nZc4I', 'xXx09ejihbvx7yUX9JYxtIVPGDs'),
    ('xXx0LFCBTNE4Wx7KUjfsG1nZc4I', 'xXx04jaaxCjDADdZfQejz3gbNFD'),
]


@pytest.mark.mock_inference
@pytest.mark.parametrize('gene_id_x, gene_id_y', CROSSOVER_PAIRS)
def test_crossover_function(gene_id_x, gene_id_y, tmp_path, monkeypatch, operators, mock_codegen):
    _, crossover = operators
    monkeypatch.setattr(crossover.random, 'shuffle', lambda parts: None)
    # Frozen Titanic code has class segments rather than Jailbreak trait tags.
    import llm_mutation
    monkeypatch.setattr(llm_mutation, 'TRAIT_VALIDATION', None)
    x = Path(INDIVIDUALS_DIR) / f'model_{gene_id_x}.py'
    y = Path(INDIVIDUALS_DIR) / f'model_{gene_id_y}.py'
    parts_x, parts_y = crossover.split_file(x), crossover.split_file(y)
    idx = next((i for i in range(1, min(len(parts_x), len(parts_y)))
                if parts_x[i].strip() != parts_y[i].strip()), 1)
    changed = parts_x[idx].strip() + '\nCI_CROSSOVER_FIXTURE = True'
    generator = mock_codegen(changed)
    output = tmp_path / 'crossed' / f'model_crossed_{gene_id_x}_{gene_id_y}.py'
    output.parent.mkdir()
    crossover.augment_network(str(x), str(y), str(output), llm_model='llama3')
    actual = crossover.split_file(output)
    assert actual[idx].strip() == changed
    assert actual[1:idx] == parts_x[1:idx] and actual[idx+1:] == parts_x[idx+1:]
    ast.parse(output.read_text())
    generator.assert_called_once()


# ── 6. Mocked generation from a trait seed ────────

@pytest.mark.mock_inference
def test_create_individual(tmp_path, monkeypatch, operators, mock_codegen):
    mutation, _ = operators
    parent = tmp_path / 'seed.py'
    parent.write_text('# --OPTION--\ntrait_persona = \'<persona>old</persona>\'\n')
    changed = "trait_persona = '<persona>new fixture</persona>'"
    generator = mock_codegen(changed)
    output = tmp_path / 'model_test.py'
    mutation.augment_network(str(parent), str(output), 'tests/operators/test_prompt_1.txt', llm_model='llama3')
    assert mutation._trait_assign_name(mutation.split_file(output)[1].strip()) == ('trait_persona', '<persona>new fixture</persona>')
    generator.assert_called_once()
