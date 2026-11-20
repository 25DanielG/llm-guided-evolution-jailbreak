"""Genuine inference, excluded unless --run-live-inference is explicitly supplied.

These tests require an already authorized running /generate service. They do not
allocate GPUs, download weights, or substitute fixed model responses.
"""
from unittest.mock import Mock

import pytest
import llm_utils

pytestmark = [pytest.mark.live_inference, pytest.mark.usefixtures('live_llm')]


def test_generate_service_round_trip(live_llm):
    import requests
    result = requests.post(live_llm + '/generate', json={'prompt': 'Reply with a short greeting.',
                           'max_new_tokens': 128, 'temperature': 0.1, 'top_p': 0.9}, timeout=30)
    result.raise_for_status()
    assert result.json()['generated_text'].strip()
    assert result.json()['response_time_sec'] >= 0


@pytest.mark.parametrize('operation', ['mutation', 'crossover'])
def test_real_inference_changes_trait(operation, tmp_path, monkeypatch, operators):
    mutation, crossover = operators
    # Fix the trait position, not the answer. All model responses here are real.
    monkeypatch.setattr(mutation.np.random, 'randint', lambda *a: 1)
    monkeypatch.setattr(crossover.random, 'shuffle', lambda *a: None)
    generator, qc = llm_utils.get_llm_code_generator('llama3')
    observed = Mock(wraps=generator)
    monkeypatch.setattr(llm_utils, 'get_llm_code_generator', lambda *a: (observed, qc))
    x, y = tmp_path / 'x.py', tmp_path / 'y.py'
    x.write_text("# --OPTION--\ntrait_persona = '<persona>Explain simply</persona>'\n")
    y.write_text("# --OPTION--\ntrait_persona = '<persona>Use concrete examples</persona>'\n")
    output = tmp_path / 'child.py'
    if operation == 'mutation':
        mutation.augment_network(str(x), str(output), 'templates/Jailbreak/traits/Persona.txt', llm_model='llama3')
    else:
        crossover.augment_network(str(x), str(y), str(output), llm_model='llama3')
    observed.assert_called_once()
    assert output.is_file()
    segment = mutation.split_file(output)[1].strip()
    assert mutation.validate_generated_chunk(segment, expected_name='trait_persona')[0]
    # A transport failure or rejected output cannot pass by copying its parent.
    assert segment != mutation.split_file(x)[1].strip(), 'No verified new trait; model output fell back or was unchanged'
