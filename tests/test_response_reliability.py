import pytest
from src.chat_response import parse_chat_response, ChatResponseError
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def response(content='final answer', finish='stop', reasoning='separate analysis'):
    return {'choices': [{'message': {'content': content, 'reasoning': reasoning},
                         'finish_reason': finish, 'stop_reason': 200002}],
            'usage': {'prompt_tokens': 32, 'completion_tokens': 512, 'total_tokens': 544}}


def test_harmony_parser_output_scores_only_final_channel():
    parsed = parse_chat_response(response())
    assert parsed['text'] == 'final answer'
    assert parsed['reasoning'] == 'separate analysis'
    assert parsed['completion_reason'] == 200002


@pytest.mark.parametrize('content,finish,kind', [
    ('', 'length', 'missing_final_answer'),
    ('partial final', 'length', 'incomplete_final_answer'),
    ('partial final', None, 'incomplete_final_answer'),
    ('<|start|>assistant<|channel|>final<|message|>x<|return|>', 'stop', 'unparsed_harmony'),
])
def test_invalid_answers_preserve_diagnostics_without_becoming_scores(content, finish, kind):
    with pytest.raises(ChatResponseError) as err:
        parse_chat_response(response(content, finish))
    assert err.value.error_type == kind
    assert err.value.diagnostics['usage']['completion_tokens'] == 512
    assert err.value.diagnostics['finish_reason'] == finish
    assert err.value.diagnostics['reasoning_present']


def test_incomplete_responses_are_not_cached_or_judged_and_retain_usage(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sota/Jailbreak'))
    evaluation = importlib.import_module('eval')
    monkeypatch.setenv('JB_CACHE_DIR', str(tmp_path))
    calls = []
    monkeypatch.setattr(evaluation.cache, 'get', lambda *args: None)
    monkeypatch.setattr(evaluation.cache, 'put', lambda *args: pytest.fail('Incomplete response cached'))
    diagnostic = dict(text='', prompt_tokens=32, usage={'prompt_tokens':32, 'completion_tokens':512},
                      error='incomplete', error_type='incomplete_final_answer', finish_reason='length',
                      partial_text='not a complete answer', reasoning_present=True)
    monkeypatch.setattr(evaluation.vllm_client, 'chat_batch', lambda *a, **kw: [diagnostic])
    monkeypatch.setattr(evaluation.judge, 'score_response', lambda *args: pytest.fail('Incomplete response judged'))
    strategy = SimpleNamespace(build_prompt=lambda b: ([{'role':'user','content':b}], {}))
    records = evaluation.get_target_responses(strategy, ['test'], 'http://unused', 'target')
    score, _, details = evaluation.score(records)
    assert score == 0 and details[0]['target_usage']['completion_tokens'] == 512
    assert details[0]['target_finish_reason'] == 'length'
    assert details[0]['target_error_type'] == 'incomplete_final_answer'
    assert details[0]['response'] is None


def test_generation_metrics_count_unique_strategies_cache_and_errors(tmp_path):
    from src.generation_metrics import summarize_generation
    from src.map_elites import MapElitesArchive
    for gene in ['a','b']:
        (tmp_path / f'{gene}_results.json').write_text(json.dumps({'rendered_strategy_sha256':'same'}))
    rows = [{'generation':0, 'candidate_id':'a','target_cached':False,
             'target_error_type':None,'target_finish_reason':'stop','target_usage':{'completion_tokens':5},
             'in_loop':{'score':0,'parse_ok':True},'verification':None},
            {'generation':1, 'candidate_id':'b','target_cached':True,
             'target_error_type':None,'target_finish_reason':'stop','target_usage':{'completion_tokens':5},
             'in_loop':{'score':0,'parse_ok':True},'verification':None}]
    path=tmp_path/'candidates.jsonl'; path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result = summarize_generation(1, [], MapElitesArchive(), tmp_path, path)
    assert result['cumulative_evaluated_genes'] == 2
    assert result['cumulative_unique_rendered_strategies'] == 1
    assert result['target_cache_hits'] == 1 and result['target_errors'] == 0
