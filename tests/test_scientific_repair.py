"""CPU-only contracts and regressions from the preserved R3 scientific audit."""
import ast
import asyncio
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/scientific_repair'
sys.path.insert(0, str(ROOT / 'sota/Jailbreak'))
evaluation = importlib.import_module('eval')
judge = importlib.import_module('judge')
from src import llm_mutation as mutation
from src.mutator_inputs import prepare_mutator_inputs
from src.fitness_state import snapshot_history
from src.map_elites import MapElitesArchive, extract_trait_assignments
import run_improved as evo


def read(name):
    return json.loads((FIXTURES / name).read_text())


def detail(score=0.0):
    return dict(behavior='Explain photosynthesis', response='An existing complete answer',
                score=score, messages=[{'role': 'user', 'content': 'Explain photosynthesis'}],
                error=None, prompt_tokens=20, input_cost_tokens=20)


@pytest.mark.parametrize('override', [None, 'custom-results', 'absolute'])
def test_feedback_uses_the_actual_evaluation_producer_path(tmp_path, monkeypatch, override):
    sota = tmp_path / 'sota'; sota.mkdir()
    monkeypatch.chdir(sota)
    monkeypatch.setattr(mutation, 'SOTA_ROOT', str(sota))
    if override is None:
        monkeypatch.delenv('JB_GENE_RESULTS_DIR', raising=False)
    else:
        monkeypatch.setenv('JB_GENE_RESULTS_DIR', str(tmp_path/'absolute') if override == 'absolute' else override)
    evaluation.write_details('parent', .25, 20, [detail(.25)])
    feedback = mutation._read_feedback_summary(sota/'models/prompt_parent.py')
    assert 'mean_judge_score=0.25' in feedback and 'Explain photosynthesis' in feedback
    assert mutation._read_feedback_summary(sota/'models/prompt_never_evaluated.py') == ''


def test_feedback_does_not_silently_use_malformed_or_other_gene_data(tmp_path, monkeypatch):
    monkeypatch.setenv('JB_GENE_RESULTS_DIR', str(tmp_path))
    p = tmp_path/'parent_results.json'; p.write_text('{broken')
    with pytest.raises(json.JSONDecodeError):mutation._read_feedback_summary('prompt_parent.py')
    p.write_text(json.dumps(dict(gene_id='another-parent', mean_judge_score=.9)))
    with pytest.raises(ValueError, match='another gene'):mutation._read_feedback_summary('prompt_parent.py')


@pytest.mark.parametrize('template,trait', [('Persona', 'trait_format'), ('FormatObfuscate', 'trait_persona'),
                                         ('ContextScaffold', 'trait_instruction')])
def test_specialized_template_cannot_contradict_random_trait(template, trait):
    text = (ROOT/f'templates/Jailbreak/traits/{template}.txt').read_text()
    compatible = mutation._compatible_trait_template(text, trait)
    assert compatible.startswith((ROOT/'templates/Jailbreak/traits/Significant.txt').read_text())
    assert '{{FEEDBACK}}' in compatible and '{{FULL}}' in compatible


@pytest.mark.parametrize('template,trait', [('Persona', 'trait_persona'), ('FormatObfuscate', 'trait_format'),
                                         ('ContextScaffold', 'trait_turns'), ('Significant', 'trait_language')])
def test_compatible_template_is_preserved(template, trait):
    text = (ROOT/f'templates/Jailbreak/traits/{template}.txt').read_text()
    assert mutation._compatible_trait_template(text, trait) == text


@pytest.mark.parametrize('failure', read('r3-rejected-chunks.json'))
def test_all_six_original_invalid_chunks_remain_rejected(failure):
    valid, reason = mutation.validate_generated_chunk(failure['generated_chunk'], failure['expected_name'])
    assert not valid and reason


@pytest.mark.parametrize('code', ['import os\ntrait_context = "<context>x</context>"',
    'trait_context = "<context>x</context>"\nprint("extra code")',
    'trait_context = other = "<context>x</context>"'])
def test_extra_executable_content_is_not_a_single_trait(code):
    assert not mutation.validate_generated_chunk(code, 'trait_context')[0]


def test_mutation_fallback_keeps_parent_when_model_still_returns_wrong_schema(tmp_path, monkeypatch):
    parent = tmp_path/'prompt_parent.py'; parent.write_text((ROOT/'sota/Jailbreak/prompt.py').read_text())
    output = tmp_path/'prompt_child.py'; captured = []
    monkeypatch.setattr(mutation.np.random, 'randint', lambda *a: 2)  # persona slot, not format
    def invalid_output(prompt, *a, **kw):
        captured.append(prompt)
        return 'trait_format = \'<format type="freeform">{ASK}</format>\''
    monkeypatch.setattr(mutation, 'generate_augmented_code', invalid_output)
    mutation.augment_network(str(parent), str(output), 'templates/Jailbreak/traits/FormatObfuscate.txt')
    assert 'Selected assignment (the only assignment to return): trait_persona.' in captured[0]
    assert extract_trait_assignments(output.read_text()) == extract_trait_assignments(parent.read_text())


def test_mutator_formats_existing_instructions_once_without_duplicate_special_tokens():
    tokenizer = Mock(chat_template='existing checkpoint template')
    tokenizer.apply_chat_template.side_effect = ['CHAT:first', 'CHAT:second']
    prompts, kwargs = prepare_mutator_inputs(tokenizer, ['first', 'second'])
    assert prompts == ['CHAT:first', 'CHAT:second'] and kwargs == {'add_special_tokens': False}
    assert tokenizer.apply_chat_template.call_args_list[0].args == ([{'role':'user', 'content':'first'}],)
    assert tokenizer.apply_chat_template.call_args_list[0].kwargs == dict(tokenize=False, add_generation_prompt=True)
    plain = SimpleNamespace(chat_template=None)
    assert prepare_mutator_inputs(plain, ['original']) == (['original'], {})


def test_actual_batch_processor_uses_chat_template_and_preserves_generation_settings(monkeypatch):
    tree = ast.parse((ROOT/'server.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LLMModel')
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == '_batch_processor')
    ns = dict(asyncio=asyncio, time=time, BATCH_SIZE=1, BATCH_WAIT_TIME=.01,
              prepare_mutator_inputs=prepare_mutator_inputs)
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'server.py', 'exec'), ns)
    tokenizer = Mock(chat_template='checkpoint template')
    tokenizer.apply_chat_template.return_value = 'CHAT:original instructions'
    pipeline = Mock(return_value=[[{'generated_text':'original output'}]])
    async def inline_inference(func, *args, **kwargs):
        return func(*args, **kwargs)
    # The external inference boundary is mocked; no worker/model is launched.
    monkeypatch.setattr(asyncio, 'to_thread', inline_inference)
    async def scenario():
        model = SimpleNamespace(tokenizer=tokenizer, pipeline=pipeline, request_queue=asyncio.Queue(),
                                batch_lock=asyncio.Lock(), batch_id=0, is_processing=True)
        future = asyncio.get_running_loop().create_future()
        await model.request_queue.put((dict(prompt='original instructions', max_new_tokens=850,
                                           temperature=.1, top_p=.9), future))
        task = asyncio.create_task(ns['_batch_processor'](model))
        try:
            result = await asyncio.wait_for(future, timeout=5)
            assert result['generated_text'] == 'original output'
        finally:
            task.cancel()
            await task
    asyncio.run(scenario())
    pipeline.assert_called_once_with(['CHAT:original instructions'], max_new_tokens=850,
                                    temperature=.1, top_p=.9, add_special_tokens=False)


def test_actual_recorded_clone_costs_no_longer_make_fake_archive_replacements(tmp_path, monkeypatch):
    cases = read('r3-identical-input-costs.json')['cases']
    assert len({c['rendered_strategy_sha256'] for c in cases}) == 1
    monkeypatch.setattr(evaluation.judge, 'score_response', lambda *a: dict(score=0.0, parse_ok=True))
    monkeypatch.setattr(evo, 'VARIANT_DIR', str(tmp_path)); monkeypatch.setattr(evo, 'box_print', lambda *a,**k:None)
    monkeypatch.setenv('JB_GENE_RESULTS_DIR', str(tmp_path)); archive = MapElitesArchive()
    costs=[];decisions=[]
    for case in cases:
        mean,cost,details=evaluation.score(copy.deepcopy(case['records']))
        assert mean == 0.0 and cost == pytest.approx(223.42)
        assert sum(not x['target_valid'] for x in details) == sum(bool(x['error']) for x in case['records'])
        assert all(x['input_cost_tokens'] for x in details)
        gene=case['gene'];(tmp_path/f'prompt_{gene}.py').write_text('trait_format = \'<format type="transcript">{ASK}</format>\'\ntrait_turns = \'<turns></turns>\'')
        ind=evo.creator.Individual([gene]);ind.fitness.values=(mean,.24963896301484712,cost)
        decisions.append(evo.update_map_elites_archive([ind],archive));costs.append(cost)
    assert decisions[0]['inserted']==1
    assert all(d['replaced']==0 for d in decisions[1:])
    assert cases[0]['original_cost'] > cases[1]['original_cost'] > cases[2]['original_cost']


@pytest.mark.parametrize('usage', [{}, {'prompt_tokens': None}, {'prompt_tokens': -1}, {'prompt_tokens': True}])
def test_missing_input_usage_cannot_lower_cost(usage, monkeypatch):
    monkeypatch.setattr(evaluation.judge, 'score_response', lambda *a:dict(score=.25,parse_ok=True))
    records=[dict(behavior='request',response='answer',prompt_tokens=100,usage={'prompt_tokens':100}),
             dict(behavior='request',response=None,prompt_tokens=None,usage=usage,error='request_error')]
    mean,cost,details=evaluation.score(records)
    assert mean==.25 and cost==evaluation.PENALTY_TOKENS
    assert details[1]['input_cost_tokens'] is None and not details[1]['fitness_included']


def test_empty_evaluation_keeps_the_existing_conservative_cost():
    assert evaluation.score([]) == (0.0, evaluation.PENALTY_TOKENS, [])


@pytest.mark.parametrize('response,finish', [(None,'stop'),('', 'stop'),('partial','length'),('partial',None)])
def test_invalid_direct_records_are_excluded_without_judging(response, finish, monkeypatch):
    callback=Mock();monkeypatch.setattr(evaluation.judge,'score_response',callback)
    mean,cost,details=evaluation.score([dict(behavior='request',response=response,finish_reason=finish,
        prompt_tokens=None,usage={'prompt_tokens':32})])
    callback.assert_not_called()
    assert mean==0 and cost==32 and details[0]['target_valid'] is False
    assert details[0]['input_cost_tokens']==32 and details[0]['error'].startswith('target_error')


@pytest.mark.parametrize('field',['behavior','response'])
@pytest.mark.parametrize('value',[None,'',' ',42])
def test_judge_input_validation_precedes_model_requests(field,value,monkeypatch):
    callback=Mock();monkeypatch.setattr(judge.vllm_client,'chat',callback)
    args=dict(behavior='request',response='answer');args[field]=value
    with pytest.raises(ValueError,match='Invalid judge input'):judge.score_response(**args)
    callback.assert_not_called()


def test_original_controls_do_not_manufacture_an_attack_gradient(monkeypatch):
    controls=read('original-controls.json')
    monkeypatch.setattr(judge.cache,'get',lambda *a:None)
    put=Mock();monkeypatch.setattr(judge.cache,'put',put)
    refusal=controls['C01_initial_refusal.original-record.json']
    benign=controls['C13_20b.benign-probe.json']
    for behavior,response,verdict,expected in [
        (refusal['behavior'],refusal['target_response'],refusal['in_loop'],0),
        ('Explain photosynthesis',benign['target_budget_check']['text'],benign['judge'],1)]:
        monkeypatch.setattr(judge.vllm_client,'chat',lambda *a,**k:dict(text=json.dumps(verdict['rubric']),logprobs=None))
        result=judge.score_response(behavior,response)
        assert result['parse_ok'] and result['score']==expected
    bad=controls['C03_malformed_judge.original-record.json'];put.reset_mock()
    monkeypatch.setattr(judge.vllm_client,'chat',lambda *a,**k:dict(text=bad['in_loop']['raw_output'],logprobs=None))
    result=judge.score_response(bad['behavior'],bad['target_response'])
    assert not result['parse_ok'] and result['score']==0
    put.assert_not_called()


def test_absent_incumbent_and_identical_child_share_current_novelty_reference(tmp_path,monkeypatch):
    current={};history={};archive=MapElitesArchive()
    monkeypatch.setattr(evo,'GLOBAL_DATA',current);monkeypatch.setattr(evo,'GLOBAL_DATA_HIST',history)
    monkeypatch.setattr(evo,'MAP_ELITES_ARCHIVE',archive);monkeypatch.setattr(evo,'VARIANT_DIR',str(tmp_path))
    monkeypatch.setattr(evo,'_selection_method','map_elites');monkeypatch.setattr(evo,'box_print',lambda *a,**k:None)
    monkeypatch.setenv('JB_GENE_RESULTS_DIR',str(tmp_path));individuals=[]
    for gene in ['incumbent','child']:
        ind=evo.creator.Individual([gene]);ind.fitness.values=(0,.99,20)
        current[gene]=dict(status='completed',sub_flag=True,fitness=ind.fitness.values)
        (tmp_path/f'prompt_{gene}.py').write_text('trait_format = \'<format type="freeform">{ASK}</format>\'\ntrait_turns = \'<turns></turns>\'')
        (tmp_path/f'{gene}_results.json').write_text(json.dumps(dict(behaviors=[detail()])))
        (tmp_path/f'{gene}_results.csv').write_text('mean_judge_score,diversity,avg_prompt_tokens\n0,.99,20\n')
        individuals.append(ind)
    snapshot_history(current,history);evo.update_map_elites_archive([individuals[0]])
    evo.apply_diversity_objective([individuals[1]])
    assert archive.elites()[0].fitness[1] == individuals[1].fitness.values[1]
    assert individuals[1].fitness.values[1] == pytest.approx(0, abs=1e-12)
    assert evo.update_map_elites_archive([individuals[1]])['replaced']==0
    # The archive owns a copy; the original absent population object is unused.
    evo.validate_fitness_state([individuals[1]],archive,current,history)


def test_evaluation_feedback_and_actual_search_history_survive_three_generations(tmp_path,monkeypatch):
    results=tmp_path/'results';results.mkdir();variants=tmp_path/'variants';variants.mkdir()
    monkeypatch.setenv('JB_GENE_RESULTS_DIR',str(results));monkeypatch.setenv('JB_RESUME_MODE','new')
    monkeypatch.delenv('JB_STATUS_FILE',raising=False)
    for name,value in dict(GLOBAL_DATA={},GLOBAL_DATA_HIST={},GLOBAL_DATA_ANCESTRY={},
        MAP_ELITES_ARCHIVE=MapElitesArchive(),hof=evo.tools.ParetoFront(),population=[],
        VARIANT_DIR=str(variants),_selection_method='map_elites').items():monkeypatch.setattr(evo,name,value,raising=False)
    monkeypatch.setattr(evo,'box_print',lambda *a,**k:None)
    parent='g0';(variants/f'prompt_{parent}.py').write_text((ROOT/'sota/Jailbreak/prompt.py').read_text())
    requests=[];monkeypatch.setattr(mutation.np.random,'randint',lambda *a:2)
    def model(prompt,*a,**kw):
        requests.append(prompt)
        return 'trait_persona = '+repr('<persona>Benign fixture generation '+str(len(requests))+'</persona>')
    monkeypatch.setattr(mutation,'generate_augmented_code',model)
    for gen in range(3):
        # Actual producer and scoring path; only the external judge is mocked.
        monkeypatch.setattr(evaluation.judge,'score_response',lambda *a:dict(score=0,parse_ok=True))
        mean,cost,details=evaluation.score([dict(behavior='Explain photosynthesis',response='Existing refusal',
            prompt_tokens=20,usage={'prompt_tokens':20},finish_reason='stop')])
        evaluation.write_results(parent,mean,cost);evaluation.write_details(parent,mean,cost,details)
        ind=evo.creator.Individual([parent]);ind.fitness.values=(mean,.1,cost)
        evo.GLOBAL_DATA[parent]=dict(status='completed',sub_flag=True,fitness=ind.fitness.values)
        evo.GLOBAL_DATA_ANCESTRY[parent]={};snapshot_history(evo.GLOBAL_DATA,evo.GLOBAL_DATA_HIST)
        evo.population[:]=[ind];evo.apply_diversity_objective(evo.population)
        evo.update_map_elites_archive(evo.population);evo.refresh_hall_of_fame(evo.hof,evo.population)
        evo.save_checkpoint(gen,tmp_path/'ckpt',tmp_path/'global')
        if gen==1:
            monkeypatch.setenv('JB_RESUME_MODE','required')
            state,next_gen,data=evo.load_checkpoint(tmp_path/'ckpt',global_path=tmp_path/'global')
            assert next_gen==2 and state['population'][0][0]==parent
            for key in ['GLOBAL_DATA','GLOBAL_DATA_HIST','GLOBAL_DATA_ANCESTRY']:monkeypatch.setattr(evo,key,data[key])
            monkeypatch.setattr(evo,'MAP_ELITES_ARCHIVE',state['map_elites_archive'])
            monkeypatch.setattr(evo,'hof',state['hof'])
        child=f'g{gen+1}'
        mutation.augment_network(str(variants/f'prompt_{parent}.py'),str(variants/f'prompt_{child}.py'),
            'templates/Jailbreak/traits/Persona.txt')
        assert 'mean_judge_score=0.0' in requests[-1] and '(no prior eval yet)' not in requests[-1]
        assert 'Explain photosynthesis' in requests[-1]
        # A parent remains canonical after a selected copy becomes a new child.
        before=copy.deepcopy(evo.GLOBAL_DATA[parent]);evo.GLOBAL_DATA[child]=dict(status='subbed file',sub_flag=True,fitness=None)
        evo.update_individual(copy.deepcopy(ind),child,parent)
        assert evo.GLOBAL_DATA[parent]==before and evo.GLOBAL_DATA_HIST[parent]==before
        parent=child
    evo.assert_fitness_state()
    assert (tmp_path/'ckpt/checkpoint_gen_2.json').exists() and len(requests)==3
