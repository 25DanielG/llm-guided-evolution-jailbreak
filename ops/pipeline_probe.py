"""Live benign target/judge/reporter/mutator and cache acceptance inside Slurm."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'sota/Jailbreak'))
import eval as evaluator
import jb_config
import judge
import vllm_client


def main():
    for url, name in ((jb_config.target_base_url(),jb_config.target_model_name()),
                      (jb_config.judge_base_url(),jb_config.judge_model_name()),
                      (jb_config.reporting_base_url(),jb_config.reporting_model_name())):
        vllm_client.assert_model(url,name)
    behavior='Explain photosynthesis in two sentences for a biology student.'
    # The serving smoke uses 2048 tokens. Require a complete final answer at
    # the actual common scientific budget (512), before evaluating any genes.
    budget_check=vllm_client.chat(jb_config.target_base_url(),jb_config.target_model_name(),
        [{'role':'user','content':behavior}],max_tokens=jb_config.target_max_new_tokens(),
        temperature=jb_config.target_temperature(),timeout=jb_config.request_timeout())
    if budget_check.get('finish_reason') != 'stop' or not budget_check.get('text','').strip():
        raise RuntimeError('Common target token budget produced an incomplete final answer: '+repr(budget_check))
    strategy=SimpleNamespace(build_prompt=lambda text:([{'role':'user','content':text}],{}))
    first=evaluator.get_target_responses(strategy,[behavior],jb_config.target_base_url(),jb_config.target_model_name())
    assert first[0].get('response','').strip(),first
    verdict=judge.score_response(behavior,first[0]['response'])
    assert verdict['parse_ok'] is True,verdict
    second=evaluator.get_target_responses(strategy,[behavior],jb_config.target_base_url(),jb_config.target_model_name())
    assert second[0]['target_cached'] is True,second
    repeated=judge.score_response(behavior,first[0]['response'])
    assert repeated.get('cached') is True,repeated
    reporter=judge.reporting_score(behavior,first[0]['response'])
    assert reporter.get('parse_ok') is True,reporter
    response=requests.post(f'http://{os.environ["SERVER_HOSTNAME"]}:{os.environ.get("LLMGE_PORT","8137")}/generate',
                           json={'prompt':'Return a Python function add(a, b) that returns a + b, inside a Python code fence.',
                                 'max_new_tokens':850,'temperature':0.1,'top_p':0.15},timeout=120)
    response.raise_for_status()
    code=response.json().get('generated_text','')
    assert code.strip() and 'def add(' in code,code
    result={'status':'passed','target_final_present':True,'target_cache_hit':True,'judge_cache_hit':True,
            'judge':verdict,'reporter':reporter,'mutator_response':code,
            'target_budget':jb_config.target_max_new_tokens(),
            'target_budget_check':budget_check,'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
    path=Path(os.environ['RUN_ID'])/'pipeline-probe.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('LIVE PIPELINE PROBE PASSED',path,flush=True)
    if os.environ.get('JB_120B_MEMORY_PROBE') == '1':
        sys.path.insert(0, str(ROOT))
        from ops.serving_memory_probe import run
        run()


if __name__=='__main__':
    main()
