"""Audit full saved prompts with the pinned 120B Harmony tokenizer, offline."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ops.serving_memory import audit_context,kv_upper_bound,GIB
from src.model_serving import profile,validate_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-directory',required=True,type=Path)
    parser.add_argument('--gene-directory',action='append',default=[],type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',TOKENIZERS_PARALLELISM='false')
    from transformers import AutoTokenizer
    validate_checkpoint('gpt-oss-120b',args.model_directory)
    tokenizer=AutoTokenizer.from_pretrained(args.model_directory,local_files_only=True)
    spec=importlib.util.spec_from_file_location('audit_renderer',ROOT/'sota/Jailbreak/prompt.py')
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    data=ROOT/'ops/data/shared50/selected.csv'
    behaviors=[row['behavior'] for row in csv.DictReader(data.open())]
    if len(behaviors)!=50:raise ValueError('Frozen dataset must have 50 behaviors')
    paths=[ROOT/'sota/Jailbreak/prompt.py']
    for folder in args.gene_directory:
        genes=sorted(folder.glob('prompt_*.py'))
        if not genes:raise ValueError('No gene sources in '+str(folder))
        paths.extend(genes)
    item=profile('gpt-oss-120b')
    result=audit_context(tokenizer,renderer.render,paths,behaviors,512,item['max_model_len'])
    result.update(model_id=item['model_id'],tokenizer_revision=item['revision'],
                  dataset_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                  tokenizer_files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in args.model_directory.iterdir() if p.name in ['tokenizer.json','tokenizer_config.json','chat_template.jinja']},
                  conservative_eight_sequence_kv_bytes_per_gpu=kv_upper_bound(
                    json.loads((args.model_directory/'config.json').read_text()),item['max_model_len'],8,2),
                  configured_kv_bytes_per_gpu=8*GIB,future_prompt_lengths_unbounded=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_files','longest']},indent=2))


if __name__=='__main__':main()
