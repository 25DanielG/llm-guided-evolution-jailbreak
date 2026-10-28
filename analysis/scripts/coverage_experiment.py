#!/usr/bin/env python3
"""Prepare, run, and analyze the fixed behavior-coverage experiment."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.experiments import coverage, runner


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--output', required=True)
    prep.add_argument('--source-csv')
    prep.add_argument('--source-revision')
    prep.add_argument('--seed', type=int, default=42)
    prep.add_argument('--embedding-model', default='sentence-transformers/all-mpnet-base-v2')
    prep.add_argument('--embedding-revision')
    prep.add_argument('--device', default='cpu')
    for name in ('freeze-initial', 'run', 'evaluate-heldout', 'analyze', 'estimate'):
        q = sub.add_parser(name)
        q.add_argument('--manifest', required=True)
        if name in ('freeze-initial', 'run', 'evaluate-heldout'):
            q.add_argument('--run-root', required=True)
        if name in ('freeze-initial', 'run'):
            q.add_argument('--initial-population', required=True)
            q.add_argument('--seed', type=int, default=42)
            q.add_argument('--llm-model', default='llama3')
        if name == 'run':
            q.add_argument('--subset', required=True)
        if name != 'freeze-initial':
            q.add_argument('--final-generation', type=int, default=5)
        if name == 'estimate':
            q.add_argument('--pilot')
            q.add_argument('--repeats', type=int, default=1)
        if name == 'analyze':
            q.add_argument('--runs', nargs='+', required=True)
            q.add_argument('--output', required=True)
            q.add_argument('--device', default='cpu')
    a = p.parse_args(argv)
    if a.command == 'prepare':
        from analysis.utils.prompt_diversity import SentenceBertEmbedder
        if (Path(a.output) / 'experiment.json').exists():
            p.error('Experiment already exists')
        embedder = SentenceBertEmbedder(Path(a.output) / 'embedding_cache', a.embedding_model,
                                       a.embedding_revision, device=a.device)
        result = coverage.prepare(a.output, embedder, a.source_csv, a.source_revision, a.seed)
        print(json.dumps({'subsets': len(result['subsets']), 'manifest': str(Path(a.output) / 'experiment.json')}))
    elif a.command == 'freeze-initial':
        runner.freeze(a.manifest, a.initial_population, a.run_root, a.seed, a.llm_model)
    elif a.command == 'run':
        runner.run(a.manifest, a.initial_population, a.subset, a.run_root, a.seed, a.final_generation, a.llm_model)
    elif a.command == 'evaluate-heldout':
        runner.evaluate_heldout(a.manifest, a.run_root, a.final_generation)
    elif a.command == 'estimate':
        print(json.dumps(runner.estimate(a.manifest, a.pilot, a.repeats, a.final_generation), indent=2))
    elif a.command == 'analyze':
        from analysis.experiments.report import analyze
        analyze(a.manifest, a.runs, a.output, a.final_generation, device=a.device)

if __name__ == '__main__':
    main()
