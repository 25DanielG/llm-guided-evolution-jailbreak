"""Create a new seed cohort using an existing campaign's exact frozen inputs."""
import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.experiments.difficulty_tournament import load_manifest, validate_runtime, json_hash
from src.run_status import atomic_json


def extend(source, destination, seeds):
    manifest = load_manifest(source)
    validate_runtime(manifest)
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**32 for s in seeds):
        raise ValueError('Need distinct unsigned 32-bit seeds')
    if set(seeds) & set(manifest['seeds']):
        raise ValueError('Additional seeds must not overlap the original cohort')
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('source.tar.gz', 'behaviors.csv', 'panel.json'):
        shutil.copy2(source / name, destination / name)
    manifest.pop('identity')
    manifest['seeds'] = seeds
    manifest['identity'] = json_hash(manifest)
    atomic_json(destination / 'manifest.json', manifest)
    load_manifest(destination)
    atomic_json(destination / 'cohort.json', {
        'original_campaign': str(source.resolve()),
        'original_identity': load_manifest(source)['identity'],
        'additional_campaign': str(destination.resolve()),
        'additional_identity': manifest['identity'],
        'all_seeds': load_manifest(source)['seeds'] + seeds,
        'final_generation': manifest['final_generation'],
    })
    print(f'Prepared {destination}: seeds {seeds}, generation {manifest["final_generation"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    args = parser.parse_args()
    extend(args.source.resolve(), args.campaign.resolve(), args.seeds)
