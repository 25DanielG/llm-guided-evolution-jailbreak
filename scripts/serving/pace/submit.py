"""Build a PACE Slurm submission for one guarded deployment."""

import argparse
import os
import shlex
import subprocess
import sys

from pace_inference import SOURCE, cache_root, model_profile, read_json


def slurm_command(args, root):
    profile = model_profile(args.model)
    allocation = profile['allocation']
    gpu = args.gpu or allocation['gpu']
    serve_mode = 'suite' if profile['video'] else ('image' if profile['image'] else 'text')
    mode = args.mode or (allocation['validation'] if args.operation == 'validate' else serve_mode)
    if args.tp == 2 and (args.model != 'super-vl' or gpu != 'h100'):
        raise ValueError('TP2 is configured only for Super VL on H100')
    if (mode == 'image' and not profile['image']) or (mode == 'suite' and profile['runtime'] != 'stable'):
        raise ValueError(f'{args.model} does not support mode {mode}')
    flags = ['sbatch', '--parsable', '--export=ALL', f'--gres=gpu:{gpu}:{args.tp}',
             f'--constraint={gpu.upper()}', f'--mem={args.mem or allocation["memory"]}',
             f'--tmp={args.tmp or allocation["memory"]}', f'--time={args.time or allocation["walltime"]}',
             f'--output={root / ("slurm-" + args.model + "-%j.out")}']
    for option in ('account', 'partition', 'qos', 'exclude'):
        if value := getattr(args, option):
            flags.append(f'--{option}={value}')
    if args.test_only:
        flags.append('--test-only')
    return flags + [str(SOURCE / 'slurm/validate.sbatch'), str(SOURCE), args.model,
                    mode, str(args.tp), args.source or allocation['source'], gpu.upper(),
                    args.operation, str(args.port)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(read_json(SOURCE / 'config/models.json')))
    for option in ('account', 'partition', 'qos', 'exclude'):
        parser.add_argument(f'--{option}', default=os.environ.get(f'PACE_{option.upper()}'))
    parser.add_argument('--operation', choices=['validate', 'serve'], default='validate')
    parser.add_argument('--mode', choices=['text', 'image', 'suite'])
    parser.add_argument('--source', choices=['shared', 'node-fetch'])
    parser.add_argument('--gpu', choices=['h100', 'h200'])
    parser.add_argument('--tp', type=int, choices=[1, 2], default=1)
    parser.add_argument('--port', type=int, choices=range(65536), metavar='0-65535', default=0,
                        help='0 requests a distinct kernel-assigned port')
    for option in ('time', 'mem', 'tmp'):
        parser.add_argument(f'--{option}')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    try:
        root = cache_root()
        command = slurm_command(args, root)
        if args.dry_run:
            print(shlex.join(command))
            return
        # Leave shell, installer and job clients on the same interpreter.
        env = dict(os.environ, PACE_INFERENCE_STATE=str(root),
                   PACE_PYTHON=os.environ.get('PACE_PYTHON', sys.executable))
        os.umask(0o077)
        subprocess.run(command, env=env, check=True)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
