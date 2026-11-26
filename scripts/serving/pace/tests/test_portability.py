import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from managed_job import child_process, listener_port
from pace_inference import cache_root, disk_headroom, main, model_profile, reviewed_runtime
from submit import slurm_command


class PortabilityTests(unittest.TestCase):
    def test_registry_rejects_mutable_revision_and_unsafe_manifest(self):
        profile = model_profile('inkling')
        original = dict(profile)
        with patch('pace_inference.read_json', side_effect=[
                {'inkling': dict(profile, revision='main')}]):
            with self.assertRaises(ValueError):
                model_profile('inkling')
        for path in ('../../weights.safetensors', 'sub/*.safetensors', 'code.py'):
            with self.subTest(path=path):
                files = dict(original['files'])
                files[path] = {'bytes': 1, 'sha256': 'a' * 64}
                with patch('pace_inference.read_json', side_effect=[{'inkling': original}, files]):
                    with self.assertRaisesRegex(ValueError, 'Unsafe manifest'):
                        model_profile('inkling')

    def test_private_state_with_spaces_and_denied_shared_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'state with spaces'
            with patch.dict(os.environ, {'PACE_INFERENCE_STATE': str(folder)}):
                self.assertEqual(cache_root(), folder)
                self.assertEqual(folder.stat().st_mode & 0o777, 0o700)
                folder.chmod(0o750)
                with self.assertRaisesRegex(ValueError, '0700'):
                    cache_root()

    def test_stable_runtime_shared_and_bad_fingerprint_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stable = root / 'runtimes/stable'
            stable.mkdir(parents=True)
            fingerprint = stable / '.pace-lock-sha256'
            import hashlib
            from pace_inference import SOURCE
            fingerprint.write_text(hashlib.sha256((SOURCE / 'locks/stable.lock').read_bytes()).hexdigest())
            for alias in ('lightning', 'muse', 'super-vl'):
                self.assertEqual(reviewed_runtime(root, model_profile(alias)), stable)
            fingerprint.write_text('stale')
            with self.assertRaisesRegex(ValueError, 'dependency lock'):
                reviewed_runtime(root, model_profile('lightning'))

    def test_lustre_counts_both_filesystem_and_user_quota(self):
        quota = 'title\n/mnt/ice 896 1000 1024 - 13 16 20 -\n'
        with patch('pace_inference.subprocess.check_output', side_effect=['lustre /mnt/ice\n', quota]), \
                patch('pace_inference.shutil.disk_usage', return_value=SimpleNamespace(free=900_000)), \
                patch('pace_inference.os.statvfs', return_value=SimpleNamespace(f_favail=10)):
            self.assertEqual(disk_headroom(Path('/mnt/ice/test')), ('lustre', 104 * 1024, 3))
        with patch('pace_inference.subprocess.check_output', side_effect=['lustre /mnt/ice\n', 'unavailable\n']):
            with patch('pace_inference.shutil.disk_usage'), patch('pace_inference.os.statvfs'):
                with self.assertRaisesRegex(ValueError, 'Cannot read user quota'):
                    disk_headroom(Path('/mnt/ice/test'))

    def test_slurm_overrides_and_scheduler_defaults(self):
        args = SimpleNamespace(model='super-vl', gpu=None, operation='validate', mode=None, tp=1,
                               time=None, tmp=None, mem=None, account=None, partition=None,
                               qos=None, exclude=None, port=0, test_only=False, source=None,
                               inkling_graph=False)
        command = slurm_command(args, Path('/a path with spaces'))
        self.assertIn('--gres=gpu:h200:1', command)
        self.assertIn('--mem=128G', command)
        self.assertFalse(any(arg.startswith(('--account=', '--qos=')) for arg in command))
        self.assertEqual(command[-6:], ['suite', '1', 'node-fetch', 'H200', 'validate', '0'])
        args.gpu = 'h100'
        args.tp = 2
        args.account = 'another-account'
        args.qos = 'another-qos'
        args.exclude = 'test-node'
        args.operation = 'serve'
        args.mode = 'image'
        args.port = 20991
        args.source = 'shared'
        command = slurm_command(args, Path('/a path with spaces'))
        for flag in ('--account=another-account', '--qos=another-qos', '--exclude=test-node',
                     '--gres=gpu:h100:2', '--output=/a path with spaces/slurm-super-vl-%j.out'):
            self.assertIn(flag, command)
        self.assertEqual(command[-6:], ['image', '2', 'shared', 'H100', 'serve', '20991'])
        args.inkling_graph = True
        with self.assertRaisesRegex(ValueError, 'specific to Inkling'):
            slurm_command(args, Path('/a path with spaces'))
        args.model, args.tp = 'inkling', 1
        self.assertEqual(slurm_command(args, Path('/a path with spaces'))[-1], 'inkling-graph')

    def test_listener_filters_by_pid_socket_and_ipv4_loopback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for pid, inode in ((81, 701), (82, 702)):
                process = root / str(pid)
                (process / 'fd').mkdir(parents=True)
                (process / 'net').mkdir()
                (process / 'fd/6').symlink_to(f'socket:[{inode}]')
                (process / 'net/tcp').write_text(
                    'header\n0: 0100007F:AA12 00:00 0A 0 0 0 0 0 701\n'
                    '1: 0100007F:AA13 00:00 0A 0 0 0 0 0 702\n'
                    '2: 00000000:AA14 00:00 0A 0 0 0 0 0 701\n')
            self.assertEqual(listener_port(81, proc=root), 0xAA12)
            self.assertEqual(listener_port(82, proc=root), 0xAA13)
            self.assertIsNone(listener_port(81, 0xAA13, root))

    def test_owned_subprocess_is_reaped_on_validation_error(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / 'started'
            command = [sys.executable, '-c',
                       'from pathlib import Path; import sys, time; '
                       'Path(sys.argv[1]).touch(); time.sleep(60)', str(marker)]
            child = None
            with self.assertRaisesRegex(ValueError, 'validation failure'):
                with child_process(command, Path(temp) / 'server.log') as child:
                    deadline = time.monotonic() + 5
                    while not marker.exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertTrue(marker.exists())
                    raise ValueError('validation failure')
            self.assertIsNotNone(child.poll())

    def test_eviction_requires_marker_and_preserves_adjacent_snapshots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile = model_profile('inkling')
            snapshot = root / 'snapshots/inkling' / profile['revision']
            snapshot.mkdir(parents=True)
            preserve = snapshot.with_name('unrelated')
            preserve.mkdir()
            command = ['pace_inference.py', 'evict', 'inkling', '--confirm']
            with patch.dict(os.environ, {'PACE_INFERENCE_STATE': temp}), patch.object(sys, 'argv', command):
                with self.assertRaisesRegex(ValueError, 'Unrecognized snapshot'):
                    main()
                self.assertTrue(snapshot.exists())
                (snapshot / '.pace-snapshot.json').write_text(json.dumps(
                    {key: profile[key] for key in ('repo', 'revision')}))
                main()
            self.assertFalse(snapshot.exists())
            self.assertTrue(preserve.exists())

    def test_unknown_listener_exits_without_requesting_other_process(self):
        from managed_job import await_endpoint
        process = SimpleNamespace(pid=103, poll=lambda: 1)
        with patch('managed_job.api_open') as request, self.assertRaisesRegex(ValueError, 'exited'):
            await_endpoint(process, 'inkling', 0, time.monotonic() + 1)
        request.assert_not_called()


if __name__ == '__main__':
    unittest.main()
