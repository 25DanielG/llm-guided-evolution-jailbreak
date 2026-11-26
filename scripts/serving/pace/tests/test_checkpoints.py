import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pace_inference import assert_managed, fetch, model_profile, stage_snapshot, verify
from validate import red_png


class CheckpointTests(unittest.TestCase):
    def test_registry_has_pinned_weight_and_config_files(self):
        for alias in ('inkling', 'lightning', 'muse', 'super-vl'):
            with self.subTest(alias=alias):
                profile = model_profile(alias)
                self.assertRegex(profile['revision'], r'^[0-9a-f]{40}$')
                self.assertIn('config.json', profile['files'])
                self.assertTrue(any(name.endswith('.safetensors') for name in profile['files']))
                for name, data in profile['files'].items():
                    self.assertFalse(Path(name).is_absolute())
                    self.assertNotIn('..', Path(name).parts)
                    self.assertGreater(data['bytes'], 0)
                    if 'sha256' in data:
                        self.assertRegex(data['sha256'], r'^[0-9a-f]{64}$')
                    else:
                        self.assertRegex(data['git_blob'], r'^[0-9a-f]{40}$')
        self.assertNotIn('mtp.safetensors', model_profile('inkling')['files'])

    def test_verify_fails_for_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'weights').write_bytes(b'original')
            (root / 'readme').write_bytes(b'git')
            files = {
                'weights': {'bytes': 8, 'sha256': hashlib.sha256(b'original').hexdigest()},
                'readme': {'bytes': 3, 'git_blob': hashlib.sha1(b'blob 3\0git').hexdigest()},
            }
            verify(root, files, digest=True)
            (root / 'weights').write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                verify(root, files, digest=True)

    def test_verify_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'content').write_text('x')
            (root / 'linked').symlink_to('content')
            with self.assertRaisesRegex(ValueError, 'linked'):
                verify(root, {'linked': {'bytes': 1}})

    def test_stage_rejects_same_size_cache_corruption_before_serving(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            snapshot = root / 'snapshot'
            snapshot.mkdir()
            metadata = {'repo': 'example/test', 'revision': '1' * 40}
            (snapshot / '.pace-snapshot.json').write_text(json.dumps(metadata))
            weights = snapshot / 'model.safetensors'
            weights.write_bytes(b'original')
            profile = dict(metadata, files={'model.safetensors': {
                'bytes': 8, 'sha256': hashlib.sha256(b'original').hexdigest(),
            }})
            with patch('pace_inference.check_local_disk'):
                staged = stage_snapshot(root, snapshot, profile, root / 'good')
                self.assertEqual((staged / weights.name).read_bytes(), b'original')
                weights.write_bytes(b'changed!')
                with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                    stage_snapshot(root, snapshot, profile, root / 'corrupt')

    def test_fetch_publishes_only_verified_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            final = root / 'snapshot'
            partial = root / '.snapshot.partial'
            profile = {'repo': 'example/test', 'revision': '1' * 40,
                       'files': {'model.safetensors': {
                           'bytes': 1, 'sha256': hashlib.sha256(b'x').hexdigest()}}}

            def download(**kwargs):
                (kwargs['local_dir'] / 'model.safetensors').write_bytes(b'x')

            hub = Mock(snapshot_download=Mock(side_effect=download))
            with patch.dict(sys.modules, huggingface_hub=hub), patch.dict(os.environ), \
                    patch('pace_inference.disk_headroom', return_value=('xfs', 8 * 1024**3, 1024)):
                fetch(root, final, partial, profile)
                fetch(root, final, partial, profile)
            hub.snapshot_download.assert_called_once_with(
                repo_id=profile['repo'], revision=profile['revision'], token=False,
                local_dir=partial, allow_patterns=['model.safetensors'], max_workers=2,
            )
            assert_managed(final, profile)
            self.assertFalse(partial.exists())

    def test_offline_image_has_png_header(self):
        self.assertTrue(red_png().startswith(b'\x89PNG\r\n\x1a\n'))
