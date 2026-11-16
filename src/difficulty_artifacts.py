"""Committed generation artifacts for faithful resume of isolated experiments."""
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from src.checkpointing import digest

DIRECTORIES = ('sota/Jailbreak/models/llmge_models', 'sota/Jailbreak/results', 'cache', 'jailbreak_test')


def snapshot(work, checkpoint_dir, generation):
    work, checkpoint_dir = Path(work).resolve(), Path(checkpoint_dir).resolve()
    if not checkpoint_dir.is_relative_to(work):
        raise ValueError('Artifact checkpoint must be inside isolated work directory')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_dir / f'artifacts_gen_{generation}.tar.gz'
    fd, temporary = tempfile.mkstemp(dir=checkpoint_dir, prefix='.artifacts-')
    os.close(fd)
    try:
        with tarfile.open(temporary, 'w:gz') as archive:
            for directory in DIRECTORIES:
                path = work / directory
                if path.exists():
                    archive.add(path, arcname=directory, filter=lambda info: None if '__pycache__' in info.name else info)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {'path': str(destination.relative_to(work)), 'sha256': digest(destination)}


def restore(work, metadata):
    work = Path(work).resolve()
    archive_path = (work / metadata['path']).resolve()
    if not archive_path.is_relative_to(work) or digest(archive_path) != metadata['sha256']:
        raise ValueError('Committed artifact archive missing or changed')
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if not any(member.name == directory or member.name.startswith(directory + '/') for directory in DIRECTORIES):
                raise ValueError('Unexpected artifact archive member')
        for directory in DIRECTORIES:
            path = work / directory
            if path.exists():
                shutil.rmtree(path)
        archive.extractall(work, filter='data')
