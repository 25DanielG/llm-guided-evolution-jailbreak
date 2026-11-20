"""Create a read-only source export for an independent reviewer, without .git/auth."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile


def create(destination, ref='HEAD'):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Review destination must be new; never reuse an active checkout')
    sha = subprocess.check_output(['git','rev-parse', '--verify', ref + '^{commit}'], text=True).strip()
    data = subprocess.check_output(['git','archive','--format=tar',sha])
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(destination, filter='data')
    (destination/'REVIEW-SNAPSHOT.json').write_text(json.dumps({'commit':sha,'source_only':True,
                                        'git_metadata_copied':False,'untracked_files_copied':False},indent=2)+'\n')
    for path in destination.rglob('*'):
        if not path.is_symlink():
            path.chmod(0o555 if path.is_dir() else 0o444)
    destination.chmod(0o555)
    return sha


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--ref', default='HEAD')
    args = parser.parse_args()
    print(create(args.destination, args.ref))
