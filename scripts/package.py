"""Build code and data archives from the explicit release file whitelist."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from validate import safe_path, verify_hash

ROOT = Path(__file__).resolve().parents[1]


def archive(root, output):
    entries = json.loads((root / 'artifact_manifest.json').read_text())['files']
    allowed = json.loads((root / 'release-files.json').read_text())['files']
    if {e['path'] for e in entries} != set(allowed):
        raise ValueError('Manifest and release whitelist differ.')
    for entry in entries:
        verify_hash(safe_path(root, entry['path']), entry['sha256'])
    output.mkdir(parents=True, exist_ok=True)
    checksums = []
    for name, want_data in (('veruseval-code.tar.gz', False), ('veruseval-data.tar.gz', True)):
        path = output / name
        if path.exists():
            raise ValueError(f'Archive already exists: {path}')
        files = sorted(allowed + ['artifact_manifest.json'])
        with path.open('wb') as stream:
            compressor = (subprocess.Popen([shutil.which('pigz'), '-n', '-p', '8'],
                                           stdin=subprocess.PIPE, stdout=stream)
                          if shutil.which('pigz') else None)
            compressed = compressor.stdin if compressor else gzip.GzipFile(
                fileobj=stream, mode='wb', filename='', mtime=0)
            try:
                with tarfile.open(fileobj=compressed, mode='w|') as tar:
                    for relative in files:
                        is_data = relative.startswith('data/') or relative == 'artifact_manifest.json'
                        if is_data != want_data:
                            continue
                        source = safe_path(root, relative)
                        info = tar.gettarinfo(str(source), arcname=relative)
                        info.uid = info.gid = info.mtime = 0
                        info.uname = info.gname = ''
                        with source.open('rb') as content:
                            tar.addfile(info, content)
            finally:
                compressed.close()
                if compressor and compressor.wait() != 0:
                    raise RuntimeError('Archive compression failed.')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        checksums.append(f'{digest}  {name}')
        print(f'Created {name}: {path.stat().st_size:,} bytes', flush=True)
    (output / 'SHA256SUMS').write_text('\n'.join(checksums) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        archive(ROOT, args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Packaging failed: {error}\n')
