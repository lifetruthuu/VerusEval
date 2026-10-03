"""Download and verify the matching VerusEval code and data archives.

Python 3.9 or later and curl for network downloads; no third-party Python
packages. A complete Git checkout uses
local parts. A standalone copy of this script uses the anonymous mirror.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess

DEFAULT_BASE_URL = 'https://anonymous.4open.science/api/repo/VerusEval-A7E4/file/downloads/'
CHUNK_SIZE = 1024 * 1024


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b''):
            value.update(block)
    return value.hexdigest()


def matches(path, entry):
    return path.is_file() and path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256']


def validate_manifest(manifest):
    if manifest.get('format') != 1 or not manifest.get('artifacts'):
        raise ValueError('Unsupported or empty download manifest.')
    names = set()
    for artifact in manifest['artifacts']:
        if not artifact.get('parts'):
            raise ValueError('Archive has no parts.')
        for entry in [artifact, *artifact['parts']]:
            name = entry['name']
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name):
                raise ValueError('Unsafe filename in download manifest.')
            if name in names:
                raise ValueError('Duplicate filename in download manifest.')
            names.add(name)
            if not isinstance(entry['bytes'], int) or entry['bytes'] <= 0:
                raise ValueError('Invalid file size in download manifest.')
            if not re.fullmatch(r'[a-f0-9]{64}', entry['sha256']):
                raise ValueError('Invalid SHA-256 in download manifest.')
        if sum(part['bytes'] for part in artifact['parts']) != artifact['bytes']:
            raise ValueError('Part sizes do not match the archive size.')
    return manifest


def read_remote(url):
    curl = shutil.which('curl')
    if not curl:
        raise OSError('Install curl for network downloads, or use a complete Git checkout.')
    # Use the same client as the documented bootstrap command. The mirror
    # accepts curl but can reject Python urllib requests with HTTP 403.
    result = subprocess.run(
        [curl, '--fail', '--location', '--silent', '--show-error',
         '--connect-timeout', '30', '--max-time', '180', '--max-filesize',
         str(8 * 1024 * 1024), url], capture_output=True)
    if result.returncode:
        raise OSError(result.stderr.decode(errors='replace').strip() or
                      f'curl failed with exit code {result.returncode}: {url}')
    return io.BytesIO(result.stdout)


def restore(manifest, output, source_dir=None, base_url=DEFAULT_BASE_URL):
    validate_manifest(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / '.parts'
    cache.mkdir(exist_ok=True)
    for artifact in manifest['artifacts']:
        destination = output / artifact['name']
        if matches(destination, artifact):
            print(f"Already verified: {artifact['name']}", flush=True)
            continue
        if destination.exists():
            raise ValueError(f'Existing archive does not match; move it aside first: {destination}')
        parts = []
        for number, part in enumerate(artifact['parts'], 1):
            path = cache / part['name']
            if not matches(path, part):
                temporary = path.with_name(path.name + '.partial')
                source = (Path(source_dir) / 'parts' / part['name']).open('rb') if source_dir else read_remote(
                    base_url.rstrip('/') + '/parts/' + part['name'])
                with source, temporary.open('wb') as target:
                    total = 0
                    for block in iter(lambda: source.read(CHUNK_SIZE), b''):
                        total += len(block)
                        if total > part['bytes']:
                            raise ValueError(f"Part exceeds its expected size: {part['name']}")
                        target.write(block)
                if not matches(temporary, part):
                    raise ValueError(f"Size or SHA-256 mismatch: {part['name']}")
                temporary.replace(path)
            parts.append(path)
            print(f"{artifact['name']}: part {number}/{len(artifact['parts'])} verified", flush=True)
        temporary = destination.with_name(destination.name + '.partial')
        with temporary.open('wb') as target:
            for part in parts:
                with part.open('rb') as source:
                    shutil.copyfileobj(source, target, length=CHUNK_SIZE)
        if not matches(temporary, artifact):
            raise ValueError(f"Assembled archive SHA-256 mismatch: {artifact['name']}")
        temporary.replace(destination)
        print(f"Verified archive: {destination}", flush=True)
    (output / 'SHA256SUMS').write_text(''.join(
        f"{a['sha256']}  {a['name']}\n" for a in manifest['artifacts']), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('packages'))
    parser.add_argument('--base-url', help='Override the anonymous download directory URL.')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    local = directory if not args.base_url and (directory / 'manifest.json').is_file() else None
    base = args.base_url or DEFAULT_BASE_URL
    try:
        if local:
            manifest = json.loads((local / 'manifest.json').read_text(encoding='utf-8'))
        else:
            with read_remote(base.rstrip('/') + '/manifest.json') as response:
                manifest = json.loads(response.read(1024 * 1024))
        restore(manifest, args.output_dir, source_dir=local, base_url=base)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Download failed: {error}\n'
                    'Verified parts are kept; rerun the same command to resume. '
                    'If the mirror returns 404, it may not have refreshed yet.\n')
    print('Extract both archives into the same empty directory, then follow its README.')


if __name__ == '__main__':
    main()
