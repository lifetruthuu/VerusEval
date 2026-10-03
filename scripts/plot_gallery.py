"""Regenerate the complete metric images from the matching released records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.reporting import metric_overview as overview
from scripts.validate import require, safe_path, verify_hash


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_group(table, expected, root):
    """Check table identities and every displayed score against primary JSON."""
    records = sorted((table.parent / 'per_file').glob('*.json'))
    paths = [table, *records]
    lineage = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        require(relative in expected, f'Unlisted gallery input: {relative}')
        verify_hash(safe_path(root, relative), expected[relative])
        lineage.append(f'{relative}\t{expected[relative]}\n')
    with table.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    names = [Path(row['filename']).stem for row in rows]
    require(len(names) == len(set(names)) and set(names) == {p.stem for p in records},
            f'CSV/JSON identity mismatch: {table}')
    for row in rows:
        path = table.parent / 'per_file' / (Path(row['filename']).stem + '.json')
        record = json.loads(path.read_text())
        for key, value in row.items():
            if not key.endswith('_score'):
                continue
            metric = record['metrics'].get(key[:-6], {})
            score = metric.get('generated', metric).get('score')
            same = (value == '' and score is None) or (
                value != '' and score is not None and
                math.isclose(float(value), float(score), rel_tol=0, abs_tol=1e-12))
            require(same, f'CSV/JSON score mismatch: {path.name}: {key}')
    return len(records), hashlib.sha256(''.join(lineage).encode()).hexdigest()


def generate(output):
    require(output != ROOT and not output.is_relative_to(ROOT / 'data'),
            'Choose an output directory outside the released data.')
    require(not output.exists() or not any(output.iterdir()), 'Output directory must be empty.')
    manifest_path = ROOT / 'artifact_manifest.json'
    require(manifest_path.is_file(), 'Missing Release data: extract veruseval-data.tar.gz first.')
    expected = {entry['path']: entry['sha256'] for entry in json.loads(manifest_path.read_text())['files']}
    tables = sorted((ROOT / 'data/evaluation').glob('*_by_model/*/*/*/scores.csv'))
    require(len(tables) == 36, 'Expected 36 verified/unverified score tables from the matching Release.')
    index_path = 'data/evaluation/artifact_index.csv'
    verify_hash(safe_path(ROOT, index_path), expected[index_path])
    with (ROOT / index_path).open(newline='', encoding='utf-8') as stream:
        index_rows = list(csv.DictReader(stream))
    indexed = {row['result_path'] for row in index_rows}
    models = {}
    for row in index_rows:
        group = (ROOT / row['result_path']).parent.parent
        model = row['sample_id'].split('|')[1]
        require(models.setdefault(group, model) == model, f'Mixed models in {group}')
    actual = {p.relative_to(ROOT).as_posix() for t in tables for p in (t.parent / 'per_file').glob('*.json')}
    require(actual == indexed and len(actual) == 13716, 'Gallery population differs from the evaluation index.')
    # Validate every source before publishing any image.
    groups = [(table, *check_group(table, expected, ROOT)) for table in tables]
    output.mkdir(parents=True, exist_ok=True)
    plt = overview._setup_matplotlib()
    images = []
    index = ['# Metric image gallery', '',
             '36 full-size images regenerated from the final released records: 18 configurations, '
             '13,716 programs. Click an image to open its original PNG.', '',
             'The verified/unverified labels retain the original generation partitions. '
             'They are not the final target-verifier acceptance labels used in RQ1. '
             'These overview images include the 57 missing-target cases; paper quality analyses exclude them.', '',
             'Chart policy: unresolved implications contribute zero; other missing bounded non-I/O '
             'scores receive the worst value. Empty I/O categories and unverified mutation/redundancy '
             'stay N/A. Verified programs with no testable proof clauses have chart redundancy zero. '
             'Hatched bars mean lower is better; complexity uses separate raw-count scales; time is in seconds. '
             'Stored evaluation metrics are unchanged. Reference means the comparison specification.', '',
             '[Input hashes and plotted summaries](manifest.json). '
             'Regenerate with `python scripts/plot_gallery.py --output-dir runs/gallery`.', '']
    index.extend(['| Workflow | Model | Prompt | Unverified | Verified |',
                  '| --- | --- | --- | --- | --- |'])
    pairs = {}
    for table, count, _ in groups:
        workflow, shot, model, status = table.relative_to(ROOT / 'data/evaluation').parts[:-1]
        workflow = workflow.removesuffix('_by_model')
        filename = f'{workflow}__{model}__{shot}__{status}.png'
        pairs.setdefault((workflow, models[table.parent], shot), {})[status] = f'[{count} programs]({filename})'
    for (workflow, model, shot), pair in sorted(pairs.items()):
        index.append(f"| {workflow} | {model} | {shot} | {pair['unverified']} | {pair['verified']} |")
    index.append('')
    for table, count, input_digest in groups:
        workflow, shot, model, status = table.relative_to(ROOT / 'data/evaluation').parts[:-1]
        workflow = workflow.removesuffix('_by_model')
        title = f'{workflow} / {models[table.parent]} / {shot} / {status}'
        filename = f'{workflow}__{model}__{shot}__{status}.png'
        values = overview.load_scores_many([table])
        values.update(overview.load_conservative_implication_scores_many([table.parent / 'per_file']))
        values.update(overview.load_sub_scores_many([table.parent / 'per_file']))
        paths = overview.fig_mean_hbar(plt, values, output, title, count)
        paths[0].rename(output / filename)
        images.append({'image': filename, 'programs': count, 'source': table.relative_to(ROOT).as_posix(),
                       'source_sha256': sha(table), 'csv_and_records_sha256': input_digest,
                       'image_sha256': sha(output / filename),
                       'metrics': {metric: {key: value for key, value in body.items() if key != 'scores'}
                                   for metric, body in values.items()}})
        index.extend([f'## {title} ({count} programs)', '', f'[![{title}]({filename})]({filename})', ''])
        print(f'{filename}: {count} programs', flush=True)
    require(sum(item['programs'] for item in images) == 13716, 'Gallery total mismatch.')
    (output / 'README.md').write_text('\n'.join(index), encoding='utf-8')
    report = {'schema_version': 1, 'programs': 13716, 'configurations': 18, 'images': images,
              'source_index_sha256': sha(ROOT / 'data/evaluation/artifact_index.csv'),
              'renderer_sha256': sha(Path(overview.__file__)), 'generator_sha256': sha(Path(__file__))}
    (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/gallery')
    args = parser.parse_args()
    try:
        generate(args.output_dir.resolve())
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'Gallery failed: {error}\n')
