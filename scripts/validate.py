"""Validate the released corpus and compare reproduced numerical results."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = {
    '1': ['RQ1/results/rq1_acceptance_association.csv', 'RQ1/results/rq1_acceptance_artifacts.csv', 'RQ1/results/screening/accepted_profiles.csv'],
    '2': ['RQ2/results/rq2_configuration_profiles.csv', 'RQ2/results/rq2_prompt_deltas.csv'],
    '3': ['RQ3/results/natural/conditional_rates.csv', 'RQ3/results/variants/detection_matrix.csv', 'RQ3/results/variants/operator_selection.csv'],
    '4': ['RQ4/results/directions.csv', 'RQ4/results/combinations.csv', 'RQ4/results/clause_summary.csv', 'RQ4/results/reference_screen/screen_signals.csv', 'RQ4/results/reference_screen/sensitivity.csv'],
}


def rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(root, relative):
    path = root / relative
    require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
            f'Invalid release path: {relative}')
    require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
            f'Symlink or path outside release: {relative}')
    require(path.is_file(), f'Missing data or code file: {relative}')
    return path


def verify_hash(path, expected):
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    require(actual == expected, f'Hash mismatch: {path}')


def validate_references(root, catalog):
    paths = [row['reference_path'] for row in catalog]
    require(len(set(paths)) == len(catalog), 'References are not one-to-one with tasks.')
    for row in catalog:
        require(row['reference_path'].startswith('data/references/'), 'Invalid reference directory.')
        verify_hash(safe_path(root, row['reference_path']), row['reference_sha256'])
    return len(paths)


def validate(root=ROOT, all_files=False):
    data = root / 'data/evaluation'
    require((data / 'artifact_index.csv').is_file(),
            'Missing data: unpack veruseval-data.tar.gz at the repository root first.')
    manifest_path = root / 'artifact_manifest.json'
    require(manifest_path.is_file(), 'Missing release manifest: unpack the matching data archive.')
    manifest = json.loads(manifest_path.read_text())
    # Metadata is always verified before following its paths.
    metadata = {'data/evaluation/artifact_index.csv', 'data/evaluation/artifact_outcomes.csv',
                'data/evaluation/rq1_artifact_outcomes.csv', 'data/evaluation/target_functions.csv',
                'data/generated/manifest.csv', 'data/references/availability.json'}
    checked = set()
    for entry in manifest['files']:
        if all_files or entry['path'] in metadata:
            path = safe_path(root, entry['path'])
            verify_hash(path, entry['sha256'])
            checked.add(entry['path'])
    require(metadata <= checked, 'Release manifest does not cover all corpus indexes.')
    index, outcomes = rows(data / 'artifact_index.csv'), rows(data / 'artifact_outcomes.csv')
    eligible = rows(data / 'rq1_artifact_outcomes.csv')
    corpus = rows(root / 'data/generated/manifest.csv')
    catalog = rows(data / 'target_functions.csv')
    ids = {r['sample_id'] for r in index}
    require(len(index) == len(ids) == 13716, 'Expected 13,716 unique evaluation records.')
    require(len(outcomes) == len(ids) and {r['sample_id'] for r in outcomes} == ids, 'Outcome population mismatch.')
    require(len(eligible) == len({r['sample_id'] for r in eligible}) == 13659, 'Quality population mismatch.')
    require({r['sample_id'] for r in eligible} == {r['sample_id'] for r in index if r['analysis_eligible'] == 'True'}, 'Exclusion identity mismatch.')
    require(Counter(r['verification_stage'] for r in eligible)['accepted'] == 7081, 'Accepted population mismatch.')
    require(len(catalog) == len({r['task_id'] for r in catalog}) == 762, 'Target catalog mismatch.')
    references_checked = validate_references(root, catalog)
    availability = json.loads((root / 'data/references/availability.json').read_text())
    require(availability == {'tasks': 762, 'available': 762, 'missing_reference': []},
            'Reference availability metadata mismatch.')
    sources = {r['sample_id']: r for r in corpus}
    require(len(sources) == len(corpus) == len(index) and set(sources) == ids, 'Generated-source pairing mismatch.')
    configs = {(r['workflow'], r['model'], r['shot']) for r in corpus}
    require(len(configs) == 18, 'Expected 18 generation configurations.')
    task_ids = {r['task_id'] for r in catalog}
    require({r['task_id'] for r in corpus} == task_ids, 'Generated task set differs from catalog.')
    require(len({r['generated_path'] for r in corpus}) == 13716, 'Generated sources are not one-to-one.')
    for entry in index:
        source = sources[entry['sample_id']]
        path = safe_path(root, entry['result_path'])
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == entry['result_sha256'],
                f"Indexed record hash mismatch: {entry['result_path']}")
        record = json.loads(raw)
        require(source['generated_path'] == record['generated_path'], f"Source path mismatch: {entry['sample_id']}")
        for field in ('result_path', 'result_sha256', 'analysis_eligible'):
            require(source[field] == entry[field], f"{field} mismatch: {entry['sample_id']}")
        generated = safe_path(root, source['generated_path'])
        verify_hash(generated, source['generated_sha256'])
    return {'status': 'passed', 'artifacts': len(index), 'tasks': len(catalog), 'configurations': len(configs),
            'analysis_artifacts': len(eligible), 'excluded': len(index) - len(eligible),
            'generated_source_hashes_checked': len(corpus), 'per_file_hashes_checked': len(index),
            'reference_hashes_checked': references_checked,
            'release_files_checked': len(checked)}


def equal_cell(left, right):
    if left == right:
        return True
    try:
        a, b = float(left), float(right)
        return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
    except (ValueError, TypeError):
        return False


def compare_results(work, frozen, selected):
    compared = []
    for rq in sorted(selected):
        for name in TABLES[rq]:
            current, previous = rows(work / 'RQs' / name), rows(frozen / name)
            require(len(current) == len(previous), f'Changed row count: {name}')
            for i, (a, b) in enumerate(zip(current, previous)):
                require(a.keys() == b.keys(), f'Changed columns: {name}')
                for field in a:
                    require(equal_cell(a[field], b[field]), f'Changed result: {name}, row {i}, {field}')
            compared.append(name)
    return compared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-files', action='store_true', help='Hash every published file, including proof evidence.')
    parser.add_argument('--results', type=Path, help='Reproduction work directory to compare with frozen tables.')
    parser.add_argument('--rq', choices=['all', '1', '2', '3', '4'], default='all')
    args = parser.parse_args()
    try:
        result = validate(all_files=args.all_files)
        if args.results:
            selected = set('1234') if args.rq == 'all' else {args.rq}
            result['numerical_tables_compared'] = compare_results(args.results, ROOT / 'data/evidence/paper_results', selected)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'Validation failed: {error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
