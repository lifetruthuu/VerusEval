"""Build RQ1 screening and sensitivity results from released evaluation records."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import refresh_rq1_paper as paper
from analyze_rq1_accepted_profiles import FORMAL, IO, formal_state, io_state, triviality_state
from analyze_rq1_acceptance_association import pair_counts, tau_b
from rq1_paths import RESULTS, SCREENING

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from RQs.shared import records

OUT = SCREENING


def evidence_for(item):
    sample, index, outcome = item
    failures, reasons = [], []
    if any(int(outcome[k + '_' + s]) for k in IO for s in ('failed', 'unresolved')):
        payload = records.read_json(ROOT / index['result_path'], index['result_sha256'])
        for prefix, metric in records.IO.items():
            node = records.generated(payload, metric)
            bad, unknown = [], []
            for case in node.get('details') or []:
                evaluation = case.get('evaluation') or {}
                if case.get('success') is False:
                    bad.append(prefix != 'io_invalid' or evaluation.get('requires_accepted') is True)
                elif case.get('success') is None:
                    unknown.append(evaluation.get('reason') or case.get('reason'))
            if not node.get('details'):
                unknown = [r for r, n in (node.get('unknown_reason_summary') or {}).items() for _ in range(n)]
            assert len(bad) == int(outcome[prefix + '_failed']), (sample, prefix)
            assert len(unknown) == int(outcome[prefix + '_unresolved']), (sample, prefix)
            failures.extend(bad)
            reasons.extend(unknown)
    return sample, {'proved_failure': any(failures), 'reasons': reasons,
                    'undecided': any(r in paper.UNDECIDED for r in reasons)}


def main(statistics_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    labels = records.read_csv(records.DEST / 'rq1_artifact_labels.csv')
    outcomes = {r['sample_id']: r for r in records.read_csv(records.DEST / 'rq1_artifact_outcomes.csv')}
    index = {r['sample_id']: r for r in records.read_csv(records.DEST / 'artifact_index.csv')}
    selected = [r for r in labels if r['verification_stage'] in ('accepted', 'proof_failed')]
    with ThreadPoolExecutor(16) as pool:
        evidence = dict(pool.map(evidence_for, [(r['sample_id'], index[r['sample_id']], outcomes[r['sample_id']]) for r in selected]))
    profiles = []
    for label in selected:
        if label['verification_stage'] != 'accepted':
            continue
        sample = label['sample_id']
        outcome, ev = outcomes[sample], evidence[sample]
        p = {'sample_id': sample, 'task_id': label['task_id']}
        p.update({name: formal_state(outcome, keys) for name, keys in FORMAL.items()})
        p.update(io=io_state(outcome), triviality=triviality_state(label))
        p['complete'] = int(all(p[k] in ('pass', 'fail') for k in paper.CHECKS))
        p['flag_mask'] = ''.join('1' if p[k] == 'fail' else '0' for k in paper.CHECKS)
        p['available_io_kinds'] = sum(any(int(outcome[k + '_' + s]) for s in ('passed', 'failed', 'unresolved')) for k in IO)
        p['io_failure_evidence'] = 'proved' if ev['proved_failure'] else 'unproved' if p['io'] == 'fail' else ''
        p['flag_evidence'] = '' if p['flag_mask'] == '000' else (
            'counterexample_or_proved_triviality' if ev['proved_failure'] or p['triviality'] == 'fail' else
            'unproved_io_failure' if p['io'] == 'fail' else 'reference_rejection_only')
        directions = {outcome[d] for d in paper.DIRECTIONS}
        p['indeterminate_cause'] = '' if p['complete'] or p['flag_mask'] != '000' else (
            'unresolved' if ev['undecided'] or ('unknown' in directions and 'unavailable' not in directions) else 'unevaluable_or_missing')
        profiles.append(p)
    regions = {k: sum(p['flag_mask'] == k for p in profiles) for k in paper.ORDER}
    incomplete = {k: sum(p['flag_mask'] == k and not p['complete'] for p in profiles) for k in paper.ORDER}
    joint = [p for p in profiles if p['complete'] and p['flag_mask'] == '000']
    summary = {
        'venn_population': len(profiles), 'venn_regions': regions,
        'venn_regions_with_indeterminate_check': incomplete,
        'venn_marginals': {k: sum(p[k] == 'fail' for p in profiles) for k in paper.CHECKS},
        'joint_pass_available_cases': len(joint), 'at_least_one_flag': len(profiles) - regions['000'],
        'no_flag_but_indeterminate': incomplete['000'],
        'flag_evidence': dict(Counter(p['flag_evidence'] for p in profiles if p['flag_evidence'])),
        'unflagged_indeterminate_causes': dict(Counter(p['indeterminate_cause'] for p in profiles if p['indeterminate_cause'])),
        'unflagged_incomplete_decided_cases_pass': sum(
            p['flag_mask'] == '000' and not p['complete'] and p['equivalence'] == 'pass' and p['triviality'] == 'pass'
            and sum(int(outcomes[p['sample_id']][k + '_passed']) for k in IO) > 0 for p in profiles),
        'joint_pass_io_kinds': dict(Counter(p['available_io_kinds'] for p in joint)),
        'joint_pass_missing_io_kind': {k: sum(not any(int(outcomes[p['sample_id']][k + '_' + s]) for s in ('passed', 'failed', 'unresolved')) for p in joint) for k in IO},
        'rq3_all_three_io_kinds_nonempty_and_pass': sum(p['io'] == 'pass' and p['available_io_kinds'] == 3 for p in profiles),
        'reference_only_io_states': dict(Counter(p['io'] for p in profiles if p['flag_mask'] == '100')),
        'outside_reference_flagged_states': dict(Counter(p['equivalence'] for p in profiles if p['flag_mask'] in ('010', '001', '011'))),
        'venn_unproved_io_flags': dict(Counter(p['flag_mask'] for p in profiles if p['io_failure_evidence'] == 'unproved')),
    }
    encoded_path = RESULTS / 'rq1_acceptance_artifacts.csv'
    encoded = records.read_csv(encoded_path)
    ill_formed = {s for s, ev in evidence.items() if 'contract_ill_formed' in ev['reasons']}
    sensitivity = {'artifacts': len(ill_formed), 'accepted': sum(r['accepted'] == '1' for r in encoded if r['sample_id'] in ill_formed)}
    for metric in ('Correct-I/O acceptance', 'Wrong-output rejection', 'Invalid-input rejection', 'Equivalence'):
        usable = [r for r in encoded if r[metric] and r['sample_id'] not in ill_formed]
        sensitivity[metric] = float(tau_b(pair_counts(np.array([r['accepted'] == '1' for r in usable]), np.array([float(r[metric]) for r in usable]))))
    summary['association_without_ill_formed'] = sensitivity
    summary['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               (records.DEST / 'artifact_index.csv', records.DEST / 'rq1_artifact_outcomes.csv',
                                records.DEST / 'rq1_artifact_labels.csv', encoded_path, Path(__file__))}
    assert len(joint) + summary['at_least_one_flag'] + incomplete['000'] == len(profiles)
    records.write_csv(OUT / 'accepted_profiles.csv', profiles)
    records.write_json(OUT / 'summary.json', summary)
    paper.OUTPUT = OUT
    if not statistics_only:
        paper.draw_venn(summary)
        paper.write_figure(summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statistics-only', action='store_true')
    main(parser.parse_args().statistics_only)
