"""Independently validate Table 4 ranks and task-bootstrap binary intervals."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from rq1_paths import RESULTS

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = RESULTS


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def rank_tau(accepted, values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ranks = (2 * counts.cumsum() - counts + 1) / 2
    high, low = accepted.sum(), (~accepted).sum()
    u = ranks[inverse[accepted]].sum() - high * (high + 1) / 2
    untied = (len(values) * (len(values) - 1) - (counts * (counts - 1)).sum()) / 2
    denominator = np.sqrt(high * low * untied)
    return (2 * u - high * low) / denominator if denominator else np.nan


def validate():
    summary = json.loads((OUTPUT / 'summary.json').read_text())
    assert summary['status'] == 'complete'
    for path, expected in {**summary['source_sha256'], **summary['code_sha256'], **summary['result_sha256']}.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    assert hashlib.sha256((ROOT / summary['table']).read_bytes()).hexdigest() == summary['table_sha256']
    records = read_csv(OUTPUT / 'rq1_acceptance_artifacts.csv')
    metrics = read_csv(OUTPUT / 'rq1_acceptance_association.csv')
    assert len({r['sample_id'] for r in records}) == len(records)
    task_ids = sorted({r['task_id'] for r in records})
    task_codes = np.searchsorted(task_ids, [r['task_id'] for r in records])
    accepted = np.array([r['accepted'] == '1' for r in records])
    rng = np.random.default_rng(summary['bootstrap_seed'])
    weights = np.array([np.bincount(rng.integers(len(task_ids), size=len(task_ids)), minlength=len(task_ids))
                        for _ in range(summary['bootstrap_replicates'])])
    checks, bootstrap = [], []
    for metric in metrics:
        name = metric['metric']
        values = np.array([float(r[name]) if r[name] else np.nan for r in records])
        ok = np.isfinite(values)
        a, y = accepted[ok], values[ok]
        tau = rank_tau(a, y)
        assert int(a.sum()) == int(metric['accepted_n'])
        assert int((~a).sum()) == int(metric['proof_failed_n'])
        assert np.isclose(y[a].mean(), float(metric['accepted_mean']), atol=1e-12, rtol=0)
        assert np.isclose(y[~a].mean(), float(metric['proof_failed_mean']), atol=1e-12, rtol=0)
        assert np.isclose(tau, float(metric['tau_b']), atol=1e-12, rtol=0)
        checks.append({'metric': name, 'accepted_n': int(a.sum()), 'proof_failed_n': int((~a).sum()),
                       'tau_b': float(tau), 'status': 'passed'})
        if metric['kind'] != 'binary':
            continue
        assert set(y) <= {0, 1}
        cells = np.zeros((len(task_ids), 4), dtype=int)
        category = (~a).astype(int) * 2 + (y == 0).astype(int)
        np.add.at(cells, (task_codes[ok], category), 1)
        aa, bb, cc, dd = (weights @ cells).astype(float).T
        with np.errstate(invalid='ignore', divide='ignore'):
            estimates = (aa * dd - bb * cc) / np.sqrt((aa + bb) * (cc + dd) * (aa + cc) * (bb + dd))
        finite = estimates[np.isfinite(estimates)]
        interval = np.quantile(finite, (.025, .975)) if len(finite) >= np.ceil(.95 * len(estimates)) else None
        expected = np.array([float(metric['ci_low']), float(metric['ci_high'])])
        if interval is None:
            assert np.isnan(expected).all()
        else:
            assert np.allclose(interval, expected, atol=1e-12, rtol=0), name
        bootstrap.append({'metric': name, 'valid_replicates': len(finite), 'total_replicates': len(estimates),
                          'interval': interval.tolist() if interval is not None else None, 'status': 'passed'})
    result = {'status': 'complete', 'method': 'Independent rank-sum identity for Kendall tau-b and task-weighted contingency-table phi for binary bootstrap; no production pair-count functions imported.',
              'source_and_code_hashes_verified': True, 'paper_table_hash_verified': True,
              'groups': summary['groups'], 'tasks': len(task_ids), 'metric_checks': checks,
              'bootstrap_unit': 'task', 'bootstrap_replicates': summary['bootstrap_replicates'],
              'bootstrap_seed': summary['bootstrap_seed'], 'bootstrap_checks': bootstrap,
              'summary_sha256': hashlib.sha256((OUTPUT / 'summary.json').read_bytes()).hexdigest()}
    (OUTPUT / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Validated {len(checks)} metrics and {len(bootstrap)} binary bootstrap intervals.')


if __name__ == '__main__':
    validate()
