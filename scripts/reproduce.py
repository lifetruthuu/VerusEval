"""Reproduce the four research questions from the released final records."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from validate import compare_results, validate

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rq', choices=['all', '1', '2', '3', '4'], default='all')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/reproduced')
    parser.add_argument('--statistics-only', action='store_true', help='Generate statistics and tables without figures.')
    parser.add_argument('--rscript', default=os.environ.get('RSCRIPT', 'Rscript'))
    args = parser.parse_args()
    selected = set('1234') if args.rq == 'all' else {args.rq}
    destination = args.output_dir.resolve()
    if destination == ROOT or destination.is_relative_to(ROOT / 'data'):
        parser.error('Choose an output directory outside the released data.')
    if destination.exists() and any(destination.iterdir()):
        parser.error('Output directory must be empty; choose a new directory.')
    if '2' in selected and not args.statistics_only and shutil.which(args.rscript) is None:
        parser.error('Rscript is missing; set --rscript or use --statistics-only.')
    try:
        integrity = validate(all_files=True)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'Input validation failed: {error}\n')
    destination.mkdir(parents=True, exist_ok=True)
    work = destination / 'work'
    work.mkdir()
    for name in ('scripts', 'metrics_rebuild', 'RQs'):
        shutil.copytree(ROOT / name, work / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (work / 'data').symlink_to(ROOT / 'data', target_is_directory=True)
    for rq in range(1, 5):
        for name in ('results', 'tables', 'figures', 'figure_captions'):
            (work / f'RQs/RQ{rq}' / name).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(work), MPLBACKEND='Agg', PYTHONHASHSEED='0')
    stages = []
    report = {'status': 'running', 'research_questions': sorted(selected), 'stages': stages,
              'input_validation': integrity, 'verifier_or_llm_invoked': False,
              'figures_requested': not args.statistics_only}

    def save():
        (destination / 'reproduction.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(script, *arguments, executable=None):
        start = time.monotonic()
        log = destination / f'{len(stages) + 1:02d}_{Path(script).stem}.log'
        print(f'Running {script}', flush=True)
        with log.open('w') as stream:
            result = subprocess.run([executable or sys.executable, script, *arguments], cwd=work,
                                    env=env, stdout=stream, stderr=subprocess.STDOUT)
        stages.append({'script': script, 'exit_code': result.returncode, 'log': log.name,
                       'seconds': round(time.monotonic() - start, 2)})
        if result.returncode:
            report['status'] = 'failed'
            save()
            parser.exit(1, log.read_text()[-6000:] + f'\nStage failed; see {log}\n')
        save()

    stats = ['--statistics-only'] if args.statistics_only else []
    # RQ2 cross-checks RQ1's score encodings; RQ3 checks RQ1's accepted profiles.
    if selected & set('123'):
        run('RQs/RQ1/scripts/refresh_rq1_paper.py')
        run('RQs/RQ1/scripts/refresh_rq1_screening.py', *stats)
        run('RQs/RQ1/scripts/validate_rq1_table4.py')
    if '2' in selected:
        run('RQs/RQ2/scripts/analyze_rq2_configurations.py')
        run('RQs/RQ2/scripts/validate_rq2_results.py')
        if not args.statistics_only:
            run('RQs/RQ2/scripts/plot_rq2_figures.R', executable=args.rscript)
    if '3' in selected:
        run('RQs/RQ3/scripts/analyze_rq3_natural.py')
        run('RQs/RQ3/scripts/analyze_rq3_variants.py')
        run('RQs/RQ3/scripts/validate_rq3_results.py')
        run('RQs/RQ3/scripts/refresh_rq3_paper.py')
        if not args.statistics_only:
            run('RQs/RQ3/scripts/plot_rq3.py')
    if '4' in selected:
        run('RQs/RQ4/scripts/build_rq4.py')
        run('RQs/RQ4/scripts/refresh_rq4_paper.py', *stats)
        run('RQs/RQ4/scripts/validate_rq4.py')
        if not args.statistics_only:
            run('RQs/RQ4/scripts/build_reference_defects_figure.py')
    try:
        report['numerical_tables_compared'] = compare_results(work, ROOT / 'data/evidence/paper_results', selected)
    except (ValueError, OSError) as error:
        report['status'] = 'failed'
        save()
        parser.exit(1, f'Result comparison failed: {error}\n')
    report['status'] = 'passed'
    save()
    print(f'Completed. Results, tables and figures: {work / "RQs"}')


if __name__ == '__main__':
    main()
