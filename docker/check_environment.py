"""Fail the image build if the verifier, compiler, browser or plots are unusable."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import tempfile

from playwright.sync_api import sync_playwright


def run(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def main():
    rust = run('rustc', '--version')
    verus = run('verus', '--version')
    if not rust.startswith('rustc 1.88.0 ') or '0.2025.09.25.04e8687' not in verus:
        raise RuntimeError(f'Wrong toolchain: {rust}; {verus}')
    proof = run('verus', '/opt/identity.rs')
    if '0 errors' not in proof:
        raise RuntimeError(proof)
    # I/O harnesses also compile and execute ordinary Rust programs.
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / 'io.rs'
        source.write_text('fn main() { assert_eq!(2 + 2, 4); }\n')
        run('rustc', str(source), '-o', str(Path(directory) / 'io'))
        run(str(Path(directory) / 'io'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'pgf.texsystem': 'pdflatex', 'pgf.rcfonts': False,
                             'pgf.preamble': r'\usepackage[T1]{fontenc}\usepackage{libertine}'})
        figure, axes = plt.subplots()
        axes.text(0.5, 0.5, 'VerusEval 0.5')
        figure.savefig(Path(directory) / 'plot.pdf', backend='pgf')
        plt.close(figure)
    r = run('Rscript', '-e', 'library(ggplot2); library(dplyr); library(patchwork); cat(R.version.string)')
    r_packages = run('Rscript', '-e', 'stopifnot(packageVersion("ggplot2") == "4.0.3", '
                     'packageVersion("patchwork") == "1.3.2"); '
                     'ggplot2::position_dodge(orientation="y"); '
                     'cat(paste(c("ggplot2", "dplyr", "patchwork"), '
                     'sapply(c("ggplot2", "dplyr", "patchwork"), function(p) as.character(packageVersion(p))), collapse="; "))')
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content('<p>VerusEval</p>')
        assert page.locator('p').inner_text() == 'VerusEval'
        browser.close()
    run('/opt/tools/lynette', '--help')
    import torch
    import z3
    assert torch.softmax(torch.tensor([1.0, 1.0]), dim=0).tolist() == [0.5, 0.5]
    assert z3.simplify(z3.IntVal(2) + 2).as_long() == 4
    print(json.dumps({'status': 'passed', 'rust': rust, 'verus': verus,
                      'tex': run('pdflatex', '--version').splitlines()[0],
                      'r': r.splitlines()[-1], 'r_packages': r_packages.splitlines()[-1], 'python_packages': {
                          name: importlib.metadata.version(name)
                          for name in ('numpy', 'matplotlib', 'playwright', 'torch', 'z3-solver')
                      }}, indent=2))


if __name__ == '__main__':
    main()
