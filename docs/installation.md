# Installation

Install Docker Engine (or Docker Desktop) with Docker Compose, then run from the
checkout root:

```bash
docker compose run --build --rm veruseval
```

The image targets Linux x86-64 (`linux/amd64`). The build downloads dependencies
and runs an environment check. Allow about 25 GB for Docker layers, extracted
data and results. Other processor architectures need Docker's amd64 emulation;
native ARM Verus is not substituted. The code directory is mounted at
`/workspace`, so commands and outputs use the same relative paths as the host.

Verus is fixed to `0.2025.09.25.04e8687`; its official Linux archive is checked
against SHA-256 `99e558140efe90ea58ae53a3aac6fe2d6c2123bd6840afe00549179e9a8c5948`.
Rust is fixed to `1.88.0`, matching the release's
[toolchain file](https://github.com/verus-lang/verus/blob/04e8687/rust-toolchain.toml).
The image includes `rustc-dev`, LLVM tools and rustfmt. `PATH`, `VERUS_PATH` and
`RUSTUP_TOOLCHAIN` select these versions for verifier calls and Rust I/O harnesses.
The build checks both version strings, verifies `examples/identity.rs`, and
compiles and runs a Rust program. Do not upgrade either tool independently when
reproducing this corpus.

Direct Python 3.11 dependencies are pinned in `docker/requirements.lock`. The image also
contains Debian's R 4.2.2 with ggplot2 4.0.3, dplyr 1.2.1 and patchwork 1.3.2,
CJK and Liberation fonts,
TeX Live with PGF and Libertine for RQ1's PDF export, Playwright Chromium,
CPU PyTorch for AlphaVerus, and the bundled Lynette built from its
Cargo lockfile. Browser launch, Z3, PyTorch and Lynette are checked during the
build, including a PGF-to-PDF rendering check. Model generation and LLM judging use remote APIs; supply your own model
endpoint and credentials at runtime with individual `-e NAME` flags (export the
variables on the host first). Credentials and data are excluded from the Docker
build context.

The R package versions and source archive hashes are fixed in
`docker/r-packages.lock.json`, including their dependencies. RQ2 needs the pinned
ggplot2 API; an older system ggplot2 is not sufficient.

Use the [download helper](../downloads/README.md) to obtain the matching code
and data archives, then extract both into the same empty directory. Run the
commands below from that extracted snapshot. Data are needed for reproduction
and gallery regeneration, but not for building the environment:

```bash
docker compose run --rm veruseval python scripts/validate.py --all-files
docker compose run --rm veruseval python scripts/reproduce.py --rq all --output-dir runs/full
docker compose run --rm veruseval python scripts/plot_gallery.py --output-dir runs/gallery
```

On Linux, add `--user "$(id -u):$(id -g)"` after `run --rm` to give generated
files your host ownership. Use `docker compose run --rm veruseval bash` for an
interactive shell. Frozen-result reproduction and gallery generation work
without network access after setup; LLM operations require a network connection.

For manual installation, `conda env create -f environment.yml` provides Python
and `conda env create -f environment-r.yml` provides R. Install Chromium with
`python -m playwright install --with-deps chromium`; pass the R environment's
`Rscript` path via `--rscript`. Install the same pinned Verus release and Rust
toolchain above, then place their executables on `PATH`. Docker is the checked
complete environment; these Conda files are a convenience for existing setups.
Manual figure reproduction also needs `pdflatex`, PGF and the Libertine TeX fonts
(Debian packages `texlive-latex-base`, `texlive-latex-recommended`,
`texlive-latex-extra`, `texlive-pictures` and `texlive-fonts-extra`).
