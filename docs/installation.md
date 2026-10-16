# Installation

Install Docker with Docker Compose and extract the matching
[code and data archives](../downloads/README.md) into one directory.
From that directory, run:

```bash
docker compose run --build --rm veruseval
```

The environment check should print `"status": "passed"`. Allow about 25 GB of
disk space. The image targets Linux x86-64; ARM hosts need Docker's amd64
emulation. The project is mounted at `/workspace`, with outputs saved on the host.

Docker includes Verus `0.2025.09.25.04e8687`, Rust `1.88.0`, Python 3.11,
R, and the plotting and baseline dependencies. Dependency versions and setup
commands are recorded in the [Dockerfile](../Dockerfile),
[Python lockfile](../docker/requirements.lock), and
[R lockfile](../docker/r-packages.lock.json).

Continue with [paper result reproduction](reproduction.md) or
[baseline generation and evaluation](../baselines/README.md). Paper result
reproduction works offline after setup; model calls require API credentials
and network access.

For an interactive shell, run `docker compose run --rm veruseval bash`.
On Linux, add `--user "$(id -u):$(id -g)"` after `run --rm` to give outputs
your host ownership.

## Local environment

For local evaluation on Linux x86-64, install Conda and rustup, extract the
matching code and data archives, and run from the project root:

```bash
conda env create -f environment.yml
conda activate veruseval
rustup toolchain install 1.88.0 --profile minimal
rustup component add --toolchain 1.88.0 rustc-dev llvm-tools rustfmt
export RUSTUP_TOOLCHAIN=1.88.0
```

This creates a Python 3.11 environment with the dependencies in
`requirements.txt`. Download and extract the
[Verus 0.2025.09.25.04e8687 Linux release](https://github.com/verus-lang/verus/releases/download/release/0.2025.09.25.04e8687/verus-0.2025.09.25.04e8687-x86-linux.zip),
keeping the executable and its bundled files together. Replace the placeholder
below with its absolute path:

```bash
export VERUS_PATH=/absolute/path/to/verus-x86-linux/verus
export PATH="$(dirname "$VERUS_PATH"):$PATH"
"$VERUS_PATH" --version
rustc --version
"$VERUS_PATH" examples/identity.rs
```

The versions should be Verus `0.2025.09.25.04e8687` and Rust `1.88.0`; the example
should report `0 errors`. Keep these exports in each terminal used for
evaluation, then follow the [local Python commands](usage.md#local-python).

For baseline generation, install the additional baseline dependencies and build
Lynette as shown in the [Dockerfile](../Dockerfile). For paper figures, also
prepare [R](../environment-r.yml), the versions in the
[R lockfile](../docker/r-packages.lock.json), and the TeX and Chromium dependencies
listed in the Dockerfile. Use `--rscript` to select your R executable when
reproducing figures.
