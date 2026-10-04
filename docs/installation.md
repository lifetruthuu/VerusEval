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

For manual setup, start with [the Python environment](../environment.yml) and
[the R environment](../environment-r.yml), then install the matching Verus, Rust,
and remaining dependencies listed in the Dockerfile. Use `--rscript` to select
your R executable when reproducing figures.
