# VerusEval

Evaluate generated Verus specifications, reproduce four research questions,
and run the included baseline adapters. The released corpus covers 13,716
programs, 762 tasks and 18 configurations.

[Browse the full metric images](gallery/README.md) ·
[Installation](docs/installation.md) · [Usage](docs/usage.md) ·
[Reproduction](docs/reproduction.md) · [Data](docs/data.md)

With Docker and Docker Compose installed, build and check the environment:

```bash
docker compose run --build --rm veruseval
```

This installs Verus `0.2025.09.25.04e8687`, Rust `1.88.0`, Python 3.11,
R, and the plotting and baseline dependencies. Download
[veruseval-data.tar.gz](https://github.com/lifetruthuu/VerusEval/releases/download/v1.0.0/veruseval-data.tar.gz)
and [SHA256SUMS](https://github.com/lifetruthuu/VerusEval/releases/download/v1.0.0/SHA256SUMS)
from the [v1.0.0 Release](https://github.com/lifetruthuu/VerusEval/releases/tag/v1.0.0), then run:

```bash
sha256sum --check --ignore-missing SHA256SUMS
tar -xzf veruseval-data.tar.gz
docker compose run --rm veruseval python scripts/reproduce.py --rq all --output-dir runs/full
```

The four RQs cover acceptance and quality, generation configurations, I/O
detection, and contract differences with reference screening and repair.
Use `--rq 1|2|3|4` to select one, or `--statistics-only` to skip figures.
Results are saved in `runs/full/`.

Code: [MIT](LICENSE). Data and third-party terms:
[data license](DATA_LICENSE.md), [third-party notices](THIRD_PARTY.md).
