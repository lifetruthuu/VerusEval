# VerusEval

Evaluate generated Verus specifications, reproduce four research questions,
and run the included baseline adapters. The released corpus covers 13,716
programs, 762 tasks, 762 reference files and 18 configurations.

[Browse the full metric images](gallery/README.md) ·
[Download code and data](downloads/README.md) ·
[Installation](docs/installation.md) · [Usage](docs/usage.md) ·
[Reproduction](docs/reproduction.md) · [Data](docs/data.md)

Download the [matching code and data archives](downloads/README.md) and extract
both into the same empty directory. With Docker and Docker Compose installed,
run from that directory:

```bash
docker compose run --build --rm veruseval
docker compose run --rm veruseval python scripts/reproduce.py --rq all --output-dir runs/full
```

This installs Verus `0.2025.09.25.04e8687`, Rust `1.88.0`, Python 3.11,
R, and the plotting and baseline dependencies. Downloads are available directly
from this repository through the anonymous mirror.

The four RQs cover acceptance and quality, generation configurations, I/O
detection, and contract differences with reference screening and repair.
Use `--rq 1|2|3|4` to select one, or `--statistics-only` to skip figures.
Results are saved in `runs/full/`.

Code: [MIT](LICENSE). Data and third-party terms:
[data license](DATA_LICENSE.md), [third-party notices](THIRD_PARTY.md).
