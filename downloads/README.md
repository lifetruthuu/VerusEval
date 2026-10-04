# Download code and data

The matching code and data archives are stored as parts of at most 7 MiB so they
can be fetched from the anonymous mirror. The helper needs Python 3.9 or later
and curl; no extra Python packages are required. It downloads, joins, and checks
every part and both archives with SHA-256. Interrupted downloads resume from
verified parts.

```bash
curl -fL https://anonymous.4open.science/api/repo/VerusEval-A7E4/file/downloads/download.py -o download.py
python download.py
```

This writes `packages/veruseval-code.tar.gz`, `packages/veruseval-data.tar.gz`,
and `packages/SHA256SUMS`. Archive sizes are listed in `downloads/manifest.json`.
Extract both into one directory and start the pinned environment:

```bash
mkdir veruseval-release
tar -xzf packages/veruseval-code.tar.gz -C veruseval-release
tar -xzf packages/veruseval-data.tar.gz -C veruseval-release
cd veruseval-release
docker compose run --build --rm veruseval
docker compose run --rm veruseval python scripts/reproduce.py --rq all --output-dir runs/full
```

Then follow [Part 1: reproduce paper results](../docs/reproduction.md) or
[Part 2: run and evaluate baselines](../baselines/README.md).

From a complete Git checkout, `python downloads/download.py` joins the local
parts without network access. A standalone copy of the script downloads the
parts from the anonymous mirror. The
[download manifest](https://anonymous.4open.science/api/repo/VerusEval-A7E4/file/downloads/manifest.json)
and [archive checksums](https://anonymous.4open.science/api/repo/VerusEval-A7E4/file/downloads/SHA256SUMS)
stay in the repository and are not inside the code archive.
