# Download code and data

The matching code and data archives are stored in this repository as parts of
at most 7 MiB, so they can be downloaded through the anonymous mirror.
The helper downloads, joins and checks them automatically. It uses Python 3.9
or later and curl; no extra Python packages are needed.

Save [download.py](download.py) and run it, or use these commands:

```bash
curl -fL https://anonymous.4open.science/api/repo/VerusEval-A7E4/file/downloads/download.py -o download.py
python download.py
```

This creates `packages/veruseval-code.tar.gz`, `packages/veruseval-data.tar.gz`
and `packages/SHA256SUMS`. The two archives total approximately 511 MiB.
Interrupted downloads resume from verified parts when the command is repeated.
Every part and both assembled archives are checked with SHA-256.

Extract both archives into a new directory and run the pinned Docker environment:

```bash
mkdir veruseval-release
tar -xzf packages/veruseval-code.tar.gz -C veruseval-release
tar -xzf packages/veruseval-data.tar.gz -C veruseval-release
cd veruseval-release
docker compose run --build --rm veruseval
docker compose run --rm veruseval python scripts/reproduce.py --rq all --output-dir runs/full
```

Use the code and data archives together: their combined manifest verifies the
exact frozen code and records. The code archive includes this guide and the
download helper. If both archives are already extracted, start with the Docker
commands above.

From a complete Git checkout, `python downloads/download.py` joins the local
parts without network access or curl. From an extracted code archive, the same
command downloads the parts from the anonymous mirror. The parts,
[download manifest](https://anonymous.4open.science/api/repo/VerusEval-A7E4/file/downloads/manifest.json)
and [archive checksums](https://anonymous.4open.science/api/repo/VerusEval-A7E4/file/downloads/SHA256SUMS)
are hosted in the repository and are not embedded in the code archive.
