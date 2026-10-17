# Part 3: Generate programs with the baselines

Use this guide to generate programs with AutoVerus, VeruSAGE, StarVerus, and
AlphaVerus, then evaluate them with VerusEval. To check the paper's released
results, follow [Part 1](../docs/reproduction.md). To re-evaluate released
programs or new outputs, follow [Part 2](../docs/usage.md).

## Prepare

Extract the matching [code and data archives](../downloads/README.md) into one
directory and set up the [Docker environment](../docs/installation.md). Run all
commands from that directory.

Each configuration generates programs for the 762 unannotated tasks in
`data/generation/X_code/`. Few-shot runs add five examples from
`data/generation/`; zero-shot runs add none.

Check all 18 configurations without calling a model:

```bash
docker compose run --rm veruseval python scripts/generation/run_dataset.py \
  --dry-run --output-root runs/generation-check
```

A successful check writes `"status": "checked"` to
`runs/generation-check/generation.json`. Use a new output directory for the real run.

Set the API key and endpoint on the host:

```bash
export OPENAI_API_KEY='YOUR_API_KEY'
export OPENAI_BASE_URL='https://your-provider.example/v1'
```

Keys stay in environment variables and never go into JSON files.
[generation.json](generation.json) names the variables each model uses. Pass
them into Docker with `-e VARIABLE`.

## Run generation

[generation.json](generation.json) defines the 18 workflow, model, and shot
combinations, including models, endpoints, and budgets. To change settings, copy
the file under `runs/`, edit the copy, and pass it with `--config`.

Before each run, make sure the endpoint serves the selected model. Run one
baseline with one model:

```bash
# AutoVerus
docker compose run --rm -e OPENAI_API_KEY -e OPENAI_BASE_URL veruseval \
  python scripts/generation/run_dataset.py --baseline autoverus --model gpt-4o \
  --shot zero-shot --output-root runs/autoverus

# VeruSAGE
docker compose run --rm -e OPENAI_API_KEY -e OPENAI_BASE_URL veruseval \
  python scripts/generation/run_dataset.py --baseline verusage --model qwen-coder \
  --shot zero-shot --output-root runs/verusage

# StarVerus
docker compose run --rm -e OPENAI_API_KEY -e OPENAI_BASE_URL veruseval \
  python scripts/generation/run_dataset.py --baseline starverus --model gpt-4o \
  --shot zero-shot --output-root runs/starverus

# AlphaVerus
docker compose run --rm -e OPENAI_API_KEY -e OPENAI_BASE_URL veruseval \
  python scripts/generation/run_dataset.py --baseline alphaverus --model llama \
  --shot zero-shot --output-root runs/alphaverus
```

Use `--shot few-shot` for the few-shot setting, or omit `--shot` to run both.
Use `--benchmark HumanEval-Verus` to limit the tasks, and `--workers 4` to run
four tasks at once. Omitting `--baseline`, `--model`, and `--shot` runs all 18
configurations and requires access to every model. Always use a new output root.
Generation sends paid requests to your provider.

To test a different provider model, keep the dataset label and override the
model ID and endpoint:

```bash
docker compose run --rm -e OPENAI_API_KEY veruseval \
  python scripts/generation/run_dataset.py --baseline verusage --model deepseek-chat \
  --api-model YOUR_MODEL_ID --base-url https://your-provider.example/v1 \
  --shot zero-shot --benchmark HumanEval-Verus --output-root runs/provider-check
```

This runs the whole HumanEval-Verus subset. Output from a different model is a
new test run, not a reproduction of the paper's outputs.

## Find outputs

Each configuration writes to `<output-root>/<baseline>/<model>/<shot>/`, which
contains `generation.json` (resolved commands, parameters, and exit status),
`inputs.json` (input hashes), and `logs/`.

Final programs are in `results/` for AutoVerus and AlphaVerus, under `verusage/`
for VeruSAGE, and at the `selected=` path in the StarVerus logs.

A completed pipeline can still keep an unverified program. Verification alone
does not show that the target function was preserved or that its contract is
correct, so check the per-task records as well.

## Evaluate new programs

[target_functions.csv](../data/evaluation/target_functions.csv) maps task IDs to
the references in `data/references/`. Generation examples in `data/generation/Y/`
differ from the evaluation references for some tasks, so always evaluate against
the catalog.

To evaluate one generated file, give its task ID's reference:

```bash
docker compose run --rm veruseval python scripts/reevaluate.py file \
  -g runs/autoverus/autoverus/gpt-4o/zero-shot/results/VeriCoding/VeriCoding_VV0178_vericoded.rs \
  -r data/references/VeriCoding/VeriCoding_VV0178_vericoded.rs \
  --workflow AutoVerus --model gpt-4o --shot zero-shot --io-root data/io \
  -o runs/evaluation-one --verus-path /opt/verus/verus
```

For a batch, first write a manifest with one final file per task. Set `baseline`
and `run` in the script, and choose a new `pairs` directory. The script includes
verified and unverified final programs, skips intermediate copies, and stops on
failed tasks or duplicates. Run it after generation has finished, in the same
Docker workspace used for generation.

```bash
docker compose run --rm -T veruseval python - <<'PY'
import csv
import json
from pathlib import Path
import re

baseline = "autoverus"
run = Path("runs/autoverus/autoverus/gpt-4o/zero-shot")
pairs = Path("runs/evaluation-inputs/autoverus-gpt4o-zero")
with Path("data/evaluation/target_functions.csv").open() as stream:
    references = {row["task_id"]: Path(row["reference_path"])
                  for row in csv.DictReader(stream)}
selected = {}
def add(task, source):
    if task in selected:
        raise ValueError(f"Duplicate final output: {task}")
    if task not in references or not source.is_file():
        raise ValueError(f"Missing task/reference/output: {task}: {source}")
    selected[task] = source

if baseline in ("autoverus", "alphaverus", "verusage"):
    pattern = (".completion_work/*/*/*/state.json" if baseline == "verusage"
               else "tasks/*/*/state.json")
    for path in sorted(run.glob(pattern)):
        state = json.loads(path.read_text())
        if state["status"] != "complete":
            raise ValueError(f"Task did not complete: {path}")
        task = state["task_name"] if baseline == "verusage" else state["task_id"]
        add(task, Path(state["output"]))
elif baseline == "starverus":
    for path in sorted((run / "generation").rglob("*.log")):
        log = path.read_text()
        if "[TASK FAIL]" in log:
            raise ValueError(f"Task failure: {path}")
        for source, final in re.findall(
                r"\[TASK (?:VERIFIED|DONE)\] (.*?): generated=\d+, selected=(.*)", log):
            add(Path(source).stem, Path(final))
else:
    raise ValueError(f"Unknown baseline: {baseline}")
if not selected:
    raise ValueError("No final outputs found; check baseline and run")
pairs.mkdir(parents=True, exist_ok=False)
workflow = {"autoverus": "AutoVerus", "alphaverus": "AlphaVerus",
            "verusage": "VeruSAGE", "starverus": "StarVerus"}[baseline]
with (pairs / "programs.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=["workflow", "model", "shot", "task_id", "generated_path"])
    writer.writeheader()
    for task, source in sorted(selected.items()):
        writer.writerow({"workflow": workflow, "model": run.parent.name, "shot": run.name,
                         "task_id": task, "generated_path": str(source)})
print(f"Prepared {len(selected)} samples in {pairs / 'programs.csv'}")
PY

docker compose run --rm veruseval python scripts/reevaluate.py dataset \
  --manifest runs/evaluation-inputs/autoverus-gpt4o-zero/programs.csv \
  -o runs/evaluation-autoverus --workers 4 --verus-path /opt/verus/verus
```

Check the prepared count against the selected tasks in `generation.json`, and
resolve any missing-task errors before evaluating.

Dataset evaluation uses the released offline I/O suites in `data/io` by default.
LLM intent judgments need `VERUSEVAL_LLM_API_KEY`, `VERUSEVAL_LLM_BASE_URL`, and
`VERUSEVAL_LLM_MODEL_NAME`, passed on the host with `-e` before `veruseval`. These
are separate from the generation credentials. Without them, judgments that need
a model are unavailable.

Check `summary.json`, `artifact_index.csv`, and `per_file/*.json` in the
evaluation output. "Unresolved" and "unavailable" are not passes, and a missing
I/O category is not a successful check. New evaluations can differ from the
released records. See [evaluation options](../docs/usage.md) and
[data details](../docs/data.md).

## Sources

Model settings are in [generation.json](generation.json).

Upstream: [AutoVerus and VeruSAGE](https://github.com/microsoft/verus-proof-synthesis),
[StarVerus](https://github.com/Je5s1e/KDD26-ADS-StarVerus), and
[AlphaVerus](https://github.com/cmu-l3/alphaverus). See
[third-party notices](../THIRD_PARTY.md) for licenses and adaptations.
