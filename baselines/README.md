# Part 2: Run and evaluate the baselines

Use this guide to generate new programs with AutoVerus, VeruSAGE, StarVerus,
and AlphaVerus, then evaluate them with VerusEval. To check the paper's released
results first, follow [Part 1](../docs/reproduction.md).

## Prepare the dataset and model service

Extract the matching [code and data archives](../downloads/README.md) into one
directory and prepare the [Docker environment](../docs/installation.md).
Run commands from that directory. Docker includes Verus `0.2025.09.25.04e8687`,
Rust `1.88.0`, and the baseline dependencies.

Each configuration processes 762 unannotated programs under
`data/generation/X_code/<benchmark>/<task_id>.rs`. Few-shot generation uses five
ordered examples from `data/generation/knn_similar.json`, with inputs from
`X_code/` and annotated examples from `Y/`. Zero-shot generation uses no examples.
The output is a complete program with contracts and proof annotations.

First check all 18 configurations without model calls:

```bash
docker compose run --rm veruseval python scripts/generation/run_dataset.py \
  --dry-run --output-root runs/generation-check
```

A successful check writes `"status": "checked"` to
`runs/generation-check/generation.json`. It still requires the dataset and pinned
Verus. Choose a different output directory for the real run.

For generation, export credentials and the endpoint of your OpenAI-compatible
service on the host, then pass them to Docker:

```bash
export OPENAI_API_KEY='YOUR_API_KEY'
export OPENAI_BASE_URL='https://your-provider.example/v1'
```

Replace the placeholders with your provider's values. The API key remains in the
environment. In [generation.json](generation.json), `api_key_env` names an
environment variable; put the key in that variable, never in the JSON field.
For separate providers, use the model-specific environment variables in that
file and pass each into Docker with `-e VARIABLE`. The shared fallbacks are
`BASELINE_API_KEY` / `OPENAI_API_KEY` and `OPENAI_BASE_URL`.

## Configurations and commands

[generation.json](generation.json) defines the 18 workflow/model/shot combinations.
`--model` selects a dataset label; `--api-model` selects a provider's actual model
ID and requires `--model`. `--base-url` overrides the endpoint. These overrides
are recorded in the output manifest.

| Tool | Model labels | Shots | Generation and repair settings |
| --- | --- | --- | --- |
| AutoVerus | `gpt-4o` | zero / few | 5 candidates per inference attempt; up to 5 repair rounds; 32,000 token limit |
| VeruSAGE | `deepseek-chat`, `qwen-coder` | zero / few | 1 initial candidate; up to 5 repair rounds; 20,000 token limit |
| StarVerus | `deepseek-chat`, `deepseek-reasoner`, `gpt-4o`, `llama`, `qwen-coder` | zero / few | 5 initial candidates; 3 alignment and 3 repair rounds; actor width 3; one rewrite with 3 extra rounds |
| AlphaVerus | `llama` | zero / few | 1 initial candidate; 1,024 initial response tokens; repair width 3 and depth 3 |

The generation temperature is 1.0; StarVerus repair uses 0.3.
`arguments` in the configuration are passed to launchers, and `fixed_settings`
documents implementation constants. To adjust budgets, copy the configuration
to a file under `runs/`, edit its `arguments`, and pass `--config`. For example,
`workflows.alphaverus.arguments.alpha-max-tokens` controls the initial response
budget, including reasoning tokens charged by the provider. An empty or
truncated AlphaVerus response fails explicitly.

Run one tool and prompting mode using the corresponding command below.
Configure an endpoint that serves the selected model before each invocation.

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

Use `--shot few-shot` for the five-example setting; omit `--shot` to run both.
Add `--benchmark HumanEval-Verus` to restrict the task set, and `--workers 4`
to run four tasks concurrently. Omitting baseline/model/shot selectors runs all
18 configurations and requires access to every configured model. Always use a
fresh output root. Generation makes paid model requests according to your provider.

For a service check using a different model, keep the dataset label and override
the provider model and endpoint, for example:

```bash
docker compose run --rm -e OPENAI_API_KEY veruseval \
  python scripts/generation/run_dataset.py --baseline verusage --model deepseek-chat \
  --api-model deepseek-flash --base-url https://api.deepseek.com/v1 \
  --shot zero-shot --benchmark HumanEval-Verus --output-root runs/deepseek-check
```

This example runs the entire HumanEval-Verus subset. For a single-task check,
the AutoVerus/AlphaVerus and VeruSAGE launchers linked below accept `--task-id`
and `--limit`; StarVerus's benchmark entry accepts a `--source_bench_dir`
containing the selected input file, with `--x_dir`, `--y_dir`, and
`--knn_json_path` still pointing to the complete dataset. A service check with
an alternative model is a new test run, not an exact regeneration of paper outputs.

## Locate outputs and source code

At the output root, `generation.json` records actual models, resolved commands,
parameters and exit status; `inputs.json` records input/example hashes; `logs/`
contains launcher logs. Check the per-task results as well: a completed pipeline
may retain an unverified program. Verification alone does not establish that
the target function was preserved or that its contract is correct.

Paths in this table are relative to each configuration directory,
`<output-root>/<baseline>/<model>/<shot>/`.

| Tool | Final files and task records | Code entry points |
| --- | --- | --- |
| AutoVerus | `results/<benchmark>/<task_id>.rs`; `tasks/<benchmark>/<task_id>/state.json` | [launcher](../scripts/generation/run_spec_baselines.py), [generation and prompts](verus-proof-synthesis/autoverus/generation.py), [repair](verus-proof-synthesis/autoverus/refinement.py) |
| VeruSAGE | `<shot>/<model>/<verified-or-unverified>/verusage/*.rs`; `.completion_work/<shot>/<model>/<task_id>/state.json`; `completion_manifest.csv` | [launcher](../scripts/generation/run_verusage_completion.py), [specification prompts](verus-proof-synthesis/verusage/spec_generation.py), [agent entry](verus-proof-synthesis/verusage/main.py) |
| StarVerus | Final `selected=` path in `generation/**/*.log`; successful proofs also have `repair/**/correct.rs` | [launcher](../scripts/generation/run_starverus.py), [benchmark entry](starverus/workflows/benchmark/run.py), [prompt](starverus/workflows/benchmark/prompt/system_prompt.txt), [repair pipeline](starverus/workflows/benchmark/pipeline/pipeline.py) |
| AlphaVerus | `results/<benchmark>/<task_id>.rs`; `tasks/<benchmark>/<task_id>/state.json` | [launcher](../scripts/generation/run_spec_baselines.py), [generation and prompts](alphaverus/inference/solve.py), [tree repair](alphaverus/inference/rebase.py) |

## Evaluate newly generated programs

Use `data/evaluation/target_functions.csv` to map task IDs to the evaluation
references in `data/references/`. Generation examples in `data/generation/Y/`
differ from the evaluation references for some tasks.

For one generated file, choose its task ID and corresponding reference:

```bash
docker compose run --rm veruseval python scripts/evaluation/evaluate.py file \
  -g runs/autoverus/autoverus/gpt-4o/zero-shot/results/VeriCoding/VeriCoding_VV0178_vericoded.rs \
  -r data/references/VeriCoding/VeriCoding_VV0178_vericoded.rs \
  -o runs/evaluation-one --verus-path /opt/verus/verus
```

For batch evaluation, first collect one final file per task into a flat directory
and copy its reference under the same filename. The following command handles
all four output layouts. Set `baseline` and `run` to the configuration being
evaluated, and choose a new `pairs` directory. It includes verified and unverified
final programs, rejects failed tasks and duplicates, and omits intermediate copies.

```bash
docker compose run --rm -T veruseval python - <<'PY'
import csv
import json
from pathlib import Path
import re
import shutil

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
for kind in ("generated", "references"):
    (pairs / kind).mkdir()
for task, source in sorted(selected.items()):
    shutil.copyfile(source, pairs / "generated" / f"{task}.rs")
    shutil.copyfile(references[task], pairs / "references" / f"{task}.rs")
print(f"Prepared {len(selected)} task pairs in {pairs}")
PY

docker compose run --rm veruseval python scripts/evaluation/evaluate.py dir \
  -g runs/evaluation-inputs/autoverus-gpt4o-zero/generated \
  -r runs/evaluation-inputs/autoverus-gpt4o-zero/references \
  -o runs/evaluation-autoverus --workers 4 --verus-path /opt/verus/verus
```

The directory evaluator matches only top-level files with identical names.
Check the prepared count against the selected tasks in `generation.json` and
inspect any missing-task warnings. Run preparation after generation has finished.
Run evaluation in the same Docker workspace and output location used for generation.

To use the released offline I/O suites, copy `config.example.yaml` to `config.yaml`,
set `verus_path: /opt/verus/verus`, and retain `io_suite_dir: data/io`. For LLM intent
judgments, also set `VERUSEVAL_LLM_API_KEY`, `VERUSEVAL_LLM_BASE_URL`, and
`VERUSEVAL_LLM_MODEL_NAME` on the host and pass each with `-e` before `veruseval`
in the evaluation command. These are separate from generation credentials.
Without a configured judge, the LLM metric is unavailable.

Inspect `summary.json`, `scores.csv`, `statistics.json`, and `per_file/*.json` in
the evaluation output. Metric states distinguish passed, failed, unresolved, and
unavailable results. A missing I/O category or unavailable judgment is not a
successful check. New evaluations can differ from the released records.
See [evaluation options](../docs/usage.md) and [data details](../docs/data.md).

## Model configuration and sources

The executable settings are defined in [generation.json](generation.json).
AlphaVerus uses Llama 4 Maverick by default.
VeruSAGE's `deepseek-chat` label defaults to `deepseek-v4-flash` on DashScope;
the general `deepseek-chat` mapping uses `deepseek-flash` on DeepSeek. Override both
`--api-model` and `--base-url` when using another provider. Model aliases and
stochastic generation can change outputs.

Upstream: [AutoVerus and VeruSAGE](https://github.com/microsoft/verus-proof-synthesis),
[StarVerus](https://github.com/Je5s1e/KDD26-ADS-StarVerus), and
[AlphaVerus](https://github.com/cmu-l3/alphaverus).
See [third-party notices](../THIRD_PARTY.md) for licenses and adaptations.
