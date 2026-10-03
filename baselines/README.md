# Baseline generation

The bundled AlphaVerus, AutoVerus, VeruSAGE and StarVerus workflows generate
**contracts and proof code together** from the 762 unannotated programs in
`data/generation/X_code/`. Each returns complete Verus programs with contracts,
loop invariants and proof annotations, followed by verification and repair.
Few-shot runs use five ordered X/Y examples from `knn_similar.json`; zero-shot
runs use none. Generated specifications can still be incorrect or unverifiable.

[generation.json](generation.json) lists all 18 configurations, API model names,
endpoint defaults and generation/repair parameters. The configuration matrix is
checked against the released 13,716-program manifest.

| Workflow | Model labels | Prompting | Initial candidates | Repair limit |
| --- | --- | --- | --- | --- |
| AlphaVerus | llama | zero/few-shot | 1 | 3 rounds, width 3 |
| AutoVerus | gpt-4o | zero/few-shot | 5 per inference attempt | 5 rounds |
| VeruSAGE | deepseek-chat, qwen-coder | zero/few-shot | 1 | 5 rounds |
| StarVerus | deepseek-chat, deepseek-reasoner, gpt-4o, llama, qwen-coder | zero/few-shot | 5 | 3 alignment, 3 proof repair; one rewrite with 3 extra rounds |

After extracting the code and data archives, check all configurations without
model calls:

```bash
docker compose run --rm veruseval python scripts/generation/run_dataset.py \
  --dry-run --output-root runs/generation-check
```

For a generation run, export `OPENAI_API_KEY` and `OPENAI_BASE_URL` for your
OpenAI-compatible service, then select a baseline, model and prompting mode:

```bash
docker compose run --rm -e OPENAI_API_KEY -e OPENAI_BASE_URL veruseval \
  python scripts/generation/run_dataset.py --baseline autoverus --model gpt-4o \
  --shot zero-shot --output-root runs/autoverus-gpt4o-zero
```

Use `--baseline verusage --model qwen-coder` or `--baseline starverus --model gpt-4o`
to select the other workflows. Omit `--shot` for both prompting modes. Omit all
three selectors to run all 18 configurations. `--benchmark HumanEval-Verus`
limits a run to that benchmark. Always choose a fresh output directory.
`--api-model` overrides the selected model's provider identifier; `--base-url`
overrides its endpoint. Separate provider keys and endpoints use the environment
variable names listed in `generation.json`; pass those variables into Docker
with `-e VARIABLE`. Configuration files and run manifests contain no API keys.

Results are under `<output-root>/<baseline>/<model>/<shot>/`. `generation.json`
records resolved commands, model identifiers, parameters and exit status;
`inputs.json` records input/example hashes and `logs/` contains launcher logs.
All workflows use Verus `0.2025.09.25.04e8687` and the bundled Docker dependencies.
The `arguments` entries in the configuration are passed to the launchers;
`fixed_settings` describes constants in the bundled implementations.

The settings combine the included adapters and StarVerus's upstream example
configuration. They are executable rerun settings. Complete original API-call
logs were not retained. In particular, the archived AlphaVerus launcher named
Llama 3.3 while the paper labels Llama 4 Maverick; the rerun configuration selects
Llama 4 explicitly. VeruSAGE's recorded `deepseek-chat` mapping uses
`deepseek-v4-flash`. Model aliases and stochastic generation can produce outputs
different from the frozen corpus.

The specification prompts are in
[AlphaVerus solve.py](alphaverus/inference/solve.py),
[AutoVerus generation.py](verus-proof-synthesis/autoverus/generation.py),
[VeruSAGE spec_generation.py](verus-proof-synthesis/verusage/spec_generation.py)
and [StarVerus system_prompt.txt](starverus/workflows/benchmark/prompt/system_prompt.txt).
The launchers in `scripts/generation/` enable specification generation and invoke
the corresponding proof and repair pipelines.

Upstream: [AlphaVerus](https://github.com/cmu-l3/alphaverus),
[AutoVerus and VeruSAGE](https://github.com/microsoft/verus-proof-synthesis),
[StarVerus](https://github.com/Je5s1e/KDD26-ADS-StarVerus).
The imported AlphaVerus and Microsoft snapshots do not identify an upstream commit.
StarVerus's benchmark source is at `0c0ce03c7f68027085bdb70c82075510cf0a8f57`.
See [third-party notices](../THIRD_PARTY.md) for licenses and adaptations.
