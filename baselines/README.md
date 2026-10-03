# Generation adapters

This directory contains the adapted AlphaVerus, AutoVerus and VeruSAGE workflows.
The launchers in `scripts/generation/` use inputs from `data/generation/` and
write new outputs under `runs/generation/`. Their specification prompts and
proof-repair adaptations are included. See [usage](../docs/usage.md) for configuration.

Upstream code: https://github.com/cmu-l3/alphaverus and
https://github.com/microsoft/verus-proof-synthesis . Existing third-party notices
and licenses apply; see [THIRD_PARTY.md](../THIRD_PARTY.md).
The imported local snapshot does not identify an upstream commit.
The StarVerus output corpus is included, but a complete generation launcher is unavailable.
