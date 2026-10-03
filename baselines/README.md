# Generation adapters

This directory contains the AlphaVerus, AutoVerus, StarVerus and VeruSAGE workflows.
The launchers in `scripts/generation/` use inputs from `data/generation/` and
write new outputs under `runs/generation/`. Their specification prompts and
proof-repair adaptations are included. See [usage](../docs/usage.md) for configuration.

Upstream code: https://github.com/cmu-l3/alphaverus and
https://github.com/microsoft/verus-proof-synthesis . Existing third-party notices
and licenses apply; see [THIRD_PARTY.md](../THIRD_PARTY.md).
The imported AlphaVerus and Microsoft snapshots do not identify an upstream commit.
The [StarVerus benchmark workflow](starverus/README.md) is vendored from
https://github.com/Je5s1e/KDD26-ADS-StarVerus at commit
`0c0ce03c7f68027085bdb70c82075510cf0a8f57`, with a launcher for the released
762-task generation dataset. Its MIT license is included alongside the source.
