# Third-party material

The root MIT license covers the original VerusEval implementation. The CC BY
4.0 grant covers original measurements and annotations. Neither grant relicenses
third-party programs, source excerpts, fonts, or libraries. Existing source-file
notices are retained.

## Benchmarks and generated programs

The task catalog identifies every benchmark, reference file, target function,
and original SHA-256. Reference copies were recovered from the baseline dataset
and registered evaluation evidence only when those hashes match. The collection
contains DAFNY2VERUS-COLLECTION, HumanEval-Verus, MBPP-verified, VeriCoding, and
VerusBench tasks. Their original source and translation notices apply to the
reference copies and to source material embedded in generated programs.

Relevant upstream resources cited by the experiment include:

- HumanEval-Verus: https://github.com/secure-foundations/human-eval-verus
- Verus proof synthesis and benchmark collection: https://github.com/microsoft/verus-proof-synthesis
- VeriCoding benchmark: https://arxiv.org/abs/2509.22908
- Dafny synthesis benchmark: https://doi.org/10.1145/3643763
- StarVerus: https://doi.org/10.1145/3770855.3818485

The local aggregate did not supply a single license covering all benchmark
translations. This artifact does not assert a new blanket license for them;
reuse of benchmark source must follow the applicable upstream terms.
`baselines/LICENSE.verus-proof-synthesis` preserves the Microsoft MIT notice.
Adapted third-party generation implementations and their runtime source
dependencies are included under `baselines/`; see its README for their origins
and the scope of the local adaptations. The upstream AlphaVerus snapshot did
not contain a standalone license file. This release does not assign the root
MIT license to that code. The Microsoft license is also retained at
`baselines/verus-proof-synthesis/LICENSE`.

## Figure renderer and fonts

The bundled draw.io viewer is from https://viewer.diagrams.net/js/viewer-static.min.js
(JGraph draw.io, https://github.com/jgraph/drawio). Its Apache-2.0 license is in
`RQs/RQ4/figures/assets/vendor/LICENSE.drawio`. The release manifest pins the
exact viewer bytes used by the renderer.

Comic Relief regular/bold fonts are distributed under the SIL Open Font License
1.1; see `RQs/RQ4/figures/assets/fonts/OFL.txt`. Liberation Sans is installed as
an environment dependency. Chromium and Python/R dependencies are obtained
separately through their normal installers and retain their respective licenses.
