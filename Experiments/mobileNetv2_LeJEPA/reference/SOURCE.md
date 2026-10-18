# Source of the LeJEPA reference files

- Repository: official LeJEPA code, https://github.com/rbalestr-lab/lejepa (linked from the paper and its README).
- Obtained: downloaded by Nischal as a ZIP and extracted to `SHREC/2026_2027/lejepa-main` (2026-10-16). A ZIP has no Git history, so no commit hash is available; the files are identified by the SHA-256 hashes below.
- Paper: arXiv 2511.08544 **v3** (14 Nov 2025), read on arxiv.org (HTML). Not stored locally.
- Licence: `LICENSE` is **Creative Commons Attribution-NonCommercial 4.0 International** (CC BY-NC 4.0).

The files in `lejepa-main_original/` are untouched copies (same bytes as in `lejepa-main`). The three files in `../lejepa_lib/` are byte-identical to the corresponding files here; `smoke_test.py` checks that.

| File | SHA-256 |
|---|---|
| LICENSE | 1d8ca66080a2701508a70be4ddad4be28c509bcc3397a682fb8edf767fdd7a00 |
| MINIMAL.md | 5fdbd73825ae2e49e25a63cdc1f0caba17438181a817de78f08c11f58dd4224f |
| README.md | a6dac7b7a2053d67a4f06235696b341248dcc9f6d3711e3b5b57242f25cad6c3 |
| pyproject.toml | 2193b9c5089c03180e70e053d800fc034a1b6d56c96b402a8ae15a2c806b1947 |
| lejepa/__init__.py | 052379df9f805e202619d5c4d65b0ad8c6851ae45a81acdcd25d0c31e3d6351f |
| lejepa/univariate/__init__.py | 297d5516c47627b680ae635f564dfac85c63ca98b4f57cb2800523c03f873acc |
| lejepa/univariate/base.py | 08d4f115990656ea3459dacbba5991622725f9040ebc64ae8d34f4e76299eef6 |
| lejepa/univariate/epps_pulley.py | e6554ee42de27b74d62befb5f353d4d7a4f92c6c1eade25edaa6595a6b593149 |
| lejepa/multivariate/__init__.py | 1a464456c04ad0236ac3b2d6e8a21f018bacca7e4a8c6181aebe64393d2d9365 |
| lejepa/multivariate/slicing.py | 86c0fe3a714dc945ba3e23ab4093f6ed41966039f9681bd53c733e3ca5dff56b |
| scripts/launch_inet10.py | 2628081b869b0d2a0dcca84608d73b2e71d109c9f46fc44f0397db118279a224 |
| scripts/launch_epps_ablation.md | 916646599f3ceb945c0480e4b5bc4bccb8cd8f2e18a0f3a5f623cd74324d28a4 |
| scripts/launch_proj_ablation.md | a390fadfa24b2088bb2f8cadddd7c47fdc0f2c854c85545208b59032f5166e8d |
| scripts/launch_rtokens_ablation.md | efd0f8b3cf4c26911a2885319b7827ffff76a832377eb41ebd2f37ccca4d988f |
| scripts/launch_views_ablation.md | 24e185e9ca574cf8bd6a27ce6e51bde8093c24676ad38b32c14ec3eb803476f8 |

Not copied: the other statistical tests in `lejepa/` (not used), `figures/`, `tests/`, `eval/` media. The training script the launch files call (`scripts/je.py`, built on the `stable_pretraining` library) is **not in the repository**, so the exact meaning of some launch-script flags (for example `embedding_dim`, `teacher_student`) cannot be checked.
