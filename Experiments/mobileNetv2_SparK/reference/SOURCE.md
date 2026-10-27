# Source of the SparK reference files

- Repository: official SparK code, https://github.com/keyu-tian/SparK (linked from the paper). We use the pre-training code in `pretrain/`.
- Obtained: downloaded by Nischal as a ZIP and extracted to `SHREC/2026_2027/SparK-main` (2026-10-27). A ZIP has no Git history, but every file below was compared on 2026-10-27 with the same file at commit **a63e386f8e5186bc07ad7fce86e06b08f48a61ea** (2024-01-23, "[fix] (#58) fix default hyperparameters", the latest commit on `main`) via raw.githubusercontent.com: **18 of 18 identical**. That commit changed only downstream fine-tuning defaults and file renames; the pre-training recipe is the one published with the paper.
- Paper: Tian et al., "Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling", ICLR 2023 (spotlight), arXiv 2301.03580 **v2** (10 Jan 2023). Read on arxiv.org; the PDF is not in `Papers/`.
- Licence: `LICENSE` is the **MIT licence** (Copyright (c) 2023 Keyu Tian). A copy sits next to the vendored files in `../spark_lib/LICENSE`.

The files in `SparK-main_original/` are untouched copies (same bytes as in `SparK-main`). `smoke_test.py` checks these hashes, that `../spark_lib/lamb.py`, `lr_control.py` and `LICENSE` are byte-identical to the originals, and that `../spark_lib/encoder.py`, `spark.py` and `decoder.py` differ only in their documented import lines (`SPARK_CHANGES.md`, section 1).

| File | SHA-256 |
|---|---|
| LICENSE | 7c30bb95eed6002f42ab1f68cb18a7c580e75a409bc3fc7e93fb1925b510abb7 |
| README.md | 5a7ac77a63b28584c9454448e53b523f7e4c08af18fa7aeff4d48d6946765052 |
| INSTALL.md | b6d6c888e5a21699df99942ea9b15d5d3b8f3fb48046716ba9027944497f9c25 |
| pretrain/README.md | 9e069a8ad8be45723da18f9441c65cc6bfabc9362c6478f08061b3fdc441658f |
| pretrain/requirements.txt | 067733939388728ed8e80796060227e888fb85be8ef5d9ba07cd8db9bfce43ce |
| pretrain/main.py | 8d1622a0335e37b32ff2e721d1ac8d30c60fcd1849e0879171358171513bab5e |
| pretrain/encoder.py | f5824949066b84881668c8efada63c785783e822c2856af822e63a8f7cfe7a4b |
| pretrain/spark.py | ffa3854c599c57ae084eed025da3c92b72262e7ed112bbe8bf27f3df9c5588cf |
| pretrain/decoder.py | 4c913585e2b236737317c1bc4bafd5e953601a6615ba5463bd891a3fc2ac47d9 |
| pretrain/sampler.py | 13e81d33085d6a401e262acb0301337ae85a53498d85b2445855d970789c376f |
| pretrain/models/__init__.py | 359dd95a4a32c016c2346358af74f1396bea15dfe70d43e28f84d78f5325e0ab |
| pretrain/models/custom.py | d4df670bff0147e8d3a0a4a7ecfdfabd627e356e17447dd91d3f73e5045c4efa |
| pretrain/models/resnet.py | 71dd46c5556f5df1dac63e3ff7af877f0f2a788506d4edd0aad51ff810461b97 |
| pretrain/utils/arg_util.py | 37b7566f1f550a962e55aaab509211761c0a90483034b3dc8f1346c87205caf5 |
| pretrain/utils/lamb.py | ea0d56d613b4b11fe937c82f279a62815fbd62b67fac160841c6d8c6de73d34c |
| pretrain/utils/lr_control.py | 336cf1d35d4bb5e4eac6e6df99f375a7c701cbd15643c204429d88c1ca4dd0be |
| pretrain/utils/imagenet.py | 9647d749a3cb6808744eb9212106a47c004fdc8f0bc9d87e82d95ea5d81983f9 |
| pretrain/utils/misc.py | 2f1db1e7e54cbeaa852bd56f9371c7e10cc91aedb8bde36abb802c425db98e1d |

Not copied: `downstream_*` (their fine-tuning recipes; A3 uses the same fine-tuning as A0, A1 and A2), the ConvNeXt model file, `dist.py`, the notebooks and images. The two notebooks (`viz_reconstruction.ipynb`, `viz_spconv.ipynb`) were not needed: `preview_views.py` draws reconstructions with the `vis=True` path of `SparK.forward`.
