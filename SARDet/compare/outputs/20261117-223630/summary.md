# SARDet-100K detection: method comparison
generated 2026-11-17 22:36; 54 seed-0 runs

Official test mAP@[.5:.95] (AP50 in brackets); per split; mean +/- std over the splits present.

| method | budget | split 1 | split 2 | split 3 | mean +/- std | n |
|---|---|---|---|---|---|---|
| baseline_SARDet | 1pct | 0.153 (0.369) | 0.155 (0.359) | 0.161 (0.381) | 0.157 +/- 0.004 | 3 |
| baseline_SARDet | 3pct | 0.207 (0.466) | 0.214 (0.479) | 0.218 (0.482) | 0.213 +/- 0.005 | 3 |
| baseline_SARDet | 10pct | 0.284 (0.586) | 0.285 (0.591) | 0.287 (0.597) | 0.285 +/- 0.002 | 3 |
| CondDINO_SARDet:condcenter_perm | 1pct | 0.163 (0.381) | 0.159 (0.370) | 0.167 (0.385) | 0.163 +/- 0.004 | 3 |
| CondDINO_SARDet:condcenter_perm | 3pct | 0.210 (0.467) | 0.217 (0.486) | 0.225 (0.496) | 0.217 +/- 0.008 | 3 |
| CondDINO_SARDet:condcenter_perm | 10pct | 0.285 (0.582) | 0.281 (0.585) | 0.283 (0.578) | 0.283 +/- 0.002 | 3 |
| CondDINO_SARDet:condcenter_source_pol | 1pct | 0.159 (0.375) | 0.155 (0.360) | 0.162 (0.372) | 0.159 +/- 0.003 | 3 |
| CondDINO_SARDet:condcenter_source_pol | 3pct | 0.208 (0.463) | 0.217 (0.476) | 0.224 (0.492) | 0.216 +/- 0.008 | 3 |
| CondDINO_SARDet:condcenter_source_pol | 10pct | 0.282 (0.574) | 0.279 (0.577) | 0.287 (0.579) | 0.283 +/- 0.004 | 3 |
| DINO_SARDet | 1pct | 0.157 (0.358) | 0.159 (0.364) | 0.164 (0.382) | 0.160 +/- 0.004 | 3 |
| DINO_SARDet | 3pct | 0.212 (0.468) | 0.215 (0.481) | 0.221 (0.490) | 0.216 +/- 0.005 | 3 |
| DINO_SARDet | 10pct | 0.279 (0.583) | 0.285 (0.587) | 0.284 (0.578) | 0.283 +/- 0.003 | 3 |
| LeJEPA_SARDet | 1pct | 0.166 (0.387) | 0.164 (0.381) | 0.174 (0.396) | 0.168 +/- 0.005 | 3 |
| LeJEPA_SARDet | 3pct | 0.209 (0.465) | 0.217 (0.476) | 0.219 (0.480) | 0.215 +/- 0.005 | 3 |
| LeJEPA_SARDet | 10pct | 0.275 (0.568) | 0.269 (0.565) | 0.283 (0.575) | 0.276 +/- 0.007 | 3 |
| SparK_SARDet | 1pct | 0.160 (0.372) | 0.161 (0.377) | 0.172 (0.387) | 0.164 +/- 0.006 | 3 |
| SparK_SARDet | 3pct | 0.214 (0.474) | 0.217 (0.487) | 0.225 (0.491) | 0.219 +/- 0.006 | 3 |
| SparK_SARDet | 10pct | 0.292 (0.594) | 0.289 (0.603) | 0.294 (0.602) | 0.292 +/- 0.002 | 3 |

## Paired differences vs. baseline (same split, same labeled set): test mAP points x100

| method | budget | per split | mean | min..max | sign |
|---|---|---|---|---|---|
| CondDINO_SARDet:condcenter_perm | 1pct | s1: +0.9, s2: +0.4, s3: +0.5 | +0.6 | +0.4..+0.9 | all + |
| CondDINO_SARDet:condcenter_perm | 3pct | s1: +0.2, s2: +0.3, s3: +0.7 | +0.4 | +0.2..+0.7 | all + |
| CondDINO_SARDet:condcenter_perm | 10pct | s1: +0.2, s2: -0.4, s3: -0.4 | -0.2 | -0.4..+0.2 | mixed |
| CondDINO_SARDet:condcenter_source_pol | 1pct | s1: +0.6, s2: -0.0, s3: +0.0 | +0.2 | -0.0..+0.6 | mixed |
| CondDINO_SARDet:condcenter_source_pol | 3pct | s1: +0.0, s2: +0.3, s3: +0.6 | +0.3 | +0.0..+0.6 | all + |
| CondDINO_SARDet:condcenter_source_pol | 10pct | s1: -0.2, s2: -0.6, s3: -0.1 | -0.3 | -0.6..-0.1 | all - |
| DINO_SARDet | 1pct | s1: +0.3, s2: +0.4, s3: +0.3 | +0.3 | +0.3..+0.4 | all + |
| DINO_SARDet | 3pct | s1: +0.4, s2: +0.1, s3: +0.3 | +0.3 | +0.1..+0.4 | all + |
| DINO_SARDet | 10pct | s1: -0.4, s2: +0.0, s3: -0.3 | -0.2 | -0.4..+0.0 | mixed |
| LeJEPA_SARDet | 1pct | s1: +1.3, s2: +0.9, s3: +1.2 | +1.1 | +0.9..+1.3 | all + |
| LeJEPA_SARDet | 3pct | s1: +0.2, s2: +0.3, s3: +0.1 | +0.2 | +0.1..+0.3 | all + |
| LeJEPA_SARDet | 10pct | s1: -0.9, s2: -1.6, s3: -0.5 | -1.0 | -1.6..-0.5 | all - |
| SparK_SARDet | 1pct | s1: +0.6, s2: +0.6, s3: +1.0 | +0.7 | +0.6..+1.0 | all + |
| SparK_SARDet | 3pct | s1: +0.6, s2: +0.4, s3: +0.7 | +0.6 | +0.4..+0.7 | all + |
| SparK_SARDet | 10pct | s1: +0.9, s2: +0.4, s3: +0.7 | +0.7 | +0.4..+0.9 | all + |

## Selection sensitivity and secondary numbers

| method | budget | split | best-ckpt test | last-ckpt test | test_clean | full val | best epoch |
|---|---|---|---|---|---|---|---|
| baseline_SARDet | 1pct | 1 | 0.153 | 0.151 | 0.153 | 0.151 | 52 |
| baseline_SARDet | 1pct | 2 | 0.155 | 0.152 | 0.155 | 0.154 | 58 |
| baseline_SARDet | 1pct | 3 | 0.161 | 0.162 | 0.161 | 0.160 | 52 |
| baseline_SARDet | 3pct | 1 | 0.207 | 0.206 | 0.207 | 0.206 | 36 |
| baseline_SARDet | 3pct | 2 | 0.214 | 0.214 | 0.214 | 0.222 | 38 |
| baseline_SARDet | 3pct | 3 | 0.218 | 0.216 | 0.218 | 0.221 | 38 |
| baseline_SARDet | 10pct | 1 | 0.284 | 0.284 | 0.283 | 0.290 | 24 |
| baseline_SARDet | 10pct | 2 | 0.285 | 0.285 | 0.285 | 0.306 | 24 |
| baseline_SARDet | 10pct | 3 | 0.287 | 0.287 | 0.287 | 0.302 | 24 |
| CondDINO_SARDet:condcenter_perm | 1pct | 1 | 0.163 | 0.160 | 0.163 | 0.158 | 58 |
| CondDINO_SARDet:condcenter_perm | 1pct | 2 | 0.159 | 0.159 | 0.159 | 0.158 | 52 |
| CondDINO_SARDet:condcenter_perm | 1pct | 3 | 0.167 | 0.166 | 0.167 | 0.167 | 58 |
| CondDINO_SARDet:condcenter_perm | 3pct | 1 | 0.210 | 0.211 | 0.209 | 0.210 | 30 |
| CondDINO_SARDet:condcenter_perm | 3pct | 2 | 0.217 | 0.218 | 0.217 | 0.224 | 38 |
| CondDINO_SARDet:condcenter_perm | 3pct | 3 | 0.225 | 0.225 | 0.225 | 0.224 | 40 |
| CondDINO_SARDet:condcenter_perm | 10pct | 1 | 0.285 | 0.285 | 0.285 | 0.291 | 22 |
| CondDINO_SARDet:condcenter_perm | 10pct | 2 | 0.281 | 0.280 | 0.281 | 0.304 | 22 |
| CondDINO_SARDet:condcenter_perm | 10pct | 3 | 0.283 | 0.283 | 0.283 | 0.302 | 24 |
| CondDINO_SARDet:condcenter_source_pol | 1pct | 1 | 0.159 | 0.163 | 0.159 | 0.158 | 46 |
| CondDINO_SARDet:condcenter_source_pol | 1pct | 2 | 0.155 | 0.155 | 0.155 | 0.155 | 52 |
| CondDINO_SARDet:condcenter_source_pol | 1pct | 3 | 0.162 | 0.163 | 0.162 | 0.165 | 56 |
| CondDINO_SARDet:condcenter_source_pol | 3pct | 1 | 0.208 | 0.211 | 0.208 | 0.207 | 28 |
| CondDINO_SARDet:condcenter_source_pol | 3pct | 2 | 0.217 | 0.217 | 0.216 | 0.220 | 40 |
| CondDINO_SARDet:condcenter_source_pol | 3pct | 3 | 0.224 | 0.224 | 0.224 | 0.230 | 40 |
| CondDINO_SARDet:condcenter_source_pol | 10pct | 1 | 0.282 | 0.285 | 0.282 | 0.291 | 22 |
| CondDINO_SARDet:condcenter_source_pol | 10pct | 2 | 0.279 | 0.281 | 0.279 | 0.305 | 22 |
| CondDINO_SARDet:condcenter_source_pol | 10pct | 3 | 0.287 | 0.287 | 0.287 | 0.302 | 24 |
| DINO_SARDet | 1pct | 1 | 0.157 | 0.156 | 0.157 | 0.154 | 56 |
| DINO_SARDet | 1pct | 2 | 0.159 | 0.158 | 0.159 | 0.155 | 52 |
| DINO_SARDet | 1pct | 3 | 0.164 | 0.164 | 0.164 | 0.166 | 60 |
| DINO_SARDet | 3pct | 1 | 0.212 | 0.210 | 0.212 | 0.214 | 34 |
| DINO_SARDet | 3pct | 2 | 0.215 | 0.216 | 0.215 | 0.222 | 38 |
| DINO_SARDet | 3pct | 3 | 0.221 | 0.225 | 0.221 | 0.223 | 32 |
| DINO_SARDet | 10pct | 1 | 0.279 | 0.279 | 0.279 | 0.284 | 20 |
| DINO_SARDet | 10pct | 2 | 0.285 | 0.285 | 0.285 | 0.304 | 24 |
| DINO_SARDet | 10pct | 3 | 0.284 | 0.284 | 0.284 | 0.302 | 24 |
| LeJEPA_SARDet | 1pct | 1 | 0.166 | 0.169 | 0.166 | 0.161 | 46 |
| LeJEPA_SARDet | 1pct | 2 | 0.164 | 0.164 | 0.164 | 0.161 | 52 |
| LeJEPA_SARDet | 1pct | 3 | 0.174 | 0.172 | 0.173 | 0.173 | 42 |
| LeJEPA_SARDet | 3pct | 1 | 0.209 | 0.207 | 0.209 | 0.210 | 38 |
| LeJEPA_SARDet | 3pct | 2 | 0.217 | 0.215 | 0.216 | 0.225 | 36 |
| LeJEPA_SARDet | 3pct | 3 | 0.219 | 0.219 | 0.219 | 0.225 | 40 |
| LeJEPA_SARDet | 10pct | 1 | 0.275 | 0.275 | 0.274 | 0.287 | 24 |
| LeJEPA_SARDet | 10pct | 2 | 0.269 | 0.269 | 0.269 | 0.295 | 24 |
| LeJEPA_SARDet | 10pct | 3 | 0.283 | 0.283 | 0.282 | 0.302 | 24 |
| SparK_SARDet | 1pct | 1 | 0.160 | 0.159 | 0.160 | 0.159 | 56 |
| SparK_SARDet | 1pct | 2 | 0.161 | 0.160 | 0.161 | 0.158 | 58 |
| SparK_SARDet | 1pct | 3 | 0.172 | 0.170 | 0.171 | 0.169 | 58 |
| SparK_SARDet | 3pct | 1 | 0.214 | 0.214 | 0.214 | 0.210 | 40 |
| SparK_SARDet | 3pct | 2 | 0.217 | 0.215 | 0.217 | 0.224 | 36 |
| SparK_SARDet | 3pct | 3 | 0.225 | 0.224 | 0.224 | 0.224 | 32 |
| SparK_SARDet | 10pct | 1 | 0.292 | 0.292 | 0.292 | 0.297 | 24 |
| SparK_SARDet | 10pct | 2 | 0.289 | 0.290 | 0.288 | 0.313 | 22 |
| SparK_SARDet | 10pct | 3 | 0.294 | 0.294 | 0.293 | 0.309 | 22 |

## Cost

| method | SSL GPU-hours (per split run) | fine-tune minutes (10 %) |
|---|---|---|
| baseline_SARDet | - | 61 |
| CondDINO_SARDet:condcenter_perm | 8.5 | 62 |
| CondDINO_SARDet:condcenter_source_pol | 13.2 | 62 |
| DINO_SARDet | 12.4 | 61 |
| LeJEPA_SARDet | 18.1 | 62 |
| SparK_SARDet | 19.0 | 62 |
