# Running on Pitt CRC

Layout on CRC (set up and verified 2026-10-18):

| What | Where |
|---|---|
| Code (this repository) | `/ihome/ageorge/nik96/mobilenetv2_SAR_SSL` |
| Dataset (SOC_40classes, 194,750 files) | `/ix/ageorge/nik96/data/ATRNet-STAR/Ground_Range/Amplitude_8bit/<encoded name>/SOC_40classes` |
| ImageNet weights (sha256 `1f19491f...`) | `/ix/ageorge/nik96/data/mobilenet_v2_imagenet1k_v2.pth` (read directly via `MOBILENETV2_IMAGENET_WEIGHTS` in `env.sh`; no copy) |
| Python environment (Python 3.13.16, torch 2.10.0+cu128) | `/ix/ageorge/nik96/envs/sarssl` (conda-forge Python; packages by pip) |

`env.sh` holds everything CRC-specific (Python environment put first on PATH, `ATRNET_GROUND_RANGE`, `MOBILENETV2_IMAGENET_WEIGHTS`); the job scripts source it.

## Workflow

1. PC: change code, commit, `git push`.
2. CRC: `cd ~/mobilenetv2_SAR_SSL && git pull`, then `sbatch crc/<job>.sbatch` from the repository root.
3. Watch: `crc-squeue` (or `squeue -M gpu -u $USER`); log in `crc/logs/<job name>_<job id>.out`.
4. CRC: commit the small result files (configs, history, metrics, charts, logs) and `git push`.
   Checkpoints (`*.pth`, `*.pt`) are ignored by Git; copy them with scp only when needed.
5. PC: `git pull`.

## Jobs

| Script | What | Time limit |
|---|---|---|
| `baseline.sbatch` | A0: fine-tune ImageNet MobileNetV2, validation evaluation | 2 h |
| `dino.sbatch` | A1: smoke test, DINO SSL 100 epochs, fine-tune, validation evaluation | 1 day |
| `lejepa.sbatch` | A2: smoke test, LeJEPA SSL 100 epochs (batch 64), fine-tune, validation evaluation | 36 h (QoS normal) |
| `label_budgets.sbatch` | A0/A1/A2 fine-tunes on the 1% and 10% labeled sets from the existing SSL backbones (hash-checked); validation + test for each; test for the existing 3% runs | 6 h |

Cancel a job: `scancel -M gpu <job id>`. Test-set rule (Nischal, 2026-10-26): test scores are recorded for finished runs, but every decision (best epoch, settings, ablations) is made on validation only. A setting changed after seeing a test score must be reported as such.
