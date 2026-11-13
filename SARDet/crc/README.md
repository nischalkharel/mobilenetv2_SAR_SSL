# Running the SARDet suite on Pitt CRC

Same conventions as the repository's own `crc/`: code moves through GitHub (push on the PC, pull on CRC), data
and weights through scp/rsync, small result files are committed on CRC and pulled on the PC. Large data goes to
`/ix/ageorge/nik96` (never a shared drive: CRC notice of 2026-11-02). `env.sh` holds every CRC-specific path.

## Expected layout on CRC (confirm with `ls` after the transfer; then adjust `env.sh` if anything differs)

| What | Where |
|---|---|
| Code (this suite) | `/ihome/ageorge/nik96/mobilenetv2_SAR_SSL/SARDet` (the existing clone, after `git pull`) |
| Dataset (JPEGImages 116,598 files, Annotations/, test.json, mapping.json; 7.2 GB) | `/ix/ageorge/nik96/data/SARDet_100K` |
| ImageNet weights (already there from the ATRNet-STAR work) | `/ix/ageorge/nik96/data/mobilenet_v2_imagenet1k_v2.pth` |
| Python env (already there; needs pycocotools added) | `/ix/ageorge/nik96/envs/sarssl` |

## One-time setup on CRC

```
/ix/ageorge/nik96/envs/sarssl/bin/pip install pycocotools
cd ~/mobilenetv2_SAR_SSL && git pull && cd SARDet
source crc/env.sh                       # prints host, commit, GPU (none on a login node), torch, and every path with ok/MISSING
python data/audit_splits.py             # must end with "ok" (the committed manifests match the dataset copy)
mkdir -p crc/logs                       # Slurm writes the job logs here (also created by every job script)
```

## Jobs (submit from the suite root: `cd ~/mobilenetv2_SAR_SSL/SARDet` first)

| Script | What | Time / QoS |
|---|---|---|
| `lr_check.sbatch [ssl backbone]` | the shared fine-tune lr (1e-4 / 2e-4 / 5e-4) on val_select, 10 %, split 1; decide once, write into `shared/finetune_config.json`, commit | 20 h short |
| `ssl.sbatch <folder> <split> [config]` | one SSL run; checkpoints each epoch; re-submits itself (afterany) and resumes until complete | 23.5 h short, chained |
| `finetune.sbatch <folder> <split> <imagenet|auto|path> [ssl run_name match]` | 10 %, 3 %, 1 % fine-tunes with full scoring, panels and probes | 12 h short |
| `finetune_repeat.sbatch` | baseline 10 % split 1 with seed 1 (fine-tune seed noise) | 6 h short |

Order of operations:
1. `sbatch crc/lr_check.sbatch` (baseline only; ~10 h). Read `best_val_select_map` of the three lrcheck runs, write
   the winner into `shared/finetune_config.json` (`train.lr` and the `_lr_decision` note), commit and push.
   The baseline decides; a later `sbatch crc/lr_check.sbatch <DINO backbone>` is a robustness report only.
2. Then, per split s (1, 2, 3):
```
sbatch --job-name=base_ft_s1 crc/finetune.sbatch baseline_SARDet 1 imagenet
sbatch --job-name=dino_s1    crc/ssl.sbatch DINO_SARDet 1
sbatch --job-name=lejepa_s1  crc/ssl.sbatch LeJEPA_SARDet 1
sbatch --job-name=spark_s1   crc/ssl.sbatch SparK_SARDet 1
sbatch --job-name=cond_s1    crc/ssl.sbatch CondDINO_SARDet 1 config_ssl.json
sbatch --job-name=condperm_s1 crc/ssl.sbatch CondDINO_SARDet 1 config_ssl_perm.json
# when an SSL chain's last job has finished (crc-squeue shows none left for that name):
sbatch --job-name=dino_ft_s1 crc/finetune.sbatch DINO_SARDet 1 auto
sbatch --job-name=cond_ft_s1 crc/finetune.sbatch CondDINO_SARDet 1 auto condcenter_source_pol
```
Watch: `crc-squeue`; logs in `crc/logs/<job name>_<job id>.out`. Cancel a whole SSL chain with
`scancel -M gpu --name=<job name>` (cancelling only the running link lets the pending continuation resume it).
The partial epoch running when a job hits its time limit is lost and is NOT counted in the recorded GPU-hours;
the resumed job repeats it. Resume is state-exact (same batches, schedules, RNG, optimizer, centers), not
bitwise (cuDNN benchmark, mixed precision).
SSL cost estimates: on the PC (RTX 4070 SUPER) SparK runs at 8-9 ms/image, so 100 epochs over 94k images is
about 21-24 GPU-hours there; the A100 is faster, and the ATRNet runs put LeJEPA at about 1.9x DINO's time.
Every run prints "measured speed" after 25 steps: check it against the 23.5 h limit;
the chain handles overruns, but an estimate far above one day means the shared `epochs` value should be
reconsidered BEFORE the other methods start (equal epochs for all).

## Results back to the PC

On CRC, after a job: commit the small files only (`*.pt`, `*.pth`, `*.npz` are git-ignored; `outputs/` is
ignored as a whole, so add with `-f`):
```
git add -f */outputs/ssl/*/{config.json,run_info.json,manifests.json,history.csv,charts} \
           */outputs/finetune/*/{config.json,run_info.json,history.csv,charts,probes*,eval_*/metrics.json,eval_*/per_class.csv,eval_*/pr_curves.png} \
           */outputs/probes */outputs/cell_report crc/logs
git commit -m "CRC results: <what>" && git push
```
On the PC: `git pull`, then `python compare/compare.py`.

Test-set rule (as on ATRNet-STAR): every decision (lr, best epoch, settings) is made on val_select / val only;
test scores are recorded for finished runs and never used to choose anything.

## Data staging (added 2026-11-12)

`env.sh` copies the dataset to the node-local `$SLURM_SCRATCH` at the start of every job (7.2 GB, a few minutes)
and points `SARDET100K_ROOT` there. Reason: with six jobs reading 116k small files from the shared `/ix` at once,
SSL epochs took 9-21 min instead of 4-5 (GPU idle, waiting for files) and identical jobs on different nodes
differed 3x. The copy is wiped when the job ends; `STAGE_DATA=0 sbatch ...` disables it. The run record's
`dataset_root` then shows the scratch path; the manifest hashes identify the data, not the path.
