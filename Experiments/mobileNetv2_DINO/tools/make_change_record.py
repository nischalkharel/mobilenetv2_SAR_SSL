"""Generate the exact record of how copied DINO and baseline code differs from its source.

Run from the repository root after any edit to a copied block:
    python Experiments/mobileNetv2_DINO/tools/make_change_record.py

1. DINO: every block between "# >>> DINO extract <ID>: <file> L<a>-L<b> (<name>), <unchanged|modified>"
   and "# <<< end <ID>" (in dino.py and train_ssl.py) is compared line by line with lines a..b of the
   untouched original in reference/dino-main_original/. Leading indentation common to the whole block
   is removed on both sides first (a function moved into a different nesting level is not a change).
   Writes diffs/dino_<ID>_<name>.diff and the table between the GENERATED markers in DINO_CHANGES.md.
   Fails if a block marked "unchanged" differs, or if a changed hunk has no explanation in REASONS.
2. Baseline: unified diff of each copied downstream file against reference/baseline_original/,
   written to diffs/baseline_<file>.diff.
3. Checks that the reference copies still have the SHA-256 hashes recorded in reference/SOURCE.md.
"""
import difflib
import hashlib
import re
import sys
import textwrap
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REF_DINO = EXP / "reference" / "dino-main_original"
REF_BASE = EXP / "reference" / "baseline_original"
DIFFS = EXP / "diffs"
DOC = EXP / "DINO_CHANGES.md"
SOURCES_WITH_EXTRACTS = ["dino.py", "train_ssl.py"]
BASELINE_COPIES = {"data.py": "data.py", "model.py": "model.py", "engine.py": "engine.py", "utils.py": "utils.py",
                   "plots.py": "plots.py", "evaluate.py": "evaluate.py", "train.py": "train_supervised.py",
                   "config.json": "config_finetune.json"}
START = re.compile(r"#\s*>>> DINO extract (E\d+): (\S+) L(\d+)-L(\d+) \((.*?)\), (unchanged|modified)")
END = re.compile(r"#\s*<<< end (E\d+)")

# Category codes: C = compatibility (API or library), D = data handling, M = model structure,
# X = execution (single process, device, logging, records), L = learning method.
# Key: (extract ID, first original line of the hunk; for a pure insertion, the original line it follows).
REASONS = {}


def reason(block, line, category, why):
    REASONS[(block, line)] = (category, why)


# ---- E6 MultiCropWrapper (utils.py) ----
reason("E6", 605, "M", "The original disabled the ImageNet classifier by setting backbone.fc and backbone.head to "
       "Identity. MobileNetV2 has no fc, so its classifier (Dropout + Linear 1280->1000) would still run and "
       "return 1000 ImageNet logits. The backbone passed in here (MobileNetV2Backbone) already stops at global "
       "average pooling, so there is nothing to disable.")
reason("E6", 608, "X", "Adds an attribute that holds the last pooled backbone features, for the collapse "
       "diagnostics only.")
reason("E6", 627, "X", "Stores the pooled features (detached: no gradient flows through this copy) after the loop "
       "over resolutions. Outputs and gradients are unchanged.")
# ---- E8 DINOHead (vision_transformer.py) ----
reason("E8", 276, "C", "torch.nn.utils.weight_norm is deprecated in torch 2.10 (FutureWarning). "
       "torch.nn.utils.parametrizations.weight_norm computes the same weight = g * v / ||v|| along dim 0. Only "
       "the parameter names change (weight_g -> parametrizations.weight.original0, weight_v -> original1). They "
       "still contain 'last_layer', so freeze_last_layer works as before. smoke_test.py checks that the "
       "initialization and outputs are identical to the original for the same seed.")
reason("E8", 279, "C", "Same API change: the norm scale g is now named original0. As in the original, it is fixed "
       "at 1 and not trained when norm_last_layer is true.")
reason("E8", 283, "C", "The original used its own copy of trunc_normal_ (in its utils.py, itself copied from "
       "PyTorch). torch.nn.init.trunc_normal_ is the same algorithm with the same defaults (a=-2, b=2); "
       "smoke_test.py checks that the initialization is bit-identical.")
# ---- E9 DINOLoss (main_dino.py) ----
reason("E9", 412, "X", "Single process. The original summed the teacher outputs over all GPUs with dist.all_reduce "
       "and divided by (batch per GPU x world size). With one process the local sum is the whole sum and the "
       "world size is 1, so the center is numerically the same, without needing a process group. smoke_test.py "
       "checks the loss and center against the original code run with world size 1.")
# ---- E10 DataAugmentationDINO (main_dino.py) ----
reason("E10", 420, "D", "All settings come from the config (crop sizes, crop scales, flip, jitter, blur, "
       "solarization, normalization) instead of being hard-coded, so every augmentation choice is visible and "
       "recorded in each run's config.json.")
reason("E10", 422, "D", "Flip probability from the config: 0.5, the same as the original and the A0 baseline.")
reason("E10", 424, "D", "SAR chips have one channel. Brightness and contrast jitter keep the original strengths "
       "(0.4, 0.4) and probability (0.8), now from the config; they act as radar gain changes. Saturation (0.2) "
       "and hue (0.1) are removed: on a gray image copied into 3 identical channels they do nothing.")
reason("E10", 427, "D", "RandomGrayscale (p 0.2) removed: the image is already gray, so it does nothing.")
reason("E10", 429, "D", "Grayscale(3) copies the one SAR channel into R, G and B, exactly as the baseline does, so "
       "the ImageNet-pretrained first layer receives the same kind of input in SSL and in fine-tuning.")
reason("E10", 431, "D", "Normalization values from the config: the same ImageNet mean/std as the original and the "
       "baseline.")
reason("E10", 432, "D", "Bilinear interpolation (with antialias) instead of bicubic, to match the baseline's "
       "resize. A SAR-specific choice for consistency with fine-tuning, not a compatibility requirement.")
reason("E10", 436, "D", "First global crop: size and scale from the config (224 px and 0.4-1.0: the original values), "
       "bilinear interpolation (see above).")
reason("E10", 438, "D", "Blur probability from the config (1.0 for the first global view, as in the original). "
       "GaussianBlur now lives in dino.py.")
reason("E10", 443, "D", "Second global crop: same as the first.")
reason("E10", 445, "D", "Blur probability from the config (0.1, as in the original). Solarization probability from "
       "the config, set to 0 for SAR: it inverts every pixel above 128, i.e. exactly the bright scattering "
       "returns that identify a vehicle in these mostly dark chips. The transform itself is unchanged.")
reason("E10", 450, "D", "Number of local crops from the config (8, the original default).")
reason("E10", 452, "D", "Local crops: size from the config (96 px, as in the original); scale 0.15-0.4 instead of "
       "0.05-0.4 (SAR-specific: with 0.05 about 14% of local crops contained no part of the vehicle; with 0.15 "
       "about 7%; see README); bilinear interpolation.")
reason("E10", 454, "D", "Blur probability from the config (0.5, as in the original).")
# ---- E11 building student and teacher (main_dino.py) ----
reason("E11", 183, "X", "utils.MultiCropWrapper -> MultiCropWrapper (the class now lives in dino.py). The DINOHead "
       "arguments are unchanged.")
reason("E11", 189, "X", "Same for the teacher.")
reason("E11", 194, "X", "One process, one GPU. .to(device) instead of .cuda(). No SyncBatchNorm conversion: "
       "SyncBatchNorm only shares batch statistics between processes, so with one GPU it computes the same "
       "statistics as the ordinary BatchNorm that MobileNetV2 already has. No DistributedDataParallel wrappers "
       "(they only average gradients across processes and, for the teacher, broadcast buffers). teacher_without_ddp "
       "is therefore just the teacher. As in the original, the teacher is never put in eval mode.")
reason("E11", 208, "X", "student.module -> student, teacher_without_ddp -> teacher (no DDP wrapper). Still a full "
       "state_dict copy: parameters and BatchNorm buffers.")
reason("E11", 212, "X", "The message names MobileNetV2 (there is no args.arch here).")
# ---- E12 loss, optimizer, schedules (main_dino.py) ----
reason("E12", 222, "X", "DINOLoss receives student_temp and center_momentum from the config (the original used the "
       "class defaults; the config holds the same values, 0.1 and 0.9) so they are recorded. .to(device) instead "
       "of .cuda().")
reason("E12", 225, "X", "get_params_groups now lives in dino.py (no utils. prefix). Same groups: biases and other "
       "1-D parameters (BatchNorm weights) get no weight decay.")
reason("E12", 230, "X", "LARS option removed: the original suggests it for large multi-GPU convnet batches; it is "
       "not used here (the config accepts adamw or sgd).")
reason("E12", 234, "C", "torch.cuda.amp.GradScaler is deprecated; torch.amp.GradScaler('cuda') is the same scaler. "
       "Mixed precision is only used on the GPU.")
reason("E12", 238, "X", "Linear scaling rule kept: effective lr = lr x total batch / 256. The original's total batch "
       "was batch_size_per_gpu x number of GPUs; with one GPU it is batch_size_per_gpu. cosine_scheduler now lives "
       "in dino.py.")
reason("E12", 241, "X", "len(data_loader) -> niter_per_ep, the same number (images // batch, because drop_last) "
       "computed once, so it can be recorded and compared on resume.")
reason("E12", 244, "X", "cosine_scheduler now lives in dino.py.")
reason("E12", 247, "X", "len(data_loader) -> niter_per_ep (same number).")
reason("E12", 250, "X", "cosine_scheduler from dino.py; niter_per_ep (same number).")
# ---- E13 train_one_epoch (main_dino.py) ----
reason("E13", 301, "X", "teacher_without_ddp removed from the arguments (no DDP).")
reason("E13", 303, "X", "device and an optional timing callback are passed in (the callback prints the measured "
       "speed and peak GPU memory early in the run). MetricLogger replaced by plain running sums. The loader "
       "yields only the list of views, because the SSL dataset returns no label; `step` counts batches for "
       "printing. The global iteration index `it` is computed exactly as before.")
reason("E13", 315, "X", ".to(device) instead of .cuda(). The center is copied before the loss updates it, so the "
       "diagnostics use the center that this step actually used.")
reason("E13", 317, "C", "torch.cuda.amp.autocast is deprecated; torch.autocast(device, dtype=float16) is the same.")
reason("E13", 323, "X", "A non-finite loss raises an error (the run is then recorded as failed in run_info.json) "
       "instead of print(force=True) + sys.exit; force= only exists in the original's distributed print.")
reason("E13", 332, "X", "clip_gradients and cancel_gradients_last_layer now live in dino.py (no utils. prefix). "
       "Same order: backward, clip, cancel last-layer gradients during the first epoch(s), step.")
reason("E13", 340, "X", "Same, in the mixed-precision branch (unscale, clip, cancel, step, update: same order).")
reason("E13", 349, "X", "student.module -> student, teacher_without_ddp -> teacher (no DDP). The EMA still runs over "
       "parameters only, exactly as in the original: BatchNorm running statistics are not averaged; the teacher "
       "updates them itself in its forward passes.")
reason("E13", 353, "X", "torch.cuda.synchronize and MetricLogger replaced by the collapse diagnostics, running sums "
       "and progress printing (loss.item() already waits for the GPU). The returned dictionary also reports the "
       "lr, weight decay, momentum and teacher temperature used. No effect on training.")
# ---- E14 checkpoint (main_dino.py) ----
reason("E14", 283, "X", "args saved as a plain dict (the flat config) so it can be compared on resume.")
reason("E14", 284, "X", "Added for exact epoch-boundary resume: the scaler state (None when not used), the global "
       "step, the schedule fingerprint and the random number generator states.")
reason("E14", 286, "X", "Written to a temporary file and then renamed, so an interruption never leaves a "
       "half-written checkpoint. No periodic copies (saveckp_freq): the final epoch is the checkpoint chosen in "
       "advance, and DINO loss is not used to pick another one.")


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def extracts():
    for src in SOURCES_WITH_EXTRACTS:
        lines = read_lines(EXP / src)
        i = 0
        while i < len(lines):
            m = START.search(lines[i])
            if m:
                j = next(k for k in range(i + 1, len(lines)) if END.search(lines[k]))
                if END.search(lines[j]).group(1) != m.group(1):
                    raise ValueError(f"marker mismatch for {m.group(1)} in {src}")
                yield {"id": m.group(1), "orig_file": m.group(2), "a": int(m.group(3)), "b": int(m.group(4)),
                       "name": m.group(5), "status": m.group(6), "dest_file": src,
                       "dest_first": i + 2, "dest": lines[i + 1:j]}
                i = j
            i += 1


def dedent(lines):
    text = textwrap.dedent("\n".join(lines))
    return text.split("\n")


def hunks(e):
    orig = dedent(read_lines(REF_DINO / e["orig_file"])[e["a"] - 1:e["b"]])
    dest = dedent(e["dest"])
    sm = difflib.SequenceMatcher(None, orig, dest, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        out.append({"tag": tag, "orig_lines": orig[i1:i2], "dest_lines": dest[j1:j2],
                    "orig_start": e["a"] + i1 if i2 > i1 else e["a"] + i1 - 1,
                    "o1": e["a"] + i1, "o2": e["a"] + i2 - 1, "d1": e["dest_first"] + j1, "d2": e["dest_first"] + j2 - 1})
    return orig, dest, out


def unified(e, orig, dest):
    diff = difflib.unified_diff(orig, dest, fromfile=f"dino-main/{e['orig_file']} (lines {e['a']}-{e['b']})",
                                tofile=f"mobileNetv2_DINO/{e['dest_file']} (lines {e['dest_first']}-"
                                       f"{e['dest_first'] + len(e['dest']) - 1})", lineterm="", n=2)
    text = []
    for line in diff:  # rewrite hunk headers to absolute line numbers in both files
        m = re.match(r"@@ -(\d+)(,(\d+))? \+(\d+)(,(\d+))? @@", line)
        if m:
            o = int(m.group(1)) + e["a"] - 1 if m.group(3) != "0" else int(m.group(1)) + e["a"] - 1
            d = int(m.group(4)) + e["dest_first"] - 1
            line = f"@@ -{o},{m.group(3) or 1} +{d},{m.group(6) or 1} @@"
        text.append(line)
    return "\n".join(text) + "\n"


def code(s: str) -> str:
    s = s.replace("|", "\\|")
    return f"`{s.strip()}`" if s.strip() else "(blank line)"


def main():
    DIFFS.mkdir(exist_ok=True)
    for old in DIFFS.glob("*.diff"):
        old.unlink()
    rows, missing, problems, summary = [], [], [], []
    for e in extracts():
        orig, dest, hs = hunks(e)
        tag = f"{e['id']}_{e['name'].split(' (')[0].replace(' ', '_').replace(',', '')}"
        if e["status"] == "unchanged" and hs:
            problems.append(f"{e['id']} is marked unchanged but differs from the original")
        if hs:
            (DIFFS / f"dino_{tag}.diff").write_text(unified(e, orig, dest), encoding="utf-8")
        summary.append(f"| {e['id']} | `{e['orig_file']}` L{e['a']}-L{e['b']} | {e['name']} | `{e['dest_file']}` "
                       f"L{e['dest_first']}-L{e['dest_first'] + len(e['dest']) - 1} | {e['status']} | "
                       f"{len(hs)} | {'`diffs/dino_' + tag + '.diff`' if hs else 'none'} |")
        for n, h in enumerate(hs, 1):
            key = (e["id"], h["orig_start"])
            if key not in REASONS:
                missing.append(f"{key}: {h['tag']} {h['orig_lines'][:1]} -> {h['dest_lines'][:1]}")
                continue
            cat, why = REASONS[key]
            olines = (f"L{h['o1']}-L{h['o2']}" if h["o2"] > h["o1"] else f"L{h['o1']}") if h["orig_lines"] else f"after L{h['orig_start']}"
            dlines = (f"L{h['d1']}-L{h['d2']}" if h["d2"] > h["d1"] else f"L{h['d1']}") if h["dest_lines"] else "(deleted)"
            removed = "<br>".join(code(x) for x in h["orig_lines"]) or "(nothing)"
            added = "<br>".join(code(x) for x in h["dest_lines"]) or "(nothing)"
            kind = {"replace": "replaced", "delete": "deleted", "insert": "added"}[h["tag"]]
            rows.append(f"| {e['id']}.{n} | `{e['orig_file']}` {olines} | `{e['dest_file']}` {dlines} | {kind} | "
                        f"{removed} | {added} | {cat} | {why} |")
    if problems or missing:
        print("Change record NOT written:")
        for p in problems:
            print("  " + p)
        for m in missing:
            print("  no explanation for hunk " + m)
        sys.exit(1)

    base_rows = []
    for orig_name, dest_name in BASELINE_COPIES.items():
        a, b = read_lines(REF_BASE / orig_name), read_lines(EXP / dest_name)
        d = list(difflib.unified_diff(a, b, fromfile=f"Baseline/{orig_name}", tofile=f"mobileNetv2_DINO/{dest_name}",
                                      lineterm="", n=2))
        if d:
            (DIFFS / f"baseline_{dest_name}.diff").write_text("\n".join(d) + "\n", encoding="utf-8")
        plus = sum(1 for x in d if x.startswith("+") and not x.startswith("+++"))
        minus = sum(1 for x in d if x.startswith("-") and not x.startswith("---"))
        base_rows.append(f"| `Baseline/{orig_name}` | `{dest_name}` | {'+' + str(plus) + ' / -' + str(minus) if d else 'identical'} | "
                         f"{'`diffs/baseline_' + dest_name + '.diff`' if d else 'none'} |")

    source = (EXP / "reference" / "SOURCE.md").read_text(encoding="utf-8")
    for f in sorted(REF_DINO.iterdir()) + sorted(REF_BASE.iterdir()):
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        if sha not in source:
            print(f"reference file {f.name} does not match its recorded hash in reference/SOURCE.md")
            sys.exit(1)

    table = "\n".join([
        "### Copied DINO blocks", "",
        "| ID | Original | Component | Destination | Status | Changed hunks | Exact diff |",
        "|---|---|---|---|---|---|---|", *summary, "",
        "### Every changed hunk",
        "",
        "Line numbers: original = line in `reference/dino-main_original/<file>`; destination = line in this folder. "
        "Categories: C compatibility (library/API), D data handling, M model structure, X execution "
        "(single process, device, logging, records), L learning method. No hunk is in category L: "
        "the learning mechanics are unchanged.",
        "",
        "| Hunk | Original | Destination | Change | Original text (deleted) | New text (added) | Category | Why |",
        "|---|---|---|---|---|---|---|---|", *rows, "",
        "### Downstream files copied from the baseline", "",
        "| Baseline file | Copy | Lines added / removed | Exact diff |", "|---|---|---|---|", *base_rows,
    ])
    doc = DOC.read_text(encoding="utf-8")
    begin, end = "<!-- BEGIN GENERATED by tools/make_change_record.py -->", "<!-- END GENERATED -->"
    doc = doc[:doc.index(begin) + len(begin)] + "\n" + table + "\n" + doc[doc.index(end):]
    DOC.write_text(doc, encoding="utf-8")
    print(f"Wrote {len(rows)} DINO hunks, {len(list(DIFFS.glob('dino_*.diff')))} DINO diffs and "
          f"{len(list(DIFFS.glob('baseline_*.diff')))} baseline diffs; updated {DOC.name}.")


if __name__ == "__main__":
    main()
