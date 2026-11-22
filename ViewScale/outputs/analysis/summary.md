# ViewScale: gain against window extent

Sources: SARDet_to_ATR/outputs/compare/cross_dataset.csv and 20261122-124940/runs.csv.
Transfer gains are over XA DINO on SARDet (mean of 3 splits, min..max); detection gains over the no-SSL baseline (split 1).

## transfer

| row | 1 % | 3 % | 10 % |
|---|---|---|---|
| e32 | +0.6 (-0.4..+1.4) | +0.7 (+0.4..+1.2) | +0.0 (-0.5..+0.9) |
| e64 | +2.2 (+1.9..+2.5) | +2.5 (+1.9..+3.2) | +0.2 (-0.5..+1.1) |
| e128 | +3.7 (+2.5..+5.6) | +4.9 (+4.1..+5.5) | +0.9 (-0.1..+1.9) |
| e256 | +4.9 (+4.8..+4.9) | +4.7 (+4.5..+5.0) | +1.3 (+0.8..+2.3) |
| e512 | +4.3 (+4.0..+4.6) | +3.7 (+2.8..+4.4) | +0.9 (+0.1..+2.3) |
| object-only (proposed regions) | +9.4 (+8.5..+10.5) | +7.1 (+6.5..+7.5) | +1.4 (+1.0..+2.1) |
| object-only, random windows | +3.2 (+3.1..+3.2) | +3.6 (+2.3..+4.4) | +0.5 (-0.1..+1.2) |

## detection

| row | 1 % | 3 % | 10 % |
|---|---|---|---|
| e32 | +0.5 | +0.7 | +0.1 |
| e64 | +1.6 | +0.9 | -0.1 |
| e128 | +1.9 | +1.4 | +1.1 |
| e256 | +2.1 | +1.4 | +1.3 |
| e512 | +1.6 | +1.3 | +0.3 |
| object-only (proposed regions) | +1.6 (+1.5..+1.7) | +1.4 (+0.9..+1.9) | +1.2 (+1.1..+1.4) |
| object-only, random windows | +1.8 (+1.7..+1.9) | +1.6 (+1.4..+1.8) | +1.0 (+1.0..+1.1) |
