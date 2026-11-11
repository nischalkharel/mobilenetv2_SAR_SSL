# Nearest neighbours in feature space (5 nearest, cosine, own acquisition group excluded)

Test sample: 4000 official test chips (100 per class, seed 0). Share of the 5 nearest
chips that share the query's class / scene / 30-degree aspect sector, percent (chance = sum of squared frequencies).

| Model | same class | same scene | same aspect sector |
|---|---|---|---|
| A1 fine-tuned 1 % | 33.1 | 70.4 | 25.8 |
| P4 fine-tuned 1 % | 37.8 | 69.2 | 21.1 |
| chance | 2.5 | 33.6 | 8.4 |

Figure queries: test/Buick_Excelle_GT/KU_HH_15_145_518065.tif, test/Buick_Excelle_GT/KU_HH_15_150_307102.tif, test/Buick_Excelle_GT/X_HH_30_0_286399.tif, test/Chang'an_CS75_Plus/KU_HH_30_210_395547.tif, test/Chang'an_CS75_Plus/KU_HH_30_240_757486.tif, test/Chang'an_CS75_Plus/KU_HV_15_60_662.tif.
Written 2026-11-11T14:32:07.
