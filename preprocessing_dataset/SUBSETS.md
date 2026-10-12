# SOC-40 data subsets

Fixed subsets of ATRNet-STAR SOC-40 (Ground_Range, 8-bit amplitude) used by every experiment in this project.
Generated on 2026-10-12 by `preprocessing_dataset.py` with `SEED = 0`. Do not regenerate them between experiments; reuse the CSVs.

## Files

| File | Images | Contents |
|---|---:|---|
| `validation.csv` | 6,896 | About 10% of official SOC-40 train, sampled per class by whole pose groups |
| `ssl_training_data.csv` | 61,195 | All official SOC-40 train images not in validation (the SSL pool) |
| `labeled_set.csv` | 6,120 | 10% of the SSL pool, sampled per class image by image; used for labeled training |
| `class_mapping.csv` | 40 rows | Label number (0 to 39) for each class name, in case-sensitive sorted order |
| Official `test/` folder | 29,284 | Held out; not listed in any CSV |

Paths in the CSVs are relative to `paths.SOC40_DIR` (see `paths.py`).

## Rules that prevent leakage

- Validation and the SSL pool together are exactly the official SOC-40 train split, with no image in both.
- The labeled set is entirely inside the SSL pool. It shares no image with validation or test.
- No official test image appears in any CSV.
- A pose group is one class at one band, depression angle and azimuth angle (for example `Buick_GL8|X|15|110`). Its images differ only by polarization (HH, HV, VH, VV) or repeat passes, so they are near-copies. A whole pose group is in validation or in the SSL pool, never split across both.
- The same vehicle at a different angle can be in both validation and training. That is intended: the task is recognizing the vehicle type from any angle.
- All 97,375 SOC-40 images were hashed: no exact pixel duplicates exist within or across train and test.

Known limitation of the official benchmark (not fixable without abandoning it): SOC-40's train/test split was random per image, so 12,779 of 15,321 pose groups have images in both official train and test. This affects all methods equally but likely inflates SOC-40 test accuracy and should be stated when reporting results.

## Validation set per class

Validation slightly exceeds 10% per class because whole pose groups are added until each class reaches 10%.

| Label | Class name | Validation images | Class train total | % of class |
|---:|---|---:|---:|---:|
| 0 | Buick_Excelle_GT | 151 | 1,494 | 10.1% |
| 1 | Buick_GL8 | 177 | 1,766 | 10.0% |
| 2 | CNHTC_HOWO | 159 | 1,545 | 10.3% |
| 3 | Chang'an_CS75_Plus | 232 | 2,315 | 10.0% |
| 4 | Chang'an_Starlight_4500 | 180 | 1,800 | 10.0% |
| 5 | Changfeng_Cheetah_CFA6473C | 124 | 1,225 | 10.1% |
| 6 | Changlin_8228-5 | 162 | 1,599 | 10.1% |
| 7 | Chery_Arrizo 5 | 147 | 1,464 | 10.0% |
| 8 | Chery_qq3 | 179 | 1,708 | 10.5% |
| 9 | Chevrolet_Blazer_1998 | 128 | 1,254 | 10.2% |
| 10 | Dongfeng_Duolika | 137 | 1,364 | 10.0% |
| 11 | Dongfeng_EQ6608LTV | 238 | 2,342 | 10.2% |
| 12 | Dongfeng_Forthing_Lingzhi | 201 | 2,013 | 10.0% |
| 13 | Dongfeng_Tianjin_DFH2200B | 160 | 1,596 | 10.0% |
| 14 | Dongfeng_Tianjin_KR230 | 159 | 1,579 | 10.1% |
| 15 | FAW_J6P | 156 | 1,557 | 10.0% |
| 16 | FAW_Jiabao_T51 | 160 | 1,554 | 10.3% |
| 17 | Foton_BJ1045V9JB5-54 | 166 | 1,649 | 10.1% |
| 18 | Great_Wall_Voleex_C50 | 231 | 2,260 | 10.2% |
| 19 | Great_Wall_poer | 192 | 1,861 | 10.3% |
| 20 | Hawtai_EV160B | 189 | 1,893 | 10.0% |
| 21 | Hongqi_CA7180A3E | 174 | 1,741 | 10.0% |
| 22 | Hongqi_h5 | 152 | 1,513 | 10.0% |
| 23 | Huanghai_N1 | 153 | 1,527 | 10.0% |
| 24 | Hyundai_HLF25_II | 156 | 1,556 | 10.0% |
| 25 | JAC_Junling | 180 | 1,760 | 10.2% |
| 26 | JINBEI_SY5033XJH | 210 | 2,066 | 10.2% |
| 27 | Jeep_Patriot | 141 | 1,409 | 10.0% |
| 28 | Lincoln_MKC | 163 | 1,610 | 10.1% |
| 29 | Lveco_Proud_2009 | 201 | 1,986 | 10.1% |
| 30 | MAXUS_V80 | 166 | 1,633 | 10.2% |
| 31 | Mitsubishi_Outlander_2003 | 188 | 1,827 | 10.3% |
| 32 | SDLG_ZL40F | 235 | 2,352 | 10.0% |
| 33 | SHACMAN_DeLong_M3000 | 126 | 1,262 | 10.0% |
| 34 | SHACMAN_DeLong_X3000 | 148 | 1,473 | 10.0% |
| 35 | WAW_Aochi_1800 | 156 | 1,507 | 10.4% |
| 36 | WAW_Aochi_Hongrui | 219 | 2,119 | 10.3% |
| 37 | Wuling_Rongguang_V | 160 | 1,564 | 10.2% |
| 38 | Yangzi_YZK6590XCA | 170 | 1,668 | 10.2% |
| 39 | Yutong_ZK6120HY1 | 170 | 1,680 | 10.1% |
| | **Total** | **6,896** | **68,091** | **10.1%** |

## Labeled set per class

Exactly 10% (rounded to the nearest image) of each class's SSL-pool images.

| Label | Class name | Labeled images | Class SSL-pool total | % of class |
|---:|---|---:|---:|---:|
| 0 | Buick_Excelle_GT | 134 | 1,343 | 10.0% |
| 1 | Buick_GL8 | 159 | 1,589 | 10.0% |
| 2 | CNHTC_HOWO | 139 | 1,386 | 10.0% |
| 3 | Chang'an_CS75_Plus | 208 | 2,083 | 10.0% |
| 4 | Chang'an_Starlight_4500 | 162 | 1,620 | 10.0% |
| 5 | Changfeng_Cheetah_CFA6473C | 110 | 1,101 | 10.0% |
| 6 | Changlin_8228-5 | 144 | 1,437 | 10.0% |
| 7 | Chery_Arrizo 5 | 132 | 1,317 | 10.0% |
| 8 | Chery_qq3 | 153 | 1,529 | 10.0% |
| 9 | Chevrolet_Blazer_1998 | 113 | 1,126 | 10.0% |
| 10 | Dongfeng_Duolika | 123 | 1,227 | 10.0% |
| 11 | Dongfeng_EQ6608LTV | 210 | 2,104 | 10.0% |
| 12 | Dongfeng_Forthing_Lingzhi | 181 | 1,812 | 10.0% |
| 13 | Dongfeng_Tianjin_DFH2200B | 144 | 1,436 | 10.0% |
| 14 | Dongfeng_Tianjin_KR230 | 142 | 1,420 | 10.0% |
| 15 | FAW_J6P | 140 | 1,401 | 10.0% |
| 16 | FAW_Jiabao_T51 | 139 | 1,394 | 10.0% |
| 17 | Foton_BJ1045V9JB5-54 | 148 | 1,483 | 10.0% |
| 18 | Great_Wall_Voleex_C50 | 203 | 2,029 | 10.0% |
| 19 | Great_Wall_poer | 167 | 1,669 | 10.0% |
| 20 | Hawtai_EV160B | 170 | 1,704 | 10.0% |
| 21 | Hongqi_CA7180A3E | 157 | 1,567 | 10.0% |
| 22 | Hongqi_h5 | 136 | 1,361 | 10.0% |
| 23 | Huanghai_N1 | 137 | 1,374 | 10.0% |
| 24 | Hyundai_HLF25_II | 140 | 1,400 | 10.0% |
| 25 | JAC_Junling | 158 | 1,580 | 10.0% |
| 26 | JINBEI_SY5033XJH | 186 | 1,856 | 10.0% |
| 27 | Jeep_Patriot | 127 | 1,268 | 10.0% |
| 28 | Lincoln_MKC | 145 | 1,447 | 10.0% |
| 29 | Lveco_Proud_2009 | 178 | 1,785 | 10.0% |
| 30 | MAXUS_V80 | 147 | 1,467 | 10.0% |
| 31 | Mitsubishi_Outlander_2003 | 164 | 1,639 | 10.0% |
| 32 | SDLG_ZL40F | 212 | 2,117 | 10.0% |
| 33 | SHACMAN_DeLong_M3000 | 114 | 1,136 | 10.0% |
| 34 | SHACMAN_DeLong_X3000 | 132 | 1,325 | 10.0% |
| 35 | WAW_Aochi_1800 | 135 | 1,351 | 10.0% |
| 36 | WAW_Aochi_Hongrui | 190 | 1,900 | 10.0% |
| 37 | Wuling_Rongguang_V | 140 | 1,404 | 10.0% |
| 38 | Yangzi_YZK6590XCA | 150 | 1,498 | 10.0% |
| 39 | Yutong_ZK6120HY1 | 151 | 1,510 | 10.0% |
| | **Total** | **6,120** | **61,195** | **10.0%** |

## All subsets per class

| Label | Class name | Validation | SSL pool | Labeled (inside SSL pool) | Official train | Official test |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Buick_Excelle_GT | 151 | 1,343 | 134 | 1,494 | 609 |
| 1 | Buick_GL8 | 177 | 1,589 | 159 | 1,766 | 787 |
| 2 | CNHTC_HOWO | 159 | 1,386 | 139 | 1,545 | 711 |
| 3 | Chang'an_CS75_Plus | 232 | 2,083 | 208 | 2,315 | 977 |
| 4 | Chang'an_Starlight_4500 | 180 | 1,620 | 162 | 1,800 | 777 |
| 5 | Changfeng_Cheetah_CFA6473C | 124 | 1,101 | 110 | 1,225 | 464 |
| 6 | Changlin_8228-5 | 162 | 1,437 | 144 | 1,599 | 658 |
| 7 | Chery_Arrizo 5 | 147 | 1,317 | 132 | 1,464 | 660 |
| 8 | Chery_qq3 | 179 | 1,529 | 153 | 1,708 | 744 |
| 9 | Chevrolet_Blazer_1998 | 128 | 1,126 | 113 | 1,254 | 554 |
| 10 | Dongfeng_Duolika | 137 | 1,227 | 123 | 1,364 | 620 |
| 11 | Dongfeng_EQ6608LTV | 238 | 2,104 | 210 | 2,342 | 1,039 |
| 12 | Dongfeng_Forthing_Lingzhi | 201 | 1,812 | 181 | 2,013 | 916 |
| 13 | Dongfeng_Tianjin_DFH2200B | 160 | 1,436 | 144 | 1,596 | 664 |
| 14 | Dongfeng_Tianjin_KR230 | 159 | 1,420 | 142 | 1,579 | 682 |
| 15 | FAW_J6P | 156 | 1,401 | 140 | 1,557 | 655 |
| 16 | FAW_Jiabao_T51 | 160 | 1,394 | 139 | 1,554 | 666 |
| 17 | Foton_BJ1045V9JB5-54 | 166 | 1,483 | 148 | 1,649 | 717 |
| 18 | Great_Wall_Voleex_C50 | 231 | 2,029 | 203 | 2,260 | 974 |
| 19 | Great_Wall_poer | 192 | 1,669 | 167 | 1,861 | 818 |
| 20 | Hawtai_EV160B | 189 | 1,704 | 170 | 1,893 | 790 |
| 21 | Hongqi_CA7180A3E | 174 | 1,567 | 157 | 1,741 | 750 |
| 22 | Hongqi_h5 | 152 | 1,361 | 136 | 1,513 | 623 |
| 23 | Huanghai_N1 | 153 | 1,374 | 137 | 1,527 | 622 |
| 24 | Hyundai_HLF25_II | 156 | 1,400 | 140 | 1,556 | 695 |
| 25 | JAC_Junling | 180 | 1,580 | 158 | 1,760 | 822 |
| 26 | JINBEI_SY5033XJH | 210 | 1,856 | 186 | 2,066 | 873 |
| 27 | Jeep_Patriot | 141 | 1,268 | 127 | 1,409 | 667 |
| 28 | Lincoln_MKC | 163 | 1,447 | 145 | 1,610 | 638 |
| 29 | Lveco_Proud_2009 | 201 | 1,785 | 178 | 1,986 | 848 |
| 30 | MAXUS_V80 | 166 | 1,467 | 147 | 1,633 | 695 |
| 31 | Mitsubishi_Outlander_2003 | 188 | 1,639 | 164 | 1,827 | 785 |
| 32 | SDLG_ZL40F | 235 | 2,117 | 212 | 2,352 | 981 |
| 33 | SHACMAN_DeLong_M3000 | 126 | 1,136 | 114 | 1,262 | 550 |
| 34 | SHACMAN_DeLong_X3000 | 148 | 1,325 | 132 | 1,473 | 634 |
| 35 | WAW_Aochi_1800 | 156 | 1,351 | 135 | 1,507 | 624 |
| 36 | WAW_Aochi_Hongrui | 219 | 1,900 | 190 | 2,119 | 945 |
| 37 | Wuling_Rongguang_V | 160 | 1,404 | 140 | 1,564 | 662 |
| 38 | Yangzi_YZK6590XCA | 170 | 1,498 | 150 | 1,668 | 680 |
| 39 | Yutong_ZK6120HY1 | 170 | 1,510 | 151 | 1,680 | 708 |
| | **Total** | **6,896** | **61,195** | **6,120** | **68,091** | **29,284** |
