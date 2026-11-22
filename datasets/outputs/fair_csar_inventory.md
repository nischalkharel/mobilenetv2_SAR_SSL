# FAIR-CSAR-V1.0 SL inventory (from annotations under `C:\Users\nisch\Documents\SHREC\2026_2027\FAIR_CSAR_sample`)

Slice size 1024 x 1024 px (METAXmls), mode SL, nominal resolution 1.0 m. One DOTA-style txt per slice, one line per object: 8 polygon coordinates, class, difficulty flag.

## Per split

| split | scenes | slices | slices with objects | objects | objects / slice (median, p95, max) | difficult flag = 1 |
|---|---|---|---|---|---|---|
| trainval | 105 | 11959 | 11959 (100 %) | 69298 | 3, 21, 112 | 5052 |
| test | 26 | 2706 | 2706 (100 %) | 15987 | 3, 21, 150 | 1069 |

## Per class (object counts, and the long side of the oriented box in slice pixels)

| major | class | trainval | test | total | 1 % of trainval | long side median | p5 | p95 | short side median |
|---|---|---|---|---|---|---|---|---|---|
| Aircraft | Airbus_A220 | 3586 | 854 | 4440 | 35.9 | 44 | 26 | 70 | 36 |
| Aircraft | Airbus_A320 | 1543 | 240 | 1783 | 15.4 | 55 | 39 | 87 | 47 |
| Aircraft | Airbus_A330 | 1437 | 188 | 1625 | 14.4 | 75 | 51 | 103 | 67 |
| Aircraft | Airfreighter | 984 | 50 | 1034 | 9.8 | 54 | 24 | 94 | 43 |
| Aircraft | Boeing737 | 2485 | 294 | 2779 | 24.9 | 44 | 30 | 70 | 37 |
| Aircraft | Boeing747 | 332 | 58 | 390 | 3.3 | 71 | 56 | 99 | 64 |
| Aircraft | Boeing767 | 2372 | 170 | 2542 | 23.7 | 66 | 43 | 94 | 58 |
| Aircraft | Boeing777 | 589 | 100 | 689 | 5.9 | 78 | 58 | 107 | 72 |
| Aircraft | Fokker-50 | 487 | 50 | 537 | 4.9 | 33 | 24 | 42 | 29 |
| Aircraft | Gulfstream | 662 | 278 | 940 | 6.6 | 36 | 21 | 67 | 29 |
| Aircraft | Helicopter | 4902 | 318 | 5220 | 49.0 | 14 | 9 | 27 | 9 |
| Aircraft | Other_Aircraft | 1038 | 462 | 1500 | 10.4 | 35 | 22 | 67 | 29 |
| Bridge | Bridge | 2114 | 693 | 2807 | 21.1 | 84 | 34 | 850 | 28 |
| Ship | Bulk_Carrier | 976 | 249 | 1225 | 9.8 | 182 | 59 | 352 | 37 |
| Ship | Container_Ship | 1221 | 347 | 1568 | 12.2 | 151 | 67 | 281 | 34 |
| Ship | General_Cargo_Ship | 13196 | 2897 | 16093 | 132.0 | 55 | 28 | 117 | 16 |
| Ship | Motion_Defocusing_Ship | 2183 | 629 | 2812 | 21.8 | 81 | 35 | 185 | 25 |
| Ship | Oil_Tanker | 1478 | 326 | 1804 | 14.8 | 112 | 51 | 338 | 26 |
| Ship | Other_Ship | 8682 | 2772 | 11454 | 86.8 | 51 | 24 | 113 | 16 |
| Ship | Warship | 1977 | 280 | 2257 | 19.8 | 124 | 46 | 254 | 25 |
| Tank | Tank | 15364 | 4062 | 19426 | 153.6 | 45 | 22 | 99 | 35 |
| Tower_Crane | Tower_Crane | 1690 | 670 | 2360 | 16.9 | 91 | 47 | 170 | 16 |

## Object long side, all classes (slice pixels)

| bin | objects | share |
|---|---|---|
| 0 to 8 | 67 | 0.1 % |
| 8 to 16 | 3455 | 4.1 % |
| 16 to 32 | 11897 | 13.9 % |
| 32 to 64 | 37589 | 44.1 % |
| 64 to 128 | 25653 | 30.1 % |
| 128 to 256 | 5183 | 6.1 % |
| 256 to inf | 1441 | 1.7 % |

All objects: long side median 52 px, p5 18, p95 158, max 1448; short side median 26 px.

## Scenes (slices and objects per scene, largest first)

| split | scene | slices | objects | classes |
|---|---|---|---|---|
| trainval | GF3_SAY_SL_027602_E139.7_N35.5_20211106_L1A_HH_L10000000001 | 252 | 1970 | Tank 1029, General_Cargo_Ship 432, Other_Ship 254 |
| trainval | GF3_MYN_SL_028704_E139.7_N35.5_20220122_L1A_HH_L10000000001 | 257 | 1871 | Tank 846, General_Cargo_Ship 426, Other_Ship 286 |
| trainval | GF3_MYN_SL_025094_E139.7_N35.5_20210516_L1A_HH_L10000000026 | 283 | 1870 | Tank 991, General_Cargo_Ship 354, Other_Ship 244 |
| trainval | GF3_SAY_SL_027184_E139.7_N35.4_20211008_L1A_HH_L10000000001 | 235 | 1841 | Tank 993, General_Cargo_Ship 400, Other_Ship 258 |
| trainval | GF3_MYN_SL_028856_E139.7_N35.5_20220201_L1A_HH_L10000000001 | 239 | 1814 | Tank 1004, General_Cargo_Ship 438, Other_Ship 113 |
| test | GF3_KAS_SL_028368_E139.7_N35.5_20211229_L1A_HH_L10000000001 | 199 | 1792 | Tank 740, General_Cargo_Ship 558, Other_Ship 242 |
| trainval | GF3_MYN_SL_028438_E139.7_N35.5_20220103_L1A_HH_L10000000001 | 230 | 1790 | Tank 944, General_Cargo_Ship 449, Other_Ship 226 |
| trainval | GF3_MYN_SL_024503_E139.7_N35.5_20210405_L1A_HH_L10000000001 | 215 | 1677 | Tank 830, General_Cargo_Ship 427, Other_Ship 218 |
| trainval | GF3_MYN_SL_027848_E139.7_N35.5_20211123_L1A_HH_L10000000001 | 226 | 1627 | Tank 850, General_Cargo_Ship 437, Other_Ship 170 |
| trainval | GF3_MYN_SL_028540_E139.7_N35.5_20220110_L1A_HH_L10000000001 | 185 | 1596 | Tank 820, General_Cargo_Ship 354, Other_Ship 228 |
| trainval | GF3_MYN_SL_025195_E139.7_N35.5_20210523_L1A_HH_L10000000001 | 190 | 1568 | Tank 873, General_Cargo_Ship 284, Other_Ship 218 |
| test | GF3_MYN_SL_028827_E121.5_N31.4_20220130_L1A_HH_L10000000001 | 132 | 1448 | Tank 846, Other_Ship 263, General_Cargo_Ship 185 |
| trainval | GF3_SAY_SL_029101_E139.7_N35.4_20220218_L1A_HH_L10000000001 | 206 | 1443 | Tank 692, General_Cargo_Ship 376, Other_Ship 222 |
| trainval | GF3_MYN_SL_029029_E139.7_N35.5_20220213_L1A_HH_L10000000001 | 200 | 1430 | Tank 720, General_Cargo_Ship 328, Tower_Crane 108 |
| test | GF3_KAS_SL_028685_E139.7_N35.5_20220120_L1A_HH_L10000000001 | 206 | 1342 | Tank 563, General_Cargo_Ship 398, Other_Ship 186 |
| trainval | GF3_MYN_SL_036366_E129.0_N35.1_20230707_L1A_HH_L10000000001 | 162 | 1256 | Other_Ship 524, General_Cargo_Ship 304, Tower_Crane 104 |
| trainval | GF3_MYN_SL_023019_E120.9_N14.6_20201223_L1A_HH_L10000000001 | 229 | 1186 | General_Cargo_Ship 678, Other_Ship 292, Oil_Tanker 92 |
| trainval | GF3_SAY_SL_029043_E121.0_N14.6_20220214_L1A_HH_L10000000001 | 194 | 1151 | General_Cargo_Ship 722, Other_Ship 204, Container_Ship 65 |
| trainval | GF3_MYN_SL_026133_E104.0_N1.3_20210727_L1A_HH_L10000000001 | 185 | 1043 | Other_Ship 279, Airbus_A220 182, Airbus_A320 118 |
| test | GF3_KRN_SL_032500_E35.0_N32.8_20221012_L1A_HH_L10000000001 | 114 | 986 | Tank 652, Tower_Crane 106, Other_Ship 76 |
| trainval | GF3_SAY_SL_036539_E129.1_N35.1_20230719_L1A_HH_L10000000001 | 170 | 972 | Other_Ship 300, General_Cargo_Ship 222, Tower_Crane 102 |
| test | GF3_MYN_SL_021498_E121.0_N14.6_20200909_L1A_HH_L10000000001 | 155 | 964 | General_Cargo_Ship 426, Other_Ship 286, Container_Ship 62 |
| trainval | GF3_KRN_SL_019380_E104.0_N1.3_20200415_L1A_HH_L10000000001 | 204 | 950 | Boeing737 220, Boeing767 92, Oil_Tanker 86 |
| test | GF3_KAS_SL_028469_E104.0_N1.3_20220105_L1A_HH_L10000000001 | 200 | 938 | Airbus_A220 216, Other_Ship 136, Airbus_A320 112 |
| trainval | GF3_MYN_SL_025678_E121.0_N14.6_20210626_L1A_HH_L10000000001 | 183 | 923 | General_Cargo_Ship 404, Other_Ship 230, Container_Ship 94 |
| trainval | GF3_SAY_SL_017786_E121.5_N31.0_20191226_L1A_HH_L10000000001 | 92 | 911 | General_Cargo_Ship 774, Bridge 59, Other_Ship 58 |
| trainval | GF3_SAY_SL_035486_E121.0_N14.6_20230507_L1A_HH_L10000000001 | 192 | 896 | General_Cargo_Ship 366, Other_Ship 340, Container_Ship 73 |
| trainval | GF3_MYN_SL_027589_E124.0_N10.3_20211105_L1A_HH_L10000000001 | 124 | 891 | General_Cargo_Ship 484, Other_Ship 136, Tank 76 |
| test | GF3_KAS_SL_024886_E104.0_N1.3_20210502_L1A_HH_L10000000001 | 143 | 885 | Other_Ship 201, Boeing737 158, Boeing767 98 |
| trainval | GF3_MYN_SL_026508_E124.0_N10.3_20210822_L1A_HH_L10000000001 | 118 | 866 | General_Cargo_Ship 512, Tank 112, Other_Ship 58 |
| trainval | GF3_MYN_SL_022997_E124.0_N10.3_20201222_L1A_HH_L10000000001 | 107 | 863 | General_Cargo_Ship 617, Other_Ship 124, Tank 32 |
| trainval | GF3_KRN_SL_033264_W157.9_N21.4_20221204_L1A_HH_L10000000001 | 145 | 855 | Helicopter 318, Tank 156, Airfreighter 106 |
| test | GF3_SAY_SL_036121_E129.0_N35.1_20230620_L1A_HH_L10000000001 | 131 | 844 | Other_Ship 334, General_Cargo_Ship 134, Tank 94 |
| trainval | GF3_MYN_SL_036192_W157.9_N21.4_20230625_L1A_HH_L10000000001 | 123 | 838 | Helicopter 288, Tank 146, Airfreighter 110 |
| test | GF3_SAY_SL_017051_E118.7_N32.1_20191105_L1A_DH_L10000000001 | 175 | 821 | General_Cargo_Ship 519, Other_Ship 248, Motion_Defocusing_Ship 24 |
| trainval | GF3_KAS_SL_023978_E104.0_N1.4_20210228_L1A_HH_L10000000001 | 140 | 799 | Boeing767 171, Boeing737 160, General_Cargo_Ship 110 |
| trainval | GF3_MYN_SL_023214_E104.0_N1.4_20210106_L1A_HH_L10000000001 | 103 | 798 | Boeing767 194, Boeing737 186, Airbus_A330 70 |
| trainval | GF3_MYN_SL_028597_E124.0_N10.3_20220114_L1A_HH_L10000000001 | 110 | 788 | General_Cargo_Ship 476, Other_Ship 86, Tank 68 |
| trainval | GF3_KAS_SL_032260_W117.2_N32.7_20220926_L1A_HH_L10000000001 | 48 | 780 | Helicopter 636, Tank 92, Airbus_A220 12 |
| trainval | GF3_KAS_SL_021888_E104.0_N1.4_20201006_L1A_HH_L10000000001 | 100 | 777 | Boeing767 172, Airbus_A330 148, Boeing737 128 |
| trainval | GF3_MYN_SL_025542_E104.0_N1.4_20210616_L1A_HH_L10000000001 | 123 | 773 | Airbus_A220 158, Airbus_A320 155, Boeing767 136 |
| trainval | GF3_MYN_SL_026378_E104.0_N1.4_20210813_L1A_HH_L10000000001 | 125 | 758 | Airbus_A220 206, Airbus_A320 108, Boeing767 78 |
| trainval | GF3_KRN_SL_038784_W117.2_N32.7_20231222_L1A_HH_L10000000001 | 100 | 735 | Helicopter 300, Warship 244, Airfreighter 86 |
| trainval | GF3_MYN_SL_026313_E104.0_N1.3_20210809_L1A_HH_L10000000001 | 177 | 731 | Other_Ship 134, Boeing737 114, General_Cargo_Ship 82 |
| trainval | GF3_SAY_SL_037627_E124.0_N10.3_20231003_L1A_HH_L10000000001 | 136 | 729 | General_Cargo_Ship 389, Tank 86, Other_Ship 82 |
| test | GF3_MYN_SL_021476_E118.1_N24.5_20200907_L1A_HH_L10000000001 | 171 | 728 | Other_Ship 254, Tower_Crane 162, General_Cargo_Ship 138 |
| trainval | GF3_MYN_SL_038505_W157.9_N21.4_20231203_L1A_HH_L10000000001 | 126 | 706 | Helicopter 176, Tank 132, Airfreighter 104 |
| trainval | GF3_KAS_SL_028051_E104.0_N1.4_20211207_L1A_HH_L10000000001 | 118 | 703 | Boeing737 122, Gulfstream 112, Boeing767 110 |
| trainval | GF3_KAS_SL_025723_E104.0_N1.4_20210629_L1A_HH_L10000000001 | 87 | 686 | Boeing767 188, Boeing737 158, Airbus_A330 96 |
| trainval | GF3_KAS_SL_038702_W158.0_N21.3_20231216_L1A_HH_L10000000001 | 124 | 683 | Helicopter 188, Airfreighter 98, Tank 94 |
| test | GF3B_MYC_SL_005016_E117.3_N39.2_20221105_L1A_HH_L10000111814 | 107 | 675 | Airbus_A220 301, Other_Aircraft 156, Tank 102 |
| trainval | GF3_KAS_SL_026069_E104.0_N1.4_20210723_L1A_HH_L10000000001 | 130 | 672 | Boeing767 192, Boeing737 158, Airbus_A330 94 |
| trainval | GF3_MYN_SL_028821_E104.0_N1.4_20220130_L1A_HH_L10000000001 | 94 | 644 | Boeing737 166, Boeing767 120, Airbus_A330 88 |
| trainval | GF3_KRN_SL_032673_W158.0_N21.4_20221024_L1A_HH_L10000000001 | 137 | 634 | Helicopter 192, Airfreighter 100, Tank 88 |
| trainval | GF3_KAS_SL_026558_E104.0_N1.4_20210826_L1A_HH_L10000000001 | 73 | 628 | Boeing737 182, Boeing767 148, Airbus_A330 88 |
| trainval | GF3_MYN_SL_024295_E104.0_N1.3_20210322_L1A_HH_L10000000001 | 128 | 628 | Other_Ship 217, Tank 78, Bulk_Carrier 75 |
| trainval | GF3_MYN_SL_023041_E104.0_N1.3_20201225_L1A_HH_L10000000001 | 81 | 627 | Boeing767 148, Boeing737 116, Other_Ship 96 |
| trainval | GF3_KAS_SL_031006_W117.2_N32.7_20220701_L1A_HH_L10000000001 | 63 | 624 | Helicopter 432, Tank 114, Other_Aircraft 18 |
| trainval | GF3_KAS_SL_029131_E104.0_N1.4_20220220_L1A_HH_L10000000001 | 102 | 601 | Airbus_A220 248, Airbus_A320 72, Tank 68 |
| trainval | GF3_KRN_SL_031001_W157.9_N21.4_20220630_L1A_HH_L10000000001 | 123 | 594 | Helicopter 168, Warship 159, Tank 91 |
| trainval | GF3_KAS_SL_028671_W117.2_N32.7_20220120_L1A_HH_L10000000001 | 93 | 590 | Helicopter 278, Warship 81, Tank 80 |
| trainval | GF3_SAY_SL_029066_E104.0_N1.4_20220216_L1A_HH_L10000000001 | 92 | 588 | Boeing737 132, Boeing767 106, Airbus_A330 82 |
| trainval | GF3_MYN_SL_020899_E128.6_N35.1_20200729_L1A_HH_L10000000001 | 127 | 570 | General_Cargo_Ship 254, Other_Ship 180, Warship 52 |
| test | GF3_KAS_SL_029363_E124.0_N10.3_20220308_L1A_HH_L10000000001 | 109 | 568 | General_Cargo_Ship 242, Tank 70, Motion_Defocusing_Ship 64 |
| trainval | GF3_SAY_SL_037202_E124.0_N10.3_20230903_L1A_HH_L10000000001 | 121 | 564 | General_Cargo_Ship 240, Other_Ship 104, Tank 88 |
| trainval | GF3_SAY_SL_019033_E104.0_N1.3_20200322_L1A_HH_L10000000001 | 155 | 563 | General_Cargo_Ship 95, Oil_Tanker 73, Tank 72 |
| trainval | GF3_KRN_SL_038381_W157.9_N21.3_20231124_L1A_HH_L10000000001 | 92 | 558 | Tank 178, Airfreighter 80, Helicopter 60 |
| trainval | GF3B_MYC_SL_005210_E117.3_N39.2_20221119_L1A_HH_L10000118121 | 97 | 554 | Airbus_A220 358, Tank 70, Airbus_A320 54 |
| trainval | GF3_SAY_SL_031703_E120.3_N22.6_20220818_L1A_HH_L10000000001 | 175 | 540 | Other_Ship 134, General_Cargo_Ship 104, Bridge 102 |
| trainval | GF3_SAY_SL_037173_E104.0_N1.4_20230901_L1A_HH_L10000000001 | 91 | 534 | Airbus_A330 90, Airbus_A320 76, Tank 68 |
| trainval | GF3_SAY_SL_029585_E104.0_N1.4_20220324_L1A_HH_L10000000001 | 111 | 528 | Boeing767 88, Airbus_A330 84, Tank 72 |
| test | GF3_MDJ_SL_008612_E116.6_N40.1_20180330_L1A_VV_L10003093595 | 94 | 526 | Gulfstream 172, Tank 124, Other_Aircraft 64 |
| trainval | GF3B_MYC_SL_005044_E117.3_N39.2_20221107_L1A_HH_L10000113028 | 60 | 518 | Airbus_A220 364, Other_Aircraft 78, Tank 40 |
| trainval | GF3_MYN_SL_008576_E121.3_N31.2_20180327_L1A_VV_L10003088112 | 82 | 514 | Other_Aircraft 174, Airbus_A220 140, Bridge 72 |
| trainval | GF3_SAY_SL_022485_E113.8_N22.6_20201116_L1A_HH_L10000000001 | 206 | 490 | General_Cargo_Ship 192, Tank 96, Motion_Defocusing_Ship 70 |
| trainval | GF3_KAS_SL_029752_W117.2_N32.7_20220405_L1A_HH_L10000000001 | 46 | 484 | Helicopter 340, Tank 60, Airbus_A220 16 |
| trainval | GF3_KAS_SL_039199_E104.0_N1.3_20240120_L1A_HH_L10000000001 | 203 | 481 | Boeing737 79, Bulk_Carrier 77, Tank 76 |
| trainval | GF3_MYN_SL_022824_E121.0_N14.6_20201210_L1A_HH_L10000000001 | 86 | 476 | Other_Ship 294, Tower_Crane 58, Container_Ship 40 |
| trainval | GF3_KAS_SL_033514_W117.2_N32.7_20221222_L1A_HH_L10000000001 | 46 | 470 | Helicopter 342, Tank 62, Other_Aircraft 20 |
| trainval | GF3_MYN_SL_022507_E118.1_N24.5_20201118_L1A_HH_L10000000001 | 97 | 464 | Tower_Crane 128, Other_Ship 96, Tank 80 |
| test | GF3_MDJ_SL_008762_E121.3_N31.2_20180409_L1A_VV_L10003113235 | 70 | 463 | Airbus_A220 134, Other_Aircraft 108, Gulfstream 90 |
| trainval | GF3B_MYC_SL_005037_E117.3_N39.2_20221107_L1A_HH_L10000112855 | 83 | 456 | Airbus_A220 278, Tank 54, Bridge 34 |
| trainval | GF3_KRN_SL_028644_E104.0_N1.3_20220117_L1A_HH_L10000000001 | 162 | 451 | Motion_Defocusing_Ship 121, Tank 80, Bulk_Carrier 52 |
| trainval | GF3_MYN_SL_038831_E120.3_N22.7_20231225_L1A_HH_L10000000001 | 128 | 450 | General_Cargo_Ship 102, Bridge 97, Warship 89 |
| trainval | GF3_MYN_SL_023870_E104.0_N1.3_20210220_L1A_HH_L10000000001 | 207 | 427 | Motion_Defocusing_Ship 234, Tank 76, Oil_Tanker 30 |
| test | GF3B_KSC_SL_000297_W71.0_N42.4_20211213_L1A_HH_L10000129337 | 70 | 422 | Tank 282, Airbus_A220 65, Other_Aircraft 26 |
| test | GF3_KAS_SL_029340_E121.2_N25.1_20220307_L1A_HH_L10000000001 | 80 | 419 | Tank 103, Airbus_A330 88, Airbus_A320 64 |
| trainval | GF3_KAS_SL_035186_W117.2_N32.7_20230417_L1A_HH_L10000000001 | 33 | 407 | Helicopter 306, Tank 51, Airbus_A220 12 |
| trainval | GF3_MYN_SL_033043_E128.6_N35.1_20221119_L1A_HH_L10000000001 | 74 | 402 | Other_Ship 236, General_Cargo_Ship 54, Motion_Defocusing_Ship 44 |
| trainval | GF3_KAS_SL_005659_E77.1_N28.6_20170906_L1A_HH_L10002590291 | 83 | 382 | Helicopter 130, Airbus_A220 110, Tank 32 |
| trainval | GF3_KAS_SL_005609_E77.1_N28.6_20170903_L1A_HH_L10002583277 | 69 | 372 | Helicopter 138, Airbus_A220 120, Airbus_A320 50 |
| trainval | GF3_SAY_SL_008713_E116.6_N40.1_20180406_L1A_HH_L10006795873 | 94 | 361 | Gulfstream 96, Airbus_A320 94, Tank 48 |
| test | GF3_MYN_SL_038240_E120.3_N22.6_20231114_L1A_HH_L10000000001 | 116 | 355 | Bridge 136, General_Cargo_Ship 93, Other_Ship 58 |
| trainval | GF3_KAS_SL_005659_E77.1_N28.6_20170906_L1A_HH_L10004912834 | 62 | 354 | Helicopter 140, Airbus_A220 108, Tank 32 |
| trainval | GF3B_MYC_SL_009188_E121.3_N25.1_20230822_L1A_HH_L10000228817 | 71 | 326 | Tank 102, Airbus_A320 62, Boeing747 58 |
| trainval | GF3_MYN_SL_026254_E128.7_N35.1_20210805_L1A_HH_L10000000001 | 45 | 324 | Other_Ship 136, Motion_Defocusing_Ship 88, General_Cargo_Ship 50 |
| trainval | GF3_SAY_SL_000029_E100.9_N12.6_20230920_L1A_HH_L10000000001 | 62 | 320 | Other_Ship 112, Warship 74, Tank 50 |
| trainval | GF3_MYN_SL_022132_E122.5_N30.7_20201023_L1A_HH_L10000000001 | 35 | 316 | Motion_Defocusing_Ship 200, Other_Ship 106, Bridge 6 |
| test | GF3_KAS_SL_026731_E100.9_N12.6_20210907_L1A_HH_L10000000001 | 65 | 303 | Other_Ship 104, Warship 75, General_Cargo_Ship 50 |
| test | GF3_SAY_SL_028784_E140.4_N35.8_20220127_L1A_HH_L10000000001 | 36 | 288 | Tank 134, Other_Aircraft 56, Airbus_A220 56 |
| test | GF3_KAS_SL_022725_E100.9_N12.6_20201203_L1A_HH_L10000000001 | 60 | 287 | Other_Ship 140, Warship 57, Tank 46 |
| test | GF3_MYN_SL_022356_E122.5_N30.7_20201107_L1A_HH_L10000000001 | 43 | 286 | Other_Ship 132, Motion_Defocusing_Ship 128, Container_Ship 22 |
| trainval | GF3_SAY_SL_037202_E100.9_N12.6_20230903_L1A_HH_L10000000001 | 71 | 282 | Warship 96, Motion_Defocusing_Ship 60, Other_Ship 56 |
| trainval | GF3_SAY_SL_036575_E114.3_N30.6_20230722_L1A_HH_L10000000001 | 103 | 279 | General_Cargo_Ship 112, Other_Ship 82, Tank 39 |
| trainval | GF3_SAY_SL_008511_E121.3_N31.2_20180323_L1A_VV_L10003079563 | 90 | 278 | Bridge 80, Other_Aircraft 58, Airbus_A220 50 |
| trainval | GF3_KRN_SL_032744_E35.9_N34.9_20221029_L1A_HH_L10000000001 | 53 | 274 | Tank 90, Other_Ship 84, Bulk_Carrier 34 |
| trainval | GF3_KAS_SL_036540_E100.9_N12.6_20230719_L1A_HH_L10000000001 | 54 | 264 | Warship 88, Other_Ship 58, Tank 46 |
| trainval | GF3_KAS_SL_036194_E100.9_N12.6_20230625_L1A_HH_L10000000001 | 74 | 255 | Warship 70, Other_Ship 64, Tank 46 |
| trainval | GF3_SAY_SL_008684_E121.3_N31.2_20180404_L1A_HH_L10003103225 | 108 | 234 | Bridge 116, Other_Aircraft 40, Tank 32 |
| trainval | GF3C_MYC_SL_003293_E117.4_N39.2_20221121_L1A_HH_L10000083565 | 21 | 230 | Airbus_A220 186, Gulfstream 36, Other_Aircraft 6 |
| trainval | GF3_SAY_SL_035775_E100.9_N12.6_20230527_L1A_HH_L10000000001 | 79 | 220 | Motion_Defocusing_Ship 58, Tank 46, Warship 44 |
| trainval | GF3_KAS_SL_036367_E100.9_N12.6_20230707_L1A_HH_L10000000001 | 47 | 213 | Other_Ship 84, Warship 40, Motion_Defocusing_Ship 38 |
| trainval | GF3_MYN_SL_015581_E114.2_N30.8_20190726_L1A_HH_L10004143063 | 58 | 194 | Airbus_A220 92, Bridge 54, Airbus_A320 20 |
| trainval | GF3_KAS_SL_035950_E100.9_N12.6_20230608_L1A_HH_L10000000001 | 43 | 171 | Tank 46, Other_Ship 39, Motion_Defocusing_Ship 36 |
| trainval | GF3_SAY_SL_036575_E115.8_N28.7_20230722_L1A_HH_L10000000001 | 121 | 170 | Bridge 96, Other_Ship 56, General_Cargo_Ship 14 |
| test | GF3_MYN_SL_028683_E115.8_N28.7_20220120_L1A_HH_L10000000001 | 104 | 165 | Bridge 123, Other_Ship 30, General_Cargo_Ship 4 |
| trainval | GF3_MYN_SL_029814_E24.7_N48.9_20220409_L1A_HH_L10000000001 | 24 | 156 | Helicopter 136, Bridge 12, Fokker-50 8 |
| test | GF3_KRN_SL_029321_E24.7_N48.9_20220306_L1A_HH_L10000000001 | 23 | 154 | Helicopter 130, Bridge 12, Fokker-50 12 |
| trainval | GF3_KRN_SL_029393_E24.7_N48.9_20220311_L1A_HH_L10000000001 | 24 | 152 | Helicopter 128, Bridge 12, Fokker-50 12 |
| trainval | GF3_MYN_SL_028726_E126.4_N37.4_20220123_L1A_HH_L10000000001 | 19 | 147 | Other_Aircraft 78, Airbus_A220 41, Airbus_A320 22 |
| trainval | GF3_SAY_SL_036618_E100.9_N12.6_20230724_L1A_HH_L10000000001 | 51 | 144 | Tank 46, Motion_Defocusing_Ship 36, Warship 28 |
| test | GF3_KAS_SL_005464_E77.1_N28.6_20170824_L1A_HH_L10002559081 | 21 | 144 | Helicopter 128, Airbus_A220 12, Bridge 4 |
| trainval | GF3_KRN_SL_032499_E43.2_N11.5_20221012_L1A_HH_L10000000001 | 29 | 132 | Airfreighter 44, Helicopter 44, Fokker-50 20 |
| test | GF3_MYN_SL_021620_E121.1_N27.9_20200917_L1A_HH_L10000000001 | 60 | 126 | Other_Ship 34, Container_Ship 32, Motion_Defocusing_Ship 26 |
| trainval | GF3_KRN_SL_032470_E43.2_N11.5_20221010_L1A_HH_L10000000001 | 24 | 118 | Helicopter 54, Airfreighter 42, Fokker-50 22 |
| trainval | GF3_SAY_SL_016042_E121.3_N31.2_20190827_L1A_HH_L10004209496 | 43 | 104 | Gulfstream 64, Bridge 40 |
| trainval | GF3_MYN_SL_023069_E122.4_N29.9_20201227_L1A_HH_L10000000001 | 61 | 98 | General_Cargo_Ship 32, Other_Ship 30, Bridge 28 |
| trainval | GF3_KRN_SL_019362_E43.1_N11.5_20200414_L1A_HH_L10000000001 | 23 | 76 | Airfreighter 28, Fokker-50 24, Helicopter 12 |
| trainval | GF3_MYN_SL_021815_E120.8_N27.7_20201001_L1A_HH_L10000000001 | 53 | 69 | Motion_Defocusing_Ship 57, Other_Ship 8, Bridge 4 |
| test | GF3_KRN_SL_019174_E43.2_N11.5_20200401_L1A_HH_L10000000001 | 22 | 58 | Airfreighter 28, Fokker-50 22, Airbus_A320 4 |
| trainval | GF3_SAY_SL_029094_E121.0_N14.5_20220218_L1A_HH_L10000000001 | 4 | 8 | Gulfstream 8 |

Classes in trainval only: none; in test only: none.
