# P4 cell report: split 1, ssl_training_data.csv, cells = scene | band | depression | az_sector30

- images 65387, classes in the pool 40, cells 432
- images per cell min / median / mean / max: 17 / 103 / 151.4 / 668
- classes per cell min / median / max: 7 / 36 / 40 (image-weighted mean 36.1)
- cells with fewer than 5 classes: 0
- share of images in cells holding at least 20 classes: 91.3 % (= the share of the pool outside bare_soil and woodland, which hold 11 classes)
- largest-class share within a cell, median / max: 5.2 % / 23.5 %
- majority vote on the class from the cell alone: 4.86 % (scrambled cells of the same sizes 5.60 %, scene alone 3.65 %, chance 2.50 %)
- mutual information with the class, bits: cell 0.270 (scrambled cells 0.203, scene 0.161; class entropy 5.30)
- cells per scene: {'bare_soil': 96, 'factory': 96, 'sandstone': 96, 'urban': 48, 'woodland': 96}
- CSV columns the cell definition reads: ['relative_path', 'azimuth', 'band', 'depression', 'scene'] (labels read for this report only)

Per-cell table: split1_ssl_training_data_config_ssl.csv. Written 2026-11-10T19:41:49.
