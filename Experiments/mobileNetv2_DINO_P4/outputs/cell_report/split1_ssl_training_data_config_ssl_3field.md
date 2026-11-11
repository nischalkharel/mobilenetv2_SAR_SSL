# P4 cell report: split 1, ssl_training_data.csv, cells = scene | band | depression

- images 65387, classes in the pool 40, cells 36
- images per cell min / median / mean / max: 274 / 1289 / 1816.3 / 7657
- classes per cell min / median / max: 11 / 40 / 40 (image-weighted mean 37.5)
- cells with fewer than 5 classes: 0
- share of images in cells holding at least 20 classes: 91.3 % (= the share of the pool outside bare_soil and woodland, which hold 11 classes)
- largest-class share within a cell, median / max: 4.1 % / 12.4 %
- majority vote on the class from the cell alone: 3.99 % (scrambled cells of the same sizes 3.83 %, scene alone 3.65 %, chance 2.50 %)
- mutual information with the class, bits: cell 0.173 (scrambled cells 0.015, scene 0.161; class entropy 5.30)
- cells per scene: {'bare_soil': 8, 'factory': 8, 'sandstone': 8, 'urban': 4, 'woodland': 8}
- CSV columns the cell definition reads: ['relative_path', 'band', 'depression', 'scene'] (labels read for this report only)

Per-cell table: split1_ssl_training_data_config_ssl_3field.csv. Written 2026-11-10T15:06:02.
