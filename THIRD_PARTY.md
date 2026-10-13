# Third-party source

The following files come from the official [CGCNN repository](https://github.com/txie-93/cgcnn), commit `f42ab233c4ee0c416879d6bc2d22a264418413ad`:

| Release file | Upstream path | SHA-256 |
| --- | --- | --- |
| `cgcnn_model.py` | `cgcnn/model.py` | `0a773283d83fae13d274cacbf67153adcf3a4875cb64075b558fbdd6e998198f` |
| `atom_init.json` | `data/sample-regression/atom_init.json` | `cb90bd2257dbf8cfc4e6b2b6d4348d4616e8e9680f1339b1461137032e0f1894` |

The source/model and atom initialization are unchanged; only the model file name is adapted to this flat repository layout. Their MIT license, copyright (c) 2018 Tian Xie, is preserved in `LICENSE-CGCNN.txt`.

Reference: Tian Xie and Jeffrey C. Grossman, *Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties*, Physical Review Letters 120, 145301 (2018), DOI: 10.1103/PhysRevLett.120.145301.

Other imported packages are installed separately and retain their own licenses. MultiMat, ALIGNN and PotNet source trees and raw MP/AFLOW data are not redistributed in this core release.
