# Cross-Channel Unlabeled Sensing (CCUS)

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)
![Platform](https://img.shields.io/badge/platform-Linux%20|%20macOS-lightgrey.svg)

---

**Official Python implementation** accompanying:

- *Shuffled Multi-Channel Sparse Signal Recovery*  
  **T. Koka, M.C. Tsakiris, M. Muma, B. Béjar Haro**  
  _Signal Processing_, Elsevier, 2024  
- *Cross-Channel Unlabeled Sensing over a Union of Signal Subspaces*  
  **T. Koka, M.C. Tsakiris, B. Béjar Haro, M. Muma**  
  _IEEE ICASSP_, 2025

---

## ⚙️ Overview

This repository provides reference implementations for **unshuffling multi-channel signals**  
under unknown channel permutations — a generalization of *unlabeled sensing*.

The algorithms combine:
- **Robust MM estimation** for sample assignment recovery (`MM_shuffled`, `MMrse_shuffled`)
- **Sparse and structured regression** for coefficient estimation
- **Dictionary learning** (via convolutional sparse coding) to obtain signal atoms
- Simulation and **real-data experiments** (Drosophila calcium imaging)

---

## 📂 Repository Structure

```
CrossChannelUnlabeledSensing/
│
├── utils.py                    # Signal simulation, denoising, helper routines
├── mme_shuffled_signals.py     # MM / MMRSE algorithms for shuffled signals
├── csc_unshuffle_pipeline.py   # (optional) Learn dictionary + circulant embedding
├── example.ipynb               # End-to-end reproducible experiments
├── LICENSE                     # GNU GPL v3.0 license
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/<your-username>/CrossChannelUnlabeledSensing.git
cd CrossChannelUnlabeledSensing
pip install -r requirements.txt
```

### Run example
```bash
jupyter notebook example.ipynb
```

### Example: Two-channel unshuffling
```python
from mme_shuffled_signals import MM_shuffled
from utils import simulate_noisy_signal, reconstruction_params

# simulate and reconstruct
b1, b2, _, _, q_hat, losses = MM_shuffled(y1_obs, y2_obs, Z_hat, bdp=0.5)
```

---

## 🔬 Learn Circulant Dictionary

```python
from csc_unshuffle_pipeline import learn_circulant_dict_from_batch

learned = learn_circulant_dict_from_batch(samples_drosophila, n_times_atom=61)
atom = learned["atom"]
D = learned["D_circ"]
```

---

## 📊 Results (example)

| SNR (dB) | Weighted Accuracy | Normalized MSE |
|:--------:|:-----------------:|:--------------:|
| 0        | 0.69              | 8.0e-01        |
| 10       | 0.79              | 3.4e-01        |
| 20       | 0.95              | 1.3e-03        |
| 30       | 0.99              | 1.1e-04        |
| 40       | 0.999             | 8.4e-06        |
| 50       | 1.000             | 1.0e-06        |

---

## 🧠 Citation

If you use this code, please cite:

```bibtex
@article{koka2024shuffled,
  title={Shuffled Multi-Channel Sparse Signal Recovery},
  author={Koka, Taulant and Tsakiris, Manolis C. and Muma, Michael and Béjar Haro, Benjamín},
  journal={Signal Processing},
  volume={224},
  pages={109579},
  year={2024},
  publisher={Elsevier}
}

@inproceedings{koka2025ccus,
  title={Cross-Channel Unlabeled Sensing over a Union of Signal Subspaces},
  author={Koka, Taulant and Tsakiris, Manolis C. and Béjar Haro, Benjamín and Muma, Michael},
  booktitle={Proc. IEEE ICASSP},
  year={2025},
  organization={IEEE}
}
```

---

## ⚖️ License

This project is distributed under the **GNU General Public License v3.0 (GPL-3.0)**.  
See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

This work was supported by:
- TU Darmstadt, Signal Processing Group  
- Universidad de Granada, Depto. de Teoría de la Señal  
- Johns Hopkins University, ECE Department

---

> _“Recovery through structure — learning to unshuffle the unobservable.”_
