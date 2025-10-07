# 🧩 Cross-Channel Unlabeled Sensing (CCUS)

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)
![Platform](https://img.shields.io/badge/platform-Linux%20|%20macOS-lightgrey.svg)

---

**Official Python implementation** accompanying:

- 📰 *Shuffled Multi-Channel Sparse Signal Recovery*  
  **T. Koka, M.C. Tsakiris, M. Muma, B. Béjar Haro**  
  _Signal Processing_, Elsevier, 2024  
- 🎤 *Cross-Channel Unlabeled Sensing over a Union of Signal Subspaces*  
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
