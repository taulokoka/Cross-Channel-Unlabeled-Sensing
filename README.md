# Cross-Channel Unlabeled Sensing
Title

Cross-Channel Unlabeled Sensing (CCUS)
Python implementation accompanying the papers:
	1.	T. Koka et al. (2024) – Shuffled Multi-Channel Sparse Signal Recovery, Signal Processing, Elsevier.
	2.	T. Koka et al. (2025) – Cross-Channel Unlabeled Sensing over a Union of Signal Subspaces, ICASSP 2025.

⸻

Abstract

This repository contains the reference implementation of the algorithms proposed in
Koka et al. (2024, 2025) for reconstructing shuffled multi-channel sparse signals under unknown sample-to-channel correspondences.
The framework generalizes unlabeled sensing to multi-channel settings, enabling recovery of sparse or low-rank signals even when channel assignments are corrupted.

⸻

Features
	•	Implements cross-channel unlabeled sensing (CCUS) for 2 or more channels.
	•	Provides robust MM estimators (MM_shuffled, MMrse_shuffled) for sample assignment recovery.
	•	Includes dictionary learning via convolutional sparse coding (CSC) or custom atoms.
	•	Supports synthetic simulations and real calcium-imaging data (Drosophila).
	•	End-to-end example notebooks for reproducible experiments (example.ipynb).
