# FBSNet

Official PyTorch implementation of **FBSNet** for remote sensing semantic segmentation.

> **FBSNet: Re-thinking Directional Structure Modeling in Remote Sensing Segmentation via Frequency-Band State Space Networks**

---

## Highlights

- **FBSS**: Frequency-Band State Space module for modeling structured cross-band dependencies from low-frequency context to high-frequency directional details.
- **HSM**: Hierarchical Spatial Mamba block for enhancing large-area geometric continuity and cross-scale spatial consistency.
- **FGRM**: Fourier-Guided Refinement Module for preserving boundary fidelity during progressive decoding.

---

## Framework

<p align="center">
  <img src="assets/framework.pdf" width="900"/>
</p>


---

## Overview

Remote sensing images contain strong directional structures and multi-frequency cues.  
FBSNet is designed to jointly model:

- **frequency-band dependencies** for directional detail preservation,
- **spatial continuity** for large-area structural consistency,
- **Fourier-guided refinement** for sharper boundary recovery.

The proposed network adopts a **ConvNeXt-based encoder-decoder framework** and introduces three key modules:

- **FBSS**: Frequency-Band State Space
- **HSM**: Hierarchical Spatial Mamba
- **FGRM**: Fourier-Guided Refinement Module

---

## Main Modules

### 1. Frequency-Band State Space (FBSS)

FBSS first performs **Learnable Frequency-Band Decomposition (LFBD)** to generate four subbands:

- `LL`
- `LH`
- `HL`
- `HH`

Then, band-specific semantic-aligned serialization and state space modeling are used to capture ordered cross-band dependencies.

<p align="center">
  <img src="assets/fbss.pdf" width="850"/>
</p>


---

### 2. Hierarchical Spatial Mamba (HSM)

HSM complements the frequency branch by enhancing spatial structural continuity through:

- region-level propagation
- pooled global propagation

<p align="center">
  <img src="assets/hsm.pdf" width="700"/>
</p>


---

### 3. Fourier-Guided Refinement Module (FGRM)

FGRM uses Fourier-domain guidance during decoding to alleviate high-frequency attenuation and improve boundary delineation.

<p align="center">
  <img src="assets/fgrm.pdf" width="700"/>
</p>


---

## Repository Structure

```text
FBSNet/
├── README.md
├── config.py
├── models/
│   └── convnext.py
├── modules/
│   ├── fbss.py
│   ├── hsm.py
│   └── fgrm.py
├── assets/
│   ├── framework.png
│   ├── fbss.png
│   ├── hsm.png
│   └── fgrm.png

