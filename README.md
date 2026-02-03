# Slow Wave Generation a Propagation in a Model of Brain lesions

A computational framework for simulating sleep-like neuronal dynamics following brain injury using an extended Jansen-Rit neural mass model with spike-frequency adaptation.

[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)]()
[![TVB](https://img.shields.io/badge/TheVirtualBrain-2.7%2B-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

## Overview

This repository implements a multiscale neural mass model for studying the generation and propagation of slow waves (SWs) in brain networks with virtual lesions. The model combines the classical Jansen-Rit model with activity-dependent adaptation to capture transitions between wake-like and sleep-like cortical dynamics.

![Figure 1: Jansen-Rit Model with Spike-Frequency Adaptation](./figures/fig. 1 (Methods).png)

## Repository Contents

- `JR_SFA.py` - TVB implementation of the Jansen-Rit model with adaptation
- `WC_SFA.py` - TVB implementation of the Wilson-Cowan model with adaptation
- `JR_SFA.ode` - XPPAUT file of the Jansen-Rit model with adaptation
- `JR_SFA_demo.ipynb` - Multi-scale simulation showcase (isolated population → whole-brain)

## Installation

```bash
git clone https://github.com/gianlucagag/SWs-lesion_model.git
cd SWs-lesion_model

pip install -r requirements.txt
```

**Dependencies**: tvb-library, numpy, matplotlib, pandas, jupyter


## Usage

The best way to get started is through our Jupyter notebook:

```bash
jupyter notebook Jansen-Rit_SFA_demo.ipynb
```

## Model Description

The extended Jansen-Rit model includes three neuronal pools (pyramidal, excitatory interneurons, inhibitory interneurons) with an adaptation mechanism:

**dω/dt = k_ω [ Sigm( a₁ J [y₀ - g ω] ) - ω ]**

where ω is the adaptation variable and g controls adaptation strength (g=0 recovers classical JR).

Key features:
- **Up/Down state transitions**: Sleep-like cortical bistability
- **Four simulation scales**: Isolated populations → two coupled populations → toy networks → whole-brain

## Citation

```bibtex
@article{Gaglioti2026,
  title={Slow wave generation and propagation in a model of brain lesions},
  author={Gaglioti, G. and Dalla Porta, L. and Colombo, M. A. and others},
  year={2026}
}
```

## Contact

Gianluca Gaglioti - gianluca.gaglioti@unimi.it  
Marcello Massimini - marcello.massimini@unimi.it

## License

MIT License
