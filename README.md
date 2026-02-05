# Universal approximation of dynamical systems by semi-autonomous neural odes and applications

**SA-NODE** (Semi-Autonomous Neural ODE) is a research project that explores Neural Ordinary Differential Equations (Neural ODEs) with a semi-autonomous architecture. The goal is to accurately approximate the behavior of dynamical systems, including standard ODE systems and transport equations using neural networks. This repository provides a PyTorch implementation of SA-NODE and compares its performance against vanilla Neural ODEs on various simulation tasks. 
The source code is for the paper: [Z. Li, K. Liu, L. Liverani, E. Zuazua. Universal Approximation of Dynamical Systems by Semiautonomous Neural ODEs and Applications. SIAM Journal on Numerical Analysis, 64(1), 2026](https://epubs.siam.org/doi/10.1137/24M1679690)

## Installation

To get started, clone this repository and install the required dependencies. We recommend using a virtual environment. The required packages are written in `requirements.txt` file.

## Usage

This repository includes several demo scripts to run experiments. Each demo can be executed directly and does not require additional configuration. The scripts will generate simulation data, train the models, and produce output metrics/plots. Use the following commands to run each experiment:

- **ODE Approximation:**
  Run `demo_ODE.py` to train a SA-NODE model on an ODE dataset. This script will simulate a predefined ODE system (by default a nonlinear non-autonomous ODE), train the SA-NODE to learn the system’s dynamics, and then evaluate on test trajectories. It will print out error metrics (e.g., max error and end-of-simulation error) and save trajectory plots. *Usage:*

  ```bash
  python demo_ODE.py
  ```

  *Outcome:* Trains a SA-NODE on a set of ODE trajectories and saves the results plots (`figures/ODE.png` and `figures/ODE_error.png`) comparing the learned trajectories to the true solution.

- **SA-NODE vs. Neural ODE Comparison on ODEs:**
  Run `demo_ODE_compare.py` to compare the performance of a **vanilla Neural ODE** vs. the **SA-NODE** on the same ODE system. This script trains two models (one vanilla Neural ODE and one SA-NODE) on identical data and prints a comparison of their errors. It also generates plots overlaying the results of both models against the ground truth. *Usage:*

  ```bash
  python demo_ODE_compare.py
  ```

  *Outcome:* Trains both models for a fixed number of epochs and prints their error statistics. It will save comparative plots (`figures/ODE_compare.png` and `figures/ODE_compare_error.png`) showing how the SA-NODE and standard Neural ODE trajectories differ and how their errors evolve over time.

- **Transport Equation Approximation:**
  Run `demo_Transport.py` to apply the SA-NODE model to a 2D transport equation scenario. This script treats the transport equation in a semi-autonomous ODE framework. It will train a SA-NODE to predict the evolution of the transport process and evaluate on a test dataset. *Usage:*

  ```bash
  python demo_Transport.py
  ```

  *Outcome:* Trains a SA-NODE on the transport equation data and computes the approximation error. It will produce plots (`figures/Transport2D_combined.png` and `figures/Transport2D_error.png`) that visualize the learned solution vs. the true solution over time, as well as the error over the duration of the transport process.

- **SA-NODE vs. Neural ODE Comparison on Transport Equations:**
  Run `demo_Transport_compare.py` to compare the performance of a **vanilla Neural ODE** vs. the **SA-NODE** on the same transport equation. This script trains two models (one vanilla Neural ODE and one SA-NODE) on identical data and prints a comparison of their errors. It also generates plots overlaying the results of both models against the ground truth. *Usage:*

  ```bash
  python demo_Transport_compare.py
  ```

  *Outcome:* Trains both models for a fixed number of epochs and prints their error statistics. It will save comparative plots (`figures/Transport2D_compare_error.png` and `figures/Transport2D_compare.png`) showing how the SA-NODE and standard Neural ODE trajectories differ and how their errors evolve over time.
  

## Citation

If you use this project in your research, please cite us using the following BibTeX entry:

```bibtex
@article {MR5026265,
    AUTHOR = {Li, Ziqian and Liu, Kang and Liverani, Lorenzo and Zuazua,
              Enrique},
     TITLE = {Universal {A}pproximation of {D}ynamical {S}ystems by
              {S}emiautonomous {N}eural {ODE}s and {A}pplications},
   JOURNAL = {SIAM J. Numer. Anal.},
  FJOURNAL = {SIAM Journal on Numerical Analysis},
    VOLUME = {64},
      YEAR = {2026},
    NUMBER = {1},
     PAGES = {193--223},
      ISSN = {0036-1429,1095-7170},
   MRCLASS = {99-06},
  MRNUMBER = {5026265},
       DOI = {10.1137/24M1679690},
       URL = {https://doi.org/10.1137/24M1679690},
}
```
