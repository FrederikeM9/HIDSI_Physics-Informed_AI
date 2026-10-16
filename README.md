# HIDSI Workshop: Physics-Informed AI for Environmental Data Science

**Friday, October 16, 2026**

Presenter: Frederike Monte, PhD candidate, Atmospheric Sciences, University of Hawaiʻi at Mānoa · fmonte@hawaii.edu

## Overview

Physics-informed AI combines machine learning with known physical laws. Instead of learning purely from data, the model is also asked to respect physics — for example through an extra term in the loss function. This can make models more physically consistent and more reliable outside the conditions they were trained on.

This repository contains the hands-on notebook for the workshop: we train the same simple neural network with and without a physical constraint, and compare when and why the constraint improves its predictions.

## Example task

Predict **specific humidity** *q* from temperature *T*, pressure *p* and relative humidity *RH* — a simple **emulator** for a calculation we already know. Because we know the true answer, we can measure exactly what physics constraints add.

The model is trained on cooler conditions (*T* < 20 °C) and tested on warmer conditions it has never seen, to test **extrapolation**.

| Model | Physics in the loss |
|---|---|
| Model 1 | none (data only) |
| Model 2 | Clausius–Clapeyron: *q* grows ~6–7 % per K at fixed *RH* and *p* |
| Constraint ladder | A: positivity (*q* ≥ 0) + monotonicity (*q* increases with *T*) · B: A + convexity · C: Clausius–Clapeyron |

## Quick start

**Option 1 — Google Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FrederikeM9/HIDSI_Physics-Informed_AI/blob/main/physics_informed_humidity.ipynb)

Click the badge, then run the first cell. You need a Google account. The data downloads automatically.
If Colab warns that the notebook was not authored by Google, click *Run anyway*.

**Option 2 — Local Jupyter**

```bash
git clone https://github.com/FrederikeM9/HIDSI_Physics-Informed_AI.git
cd HIDSI_Physics-Informed_AI
pip install -r requirements.txt
jupyter notebook physics_informed_humidity.ipynb
```

No GPU or HPC resources needed — everything runs on a laptop CPU in a few minutes.

**Before the workshop:** please open the notebook and run the first cell ("Setup check") to make sure everything works.

## Repository contents

| File | Description |
|---|---|
| `physics_informed_humidity.ipynb` | Workshop notebook (start here) |
| `physics_informed_humidity_with_outputs.ipynb` | Same notebook with all outputs (as backup) |
| `HIDSI_workshop_slides.pdf` | Workshop slides |
| `requirements.txt` | Python packages for local use |
| `data/jena_climate_2009_2016.csv.zip` | Backup copy of the dataset (the notebook downloads it automatically) |

## Prerequisites

- Basic Python (NumPy, pandas, Matplotlib)
- Basic idea of what a neural network is
- Tools: Python, Jupyter notebooks, NumPy, pandas, Matplotlib, PyTorch

## Data

**Jena Climate dataset** — weather station of the Max Planck Institute for Biogeochemistry, Jena, Germany; 2009–2016, 10-minute resolution.
Source: https://www.bgc-jena.mpg.de/wetter/ — all credit for the data goes to MPI-BGC.
Also available on Kaggle: https://www.kaggle.com/datasets/mnassrib/jena-climate

If the automatic download fails, download the backup from `data/` and place it next to the notebook.

## Further reading

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045
- Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. Nature Reviews Physics, 3, 422–440. https://doi.org/10.1038/s42254-021-00314-5
- Kashinath, K., Mustafa, M., Albert, A., Wu, J.-L., Jiang, C., Esmaeilzadeh, S., Azizzadenesheli, K., Wang, R., Chattopadhyay, A., Singh, A., Manepalli, A., Chirila, D., Yu, R., Walters, R., White, B., Xiao, H., Tchelepi, H. A., Marcus, P., Anandkumar, A., Hassanzadeh, P., & Prabhat (2021). Physics-informed machine learning: case studies for weather and climate modelling. Phil. Trans. R. Soc. A, 379(2194), 20200093. https://doi.org/10.1098/rsta.2020.0093
- Krishnapriyan, A. S., Gholami, A., Zhe, S., Kirby, R. M., & Mahoney, M. W. (2021). Characterizing possible failure modes in physics-informed neural networks. Advances in Neural Information Processing Systems (NeurIPS), 34, 26548–26560.
