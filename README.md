# SALS-BioC

**A soft-sensor based adaptation learning simulator for predicting bacterial concentrations in membrane bioreactors.**

SALS-BioC is an interactive, browser-based simulator that couples a synthetic data generator with adaptive soft-sensor models to predict biological contaminant concentrations from routine physicochemical water-quality measurements. It is built for limited wastewater datasets, where a synthetic augmentation step and a knowledge-based adaptation mechanism help models generalize across biologically independent replicates and process drift.

The tool runs as a local web application with a side-by-side comparison dashboard, organized into three workflows:

- **Synthetic data generation** — augment a dataset with the EMCM-PS generator (a Markov-chain / Gaussian-copula model using a joint-state representation and probabilistic sampling). Inspect quality through marginal distributions, Pearson correlation heatmaps, PCA / t-SNE projections, and a fidelity scorecard. Export generated data as CSV.
- **LSTM soft-sensor** — train and test an LSTM to predict a selected target from process variables, validate on unseen data, and download trained models.
- **Lifelong-learning adaptation** — train a source-domain model, adapt it to an unseen target replicate, and track per-batch adaptation performance, with optional Bayesian hyperparameter optimization.

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt` (Flask, NumPy, pandas, SciPy, scikit-learn, copulas, PyTorch, bayesian-optimization)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Start the server:

```bash
python synthetic-data-comparison/server.py
```

(or run `run.bat` on Windows), then open:

```
http://localhost:5000
```

Upload a numeric CSV, select the target column, adjust parameters, and run the generation, training, and adaptation steps from the dashboard.

## Project layout

| Path | Description |
|------|-------------|
| `emcm_ps.py` | EMCM-PS synthetic data generator |
| `emcm_gcrw.py` | Copula-based Markov generator |
| `lstm.py` | LSTM soft-sensor model |
| `lifelong_lstm.py` | Lifelong-learning adaptation model |
| `synthetic-data-comparison/` | Flask server and web interface |

## License

This project is licensed under the [MIT License](LICENSE).
