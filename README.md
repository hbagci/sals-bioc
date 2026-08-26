# SALS-BioC

**A soft-sensor based adaptation learning simulator for predicting bacterial concentrations in membrane bioreactors.**

SALS-BioC is an interactive, browser-based simulator that couples a synthetic data generator with adaptive soft-sensor models to predict biological contaminant concentrations from routine physicochemical water-quality measurements. It is built for limited wastewater datasets, where a synthetic augmentation step and a knowledge-based adaptation mechanism help models generalize across biologically independent replicates and process drift.

The tool runs as a local web application dashboard, organized into three workflows:

- **Synthetic data generation** — augment collected real samples [3] with the EMCM-PS generator (an extended Markov chain model that uses a joint-state representation and replaces the transition matrix with probabilistic sampling from the empirical state distribution). Inspect quality through marginal distributions, Pearson correlation heatmaps, PCA / t-SNE projections, and a fidelity scorecard. Export generated data as CSV.
- **LSTM soft-sensor** — train and test an LSTM to predict a selected target from process variables, validate on unseen data, and download trained models.
- **Lifelong-learning adaptation** — train a source-domain model, adapt it to an unseen target replicate, and track per-batch adaptation performance, with optional Bayesian hyperparameter optimization [1, 2].

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt` (Flask, NumPy, pandas, SciPy, scikit-learn, PyTorch, bayesian-optimization)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Start the server:

```bash
python soft-sensor-based-adaptation-learning/server.py
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
| `lstm.py` | LSTM soft-sensor model |
| `lifelong_lstm.py` | Lifelong-learning adaptation model |
| `soft-sensor-based-adaptation-learning/` | Flask server and web interface |

## References

1. J. Chen, I. N'Doye, Y. Myshkevych, F. Aljehani, M. K. Monjed, T.-M. Laleg-Kirati, P.-Y. Hong, Viral particle prediction in wastewater treatment plants using nonlinear lifelong learning models, npj Clean Water 8 (2025) 1–13.
2. J. Chen, I. N'Doye, J. S. Medina, S. Shah, P.-Y. Hong, Model generalization paradigms for predicting viral particles and evaluating removal efficiencies in anaerobic membrane bioreactor plants, npj Emerging Contaminants 2 (10) (2026) 1–16.
3. M. Jumat, N. Hasan, P. Subramanian, C. Heberling, R.-R. Colwell, P.-Y. Hong, Membrane bioreactor-based wastewater treatment plant in Saudi Arabia: Reduction of viral diversity, load, and infectious capacity, Water 9 (2017) 534.

## License

This project is licensed under the [MIT License](LICENSE).
