# Predictive QoS under Concept Drift

A research-oriented Python simulator for **predictive Quality of Service (PQoS) in vehicular 5G networks**, with an emphasis on **federated learning (FL)**, **concept drift**, and the trade-off between prediction quality and computational/communication cost.

The simulation models vehicles as learning clients and a cloud server as the coordinator. It processes time-series network and mobility measurements, trains neural-network predictors, evaluates predictions over successive simulation rounds, and estimates the associated data-transfer and energy costs. In addition to continuous and one-shot training, the code includes drift-aware retraining and alternative FL control strategies.

This repository accompanies research on **DareFL**, a drift-aware, resource-efficient FL approach for automotive PQoS, and subsequent comparative experimentation. The simulation code is distinct from the OMNeT++/Simu5G/SUMO environment used to *generate* the datasets: it consumes pre-generated CSV traces.

## Research background

- **Published paper:** G. Drainakis *et al.*, “Distributed Predictive QoS in Automotive Environments under Concept Drift,” *IFIP Networking*, 2024. [Read the paper](https://cgi.di.uoa.gr/~ppantaz/publications/2024_Networking_pQoS.pdf).
- **Follow-up manuscript (work in progress):** G. Drainakis *et al.*, “Benchmarking Concept Drift-Aware Federated Learning for Automotive Predictive QoS.” This comparative work considers DareFL alongside other always-on, personalized, and threshold-based FL approaches under common experimental conditions. The manuscript is not cited here as a published article.
- **Public datasets:** [Zenodo record 11084689](https://zenodo.org/records/11084689), DOI [10.5281/zenodo.11084689](https://doi.org/10.5281/zenodo.11084689).

## Features

- Round-based client selection, training, validation, and inference over vehicular time-series traces.
- Centralized and federated training workflows, including variants with periodic training, drift monitoring, and convergence-based stopping.
- LSTM-based time-series prediction, configurable input features, history window, and prediction horizon; additional architectures and dataset loaders remain in the source tree.
- Prediction metrics including RMSE, normalized RMSE, MAE, SMAPE, and R², with a simple prediction baseline.
- Model aggregation, resource accounting for client/cloud processing and uplink/downlink transfers, and per-round CSV logging.
- Experiments comparing learning strategies under changing network and mobility conditions.

## Repository layout

| Path | Purpose |
| --- | --- |
| `run_simulation.py` | Entry point defining experiment configurations, repetitions, and selected learning strategies. |
| `simulation.py` | Round-based simulation orchestration and results logging. |
| `worker.py` | Client and server operations: training, inference, model exchange, aggregation, evaluation, and resource accounting. |
| `neural_ntw.py` | PyTorch neural network architectures, including LSTM implementations. |
| `dataset.py` | CSV loading, per-client dataset construction, normalization, and sliding-window preparation. |
| `core.py` | Network-path timing and energy accounting. |
| `gparams.py` | Global configuration, numerical assumptions, file paths, and experiment parameters. |
| `requirements.txt` | Original environment package export (legacy Windows/Conda format). |
| `datasets/` | Local CSV input traces; intentionally excluded from Git. |
| `logs/` | Generated run logs and CSV result summaries; intentionally excluded from Git. |
| `ml_models/` | Generated PyTorch model checkpoints; intentionally excluded from Git. |

The three data/output directories contain `.gitkeep` placeholders so their structure is visible after cloning. Their contents are ignored by `.gitignore`. Auxiliary outputs (`plots/`, `ai_sim/`, `omnet_sim/`, `plot_quanty`, and `ddm_dbg.txt`) are also ignored if created locally. Python caches, virtual environments, and local `.env` files are excluded.

## Download the datasets

The simulation reads CSV files from `datasets/`. The public [Zenodo dataset](https://zenodo.org/records/11084689) contains:

| Dataset | Description | Approximate size |
| --- | --- | ---: |
| `pqos_drift_sc1.csv` | Infrastructure-related drift: two of four 5G base stations are deactivated. | 354 MB |
| `pqos_drift_sc2.csv` | Mobility-related drift: vehicle routes shift to create a traffic hotspot. | 352 MB |

The original data was produced using **Simu5G/OMNeT++**, **SUMO**, and **OpenStreetMap**. It represents **25 vehicles**, **four 5G base stations**, and **20 hours** of simulated activity, sampled at **1 Hz**. Concept drift is introduced at approximately **10 hours**. The datasets are distributed separately under **CC BY 4.0**; refer to the Zenodo record for attribution and license details.

Download either or both CSV files and place them directly in `datasets/`:

```text
datasets/
├── .gitkeep
├── pqos_drift_sc1.csv
└── pqos_drift_sc2.csv
```

**Important:** The historical default in `gparams.py` is `asym_4bsod4_full_cl26_1hz.csv`, which is **not** one of the current public Zenodo download filenames. To use the published traces, update the `gparam['Name']['CurrDataset']` entry to `pqos_drift_sc1.csv` or `pqos_drift_sc2.csv` before running. This value is deliberately not changed automatically to preserve the existing experimental setup.

## Environment and installation

This is research code with a **legacy dependency snapshot**, not a tested modern cross-platform package. The uploaded `requirements.txt` is a Conda-style environment export (with package build strings for Windows and Python 3.8), **not** a conventional `pip install -r requirements.txt` specification.

The historical environment includes Python 3.8, PyTorch 1.12.1 (CPU), NumPy 1.23.2, pandas 1.4.4, scikit-learn 1.1.2, scikit-multiflow 0.5.3, SciPy 1.9.1, matplotlib 3.5.3, and other packages listed in `requirements.txt`. These versions are provided for reference and may require adjustment to reconstruct the original environment today. In particular, the legacy export mixes Conda-style packages and pip-origin packages, so creating a working environment may require handling dependencies individually.

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Set up a compatible Python environment, install the packages needed by the source modules, and download the traces as above. No dependency versions have been updated as part of this code cleanup.

## Running experiments

1. Place a supported CSV trace in `datasets/` and set `gparam['Name']['CurrDataset']` in `gparams.py` to its filename.
2. Review the experiment settings in `run_simulation.py` (learning strategies, client counts, repetitions, time range, model settings) and the defaults in `gparams.py`.
3. Run from the repository root:

   ```bash
   python run_simulation.py
   ```

**Before running:** The supplied `run_simulation.py` currently iterates over `range(1, 10000)` and tests `FL_ddm_all`, `FL_oneshot_with_conv_stop`, and `FL_forever`, so it is configured for a very large experiment batch. Reduce the repetition range **yourself** for a quick smoke test. The script also overrides many defaults from `gparams.py` on each run.

Generated logs are written under `logs/`; model checkpoints are written to `ml_models/`. Additional debugging output may appear in the repository root. These runtime artifacts are not tracked by Git.

### Main experiment controls

- `gparams.py`: dataset path, client pool size, model characteristics, prediction horizon, resource consumption assumptions, drift parameters, and output directories.
- `run_simulation.py`: batch selection, repetition count, dataset timing, selected learning modes, and per-experiment hyperparameters.
- `simulation.py`: how each learning mode is executed round by round.

The code is provided as an experimental baseline. Results depend on the selected dataset, seeds/client selection, hyperparameters, and environment. The script has **not** been claimed to reproduce every table or figure in either paper without additional configuration.

## Research use and citation

If you use the dataset, cite its [Zenodo DOI](https://doi.org/10.5281/zenodo.11084689). For the original DareFL methodology, cite the [IFIP Networking 2024 paper](https://cgi.di.uoa.gr/~ppantaz/publications/2024_Networking_pQoS.pdf). The newer benchmarking manuscript is still a work in progress and should not be represented as published.

## License

The source repository includes the **Apache License 2.0** (`LICENSE`). The externally hosted Zenodo datasets have their own **CC BY 4.0** license.

## Maintenance notes

This release focuses on repository hygiene and documentation. Existing algorithms, numeric assumptions, defaults, control flow, and experiment behavior have intentionally not been altered. In particular, compatibility work, algorithmic corrections, dependency upgrades, and runtime validation against the full datasets are outside the scope of this cleanup.
