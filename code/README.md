# Code Documentation

Implementation of the Federated Learning Framework for Smart Healthcare Diagnostics.

## Files

| File | Description |
|---|---|
| `FL_Healthcare_RealData.ipynb` | Full pipeline: data loading, hospital split, FedAvg, personalization, Differential Privacy, evaluation, plots |

## Requirements

Python 3.x with `torch`, `numpy`, `pandas`, `scikit-learn` and `matplotlib`. Google Colab already includes all of these. For local runs:

```bash
pip install -r ../requirements.txt
```

## How to Run

**Google Colab (recommended)**
1. Upload the notebook (File → Upload notebook).
2. Select Runtime → Run all.
3. The dataset is fetched automatically with `wget`. No Kaggle login is needed.

**Local Jupyter**
1. Install the requirements.
2. The data-loading cell uses `!wget`, which works on Linux/macOS. On Windows, download `training_data.csv` and `test_data.csv` manually into the working folder and remove the `!wget` lines.
3. Run all cells in order.

## Notebook Walkthrough

| Section | What it does |
|---|---|
| 1. Setup | Imports libraries and fixes random seeds (`SEED = 42`) |
| 2. Load Dataset | Downloads and merges the training and test CSVs |
| 3. Hospital Split | Partitions patients into 4 specialty hospitals with an 80/20 stratified train/test split |
| 4. Model | Feedforward network: 132 → 128 → 64 → 41 |
| 5. Local Training | Adam optimizer, cross-entropy loss, optional Differential Privacy |
| 6. FedAvg | Averages hospital models, weighted by dataset size |
| 7. Personalization | Each hospital fine-tunes the global model on its own data |
| 8. Experiment | Runs the full pipeline with and without Differential Privacy |
| 9. Visualization | Bar charts of global-only vs personalized accuracy |

## Configuration

| Parameter | Default | Meaning |
|---|---|---|
| `rounds` | 20 | Federated communication rounds |
| `local_epochs` | 5 | Local epochs per round |
| personalization `epochs` | 30 | Fine-tuning epochs per hospital |
| `lr` | 0.001 | Adam learning rate |
| `clip_norm` | 1.0 | Gradient clipping threshold (DP) |
| `noise_multiplier` | 0.3 | Gaussian noise scale (DP) |

**Privacy-utility experiment:** increase `noise_multiplier` (for example 0.1, 0.3, 0.6, 1.0) and re-run the experiment cell to see accuracy fall as privacy gets stronger.

## Differential Privacy Implementation

When `dp_enabled=True`, each local training step:
1. Clips the gradient norm to `clip_norm`.
2. Adds Gaussian noise with standard deviation `noise_multiplier × clip_norm` to the gradients.
3. Applies the optimizer step.

> This is a simplified DP-SGD style mechanism, chosen for transparency. It does not compute a formal privacy budget (ε). For rigorous guarantees, use Opacus.

## Using a Different Dataset

The pipeline needs:
- A shared feature schema across all hospitals (same columns)
- A label column (diagnosis)
- A mapping from labels to hospitals (`SPECIALTY_MAP`)

Replace the data-loading cell and update `SPECIALTY_MAP` and `SYMPTOM_COLS`. The model, FedAvg, personalization, DP and evaluation code work unchanged.

## Reproducibility

Seeds are fixed for Python, NumPy and PyTorch. Small numeric differences can still occur across hardware and library versions.

## Known Limitations

- Hospitals are simulated from one dataset, not real institutions.
- Only accuracy is reported (precision, recall, F1 and ROC-AUC are planned).
- Training runs in a single process and simulates the federated setting rather than using a real network.

## Troubleshooting

| Problem | Fix |
|---|---|
| `wget` fails | Check internet access, or download the CSVs manually and upload them to Colab |
| `KeyError: 'Unnamed: 133'` | Run the data-loading cell fully; the notebook already drops this empty column |
| Different accuracy values | Re-run all cells from the top so the seeds reset |
