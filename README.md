# CSEN3011 ANN Project 28 - Predictive-Maintenance Feature Learning with Autoencoder

- Dataset : AI4I 2020 Predictive Maintenance (UCI #601)
  https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- Architecture : Autoencoder + MLP
- Training technique : Mini-Batch Gradient Descent (backpropagation, written from scratch in NumPy)

## Contents of this zip
| File | Purpose |
|---|---|
| project28_ai4i_autoencoder_mlp.py | the whole project (preprocessing, models, experiments, plots, tables) |
| requirements.txt | Python libraries needed (numpy, pandas, scikit-learn, matplotlib) |
| run.bat / run.sh | one-click run for Windows / Linux-macOS (creates a virtual environment) |
| README.md | this file |

The dataset itself is NOT included (it must come from the original UCI source).
The script downloads it automatically; if that fails, see step 2 below.

## Steps
1. Install Python 3.9 or newer.
2. (Only if auto-download fails) Open the UCI link above, download the zip,
   extract `ai4i2020.csv` and place it in this folder.
3. Run:
   - Windows: double-click `run.bat`  (or: `pip install -r requirements.txt` then `python project28_ai4i_autoencoder_mlp.py`)
   - Linux/macOS: `bash run.sh`
   - Google Colab: upload the .py file (and the csv), then run
     `!pip install -q -r requirements.txt` and `!python project28_ai4i_autoencoder_mlp.py --csv ai4i2020.csv`
4. A quick test run (few epochs, 1 seed): add `--quick`.
   Options: `--seeds 5` (more random splits), `--csv path`, `--outdir name`.
5. Results appear in `results_project28/`:
   - figures/ : fig01 dataset overview, fig02 architecture diagram, fig03 loss curves,
     fig04 accuracy curves, fig05 confusion matrices, fig06 ROC/PR curves,
     fig07 latent-dimension study, fig08 learning-rate & batch-size curves,
     fig09 reconstruction-error distribution + latent space
   - tables/  : summary_E1..E7.csv (mean ± std), results_all_runs.csv, parameter_counts.csv,
     confusion_matrices_seed0.csv, dataset_sample.csv, feature_statistics.csv
   - results_summary.md, preprocessing_audit.json, run_info.json (seeds, versions, settings)

## Mapping to the report sections
Dataset -> fig01, dataset_sample.csv | Preprocessing -> preprocessing_audit.json |
Neural Architecture -> fig02, parameter_counts.csv | Optimization/Training -> run_info.json |
Experiments -> E1..E7 | Results -> fig03-fig09 + summary tables | Reproducibility -> run_info.json

## Important
- Report only numbers produced by YOUR run on the real dataset. Do not use the `--synthetic`
  flag for the report (it only tests that the code runs and writes to a separate folder).
- Report negative results honestly (e.g. if the autoencoder features do not beat the raw baseline).
- The logistic-regression row in E1 is only a yardstick, not the main model.
