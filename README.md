# Dependency-aware Schrödinger Bridge for Predictive Statistical Matching

Code accompanying the NeurIPS 2026 paper **Dependency-aware Schrödinger Bridge for Predictive Statistical Matching**.

The experiments address statistical matching from two incomplete datasets: A contains `(X, Y)`, and B contains `(X, Z)`. The proposed Schrödinger Bridge (SB) method estimates the missing variables and evaluates downstream prediction.

## Files

- `gaussian.py`, `syn1.py`, `syn2.py`, `syn3.py`: Gaussian and synthetic regression experiments (SB and comparison methods).
- `adult.py`, `celeba.py`: Classification experiments (SB).
- `utils.py`: Shared functions used by the experiment scripts.

## Environment

The code was developed with Python 3.12.7 and PyTorch 2.6.0+cu124 (CUDA 12.4, NVIDIA L4).

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

For GPU use, install a PyTorch build compatible with your system. You can also run on CPU, although training will be slower.

## Data

Gaussian and SYN1–SYN3 generate their data internally and need no downloads.

**Adult:** Download the Adult dataset from the UCI Machine Learning Repository and prepare these three header-containing, CSV-formatted files:

```text
data/adultX.txt
data/adultY.txt
data/adultZ.txt
```

`adultX.txt` contains `age`, `sex`, `race`, `native-country`, `marital-status`, and `relationship`. `adultY.txt` contains the `income` column (positive class: `>50K`). `adultZ.txt` contains `education`, `education-num`, `occupation`, `workclass`, `hours-per-week`, `capital-gain`, and `capital-loss`.

**CelebA:** Download CelebA from its official source and prepare these aligned arrays:

```text
data/X_embeddings_8d.npy  # (N, 8): 8-dimensional CAE image features
data/Y_data.npy          # (N, 1): Attractive
data/Z_data.npy          # (N, 5): five auxiliary attributes
```

The five auxiliary attributes are `Young`, `High_Cheekbones`, `Arched_Eyebrows`, `Heavy_Makeup`, and `No_Beard`. The 8-dimensional features are obtained from a convolutional autoencoder (CAE). All three files must have the same sample order. Original datasets and processed data files are not distributed in this repository; users should prepare them separately in `data/`.

## Run

Run the scripts from the repository root:

```bash
python gaussian.py
python syn1.py
python syn2.py
python syn3.py
python adult.py
python celeba.py
```

Each script uses `SEED = 42` and one fixed SB regularization parameter (`SB_LAMBDA`), which can be changed at the top of the file. Set `MY_GPU_ID` to an available physical GPU index, if needed. Results are printed to the console; the scripts do not save CSV results.

The console labels **Strong** and **Weak** refer to evaluation on the larger held-out test set `T` and the 50-sample complete set `C`, respectively. These fixed-parameter runs are not the full multi-seed or Strong/Weak lambda-selection procedure reported in the paper.
