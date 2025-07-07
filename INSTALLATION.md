# Installation Guide

This project is best run in a clean, dedicated Python environment. We recommend using **Anaconda** or **Miniconda** for robust package management and reproducibility.

## Step 1 – Install Conda (if needed)

If you haven't already, install one of the following:

- **Miniconda** (lightweight): https://docs.conda.io/en/latest/miniconda.html
- **Anaconda** (full distribution): https://www.anaconda.com/products/distribution

## Step 2 – Create and activate the environment

```bash
conda create -n dualmomentum python=3.10
conda activate dualmomentum
```

## Step 3 – Install dependencies with Conda (preferred)

```bash
conda install pandas
```

```bash
conda install anaconda::numpy
```

```bash
conda install conda-forge::seaborn
```

```bash
conda install conda-forge::matplotlib
```

```bash
conda install conda-forge::tqdm
```

This ensures that binary dependencies are installed securely using conda’s solver.

## Optional – Install with pip (if not using conda)

```bash
pip install pandas numpy matplotlib seaborn tqdm
```

If using pip, it's strongly recommended to still isolate your environment using:

```bash
python -m venv env
source env/bin/activate  # On Windows use: .\env\Scripts\activate
```

## Environment Check

Once installed, verify the environment works:

```bash
python -c "import pandas, numpy, matplotlib, seaborn, tqdm; print('All packages imported successfully.')"
```

You're now ready to run the project. Use the main script:

```bash
python dual_momentum_strategy.py
```
