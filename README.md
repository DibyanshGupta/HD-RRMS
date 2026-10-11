# HD-RRMS

Research code and experiments for **high-dimensional regret-minimizing representative set selection**.

## Objective

The project studies representative subset selection using the combined objective:

**F(S) = α · ARR(S) + (1 − α) · MRR(S)**

where:
- **ARR** is Average Regret Ratio.
- **MRR** is Maximum Regret Ratio.
- **α** controls the relative importance of ARR and MRR.

## Repository Structure

- `src/` — C++ algorithm implementations.
- `scripts/preprocessing/` — Python scripts for dataset preparation and validation.
- `data/raw/` — Original datasets.
- `data/processed/` — Prepared datasets used in experiments.

## Algorithms

### Forward Greedy

- `src/forward_greedy/main.cpp`
- `src/forward_greedy/mtp_forward_greedy_v2.cpp`
- `src/forward_greedy/mtp_forward_greedy_v3.cpp`

### K-Center with Local Search

- `src/kcenter_local_search/mtp_kcenter_local_search_v2.cpp`

## Datasets

The repository contains datasets for Austin Housing, Car, Census, DOT, Energy, Forest Cover, and Wine Quality. Some large raw dataset files are excluded through `.gitignore`.

## Preprocessing

Python preprocessing scripts are located in `scripts/preprocessing/`.

## Status

This repository is under active development. Algorithm implementations and experimental results are being validated and improved.