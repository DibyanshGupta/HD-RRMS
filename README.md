\# HD-RRMS



Research code and experiments for high-dimensional regret-minimizing representative set selection.



\## Objective



The project studies representative subset selection using the combined objective



\\\[

F(S) = \\alpha ARR(S) + (1-\\alpha)MRR(S)

\\]



where ARR is Average Regret Ratio and MRR is Maximum Regret Ratio.



\## Repository Structure



\- `src/` — C++ algorithm implementations.

\- `scripts/preprocessing/` — Python scripts for dataset preparation and validation.

\- `data/raw/` — Original datasets.

\- `data/processed/` — Prepared datasets used in experiments.



\## Algorithms



\### Forward Greedy



\- `src/forward\_greedy/main.cpp`

\- `src/forward\_greedy/mtp\_forward\_greedy\_v2.cpp`

\- `src/forward\_greedy/mtp\_forward\_greedy\_v3.cpp`



\### K-Center with Local Search



\- `src/kcenter\_local\_search/mtp\_kcenter\_local\_search\_v2.cpp`



\## Datasets



The repository includes data for Austin Housing, Car, Census, DOT, Energy, Forest Cover, and Wine Quality. Some large raw files are excluded through `.gitignore`.



\## Preprocessing



Python preprocessing scripts are located in `scripts/preprocessing/`.



\## Status



This repository is under active development. Algorithm implementations and experimental results are being validated and improved.

