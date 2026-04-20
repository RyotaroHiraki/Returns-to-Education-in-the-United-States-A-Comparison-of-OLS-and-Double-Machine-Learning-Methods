# Returns-to-Education-in-the-United-States-A-Comparison-of-OLS-and-Double-Machine-Learning-Methods
This repository contains the data analysis code and materials for my research project titled  
**"[Returns to Education in the United States A Comparison of OLS and Double Machine Learning Methods]"**.

---

## Overview

This study examines the economic returns to education in the U.S. using 2024 CPS data and
compares Ordinary Least Squares (OLS) regression with a Double Machine Learning (DML)
framework incorporating models such as random forests, boosted trees, lasso, GAMs, and
neural networks (MLP). Results show consistent returns of 8 to 9 percent per additional year
of schooling across methods. Simulations reveal that all predictors perform well under linear
assumptions if hyperparameters are optimally adjusted, while OLS/Lasso suffer from
nonlinearity. Findings suggest that OLS remains robust in low-dimensional, near-linear
contexts, offering practical guidance for economists and policymakers balancing model
complexity and interpretability in education research.

---

## Methods

- Statistical Models: OLS, Double Machine Learning (DML) (Random Forest, XGBoost, GAMs, Neural Nets(MLP))
- Tools: R (fixest, tidyverse, ggplot2)
- Data Source: CPS 2024 cleaned and preprocessed in R

---

## Repository Structure

<pre>
project-name/
├── code/              # R scripts for data cleaning, analysis, and visualization
├── data/              # Instructions or scripts for accessing datasets
├── figures/           # Output plots, charts, and tables
├── report/            # Presentation slides
├── sessionInfo.txt    # R package and version info for reproducibility
└── README.md          # Project summary and instructions
</pre>

---

## Reports

Paper: 
- [EDRE working paper](https://edre.uark.edu/_resources/pdf/edrewp-2026-02.pdf)
- [GLO Discussion Paper](https://ideas.repec.org/p/zbw/glodps/1733.html)
- [IZA Discussion Paper](https://www.iza.org/publications/dp/18523/returns-to-education-in-the-united-states-a-comparison-of-ols-and-double-machine-learning-methods)

Slides: []

---

## Reproducibility

To reproduce this analysis:

1. Clone this repository
2. Open the `.R` files in `code/` in order:
    - `01_data_prep.R`
    - `02_DML.R`
    - `03_simulation.R`
3. Install required packages (see `sessionInfo.txt`)
4. Follow any instructions in `data/README.md` to access the dataset

---

## Author

**Ryotaro Hiraki**  
M.S. in Economic Analytics
University of Arkansas
Email: rhiraki@uark.edu  
Portfolio: 
[`Github`](https://github.com/RyotaroHiraki)

---

## Notes

> Due to licensing restrictions, raw data is not included in this repository.  
> Please refer to `data/README.md` for instructions on accessing the dataset from the official source.

