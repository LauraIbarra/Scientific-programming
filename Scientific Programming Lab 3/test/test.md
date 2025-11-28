# test/

This folder contains the minimal testing environment for verifying that the project’s modules run correctly as an integrated pipeline.  
It is designed to validate imports, directory structure, preprocessing routines, visualizations, and the complete analysis workflow.

The folder includes the following elements:

---

## 1. `__init__.py`

Marks the directory as a Python package, enabling test modules to import components from the main `src/` folder using package-relative paths.

This file does not contain logic itself, but it is essential for ensuring:

- correct module discovery  
- proper Python package initialization  
- compatibility with automated testing tools  

---

## 2. `testcod.py`

This script serves as the **main testing file**.  
It loads raw data, runs preprocessing, generates plots, performs PCA, computes features, evaluates clustering, runs ANOVA/Fisher tests, and finishes with classification.

### The script performs:

- Creation of output directories  
- Loading physicochemical and spectral raw datasets  
- Raw visualization  
- Data preprocessing (negatives, outliers, Z-score, SNV)  
- Processed visualizations  
- PCA computation  
- Feature extraction  
- Correlation analysis  
- Domain distribution and index computation  
- Clustering evaluation  
- ANOVA and Fisher Ratio  
- Classification metrics  
- Saving all results into `results/figures/` and `results/tables/`

This file essentially **executes the whole project pipeline**, acting as the validation point that ensures everything inside `src/` works together correctly.

---

## 3. `__pycache__/`

Automatically generated folder containing bytecode-compiled versions of the test scripts.  
It improves performance but does not need to be edited or documented.

---

## Summary

The `test/` directory provides a lightweight yet complete environment for validating the project’s functionality.  
It ensures that preprocessing, visualization, analytical modules, and result-saving mechanisms operate properly as a unified workflow.
