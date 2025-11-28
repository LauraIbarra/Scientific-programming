# src/

This folder contains the full source code of the project. It is organized into modular components that handle preprocessing, analysis, and visualization of physicochemical and spectral datasets. Together, these modules form the project’s processing pipeline—from raw data to statistical conclusions.

The folder is organized into three main subpackages:

---

## 1. `preprocessing/`

This directory includes the tools responsible for cleaning and transforming raw datasets before any analytical procedure is applied. These preprocessing steps ensure the reliability, comparability, and consistency of the physicochemical and spectral measurements.

### The folder contains:

- **physicochemical.py**  
  Module dedicated to preparing physicochemical variables such as BOD, COD, TOC, NO₃, TN, TURB, and EC.  
  It includes the `PhysicochemicalProcessor` class, which performs:
  - Negative value correction  
  - Outlier removal using IQR  
  - Z-score normalization  

  These operations create a stable foundation for correlation analysis, PCA integration, and clustering.

- **spectral.py**  
  Module that preprocesses raw UV–Vis spectral observations.  
  It implements the `SpectralProcessor` class, responsible for:
  - Removing non-numeric metadata  
  - Applying Standard Normal Variate (SNV) normalization  

  The resulting processed spectral matrix is ready for PCA, entropy extraction, and multivariate analysis.

---

## 2. `analysis/`

This directory contains all mathematical and statistical algorithms used to extract patterns, reduce dimensionality, and evaluate relationships across domains. It forms the core analytical engine of the project.

### The folder contains:

- **analysis.py**  
  Implements the `AnalysisProcessor` class, which provides a complete analytical pipeline including:
  - PCA  
  - Merging physicochemical data with PCA components  
  - Feature extraction (mean, std, rms, spectral energy, mean power, spectral entropy)  
  - Correlation matrix computation  
  - Domain distribution analysis  
  - PhysIndex vs SpecIndex computation  
  - K-Means clustering evaluation  
  - ANOVA and Fisher Ratio statistical comparison  
  - Classification evaluation (Baseline, Logistic Regression, kNN)  

  This module reproduces the analytical logic developed in the project's notebooks.

---

## 3. `visualization/`

This directory includes all plotting utilities used to generate figures for the project. Each visualizer produces outputs saved in `results/figures/`.

### The folder contains:

- **physicochemicalViz.py**  
  Generates visual summaries of physicochemical data, including:
  - Histograms of raw variables  
  - Boxplots of processed data  

- **spectralViz.py**  
  Produces plots for UV–Vis spectral datasets, including:
  - Raw absorbance curves  
  - SNV-normalized curves for selected samples  

- **analysisViz.py**  
  Creates analysis-level visual outputs:
  - Correlation heatmaps  
  - Domain distribution violin and boxplots  
  - PhysIndex vs SpecIndex scatter plots  
  - Clustering metric curves  
  - PCA scatterplots colored by cluster  

---

## Summary

The `src/` folder contains the complete implementation of the project’s computational pipeline, covering:

- Data cleaning  
- Normalization  
- Feature extraction  
- Dimensionality reduction  
- Clustering  
- Statistical testing  
- Classification  
- Visualization  

Each subfolder contributes a specific layer of functionality, and together they produce a coherent, fully automated analysis framework.
