# Data

This folder contains all the datasets used in the project, organized into two subdirectories:

---

## 1. `data/raw/`

Original datasets, without any preprocessing.

### `actual_concen_sfwater.xlsx`
Physicochemical measurements of water samples.

Main columns:

- `LocationID`: sample identifier  
- `BOD(mg/L)`: biochemical oxygen demand  
- `COD(mg/L)`: chemical oxygen demand  
- `TOC(mg/L)`: total organic carbon  
- `NO3(mg/L)`: nitrates  
- `TN(mg/L)`: total nitrogen  
- `TURB(NTU)`: turbidity  
- `EC`: electrical conductivity  

Each row corresponds to one water sample measured in the lab.

### `specdata_undeno_sfwater.xlsx`
UV–Vis spectral data for the same samples.

- Column `wavelength`: sample identifier (integer ID)  
- Remaining columns: absorbance/intensity values at different wavelengths  
  in the range of approximately **200–799 nm**  
  (each column is one wavelength).

---

## 2. `data/processed/`

Preprocessed versions of the previous datasets:

- physicochemical data after outlier/negative-value correction and Z-score normalization  
- spectral data after SNV normalization

These files are used as input for the statistical analysis, PCA, clustering, and classification steps.
