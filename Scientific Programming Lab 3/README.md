# Water Quality Assessment Using Physicochemical Variables and UV–Vis Spectral Analysis
Scientific Programming – Final Project

---

## 1. Context

According to the FAO, agriculture is the main source of water pollution due to the use of fertilizers, pesticides, and herbicides in cultivation systems. When these substances are not absorbed by plants or retained in the soil, they end up being washed away through agricultural runoff and drainage into surface and groundwater. This introduces compounds such as nitrogen, phosphorus, potassium, ammonium, organic matter, heavy metals including cadmium, lead, mercury, arsenic, chromium, and nickel, and pathogens, particularly Escherichia coli, Salmonella, helminths, viruses, and protozoa (FAO, 2021).

To address this problem, water quality indicators are used. These include physical, chemical, and biological variables that describe the state of the water and estimate risk for agricultural use and human consumption. Measuring these parameters helps detect organic matter, nutrients, sediments, and dissolved ions. Examples include BOD, COD, TOC, nitrates, turbidity, and electrical conductivity (Navarro et al., 2024; Abidin et al., 2024).

However, physicochemical measurement alone does not always precisely identify which contaminants are present or their exact concentration, especially at low levels. For this reason, UV–Vis spectrometry is employed. Contaminants absorb light at specific wavelengths, producing a spectral signature. Integrating physicochemical and spectral analysis creates a more robust monitoring tool (Elsayed et al., 2020; Goblirsch et al., 2023).

---

## 2. Dataset Description

The dataset used comes from the scientific study “Spectral Water Quality Data” (Jiang & Tang, 2022). It includes 29 surface water samples collected along the Maozhou River in Shenzhen, China.

### Physicochemical Variables (7)
- BOD (mg/L)
- COD (mg/L)
- TOC (mg/L)
- NO₃ (mg/L)
- TN (mg/L)
- TURB (NTU)
- EC (µS/cm)

### Spectral Variables
- UV–Vis absorbance values
- 200–759 nm
- 1546 wavelengths per sample
- Stored in .xlsx format

Dataset link: https://data.mendeley.com/datasets/d4vzbcxxc

---

## 3. Repository Structure

data/
  raw/
  processed/

src/
  preprocessing/
  analysis/
  visualization/

notebooks/
  assignments/
  reporting/

results/
  figures/
  tables/

test/
  __init__.py
  testcod.py

docs/
requirements.txt

---

## 4. Installation and Setup

This section explains how to correctly configure the environment so the project can be executed without errors.

### 4.1. Clone the Repository

git clone https://github.com/LauraIbarra/Scientific-programming.git

cd "Scientific Programming Lab 3"

### 4.2. Create and Activate a Virtual Environment

Windows:

python -m venv .venv

.venv\Scripts\activate

### 4.3. Install Dependencies

pip install -r requirements.txt

---

## 5. Running the Entire Project

Once the environment is set up and the raw data is placed inside data/raw/, run:

python -m test.testcod

This will:

- Load raw data
- Preprocess physicochemical and spectral datasets
- Generate raw and processed figures
- Compute PCA
- Extract features
- Compute correlations
- Perform domain analysis
- Perform clustering
- Run ANOVA and Fisher Ratio
- Run classification models
- Save everything in results/

---

## 6. Detailed Results

### 6.1. Preprocessing Results

How:
Using PhysicochemicalProcessor and SpectralProcessor.

Outputs:
- physicochemical_processed.csv
- spectral_processed.csv
- Raw+processed plots

Observed:
- Standardized variables
- SNV-corrected spectra
- Outlier removal successful

---

### 6.2. PCA Analysis

How:
AnalysisProcessor.runPCA()

Outputs:
- pca_components.csv
- PCA scatterplots

Observed:
- First 3–5 components capture most variance
- Meaningful grouping in PCA space

---

### 6.3. Feature Extraction & Correlation

How:
extractPhysFeatures(), extractSpecFeatures(), computeCorrelation()

Outputs:
- features_phys_spec.csv
- correlation_matrix.csv
- correlation heatmap

Observed:
- Strong correlations between spectral features and COD, TOC, TN
- Redundancy in variables found

---

### 6.4. Domain Distributions & Indexes

How:
computeDomainDistributions() + computeDomainIndexes()

Outputs:
- domain_summary.csv
- domain_indexes.csv
- domain plots

Observed:
- Different spectral regions behave differently
- PhysIndex vs SpecIndex shows clear trends

---

### 6.5. Clustering

How:
runClustering() with K-Means

Outputs:
- cluster_scores.csv
- cluster metrics
- PCA clusters plot

Observed:
- 2–4 consistent clusters
- Groups differ in nutrient/organic load

---

### 6.6. Statistical Tests & Classification

How:
runAnovaFisher() and runClassification()

Outputs:
- anova_results.csv
- fisher_ratio.csv
- classification_metrics.csv

Observed:
- Features differ significantly between clusters (p < 0.05)
- Logistic Regression and kNN outperform baseline

---

## 7. Discussion

The integration of physicochemical and spectral data was effective. PCA reduced dimensions while keeping meaningful spectral patterns. Correlation and domain analyses confirmed the relationship between spectral regions and contamination indicators. Clustering revealed natural groups aligned with contamination levels. Statistical tests validated separability. Classification yielded strong predictive potential.

---

## 8. Conclusions

- Spectral data reflect water quality meaningfully
- Preprocessing improved data reliability
- PCA summarized 1546 wavelengths efficiently
- Key spectral regions identified through feature extraction
- Clusters correspond to contamination levels
- ANOVA and Fisher verify separability
- Classification models show predictive power
- Modular pipeline allows full reproducibility

---

## 9. References

FAO. (2021). El estado de los recursos de tierras y aguas del mundo para la alimentación y la agricultura - Sistemas al límite. https://openknowledge.fao.org/server/api/core/bitstreams/d6cdccdc-9f9e-4abc-b2d1-78d0351ffc37/content

OMS. (2023, September 13). Agua para consumo humano. Organizacion Mundial de La Salud. https://www.who.int/es/news-room/fact-sheets/detail/drinking-water

Navarro, J. M., Aatik, A. el, Pita, A., Martinez, R., & Vela, N. (2024). Evaluation of an IoT Device for Nitrate and Nitrite Long-Term Monitoring in Wastewater Treatment Plants. IEEE Sensors Journal. https://doi.org/10.1109/JSEN.2024.3512355

Abidin, Z., Ricardo, A., Zainuri, A., Fahanani, A. F., Markovic, M., & Miyauchi, R. (2024). Design of Water Purification System Integrated with Real-Time Quality Monitoring System Using Internet of Things. Proceedings - 2024 12th Electrical Power, Electronics, Communications, Controls and Informatics Seminar, EECCIS 2024, 63–68. https://doi.org/10.1109/EECCIS62037.2024.10839905

Elsayed, S., Hussein, H., Moghanm, F. S., Khedher, K. M., Eid, E. M., & Gad, M. (2020). Application of irrigation water quality indices and multivariate statistical techniques for surface water quality assessments in the northern nile delta, egypt. Water (Switzerland), 12(12). https://doi.org/10.3390/w12123300

Goblirsch, T., Mayer, T., Penzel, S., Rudolph, M., & Borsdorf, H. (2023). In Situ Water Quality Monitoring Using an Optical Multiparameter Sensor Probe. Sensors, 23(23). https://doi.org/10.3390/s23239545

Jiang, Jiping; Tang, Sijie (2022), “Spectral Water Quality Data”, Mendeley Data, V1, doi: 10.17632/d4vzbcxxcy.1

Chen, C., Luo, M., Wang, W., Ping, Y., Li, H., Chen, S., & Liang, Q. (2025). Characteristic Wavelength Selection and Surrogate Monitoring for UV–Vis Absorption Spectroscopy-Based Water Quality Sensing. Water (Switzerland), 17(3). https://doi.org/10.3390/w17030343

Radiation and health (RAD). (2017, 24 abril). Guidelines for drinking-water quality, 4th edition, incorporating the 1st addendum. https://www.who.int/publications/i/item/9789241549950

Resolución 2115 - 2007 | MinVivienda. (2007). https://minvivienda.gov.co/normativa/resolucion-2115-2007

EPA (US Environmental Protection Agency). (2018). YEAR IN REVIEW 2018. https://www.epa.gov/sites/default/files/2019-01/documents/epa_2018_yearinreview_0128-4.pdf

