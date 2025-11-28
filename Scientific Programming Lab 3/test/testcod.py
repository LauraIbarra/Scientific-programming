import pandas as pd
from pathlib import Path

from src.preprocessing.physicochemical import PhysicochemicalProcessor
from src.preprocessing.spectral import SpectralProcessor
from src.visualization.physicochemicalViz import PhysicochemicalVisualizer
from src.visualization.spectralviz import SpectralVisualizer
from src.analysis.analysis import AnalysisProcessor
from src.visualization.analysisviz import AnalysisVisualizer

# ============================================================
# OUTPUT FOLDERS (O(1))
# Create folders for processed data, figures and tables.
# ============================================================
Path("data/processed").mkdir(parents=True, exist_ok=True)   # O(1)
Path("results/figures").mkdir(parents=True, exist_ok=True)  # O(1)
Path("results/tables").mkdir(parents=True, exist_ok=True)   # O(1)

# ============================================================
# LOAD RAW DATA (O(n))
# Read physicochemical and spectral raw datasets.
# ============================================================
df_physico = pd.read_excel("data/raw/actual_concen_sfwater.xlsx")   # O(n)
df_spectra = pd.read_excel("data/raw/specdata_undeno_sfwater.xlsx") # O(n)

# Extract numeric physicochemical variables (O(n))
phys_num = df_physico.drop(columns=["LocationID"]).select_dtypes(include=["float64", "int64"])  # O(n)

# Extract numeric spectral intensities (O(n))
spectral_num = df_spectra.drop(columns=["wavelength"]).select_dtypes(include=["float64", "int64"])  # O(n)


# ============================================================
# RAW VISUALIZATION (O(n * m))
# Show initial behavior of raw distributions and spectra.
# ============================================================
physViz = PhysicochemicalVisualizer(phys_num)   # O(1)
specViz = SpectralVisualizer(df_spectra)        # O(1)

samples = [4, 14, 29]  # O(1)

physViz.histograms_raw("results/figures/physicochemical_hist_raw.png")  # O(n*m)
specViz.plot_raw(samples, "results/figures/spectral_raw.png")           # O(n*m)


# ============================================================
# PREPROCESSING PIPELINE
# Physicochemical: negatives → outliers → Z-score.
# Spectral: numeric filter → SNV normalization.
# ============================================================
physProcessor = PhysicochemicalProcessor(phys_num)  # O(1)
physProcessor.removeNegatives()                     # O(n)
physProcessor.removeOutliers()                      # O(n)
physProcessor.normalizeZscore()                     # O(n)
phys_processed = physProcessor.data                 # O(1)

specProcessor = SpectralProcessor(spectral_num)     # O(1)
specProcessor.removeNonNumeric()                    # O(n)
specProcessor.applySNV()                            # O(n)
spec_processed = specProcessor.data                 # O(1)

# Reattach identifiers (O(n))
phys_clean = pd.concat([df_physico["LocationID"], phys_processed], axis=1)  # O(n)
spec_clean = pd.concat([df_spectra["wavelength"], spec_processed], axis=1)  # O(n)

# Save processed data (O(n))
phys_clean.to_csv("data/processed/physicochemical_processed.csv", index=False)  # O(n)
spec_clean.to_csv("data/processed/spectral_processed.csv", index=False)         # O(n)


# ============================================================
# VISUALIZATION OF CLEAN DATA (O(n*m))
# Boxplot for physicochemical variables and SNV spectral curves.
# ============================================================
physViz.boxplot_processed(
    phys_processed, "results/figures/physicochemical_box_processed.png"
)  # O(n*m)

specViz.plot_snv(
    spec_clean, samples, "results/figures/spectral_snv.png"
)  # O(n*m)


# ============================================================
# ANALYSIS PIPELINE
# Includes PCA, feature extraction, correlation,
# domain-level statistics, clustering and classification.
# ============================================================
analysis = AnalysisProcessor()      # O(1)
analysisViz = AnalysisVisualizer()  # O(1)


# === 1) PCA on spectral data (O(n*m*k)) =======================
pcaDf, varExpl = analysis.runPCA(spec_processed, nComponents=5)  # O(n*m*k)

pca_with_id = pd.concat([df_physico["LocationID"], pcaDf], axis=1)  # O(n)
pca_with_id.to_csv("results/tables/pca_components.csv", index=False)  # O(n)


# === 2) Merge physicochemicals + PCA (O(n*m)) =================
mergedDf = analysis.mergePhysicoPCA(phys_clean, pcaDf)  # O(n*m)
mergedDf.to_csv("results/tables/merged_physico_pca.csv", index=False)  # O(n)


# === 3) Feature extraction (O(n*m)) ===========================
phys_features = analysis.extractPhysFeatures(phys_processed)  # O(n*m)
spec_features = analysis.extractSpecFeatures(pcaDf)           # O(n*m)
all_features = pd.concat([phys_features, spec_features], axis=1)  # O(n*m)
all_features.to_csv("results/tables/features_phys_spec.csv", index=False)  # O(n)


# === 4) Correlation matrix (O(k^3)) ===========================
corrDf = analysis.computeCorrelation(all_features)  # O(k^3)
corrDf.to_csv("results/tables/correlation_matrix.csv")  # O(k^2)
analysisViz.plotCorrelation(
    corrDf, "results/figures/analysis_correlation_matrix.png"
)  # O(k^2)


# === 5) Domain distributions (O(n*k)) =========================
phys_cols = ["mean", "std", "rms"]
spec_cols = ["spectral_energy", "mean_power", "spectral_entropy"]

Z, dfLong, domainSummary = analysis.computeDomainDistributions(
    all_features, phys_cols, spec_cols
)  # O(n*k)
domainSummary.to_csv("results/tables/domain_summary.csv")  # O(1)

analysisViz.plotDomainDistributions(
    dfLong, "results/figures/analysis_domain_distributions.png"
)  # O(n)

# === 6) Domain indexes (O(n*k)) ===============================
idxDf, corrIdx = analysis.computeDomainIndexes(
    all_features, phys_cols, spec_cols
)  # O(n*k)
idxDf.to_csv("results/tables/domain_indexes.csv", index=False)  # O(n)

analysisViz.plotIndexScatter(
    idxDf, corrIdx, "results/figures/analysis_index_scatter.png"
)  # O(n)


# === 7) Clustering (O(n²)) ====================================
scoresDf, clusterLabels, bestK = analysis.runClustering(
    all_features, kMin=2, kMax=5
)  # O(n²)
scoresDf.to_csv("results/tables/cluster_scores.csv", index=False)  # O(k)

analysisViz.plotClusterScores(
    scoresDf, "results/figures/analysis_cluster_scores.png"
)  # O(k)

analysisViz.plotPcaScatter(
    pcaDf, clusterLabels, "results/figures/analysis_pca_clusters.png"
)  # O(n)


# === 8) ANOVA + Fisher Ratio (O(n*k)) ==========================
anovaDf, fisherDf = analysis.runAnovaFisher(all_features, clusterLabels)  # O(n*k)
anovaDf.to_csv("results/tables/anova_results.csv", index=False)  # O(k)
fisherDf.to_csv("results/tables/fisher_ratio.csv", index=False)  # O(k)


# === 9) Classification (O(n*k)) ===============================
classMetrics = analysis.runClassification(all_features, clusterLabels)  # O(n*k)
classMetrics.to_csv("results/tables/classification_metrics.csv", index=False)  # O(1)


# ============================================================
# FINAL STATUS
# ============================================================
print("Physicochemical preprocessing completed.")
print("Spectral preprocessing completed.")
print("Visualization completed.")
print("Analysis (PCA, features, correlation) completed.")
print("Clustering, ANOVA, Fisher, and classification completed.")


# ============================================================
#  COMPUTATIONAL COMPLEXITY SUMMARY
# ============================================================
# Summing dominant blocks:
# RAW PROCESSING:           O(n) + O(n*m)
# PREPROCESSING:            O(n)
# PCA:                      O(n*m*k)
# FEATURE EXTRACTION:       O(n*m)
# CORRELATION:              O(k^3)
# DOMAIN DISTRIBUTIONS:     O(n*k)
# CLUSTERING:               O(n^2)   
# ANOVA + FISHER:           O(n*k)
# CLASSIFICATION:           O(n*k)
# ============================================================
# FINAL RESULT:
# The global computational complexity of this full pipeline is:
#
#               **O(n²)** 
# ============================================================




