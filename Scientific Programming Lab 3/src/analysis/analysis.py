import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from scipy.stats import entropy
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score,calinski_harabasz_score,davies_bouldin_score,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

class AnalysisProcessor:
    """
    Perform PCA, feature extraction, correlation, domain analysis,
    clustering and classification on physicochemical + spectral data.
    """

    def __init__(self):
        """Initialize processor (no persistent state).  O(1)"""
        pass

    def runPCA(self, spectralDf, nComponents=5):
        """
        Apply PCA over spectral processed data (SNV).

        Parameters
        ----------
        spectralDf : pandas.DataFrame
            SNV-normalized spectral intensities (no wavelength column).
        nComponents : int
            Number of PCA components.

        Returns
        -------
        pcaDf : pandas.DataFrame
            PCA_1 ... PCA_nComponents scores.
        varExplained : ndarray
            Explained variance ratio.
        """
        pca = PCA(n_components=nComponents)          # O(1)
        pcaResult = pca.fit_transform(spectralDf)    # O(n * m * k)
        pcaDf = pd.DataFrame(pcaResult, columns=[f"PCA_{i+1}" for i in range(nComponents)]) # O(n * k)
        return pcaDf, pca.explained_variance_ratio_  # O(1)

    def mergePhysicoPCA(self, physDf, pcaDf):
        """
        Concatenate physicochemical dataset (with LocationID)
        and PCA components by row index.

        Parameters
        ----------
        physDf : pandas.DataFrame
            LocationID + processed physicochemical variables.
        pcaDf : pandas.DataFrame
            PCA components.

        Returns
        -------
        mergedDf : pandas.DataFrame
        """
        mergedDf = pd.concat([physDf, pcaDf], axis=1)  # O(n * m)
        return mergedDf                                # O(1)

    def extractPhysFeatures(self, data):
        """
        Compute mean, std and RMS for each sample.

        Parameters
        ----------
        data : pandas.DataFrame
            Processed physicochemical variables.

        Returns
        -------
        df : pandas.DataFrame
            Columns: mean, std, rms.
        """
        df = pd.DataFrame(index=data.index)                      # O(n)
        df["mean"] = data.mean(axis=1)                           # O(n * m)
        df["std"] = data.std(axis=1)                             # O(n * m)
        df["rms"] = np.sqrt(np.mean(np.square(data), axis=1))    # O(n * m)
        return df                                                # O(1)

    def extractSpecFeatures(self, pcaDf):
        """
        Compute spectral_energy, mean_power, spectral_entropy
        from PCA components.

        Parameters
        ----------
        pcaDf : pandas.DataFrame
            PCA components.

        Returns
        -------
        df : pandas.DataFrame
            Columns: spectral_energy, mean_power, spectral_entropy.
        """
        df = pd.DataFrame(index=pcaDf.index)                     # O(n)

        spectralEnergy = np.sum(np.square(pcaDf), axis=1)        # O(n * m)
        df["spectral_energy"] = spectralEnergy                   # O(n)
        df["mean_power"] = spectralEnergy / pcaDf.shape[1]       # O(n)

        normMatrix = (np.square(pcaDf).T / np.square(pcaDf).sum(axis=1)).T    # O(n * m)
        df["spectral_entropy"] = normMatrix.apply( lambda x: entropy(x + 1e-12), axis=1)  # O(n * m)
        return df                                                # O(1)

    def computeCorrelation(self, X):
        """
        Compute correlation matrix of a feature dataframe.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.

        Returns
        -------
        corrDf : pandas.DataFrame
        """
        corrDf = X.corr()  # O(k^3)
        return corrDf      # O(1)

    def computeDomainDistributions(self, X, physCols, specCols):
        """
        Standardize features (z-score) and compute domain summary.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix with both domains.
        physCols : list of str
            Names of physicochemical features.
        specCols : list of str
            Names of spectral features.

        Returns
        -------
        Z : pandas.DataFrame
            Standardized features.
        dfLong : pandas.DataFrame
            Long format with columns: feature, z, domain.
        summary : pandas.DataFrame
            mean, std, median, min, max by domain.
        """
        Z = (X - X.mean()) / X.std(ddof=0)  # O(n * k)
        dfLong = Z.melt(var_name="feature", value_name="z")  # O(n * k)
        dfLong["domain"] = np.where(dfLong["feature"].isin(physCols),"Physico-chemical","Spectral")  # O(n * k)
        summary = dfLong.groupby("domain")["z"].agg( ["mean", "std", "median", "min", "max"]).round(3)  # O(n)
        return Z, dfLong, summary  # O(1)

    def computeDomainIndexes(self, X, physCols, specCols):
        """
        Compute PhysIndex and SpecIndex as average z-score per domain.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        physCols : list of str
        specCols : list of str

        Returns
        -------
        idxDf : pandas.DataFrame
            Columns: PhysIndex, SpecIndex.
        corrIdx : float
            Pearson correlation PhysIndex vs SpecIndex.
        """
        Z = (X - X.mean()) / X.std(ddof=0)          # O(n * k)

        physIndex = Z[physCols].mean(axis=1)        # O(n * k)
        specIndex = Z[specCols].mean(axis=1)        # O(n * k)

        idxDf = pd.DataFrame({"PhysIndex": physIndex, "SpecIndex": specIndex})   # O(n)
        corrIdx = idxDf.corr().loc["PhysIndex", "SpecIndex"]  # O(1)
        return idxDf, corrIdx  # O(1)

    def runClustering(self, X, kMin=2, kMax=6):
        """
        Evaluate KMeans for k in [kMin, kMax] and compute metrics.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        kMin : int
        kMax : int

        Returns
        -------
        scoresDf : pandas.DataFrame
            Metrics per k (silhouette, CH, DB).
        bestLabels : ndarray
            Labels for best k (by silhouette).
        bestK : int
        """
        Z = (X - X.mean()) / X.std(ddof=0)  # O(n * k)

        scores = []     # O(1)
        labelsDict = {} # O(1)

        for k in range(kMin, kMax + 1):                     # O(K)
            km = KMeans(n_clusters=k, n_init=20, random_state=42)                    # O(1)
            labels = km.fit_predict(Z)                      # O(n * k)
            labelsDict[k] = labels                          # O(1)

            sil = silhouette_score(Z, labels)               # O(n^2)
            ch = calinski_harabasz_score(Z, labels)         # O(n)
            db = davies_bouldin_score(Z, labels)            # O(n^2)

            scores.append(
                {
                    "k": k,
                    "silhouette": sil,
                    "calinski_harabasz": ch,
                    "davies_bouldin": db,
                }
            )                                               # O(1)

        scoresDf = pd.DataFrame(scores).round(3)            # O(K)

        bestK = int(scoresDf.sort_values("silhouette", ascending=False).iloc[0]["k"])  # O(K)
        bestLabels = labelsDict[bestK]                      # O(n)
        return scoresDf, bestLabels, bestK                  # O(1)

    def runAnovaFisher(self, X, labels):
        """
        Compute ANOVA F-test and Fisher ratio per feature.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        labels : array-like
            Cluster labels (KMeans).

        Returns
        -------
        anovaDf : pandas.DataFrame
        fisherDf : pandas.DataFrame
        """
        y = np.array(labels)                      # O(n)
        classes = np.unique(y)                    # O(n)

        def fisherRatio(values, groups):
            """Compute Fisher ratio for one feature.  O(n)"""
            cls = np.unique(groups)                                  # O(n)
            means = [np.mean(values[groups == c]) for c in cls]      # O(n)
            vars_ = [np.var(values[groups == c], ddof=1)
                     for c in cls]                                   # O(n)
            if np.mean(vars_) == 0:
                return np.nan
            return np.var(means, ddof=1) / np.mean(vars_)            # O(1)

        anovaRows = []   # O(1)
        fisherRows = []  # O(1)

        for col in X.columns:                                      # O(k)
            groupsVals = [X.loc[y == c, col].values
                          for c in classes]                        # O(n)
            F, p = stats.f_oneway(*groupsVals)                     # O(n)
            anovaRows.append(
                {"feature": col, "F": F, "p_value": p}
            )                                                      # O(1)

            fr = fisherRatio(X[col].values, y)                     # O(n)
            fisherRows.append(
                {"feature": col, "FisherRatio": fr}
            )                                                      # O(1)

        anovaDf = pd.DataFrame(anovaRows).sort_values(
            "p_value"
        ).reset_index(drop=True)                                   # O(k log k)

        fisherDf = pd.DataFrame(fisherRows).sort_values(
            "FisherRatio", ascending=False
        ).reset_index(drop=True)                                   # O(k log k)

        return anovaDf, fisherDf                                   # O(1)

    def runClassification(self, X, labels, cvSplits=5):
        """
        Evaluate baseline, logistic regression and kNN (k=5)
        using stratified cross-validation.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        labels : array-like
            Cluster labels as class targets.
        cvSplits : int
            Number of folds.

        Returns
        -------
        resultsDf : pandas.DataFrame
            Metrics per model (Accuracy, Precision, Recall, F1, ROC-AUC).
        """
        y = np.array(labels)  # O(n)

        cv = StratifiedKFold(n_splits=cvSplits, shuffle=True, random_state=42)  # O(1)

        models = {
            "Baseline": make_pipeline(
                StandardScaler(),
                DummyClassifier(strategy="most_frequent")
            ),
            "Logistic Regression": make_pipeline(
                StandardScaler(),
                LogisticRegression(solver="liblinear", random_state=42)
            ),
            "kNN (k=5)": make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=5)
            ),
        }  # O(1)

        rows = []  # O(1)

        for name, clf in models.items():                # O(3)
            yPred = cross_val_predict(clf, X, y, cv=cv)  # O(n * k)

            # Probabilities for ROC-AUC (if available)
            try:
                probaAll = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")  # O(n * k)
                if probaAll.shape[1] >= 2 and len(np.unique(y)) == 2:
                    yProba = probaAll[:, 1]             # O(n)
                    auc = roc_auc_score(y, yProba)      # O(n)
                else:
                    yProba = None
                    auc = np.nan
            except Exception:
                yProba = None
                auc = np.nan

            if len(np.unique(y)) == 2:
                avg = "binary"
            else:
                avg = "macro"

            acc = accuracy_score(y, yPred)                          # O(n)
            prec = precision_score(y, yPred, average=avg,
                                   zero_division=0)                 # O(n)
            rec = recall_score(y, yPred, average=avg,
                               zero_division=0)                     # O(n)
            f1 = f1_score(y, yPred, average=avg,
                          zero_division=0)                          # O(n)

            rows.append(
                {
                    "Model": name,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1": f1,
                    "ROC_AUC": auc,
                }
            )                                                       # O(1)

        resultsDf = pd.DataFrame(rows).round(3)                     # O(1)
        return resultsDf                                            # O(1)


# PCA:                 O(n * m * k)
# Feature extraction:  O(n * m)
# Correlation:         O(k^3)
# Domain Z + summary:  O(n * k)
# Domain indexes:      O(n * k)
# Clustering:          O(n^2)
# ANOVA/Fisher:        O(n * k)
# Classification:      O(n * k)
# This code has a computational time complexity of O(n * m + n^2)




