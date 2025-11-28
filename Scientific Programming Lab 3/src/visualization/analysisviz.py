import matplotlib.pyplot as plt
import seaborn as sns


class AnalysisVisualizer:
    """
    Visualization tools for analysis results:
    - Correlation matrix
    - Domain distributions
    - Domain index scatter
    - Clustering metrics vs k
    - PCA scatter by cluster
    """

    def __init__(self):
        """Initialize visualizer.  O(1)"""
        pass

    def plotCorrelation(self, corrDf, savepath):
        """
        Plot correlation heatmap.
        """
        plt.figure(figsize=(6, 5))  # O(1)
        sns.heatmap(
            corrDf,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            square=True,
            cbar_kws={"shrink": 0.8})  # O(k^2)
        plt.title("Correlation matrix (phys vs spec)")
        plt.tight_layout()
        plt.savefig(savepath, dpi=300)
        plt.close()

    def plotDomainDistributions(self, dfLong, savepath):
        """
        Violin + boxplot of standardized values grouped by domain.

        Parameters
        ----------
        dfLong : DataFrame
            Columns: feature, z, domain.
        """
        plt.figure(figsize=(7, 4))  # O(1)

        sns.violinplot(
            data=dfLong,
            x="domain",
            y="z",
            inner=None,
            palette={
                "Physico-chemical": "blue",
                "Spectral": "red"
            },
        )  # O(n)

        sns.boxplot(
            data=dfLong,
            x="domain",
            y="z",
            width=0.25,
            showcaps=True,
            boxprops={"zorder": 2},
            palette={"Physico-chemical": "orange", "Spectral": "orange"},
        )  # O(n)

        plt.title("Standardized distribution by domain")
        plt.tight_layout()
        plt.savefig(savepath, dpi=300)
        plt.close()

    def plotIndexScatter(self, idxDf, corrIdx, savepath):
        """
        Scatter + regression line PhysIndex vs SpecIndex.

        Parameters
        ----------
        idxDf : DataFrame
            Columns: PhysIndex, SpecIndex.
        corrIdx : float
            Correlation coefficient.

        Complexity
        ----------
        O(n)
        """
        plt.figure(figsize=(6, 5))  # O(1)
        sns.regplot(
            data=idxDf,
            x="PhysIndex",
            y="SpecIndex",
            scatter_kws={"s": 50, "alpha": 0.8},
            line_kws={"linewidth": 2},
        )  # O(n)
        plt.title(f"PhysIndex vs SpecIndex  (r = {corrIdx:.2f})")
        plt.tight_layout()
        plt.savefig(savepath, dpi=300)
        plt.close()

    def plotClusterScores(self, scoresDf, savepath):
        """
        Plot silhouette, Calinski–Harabasz and Davies–Bouldin vs k.

        Complexity
        ----------
        O(k)
        """
        plt.figure(figsize=(8, 5))  # O(1)
        plt.plot(scoresDf["k"], scoresDf["silhouette"],
                 marker="o", label="Silhouette")
        plt.plot(
            scoresDf["k"],
            scoresDf["calinski_harabasz"],
            marker="o",
            label="Calinski-Harabasz",
        )
        plt.plot(
            scoresDf["k"],
            scoresDf["davies_bouldin"],
            marker="o",
            label="Davies-Bouldin",
        )
        plt.xlabel("k")
        plt.ylabel("Score")
        plt.title("Metrics by k (Silhouette, CH, DB)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(savepath, dpi=300)
        plt.close()

    def plotPcaScatter(self, pcaDf, labels, savepath):
        """
        Scatter PCA_1 vs PCA_2 colored by clusters.

        Complexity
        ----------
        O(n)
        """
        plt.figure(figsize=(7, 6))  # O(1)
        sns.scatterplot(
            x=pcaDf["PCA_1"],
            y=pcaDf["PCA_2"],
            hue=labels,
            palette="viridis",
            s=60,
        )  # O(n)
        plt.title("PCA scatter by clusters")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.tight_layout()
        plt.savefig(savepath, dpi=300)
        plt.close()

# plotCorrelation:          O(k^2)
# plotDomainDistributions:  O(n)
# plotIndexScatter:         O(n)
# plotClusterScores:        O(k)
# plotPcaScatter:           O(n)
# This code has a computational time complexity of O(n + k^2)

