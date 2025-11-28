import os 
import math 
import pandas as pd  
import matplotlib.pyplot as plt  

class PhysicochemicalVisualizer:
    """
    Perform visualization operations on physicochemical data.

    This class generates:
    - Histograms for raw physicochemical variables
    - Boxplots for processed physicochemical variables
    
    """

    def __init__(self, rawDf):
        """
        Store the raw DataFrame and detect numeric columns.

        Parameters
        ----------
        rawDf : pandas.DataFrame
            Raw physicochemical dataset.

        Returns
        -------
        None
        """
        self.raw = rawDf.copy()  # O(n)

        # Detect numeric columns
        self.numCols = list(self.raw.select_dtypes(include=["float64", "int64"]).columns)  # O(k)

        # Detect ID column (first non-numeric)
        nonNum = self.raw.select_dtypes(exclude=["float64", "int64"]).columns.tolist()  # O(k)
        self.idCol = nonNum[0] if nonNum else None  # O(1)

        # Ensure ID column is not treated as numeric
        if self.idCol in self.numCols:  # O(1)
            self.numCols.remove(self.idCol)  # O(1)

    def _ensure(self, path):
        """
        Create directory if it does not exist.
        """
        if path and not os.path.exists(path):  # O(1)
            os.makedirs(path)  # O(1)

    def histograms_raw(self, savepath):
        """
        Generate histograms for all numeric variables (raw data).

        Parameters
        ----------
        savepath : str
            Output file path.

        Returns
        -------
        None
        """
        self._ensure(os.path.dirname(savepath))  # O(1)

        m = len(self.numCols)  # O(1)
        cols = 3  # O(1)
        rows = math.ceil(m / cols)  # O(1)

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))  # O(1)
        axes = axes.flatten()  # O(1)

        for i, col in enumerate(self.numCols):  # O(m)
            data = self.raw[col].dropna()  # O(n)

            axes[i].hist(data, bins=10, color="red", edgecolor="black")  # O(n)
            axes[i].set_title(f"Distribution of {col} in the water samples")  # O(1)
            axes[i].set_xlabel(col)  # O(1)
            axes[i].set_ylabel("Frequency (Number of samples)")  # O(1)

        for j in range(i + 1, len(axes)):  # O(m)
            axes[j].axis("off")  # O(1)

        plt.tight_layout()  # O(1)
        plt.savefig(savepath, dpi=300)  # O(1)
        plt.close()  # O(1)

    def boxplot_processed(self, processedDf, savepath):
        """
        Generate a boxplot for processed physicochemical variables.

        Parameters
        ----------
        processedDf : pandas.DataFrame
            Processed numeric dataset.
        savepath : str
            Output file path.

        Returns
        -------
        None
        """
        self._ensure(os.path.dirname(savepath))  # O(1)

        # Only use columns present in processed dataset
        cols = [c for c in self.numCols if c in processedDf.columns]  # O(m)
        numeric = processedDf[cols]  # O(n * m)

        ax = numeric.plot(  # O(n * m)
            kind="box",
            figsize=(8, 4),
            color=dict(
                boxes="royalblue",
                whiskers="black",
                medians="red",
                caps="black",
            ),
            grid=False,
        )

        ax.yaxis.grid(True, linestyle="--", alpha=0.5)  # O(1)
        plt.title("Standardized Physicochemical Variables")  # O(1)
        plt.ylabel("Z-score")  # O(1)

        plt.tight_layout()  # O(1)
        plt.savefig(savepath, dpi=300)  # O(1)
        plt.close()  # O(1)

# This code has a computational time complexity of O(n * m)
