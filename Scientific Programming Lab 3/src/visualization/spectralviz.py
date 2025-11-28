import os
import matplotlib.pyplot as plt
import pandas as pd


class SpectralVisualizer:
    """
    Visualization tools for spectral datasets.

    Generates:
    - Raw spectral plots for selected samples
    - SNV-normalized spectral plots for selected samples

    Expected dataset structure:
    - Column 0 : sample identifier (e.g. 1, 2, 3,...)
    - Columns 1..m : wavelengths as column names (floats)
    """

    def __init__(self, df: pd.DataFrame):
        """
        Store spectral dataframe and detect wavelength columns.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw spectral dataset.

        Returns
        -------
        None
        """
        self.data = df.copy()                         # O(n)
        self.idCol = self.data.columns[0]            # O(1)
        self.waveCols = self.data.columns[1:]        # O(m)
        self.wavelengths = self.waveCols.astype(float)  # O(m)

    # ---------------------------------------------------------
    # Internal helper
    # ---------------------------------------------------------
    def _ensureDir(self, path: str) -> None:
        """Create directory if it does not exist. O(1)"""
        if path and not os.path.exists(path):        # O(1)
            os.makedirs(path, exist_ok=True)         # O(1)

    def _validSamples(self, samples):
        """
        Filter sample IDs to only those present in the dataset.

        Parameters
        ----------
        samples : list of int

        Returns
        -------
        list of int
            Existing sample IDs.
        """
        available = set(self.data[self.idCol].tolist())   # O(n)
        valid = [s for s in samples if s in available]    # O(k)
        return valid                                      # O(k)

    # ---------------------------------------------------------
    # 1. RAW SPECTRA
    # ---------------------------------------------------------
    def plot_raw(self, samples, savepath: str):
        """
        Plot raw spectral curves for selected samples.

        Parameters
        ----------
        samples : list of int
            Sample IDs (values from the first column).
        savepath : str
            Output path for the figure.

        Returns
        -------
        None
        """
        samples = self._validSamples(samples)        # O(n + k)
        self._ensureDir(os.path.dirname(savepath))   # O(1)

        plt.figure(figsize=(10, 5))                  # O(1)

        for sid in samples:                          # O(k)
            row = self.data[self.data[self.idCol] == sid].iloc[0]  # O(n)
            y = row[self.waveCols].astype(float).values           # O(m)

            plt.plot(self.wavelengths, y, label=f"Sample {sid}")  # O(m)

        plt.xlabel("Wavelength (nm)")               # O(1)
        plt.ylabel("Absorbance")                    # O(1)
        plt.title("Raw spectral signatures")        # O(1)
        plt.grid(True, alpha=0.3)                   # O(1)
        plt.legend()                                # O(1)
        plt.tight_layout()                          # O(1)
        plt.savefig(savepath, dpi=300)              # O(1)
        plt.close()                                 # O(1)

    # ---------------------------------------------------------
    # 2. SNV-NORMALIZED SPECTRA
    # ---------------------------------------------------------
    def plot_snv(self, processedDf: pd.DataFrame, samples, savepath: str):
        """
        Plot SNV-normalized spectral curves.

        Parameters
        ----------
        processedDf : pandas.DataFrame
            SNV-normalized dataset with:
            - Column 0 : sample ID (same order as raw)
            - Columns 1..m : wavelengths as column names.
        samples : list of int
            Sample IDs.
        savepath : str
            Output path for the figure.

        Returns
        -------
        None
        """
        samples = self._validSamples(samples)        # O(n + k)
        self._ensureDir(os.path.dirname(savepath))   # O(1)

        # Wavelengths from SNV dataset (should match raw)
        waveCols = processedDf.columns[1:]           # O(m)
        wavelengths = waveCols.astype(float).values  # O(m)

        plt.figure(figsize=(10, 5))                  # O(1)

        for sid in samples:                          # O(k)
            row = processedDf[processedDf.iloc[:, 0] == sid].iloc[0]  # O(n)
            y = row[waveCols].astype(float).values                   # O(m)

            plt.plot(wavelengths, y, label=f"Sample {sid}")          # O(m)

        plt.xlabel("Wavelength (nm)")                     # O(1)
        plt.ylabel("SNV absorbance")                      # O(1)
        plt.title("SNV-normalized spectral signatures")   # O(1)
        plt.grid(True, alpha=0.3)                         # O(1)
        plt.legend()                                      # O(1)
        plt.tight_layout()                                # O(1)
        plt.savefig(savepath, dpi=300)                    # O(1)
        plt.close()                                       # O(1)

# This code has a computational time complexity of O(n * m)



