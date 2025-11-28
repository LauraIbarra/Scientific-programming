import pandas as pd
import numpy as np

class SpectralProcessor:
    """
    Perform preprocessing operations on spectral data.
    
    This class applies:
    - Removal of non-numeric columns
    - Standard Normal Variate (SNV) normalization
    """

    def __init__(self, data):
        """
        Store the input DataFrame.

        Parameters
        ----------
        data : pandas.DataFrame
            Raw spectral dataset (only spectral intensities or mixed).

        Returns
        -------
        None
        """
        self.data = data.copy()  # O(n)

    def removeNonNumeric(self):
        """
        Remove non-numeric columns from the spectral dataset.

        Typically used to keep only numeric spectral intensities.

        Returns
        -------
        SpectralProcessor
            Enables method chaining.
        """
        numericData = self.data.select_dtypes(include=[np.number])  # O(n)

        # Safety: if a column named 'wavelength' is numeric, drop it
        if "wavelength" in numericData.columns:                      # O(1)
            numericData = numericData.drop(columns=["wavelength"])   # O(n)

        self.data = numericData  # O(1)
        return self  # O(1)

    def applySNV(self):
        """
        Apply Standard Normal Variate (SNV) normalization row-wise.

        SNV transforms each spectrum as:
        (spectrum - mean) / std

        Returns
        -------
        SpectralProcessor
            Enables method chaining.
        """
        rowMeans = self.data.mean(axis=1)  # O(n)
        rowStds = self.data.std(axis=1)    # O(n)

        self.data = (self.data.sub(rowMeans, axis=0)).div(rowStds, axis=0)  # O(n)
        return self  # O(1)

    def toNumpy(self):
        """
        Convert processed DataFrame to NumPy array.

        Returns
        -------
        numpy.ndarray
            Spectral matrix in NumPy format.
        """
        return self.data.to_numpy()  # O(n)
    
# This code has a computational time complexity of O(n)
