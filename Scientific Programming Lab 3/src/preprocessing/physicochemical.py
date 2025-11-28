import pandas as pd
import numpy as np 

class PhysicochemicalProcessor:
    """
    Perform preprocessing operations on physicochemical data.
    
    This class applies:
    - Negative value correction
    - IQR-based outlier removal
    - Z-score normalization
    """
  
    def __init__(self, data):
        """
        Store the input DataFrame.

        Parameters
        ----------
        data : pandas.DataFrame
            Raw physicochemical dataset.

        Returns
        -------
        None
        """
        self.data = data.copy()  # O(n)
   
    def removeNegatives(self): 
        """
        Replace negative values with 0.

        Parameters
        ----------
        None
        
        Returns
        -------
        PhysicochemicalProcessor
            Returns the object to allow method chaining.
        """
        self.data = self.data.clip(lower=0)  # O(n)
        return self  #O(1)
 
    def removeOutliers(self):
        """
        Remove outliers using the IQR method (vectorized clipping).

        Q1 = 25th percentile  
        Q3 = 75th percentile  
        IQR = Q3 - Q1  
        Values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are clipped.

        Parameters
        ----------
        None
        
        Returns
        -------
        PhysicochemicalProcessor
            Returns the object to allow method chaining.
        """
        Q1 = self.data.quantile(0.25)  # O(n)
        Q3 = self.data.quantile(0.75)  # O(n)
        IQR = Q3 - Q1                  # O(n)

        lower = Q1 - 1.5 * IQR         # O(n)
        upper = Q3 + 1.5 * IQR         # O(n)

        self.data = self.data.clip(lower=lower, upper=upper,axis=1) # O(n)
        return self #O(1)
    
    def normalizeZscore(self):
        """
        Apply Z-score normalization column-wise.

        Parameters
        ----------
        None

        Returns
        -------
        PhysicochemicalProcessor
            Returns the object to allow method chaining.
        """
        means = self.data.mean()       # O(n)
        stds = self.data.std()         # O(n)

        self.data = (self.data - means) / stds   # O(n)
        return self #O(1)
    
# This code has a computational time complexity of O(n)
