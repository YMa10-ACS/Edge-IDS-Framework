'''
Description: 
Date: 2026-03-10 00:16:22
Author: Yaoquan Ma
'''
"""
Description: PCA Encoder
Date: 2026-03-10
Author: Yaoquan Ma
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAEncoder:
    def __init__(self, n_components=0.95, whiten=False, random_state=42):
        """
        n_components:
            - int: keep exact number of principal components
            - float in (0, 1]: keep enough components to preserve variance ratio
        """
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(
            n_components = n_components,
            whiten=whiten,
            random_state=random_state,
        )
    
    def fit(self, X):
        self.pca.fit(X)

    def forward(self, X):
        """
        Encode features into PCA embedding.
        """
        embedding = self.pca.transform(X).astype("float32")

        metadata = {
            "shape": list(embedding.shape),
            "dtype": str(embedding.dtype),
        }
        return embedding, metadata

