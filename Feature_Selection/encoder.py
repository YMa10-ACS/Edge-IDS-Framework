'''
Description: 
Date: 2026-03-10 00:08:59
Author: Yaoquan Ma
'''
"""
Description: Feature Selection Encoder
Date: 2026-03-10
Author: Yaoquan Ma
"""

import numpy as np


class FSEncoder:
    def __init__(self):
        """
        selected_features: list of feature indices to keep
        """
        self.selected_features = None

    def forward(self, X):
        """
        Encode input features
        """
        if self.selected_features is not None:
            X = X[:, self.selected_features]

        embedding = X.astype("float32")
        return embedding
