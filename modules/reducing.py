import numpy as np
import logging
from typing import List, Tuple, Union, Optional, Dict, Any
import umap

class Reducing():
    """
    Class for reducing the dimensionality of input data using UMAP.

    Parameters:
        X (numpy.ndarray): The input data to reduce, expected to be a NumPy array.
        settings (dict): A dictionary of settings for the UMAP reduction algorithm.

    Raises:
        ValueError: If the input X is not a NumPy array, or if the number of neighbors
                    specified in the settings is greater than or equal to the number
                    of data points in X.

    """

    def __init__(self, X: np.ndarray, settings: Dict[str, Dict]) -> None:
        self.X = X
        if not isinstance(self.X, np.ndarray):
            logging.error("Reducer: input must be a NumPy array")
            raise ValueError("Reducer: input must be a NumPy array")
        self.settings = settings

    def UMAP(self) -> np.ndarray:
        """
        Applies the UMAP reduction algorithm to the input data X.

        Returns:
            numpy.ndarray: The reduced embedding of the input data.

        """

        n_neighbors = self.settings.reducing['n_neighbors']
        min_dist = self.settings.reducing['min_dist']
        init = self.settings.reducing['init']
        metric = self.settings.reducing['metric']
        random_state = self.settings.random_state
           
        if n_neighbors >= len(self.X):
            logging.error("The number of neighbors must be smaller than the number of molecules to cluster")
            raise ValueError("The number of neighbors must be smaller than the number of molecules to cluster")

        n_components = int(np.ceil(np.log(len(self.X))/np.log(4)))
        n_neighbors = max(10, int(np.sqrt(self.X.shape[0])))

        logging.info(f'Running UMAP with {n_components} components and {n_neighbors} neighbors.')

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric,
                            init=init, random_state=random_state).fit(self.X)
        embedding = reducer.transform(self.X)

        return embedding