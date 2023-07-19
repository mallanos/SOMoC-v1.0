import numpy as np
import logging
from typing import List, Tuple, Union, Optional, Dict, Any
import umap
from sklearn.decomposition import PCA

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

    def reduce(self) -> np.ndarray:
        """
        Reduces the dimensionality of the input data using the specified method.

        Returns:
            numpy.ndarray: The reduced embedding of the input data.

        """
        if self.settings.reducing['reducer'] == 'umap':
            return self.UMAP()
        if self.settings.reducing['reducer'] == 'pca':
            return self.pca()
        else:
            raise ValueError("Invalid method. Must be 'umap' or 'pca'.")

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
        densemap = self.settings.reducing['densemap']
        random_state = self.settings.random_state

        if self.settings.reducing['n_components'] == False:
            n = 3 # Bigger n means fewer components
            n_components = int(np.ceil(np.log(len(self.X))/np.log(n)))
            logging.info(f'Approximating n_components based on dataset size')
        else:
            n_components = self.settings.reducing['n_components']

        if self.settings.reducing['n_neighbors'] == False:
            n_neighbors = max(10, int(np.sqrt(self.X.shape[0])))
            logging.info(f'Approximating n_neighbors based on dataset size')
        else:
            n_neighbors = self.settings.reducing['n_neighbors']
            if n_neighbors >= len(self.X):
                logging.error("The number of neighbors must be smaller than the number of molecules to cluster")
                raise ValueError("The number of neighbors must be smaller than the number of molecules to cluster")

        logging.info(f'Running UMAP with {n_components} components and {n_neighbors} neighbors.')

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric,
                            init=init, densmap=densemap,random_state=random_state).fit(self.X)
        embedding = reducer.transform(self.X)

        return embedding

    def pca(self) -> np.ndarray:
        """
        Applies the PCA reduction algorithm to the input data X.

        Returns:
            numpy.ndarray: The reduced embedding of the input data.

        """
        if self.settings.reducing['n_components'] == False:
            n_components = 0.95
            logging.info(f'Approximating n_components that explain 95% of the variance')
        else:
            n_components =  self.settings.reducing['n_components']

        random_state = self.settings.random_state
        
        reducer = PCA(n_components=n_components, random_state=random_state)
        embedding = reducer.fit_transform(self.X)
        logging.info(f'Running PCA with n_components={reducer.n_components_}.')

        return embedding