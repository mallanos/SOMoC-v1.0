import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from typing import List, Tuple, Union, Optional, Dict, Any

from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, pairwise_distances
from sklearn.metrics.pairwise import distance_metrics
from validclust import dunn

def calculate_CVIs(embedding: np.ndarray, metric:str, labels: List, Random: bool = True, num_iterations: int = 5, num_clusters: int = 3):

    """
    Calculate all CVIs for real or random clustering

    Parameters:
    embeddings (np.ndarray): An array of embeddings to cluster.
    labels (List): List of cluster labels.

    Returns: A nested dictionary containing the all CVIs
    """

    results = {}
    
    dist_dunn = pairwise_distances(embedding)

    if Random:
        SILs, DBs, CHs, DUNNs = np.zeros(num_iterations), np.zeros(num_iterations), np.zeros(num_iterations), np.zeros(num_iterations)

        for i in range(num_iterations):
            np.random.seed(i)
            random_clusters = np.random.randint(num_clusters, size=len(embedding))
            SILs[i], DBs[i], CHs[i], DUNNs[i] = silhouette_score(embedding, random_clusters, metric=metric), davies_bouldin_score(embedding, random_clusters), calinski_harabasz_score(embedding, random_clusters), dunn(dist_dunn, random_clusters)

        random_means = np.mean([SILs, DBs, CHs, DUNNs], axis=1)
        random_sds = np.std([SILs, DBs, CHs, DUNNs], axis=1)

        results = pd.DataFrame({'Random_mean': random_means,'Random_std':random_sds}, index=['silhouette','davies_bouldin','calinski_harabasz','dunn'])

    else:
        assert len(embedding) == len(labels), "Length of embeddings and labels must match"
        assert embedding.ndim == 2, "Embeddings must be a 2D array"

        SOMoC = {'silhouette': silhouette_score(embedding, labels, metric=metric),
                 'davies_bouldin': davies_bouldin_score(embedding, labels),
                 'calinski_harabasz': calinski_harabasz_score(embedding, labels),
                 'dunn': dunn(dist_dunn, labels)}

        results['SOMoC'] = SOMoC
        results = pd.DataFrame.from_dict(results)

    return results.round(3)

class Clustering():

    def __init__(self, name: str, embedding: np.ndarray, settings) -> None:
        self.name = name
        self.embedding = embedding
        self.settings = settings
        allowed_metrics = list(distance_metrics().keys())
        if self.settings.reducing['metric'] not in allowed_metrics:
            self.metric = 'cosine'
            logging.warning(f"{self.settings.reducing['metric']} is not available in Sklearn pairwise distances for Silhouette calculation, using Cosine metric instead !")
            # raise ValueError(f"Metric should be one of: {allowed_metrics}")
        else:
            self.metric = self.settings.reducing['metric']

    def cluster(self):
        method = self.settings.clustering['method']
        max_n_clusters = self.settings.clustering['max_n_clusters']
        n_iter = self.settings.clustering['n_iter']
        n_init = self.settings.clustering['n_init']
        covariance_type = self.settings.clustering['covariance_type']
        random_state = self.settings.random_state
        K = self.settings.optimal_K

        # Determine the optimal K using the clustering loop if K is not specified
        if K == False:
            logging.info(f"Finding the optimal K...")
            if method == 'kmeans':
                results_loop, K_loop = self._kmeans_loop()
            elif method == 'gmm':
                results_loop, K_loop = self._gmm_loop()
            else:
                logging.error("Invalid method. Must be 'gmm' or 'kmeans'.")
                raise ValueError("Invalid method. Must be 'gmm' or 'kmeans'.")
            K = K_loop
        else:
            results_loop = None
            logging.info(f"Using user-specified K = {K}")

        # Perform final clustering
        logging.info(f'Running final clustering with K = {K}...')
        if method == 'kmeans':
            data_clustered, results_CVIs, model = self._kmeans_final(K)
        else:
            data_clustered, results_CVIs, model = self._gmm_final(K)

        results_CVIs.to_csv(f'results/{self.name}/{self.name}_CVIs.csv', index=True, header=True)

        return K, results_loop, data_clustered, results_CVIs, model

    def _kmeans_loop(self) -> Tuple[pd.DataFrame, int]:
        """
        Runs KMeans clustering for a range of K values and returns the K value which maximizes the silhouette score.
        """
        max_n_clusters = self.settings.clustering['max_n_clusters']
        iterations = self.settings.clustering['iterations']
        random_state = self.settings.random_state

        temp = {i: [] for i in range(max_n_clusters+1)}  # pre-allocate the dictionary

        for n in tqdm(range(2, max_n_clusters+1), desc='Optimizing the number of clusters'):
            temp_sil = [None] * iterations # pre-allocate the list
            for x in range(iterations):
                kmeans = KMeans(n_clusters=n,init='k-means++', random_state=x).fit(self.embedding)
                labels = kmeans.predict(self.embedding)
                temp_sil[x] = silhouette_score(
                    self.embedding, labels, metric=self.metric)
            temp[n] = [int(n),np.mean(temp_sil), np.std(temp_sil)]

        results_loop = pd.DataFrame.from_dict(
            temp, orient='index', columns=['Clusters','Silhouette', 'sil_stdv']).dropna()
        results_loop = results_loop.astype({"Clusters": int})
        K_loop = results_loop.sort_values(['Silhouette'], ascending=False).index[0] # Get max Sil K      
        return results_loop, int(K_loop)

    def _gmm_loop(self) -> Tuple[pd.DataFrame, int]:
        """
        Runs GMM clustering for a range of K values and returns the K value which maximizes the silhouette score.
        """
        max_n_clusters = self.settings.clustering['max_n_clusters']
        n_iter = self.settings.clustering['n_iter']
        n_init = self.settings.clustering['n_init']
        init_params = self.settings.clustering['init_params']
        covariance_type = self.settings.clustering['covariance_type']
        warm_start = self.settings.clustering['warm_start']

        temp = {i: [] for i in range(max_n_clusters+1)}  # pre-allocate the dictionary

        for n in tqdm(range(2, max_n_clusters+1), desc='Optimizing the number of clusters'):
            temp_sil = [None] * n_iter # pre-allocate the list
            for x in range(n_iter):
                gmm = GMM(n, n_init=n_init, init_params=init_params, covariance_type=covariance_type,
                        warm_start=warm_start, random_state=x, verbose=0).fit(self.embedding)
                labels = gmm.predict(self.embedding)
                temp_sil[x] = silhouette_score(
                    self.embedding, labels, metric=self.metric)
            temp[n] = [int(n),np.mean(temp_sil), np.std(temp_sil)]

        results_loop = pd.DataFrame.from_dict(
            temp, orient='index', columns=['Clusters','Silhouette', 'sil_stdv']).dropna()
        results_loop = results_loop.astype({"Clusters": int})
        results_sorted = results_loop.sort_values(['Silhouette'], ascending=False)
        K_loop = results_sorted.index[0]  # Get max Sil K
        
        return results_loop, int(K_loop)

    def _gmm_final(self, K: int= 3):
        
        """
        Cluster the dataset using the optimal K value, and calculate several CVIs
        Parameters:
        embeddings (array-like): The input data to cluster.
        K (int): The number of clusters to form.
        n_init (int): The number of initializations to perform.
        init_params (str): The method used to initialize the weights, means, and covariances.
        warm_start (bool): Whether to reuse the last solution to initialize the next fit.
        covariance_type (str): The type of covariance parameters to use.
        random_state (int): The random seed to use.

        Returns:
            laebls (np.ndarray): Cluster labels
        """
        n_init = self.settings.clustering['n_init']
        init_params = self.settings.clustering['init_params']
        covariance_type = self.settings.clustering['covariance_type']
        warm_start = self.settings.clustering['warm_start']
        random_state = self.settings.random_state

        GMM_final = GMM(K, n_init=n_init, init_params=init_params, warm_start=warm_start,
                        covariance_type=covariance_type, random_state=random_state, verbose=0)
        GMM_final.fit(self.embedding)
        labels = GMM_final.predict(self.embedding)

        if GMM_final.converged_:
            logging.info('GMM converged.')
        else:
            logging.warning('GMM did not converge. Please check you input configuration.')
        
        logging.info('Calculating CVIs')
        results_real = calculate_CVIs(self.embedding, self.metric,labels=labels, Random = False)
        results_random = calculate_CVIs(self.embedding, self.metric, labels=None, num_clusters=K, Random = True)
        
        # concatenate DataFrames along columns axis
        results_CVIs = pd.concat([results_real, results_random], axis=1)
        results_CVIs.to_csv(f'results/{self.name}/{self.name}_CVIs.csv', index=True, header=True)

        return labels, results_CVIs, GMM_final
      
    def _kmeans_final(self, K: int= 3):
        """
        Cluster the dataset using the optimal K value, and calculate several CVIs

        Parameters:
        embeddings (array-like): The input data to cluster.
        K (int): The number of clusters to form.
        random_state (int): The random seed to use.

        Returns:
            laebls (np.ndarray): Cluster labels
        """
        random_state = self.settings.random_state

        KMeans_final = KMeans(n_clusters=K, init='k-means++', random_state=random_state)
        KMeans_final.fit(self.embedding)
        labels = KMeans_final.predict(self.embedding)

        if KMeans_final.n_iter_ < KMeans_final.max_iter:
            logging.info('K-Means converged.')
        else:
            logging.warning('K-Means did not converge. Please check your input configuration.')

        logging.info('Calculating CVIs')
        results_real = calculate_CVIs(self.embedding, self.metric, labels=labels, Random = False)
        results_random = calculate_CVIs(self.embedding, self.metric, labels=None, num_clusters=K, Random = True)
        
        # concatenate DataFrames along columns axis
        results_CVIs = pd.concat([results_real, results_random], axis=1)
        results_CVIs.to_csv(f'results/{self.name}/{self.name}_CVIs.csv', index=True, header=True)

        return labels, results_CVIs, KMeans_final
