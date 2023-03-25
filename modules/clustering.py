import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from typing import List, Tuple, Union, Optional, Dict, Any

from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, pairwise_distances
from validclust import dunn

def calculate_CVIs(embedding: np.ndarray, labels: List, Random: bool = True, num_iterations: int = 500, num_clusters: int = 3):

    """
    Calculate all CVIs for real or random clustering

    Parameters:
    embeddings (np.ndarray): An array of embeddings to cluster.
    labels (List): List of cluster labels.
    
    Returns: A nested dictionary containing the all CVIs
    """

    results = {}

    if Random:
        SILs = np.zeros(num_iterations)
        DBs = np.zeros(num_iterations)
        CHs = np.zeros(num_iterations)
        DUNNs = np.zeros(num_iterations)

        for i in range(num_iterations):
            np.random.seed(i)
            random_clusters = np.random.randint(num_clusters, size=len(embedding))
            silhouette_random = silhouette_score(embedding, random_clusters, metric='euclidean')
            SILs[i] = silhouette_random
            db_random = davies_bouldin_score(embedding, random_clusters)
            DBs[i] = db_random
            ch_random = calinski_harabasz_score(embedding, random_clusters)
            CHs[i] = ch_random
            dist_dunn = pairwise_distances(embedding)
            dunn_random = dunn(dist_dunn, random_clusters)
            DUNNs[i] = dunn_random

        sil_random = np.mean(SILs).round(4)
        sil_random_st = np.std(SILs).round(4)
        db_random = np.mean(DBs).round(4)
        db_random_st = np.std(DBs).round(4)
        ch_random = np.mean(CHs).round(4)
        ch_random_st = np.std(CHs).round(4)
        dunn_random = np.mean(DUNNs).round(4)
        dunn_random_st = np.std(DUNNs).round(4)

        random_means = [sil_random, db_random, ch_random, dunn_random]
        random_sds = [sil_random_st, db_random_st, ch_random_st, dunn_random_st]
        
        results = pd.DataFrame({'Random_mean': random_means,'Random_std':random_sds}, index=['silhouette','davies_bouldin','calinski_harabasz','dunn'])

    else:
        assert len(embedding) == len(labels), "Length of embeddings and labels must match"
        assert embedding.ndim == 2, "Embeddings must be a 2D array"

        results['SOMoC'] = {
            'silhouette': silhouette_score(embedding, labels, metric='euclidean').round(4),
            'davies_bouldin': davies_bouldin_score(embedding, labels).round(4),
            'calinski_harabasz': calinski_harabasz_score(embedding, labels).round(4),
            'dunn': dunn(pairwise_distances(embedding), labels).round(4)
        }
        results = pd.DataFrame.from_dict(results)

    return results

class Clustering():

    def __init__(self, name: str, embedding: np.ndarray, settings) -> None:
        self.name = name
        self.embedding = embedding
        self.settings = settings

    def GMM_loop(self) -> Tuple[pd.DataFrame, int]:
        """
        Runs GMM clustering for a range of K values and returns the K value which maximizes the silhouette score.

        Parameters:
        embeddings (np.ndarray): An array of embeddings to cluster.
        max_n_clusters (int): The maximum number of K values to try. Default is 10.
        iterations (int): The number of iterations to run for each K value. Default is 10.
        n_init (int): The number of initializations to perform for each K value. Default is 10.
        init_params (str): The method to initialize the model parameters. Default is 'kmeans'.
        covariance_type (str): The type of covariance to use. Default is 'full'.
        warm_start (bool): Whether to reuse the previous solution as the initialization for the next K value. Default is False.

        Returns:
        Tuple[pd.DataFrame, int]: A tuple of the results dataframe and the K value which maximizes the silhouette score.
        """
        max_n_clusters = self.settings.clustering['max_n_clusters']
        iterations = self.settings.clustering['iterations']
        n_init = self.settings.clustering['n_init']
        init_params = self.settings.clustering['init_params']
        covariance_type = self.settings.clustering['covariance_type']
        warm_start = self.settings.clustering['warm_start']

        logging.info("SOMoC will try to find the optimal K")

        temp = {i: [] for i in range(max_n_clusters+1)}  # pre-allocate the dictionary

        for n in tqdm(range(2, max_n_clusters+1), desc='Optimizing the number of clusters'):
            temp_sil = [None] * iterations # pre-allocate the list
            for x in range(iterations):
                gmm = GMM(n, n_init=n_init, init_params=init_params, covariance_type=covariance_type,
                        warm_start=warm_start, random_state=x, verbose=0).fit(self.embedding)
                labels = gmm.predict(self.embedding)
                temp_sil[x] = silhouette_score(
                    self.embedding, labels, metric='euclidean')
            temp[n] = [int(n),np.mean(temp_sil), np.std(temp_sil)]

        results = pd.DataFrame.from_dict(
            temp, orient='index', columns=['Clusters','Silhouette', 'sil_stdv']).dropna()
        results = results.astype({"Clusters": int})
        results_sorted = results.sort_values(['Silhouette'], ascending=False)
        K_loop = results_sorted.index[0]  # Get max Sil K
        
        return results, int(K_loop)

    def GMM_final(self, K: int= 3):
        
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
            data_clustered (pandas.DataFrame): The input data with an additional 'cluster' column.
        """
        n_init = self.settings.clustering['n_init']
        init_params = self.settings.clustering['init_params']
        covariance_type = self.settings.clustering['covariance_type']
        warm_start = self.settings.clustering['warm_start']
        random_state = self.settings.random_state

        logging.info(f'Running final clustering with K = {K}')

        GMM_final = GMM(K, n_init=n_init, init_params=init_params, warm_start=warm_start,
                        covariance_type=covariance_type, random_state=random_state, verbose=0)
        GMM_final.fit(self.embedding)
        labels_final = GMM_final.predict(self.embedding)

        if GMM_final.converged_:
            logging.info('GMM converged.')
        else:
            logging.warning('GMM did not converge. Please check you input configuration.')
        
        results_clustered = pd.DataFrame({'cluster': labels_final})

        logging.info('Calculating CVIs')
        results_real = calculate_CVIs(self.embedding, labels=labels_final, Random = False)
        results_random = calculate_CVIs(self.embedding, labels=None, num_clusters=K, Random = True)
        # concatenate DataFrames along columns axis
        results_CVIs = pd.concat([results_real, results_random], axis=1)
        
        results_CVIs.to_csv(f'results/{self.name}/{self.name}_CVIs.csv', index=True, header=True)

        return results_clustered, results_CVIs, GMM_final
        