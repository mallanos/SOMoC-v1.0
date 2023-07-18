import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from typing import List, Tuple, Union, Optional, Dict, Any

from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, pairwise_distances
from sklearn.metrics.pairwise import distance_metrics
from validclust import dunn, cop
from scipy.stats import multivariate_normal

def calculate_all_CVIs(embedding: np.ndarray, sil_metric: str, labels: List, Random: bool = True, num_iterations: int = 5, num_clusters: int = 3):
    """
    Calculate all CVIs for real or random clustering

    Parameters:
    embedding (np.ndarray): An array of embeddings to cluster.
    labels (List): List of cluster labels.

    Returns: A nested dictionary containing all CVIs
    """
    results = {}
    distance_matrix = pairwise_distances(embedding)

    if Random:
        SILs, DBs, CHs, DUNNs, COPs, BICs = np.zeros(num_iterations), np.zeros(num_iterations), np.zeros(num_iterations), np.zeros(num_iterations), np.zeros(num_iterations), np.zeros(num_iterations)
        for i in range(num_iterations):
            np.random.seed(i)
            random_clusters = np.random.randint(num_clusters, size=len(embedding))
            SILs[i] = silhouette_score(embedding, random_clusters, metric=sil_metric)
            DBs[i] = davies_bouldin_score(embedding, random_clusters)
            CHs[i] = calinski_harabasz_score(embedding, random_clusters)
            DUNNs[i] = dunn(distance_matrix, random_clusters)
            COPs[i] = cop(embedding, distance_matrix, random_clusters) # Gurrutxaga et al. (2010)
            BICs[i] = calculate_BIC(embedding, random_clusters) # Calculate BIC

        random_means = np.mean([SILs, DBs, CHs, DUNNs, COPs, BICs], axis=1)
        random_sds = np.std([SILs, DBs, CHs, DUNNs, COPs, BICs], axis=1)

        results = pd.DataFrame({'Random_mean': random_means, 'Random_std': random_sds}, index=['silhouette', 'davies_bouldin', 'calinski_harabasz', 'dunn', 'COP', 'BIC'])

    else:
        assert len(embedding) == len(labels), "Length of embeddings and labels must match"
        assert embedding.ndim == 2, "Embeddings must be a 2D array"

        SOMoC = {'silhouette': silhouette_score(embedding, labels, metric=sil_metric),
                 'davies_bouldin': davies_bouldin_score(embedding, labels),
                 'calinski_harabasz': calinski_harabasz_score(embedding, labels),
                 'dunn': dunn(distance_matrix, labels),
                 'COP': cop(embedding, distance_matrix, labels), # Gurrutxaga et al. (2010)
                 'BIC': calculate_BIC(embedding, labels)}

        results['SOMoC'] = SOMoC
        results = pd.DataFrame.from_dict(results)

    return results.round(3)

def calculate_individual_CVI(embedding: np.ndarray, labels: List, optimize_cvi: str='silhouette', sil_metric: str='cosine'):
    """
    Calculate individual CVIs based on user choice.

    Parameters:
    embedding (np.ndarray): An array of embeddings to cluster.
    sil_metric (str): The distance metric to use for silhouette score.
    labels (List): List of cluster labels.

    """

    if optimize_cvi == 'silhouette':
        cvi_value = silhouette_score(embedding, labels, metric=sil_metric)
    elif optimize_cvi == 'davies_bouldin':   
        cvi_value = davies_bouldin_score(embedding, labels)
    elif optimize_cvi == 'calinski_harabasz':   
        cvi_value = calinski_harabasz_score(embedding, labels)
    elif optimize_cvi == 'dunn':   
        distance_matrix = pairwise_distances(embedding)
        cvi_value = dunn(distance_matrix, labels)
    elif optimize_cvi == 'cop':   
        distance_matrix = pairwise_distances(embedding)
        cvi_value = cop(embedding, distance_matrix, labels)
    elif optimize_cvi == 'bic':   
        cvi_value = calculate_BIC(embedding, labels)      
    else:
        logging.error(f"The provided CVI '{optimize_cvi}' is not supported. Check your configuration.")

    return round(cvi_value, 3)

def calculate_BIC(embedding: np.ndarray, labels: List):
    """
    Calculate the Bayesian Information Criterion (BIC) for clustering.
    http://www.modelselection.org/bic/

    Parameters:
    embedding (np.ndarray): An array of embeddings to cluster.
    labels (List): List of cluster labels.

    Returns: The BIC value.
    """
    num_clusters = len(set(labels))
    num_samples, num_features = embedding.shape

    log_likelihood = calculate_log_likelihood(embedding, labels, num_clusters)

    num_parameters = num_clusters * num_features + (num_clusters - 1)  # cluster means + cluster covariances

    bic = -2 * log_likelihood + num_parameters * np.log(num_samples)

    return bic

def calculate_log_likelihood(embedding: np.ndarray, labels: List, num_clusters: int):
    """
    Calculate the maximized log-likelihood for clustering.

    Parameters:
    embedding (np.ndarray): An array of embeddings to cluster.
    labels (List): List of cluster labels.
    num_clusters (int): Number of clusters.

    Returns: The maximized log-likelihood.
    """
    log_likelihood = 0.0

    for cluster in range(num_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        cluster_samples = embedding[cluster_indices]

        cluster_mean = np.mean(cluster_samples, axis=0)
        cluster_cov = np.cov(cluster_samples, rowvar=False)

        mvn = multivariate_normal(mean=cluster_mean, cov=cluster_cov, allow_singular=True)
        cluster_log_likelihood = mvn.logpdf(cluster_samples)
        log_likelihood += np.sum(cluster_log_likelihood)

    return log_likelihood

class Clustering():

    def __init__(self, dataset_name: str, embedding: np.ndarray, settings) -> None:
        self.dataset_name = dataset_name
        self.embedding = embedding
        self.settings = settings

        allowed_metrics = list(distance_metrics().keys()) # Allowed metrics for Sil calculation
        if self.settings.reducing['metric'] not in allowed_metrics:
            logging.warning(f"{self.settings.reducing['metric']} is not available in Sklearn pairwise distances for Silhouette calculation, defaulting to Cosine!")
            self.sil_metric = 'cosine'
        else:
            self.sil_metric = self.settings.reducing['metric']
        
        self.optimize_cvi = self.settings.clustering['optimize_cvi']
        self.random_state = self.settings.random_state

    def estimate_optimal_clusters(self, temp: Dict[int, float], method: str='max') -> int:
        """
        Estimates the optimal number of clusters using the elbow method, maximum silhouette score, or second order derivative.

        Parameters:
        temp (Dict[int, float]): A dictionary containing the silhouette scores for each clustering method and each number of clusters.
        method (str): The method to use for estimating the optimal number of clusters. Can be 'max' (default), 'elbow', or 'second_order'.

        Returns:
        The estimated optimal number of clusters.
        """
        if method == 'elbow':
            logging.info('Finding the optimal number of clusters using the elbow method...')

            # Calculate differences between successive metric scores
            diffs = {}
            for k in range(2, len(temp)):
                diffs[k] = np.diff([temp[i] for i in range(2, k+1)])

            # Calculate ratio of differences and identify elbow point
            elbow_points = {}
            for k, v in diffs.items():
                if len(v) > 1:
                    ratios = v[1:] / v[:-1]
                    elbow_points[k] = np.argmax(ratios) + 2
                else:
                    elbow_points[k] = k

            # Take average of elbow points across methods
            elbow_idx = int(round(np.mean(list(elbow_points.values()))))

        elif method == 'max':
            logging.info('Finding the optimal number of clusters using maximum method...')
            if self.optimize_cvi in ['silhouette', 'dunn', 'calinski_harabasz']:
                elbow_idx = max(temp.items(), key=lambda x: x[1][1])[0]
            else:
                elbow_idx = min(temp.items(), key=lambda x: x[1][1])[0]

        # TODO NOT WORKING PROPERLY
        elif method == 'second_order':
            logging.info('Finding the optimal number of clusters using second order derivative...')

            # Calculate differences between successive metric scores
            diffs = {}
            for k in range(2, len(temp)):
                diffs[k] = np.diff([temp[i] for i in range(2, k+1)])

            # Calculate second order differences and identify elbow point
            second_diffs = {}
            for k, v in diffs.items():
                if len(v) > 2:
                    second_diffs[k] = np.diff(v, n=2)
                else:
                    second_diffs[k] = [0]

            elbow_idx = max(second_diffs, key=second_diffs.get)

        logging.info(f"The estimated optimal number of clusters is: {elbow_idx}")
        
        return elbow_idx
    
    def cluster(self):
        self.clustering_method = self.settings.clustering['clustering_method']
        self.max_n_clusters = self.settings.clustering['max_n_clusters']
        self.iterations = self.settings.clustering['iterations']
        self.gmm_init = self.settings.clustering['gmm_init']
        self.gmm_init_params = self.settings.clustering['gmm_init_params']
        self.gmm_covariance_type = self.settings.clustering['gmm_covariance_type']
        self.gmm_warm_start = self.settings.clustering['gmm_warm_start']

        K = self.settings.optimal_K

        # Determine the optimal K using the clustering loop if K is not specified
        if K == False:
            logging.info(f"Optimizing the {self.optimize_cvi} score...")
            if self.clustering_method == 'kmeans':
                results_loop, K_loop = self._kmeans_loop()
            elif self.clustering_method == 'gmm':
                results_loop, K_loop = self._gmm_loop()
            else:
                logging.warning("Invalid clustering method. Defaulting to 'gmm'.")
                self.clustering_method = 'gmm'
                results_loop, K_loop = self._gmm_loop()
            K = K_loop
        else:
            results_loop = None
            logging.info(f"Using user-specified K = {K}")

        # Perform final clustering
        logging.info(f'Running final clustering with K = {K}...')

        if self.clustering_method == 'gmm':
            data_clustered, results_CVIs, model = self._gmm_final(K)
        elif self.clustering_method == 'kmeans':
            data_clustered, results_CVIs, model = self._kmeans_final(K)
        # else:
        #     self.clustering_method = 'gmm'
        #     data_clustered, results_CVIs, model = self._gmm_final(K)
            
        results_CVIs.to_csv(f'results/{self.dataset_name}/{self.dataset_name}_CVIs.csv', index=True, header=True)

        return K, results_loop, data_clustered, results_CVIs, model
    
    def _kmeans_loop(self) -> Tuple[pd.DataFrame, int]:
        """
        Runs KMeans clustering for a range of K values and returns the K value which maximizes the silhouette score.
        """
        temp = {i: [] for i in range(2, self.max_n_clusters+1)}  # pre-allocate the dictionary

        for n in tqdm(range(2, self.max_n_clusters+1), desc='Optimizing the number of clusters'):
            temp_cvi = [None] * self.iterations  # pre-allocate the list
            for x in range(self.iterations):
                kmeans = KMeans(n_clusters=n,init='k-means++', n_init='auto', random_state=x).fit(self.embedding)
                labels = kmeans.predict(self.embedding)
                temp_cvi[x] = calculate_individual_CVI(self.embedding, labels, self.optimize_cvi, self.sil_metric)
            
            temp[n] = [int(n), np.mean(temp_cvi), np.std(temp_cvi)]

        results_loop = pd.DataFrame.from_dict(temp, orient='index', columns=['Clusters', f'{self.optimize_cvi}-mean', f'{self.optimize_cvi}-stdv']).dropna()
        optimal_n_clusters = self.estimate_optimal_clusters(temp, method='max')
        
        return results_loop, int(optimal_n_clusters)
    
    def _gmm_loop(self) -> Tuple[pd.DataFrame, int]:
        """
        Runs GMM clustering for a range of K values and returns the K value which maximizes the specified CVI metric.
        """
        temp = {i: [] for i in range(2, self.max_n_clusters+1)}  # pre-allocate the dictionary

        for n in tqdm(range(2, self.max_n_clusters+1), desc='Optimizing the number of clusters'):
            temp_cvi = [None] * self.iterations  # pre-allocate the list
            for x in range(self.iterations):
                gmm = GMM(n, n_init=self.gmm_init, init_params=self.gmm_init_params,
                        covariance_type=self.gmm_covariance_type, warm_start=self.gmm_warm_start,
                        random_state=x, verbose=0).fit(self.embedding)
                labels = gmm.predict(self.embedding)
                temp_cvi[x] = calculate_individual_CVI(self.embedding, labels, self.optimize_cvi, self.sil_metric)

            temp[n] = [int(n), np.mean(temp_cvi), np.std(temp_cvi)]

        results_loop = pd.DataFrame.from_dict(temp, orient='index', columns=['Clusters', f'{self.optimize_cvi}-mean', f'{self.optimize_cvi}-stdv']).dropna()
        optimal_n_clusters = self.estimate_optimal_clusters(temp, method='max')

        return results_loop, int(optimal_n_clusters)

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

        GMM_final = GMM(K, n_init=self.gmm_init, init_params=self.gmm_init_params, warm_start=self.gmm_warm_start,
                        covariance_type=self.gmm_covariance_type, random_state=self.random_state, verbose=0)
        GMM_final.fit(self.embedding)
        labels = GMM_final.predict(self.embedding)

        if GMM_final.converged_:
            logging.info('GMM converged.')
        else:
            logging.warning('GMM did not converge. Please check you input configuration.')
        
        logging.info('Calculating CVIs')
        results_real = calculate_all_CVIs(self.embedding, self.sil_metric,labels=labels, Random = False)
        results_random = calculate_all_CVIs(self.embedding, self.sil_metric, labels=None, num_clusters=K, Random = True)
        # concatenate DataFrames along columns axis
        results_CVIs = pd.concat([results_real, results_random], axis=1)
        results_CVIs.to_csv(f'results/{self.dataset_name}/{self.dataset_name}_CVIs.csv', index=True, header=True)

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

        KMeans_final = KMeans(n_clusters=K, init='k-means++', n_init='auto', random_state=self.random_state)
        KMeans_final.fit(self.embedding)
        labels = KMeans_final.predict(self.embedding)

        if KMeans_final.n_iter_ < KMeans_final.max_iter:
            logging.info('K-Means converged.')
        else:
            logging.warning('K-Means did not converge. Please check your input configuration.')

        logging.info('Calculating CVIs')
        
        results_real = calculate_all_CVIs(self.embedding, self.sil_metric,labels=labels, Random = False)
        results_random = calculate_all_CVIs(self.embedding, self.sil_metric, labels=None, num_clusters=K, Random = True)
        # concatenate DataFrames along columns axis
        results_CVIs = pd.concat([results_real, results_random], axis=1)
        results_CVIs.to_csv(f'results/{self.dataset_name}/{self.dataset_name}_CVIs.csv', index=True, header=True)

        return labels, results_CVIs, KMeans_final
