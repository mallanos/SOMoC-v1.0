#!/usr/bin/env python
# coding: utf-8
"""
@author: Manu Llanos
"""
# SOMoC is a clustering methodology based on the combination of molecular fingerprinting, 
# dimensionality reduction by the Uniform Manifold Approximation and Projection (UMAP) algorithm 
# and clustering with the Gaussian Mixture Model (GMM) algorithm.

##################################### Import packages ####################################
###########################################################################################
# The following packages are required: SKlearn, RDKit, UMAP, Molvs, validclust and Plotly.
# Please, meake sure you have them installed before running the program.
import pandas as pd
from array import array
import time
import os
from typing import List, Tuple, Union, Optional, Dict, Any
from datetime import date
from pathlib import Path
import numpy as np
import json
import logging
from tqdm import tqdm
from datetime import datetime
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse

from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score, pairwise_distances
from validclust import dunn
import umap
from rdkit import Chem
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from molvs import Standardizer

#################################### Helper functions #####################################
###########################################################################################
class Settings:
    def __init__(self, config_file: str) -> None:
        self.load_settings(config_file)

    def load_settings(self, config_file: str) -> None:
        # Load settings from the JSON file
        try:
            with open(config_file) as f:
                config = json.load(f)
        except FileNotFoundError:
            logging.warning(f"Error: {config_file} not found.")
            return
        except json.JSONDecodeError:
            logging.warning(f"Error: Invalid JSON syntax in {config_file}.")
            return

        # Set default values for the settings
        defaults: Dict[str, Any] = {
            'fingerprint_type': 'EState',
            'standardize_molec': False,
            'umap': {
                'n_neighbors': 15,
                'min_dist': 0.1,
                'n_components': 2,
                'metric': 'euclidean',
                'init': 'spectral'
            },
            'gmm': {
                'max_n_clusters': 10,
                'n_init': 10,
                'iterations': 10,
                'init_params': 'kmeans',
                'covariance_type': 'full',
                'warm_start' : False
            },
            'random_state': 10,
            'max_K': 5,
            'optimal_K': False
        }

        # Update the defaults with any values from the config file
        settings_dict: Dict[str, Any] = {}
        for k, v in defaults.items():
            if k in config:
                if isinstance(v, dict):
                    settings_dict[k] = {**v, **config[k]}
                else:
                    settings_dict[k] = config[k]
            else:
                settings_dict[k] = v

        # Set the settings as instance variables
        for k, v in settings_dict.items():
            setattr(self, k, v)

    def save_settings(self, name: str, df: pd.DataFrame = None) -> None:

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Create a dictionary with the settings
        settings_dict = {}
        settings_dict["timestamp"] = timestamp

        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                settings_dict[k] = v

        # Add the DataFrame to the dictionary if it is provided
        if df is not None:
            df = df.iloc[:,:1]
            df.columns = ['CVIs']
            for k, v in df.to_dict().items():
                settings_dict[k] = v

        # Write the dictionary to a JSON file
        with open(f'results/{name}/{name}_{timestamp}.json', 'w') as f:
            json.dump(settings_dict, f, indent=4)

def Get_name(archive):
    """strip path and extension to return the name of a file"""
    # name = os.path.splitext(os.path.basename(archive))[0]
    return os.path.basename(archive).split('.')[0]

def Make_dir(dirName: str):
    """Create a directory and not fail if it already exist"""
    try:
        os.makedirs(dirName)
    except FileExistsError:
        pass

def Get_input_data(input_file=None, test_file="test/focal_adhesion.csv"):
    """Get data from user input or use test dataset"""
    
    file_path = input_file or test_file
    if not os.path.exists(file_path):
        raise ValueError(f'File not found: {file_path}')
        
    with open(file_path) as f:
        data = pd.read_csv(f, delimiter=',', header=None)

    name = Get_name(file_path)
    logging.info(f'Loading {name} dataset')
    logging.info(f'{len(data)} molecules loaded..')

    return data, name

def Standardize_molecules(data: pd.DataFrame) -> pd.DataFrame:
    """Standardize molecules using the MolVS package https://molvs.readthedocs.io/en/latest/.

    Parameters
    ----------
    data : pandas.DataFrame
        A DataFrame containing a column of SMILES strings to be standardized in the first column.

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with an additional column of standardized RDKit molecules.

    """
    data_ = data.copy()

    list_of_smiles = data_.iloc[:, 0]
    standardized_mols = [np.nan] * len(list_of_smiles)

    for i, smiles in tqdm(enumerate(list_of_smiles), total=len(list_of_smiles), desc='Standardizing molecules'):
        try:
            mol = Chem.MolFromSmiles(smiles)
            standardized_mol = Standardizer().standardize(mol)
            # standardized_mol = Standardizer().super_parent(mol)
            standardized_mols[i] = standardized_mol
        except Exception as e:
            logging.warning(f"Failed to process molecule {i+1}: ({e})")


    data_['mol'] = standardized_mols
    data_ = data_.dropna() # Drop failed molecules

    num_processed_mols = len(standardized_mols)
    num_failed_mols = len([m for m in standardized_mols if m is np.nan])

    logging.info(f'{num_processed_mols-num_failed_mols} molecules processed')

    if num_failed_mols:
        logging.warning(f"{num_failed_mols} molecules failed to be standardized")

    return data_

def Fingerprints_calculator(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate EState molecular fingerprints using the RDKit package.

    Parameters:
        data (pd.DataFrame): A pandas DataFrame containing a FIRST column of SMILES strings.

    Returns:
        np.ndarray: The calculated EState molecular fingerprints as a NumPy array.

    Raises:
        ValueError: If the input DataFrame does not contain a column of SMILES strings.
        RuntimeError: If there is a problem with fingerprint calculation of some SMILES.
    """

    if 'smiles' in data.columns:
        smiles_list = data['smiles']
    else:
        smiles_list = data.iloc[:,0]
    
    logging.info("ENCODING")
    logging.info("Calculating EState molecular fingerprints...")

    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    fps = [None] * len(mols)
    for i, mol in tqdm(enumerate(mols), total=len(mols), desc='Calculating fingerprints'):
        try:
            fp = FingerprintMol(mol)[0]  # EState fingerprint
            fps[i] = fp
        except Exception as e:
            logging.warning(f"Failed fingerprint calculation for molecule {i+1}: ({e})")

    fingerprints = np.stack(fps, axis=0)

    return fingerprints

def UMAP_reduction(X: np.ndarray, settings) -> Tuple[np.ndarray, int]:

    """
    Reduce feature space using the UMAP algorithm.

    Parameters:
        X (np.ndarray): Input data as a NumPy array.
        n_neighbors (int): Number of neighbors to use for the UMAP algorithm.
        min_dist (float): Minimum distance threshold for the UMAP algorithm.
        metric (str): Distance metric to use for the UMAP algorithm.
        random_state (int): Random seed for the UMAP algorithm.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing the reduced feature space (as a NumPy array) and the number of components
        used for the reduction.

    Raises:
        ValueError: If the input is not a NumPy array or the number of neighbors is greater than the length of the input data.
    """
    n_neighbors = settings.umap['n_neighbors']
    min_dist = settings.umap['min_dist']
    init = settings.umap['init']
    metric = settings.umap['metric']
    random_state = settings.random_state

    logging.info('REDUCING')

    if not isinstance(X, np.ndarray):
        logging.error("Input must be a NumPy array")
        raise ValueError("Input must be a NumPy array")
        
    if n_neighbors >= len(X):
        logging.error("The number of neighbors must be smaller than the number of molecules to cluster")
        raise ValueError("The number of neighbors must be smaller than the number of molecules to cluster")

    n_components = int(np.ceil(np.log(len(X))/np.log(4)))
    n_neighbors = max(10, int(np.sqrt(X.shape[0])))

    logging.info(f'Running UMAP with {n_components} components and {n_neighbors} neighbors.')

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric,
                        init=init, random_state=random_state).fit(X)
    embedding = reducer.transform(X)

    return embedding, n_components

def Calculate_CVIs(embeddings: np.ndarray, labels: List, Random: bool = True, num_iterations: int = 500, num_clusters: int = 3):

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
            random_clusters = np.random.randint(num_clusters, size=len(embeddings))
            silhouette_random = silhouette_score(embeddings, random_clusters)
            SILs[i] = silhouette_random
            db_random = davies_bouldin_score(embeddings, random_clusters)
            DBs[i] = db_random
            ch_random = calinski_harabasz_score(embeddings, random_clusters)
            CHs[i] = ch_random
            dist_dunn = pairwise_distances(embeddings)
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
        assert len(embeddings) == len(labels), "Length of embeddings and labels must match"
        assert embeddings.ndim == 2, "Embeddings must be a 2D array"

        results['SOMoC'] = {
            'silhouette': silhouette_score(embeddings, labels, metric='cosine').round(4),
            'davies_bouldin': davies_bouldin_score(embeddings, labels).round(4),
            'calinski_harabasz': calinski_harabasz_score(embeddings, labels).round(4),
            'dunn': dunn(pairwise_distances(embeddings), labels).round(4)
        }
        results = pd.DataFrame.from_dict(results)

    return results

def GMM_clustering_loop(embeddings: np.ndarray, settings) -> Tuple[pd.DataFrame, int]:
    """
    Runs GMM clustering for a range of K values and returns the K value which maximizes the silhouette score.

    Parameters:
    embeddings (np.ndarray): An array of embeddings to cluster.
    max_K (int): The maximum number of K values to try. Default is 10.
    iterations (int): The number of iterations to run for each K value. Default is 10.
    n_init (int): The number of initializations to perform for each K value. Default is 10.
    init_params (str): The method to initialize the model parameters. Default is 'kmeans'.
    covariance_type (str): The type of covariance to use. Default is 'full'.
    warm_start (bool): Whether to reuse the previous solution as the initialization for the next K value. Default is False.

    Returns:
    Tuple[pd.DataFrame, int]: A tuple of the results dataframe and the K value which maximizes the silhouette score.
    """
    max_K = settings.max_K
    iterations = settings.gmm['iterations']
    n_init = settings.gmm['n_init']
    init_params = settings.gmm['init_params']
    covariance_type = settings.gmm['covariance_type']
    warm_start = settings.gmm['warm_start']

    logging.info("SOMoC will try to find the optimal K")

    temp = {i: [] for i in range(max_K+1)}  # pre-allocate the dictionary

    for n in tqdm(range(2, max_K+1), desc='Optimizing the number of cluters'):
        temp_sil = [None] * iterations # pre-allocate the list
        for x in range(iterations):
            gmm = GMM(n, n_init=n_init, init_params=init_params, covariance_type=covariance_type,
                      warm_start=warm_start, random_state=x, verbose=0).fit(embeddings)
            labels = gmm.predict(embeddings)
            temp_sil[x] = silhouette_score(
                embeddings, labels, metric='cosine')
        temp[n] = [int(n),np.mean(temp_sil), np.std(temp_sil)]

    results = pd.DataFrame.from_dict(
        temp, orient='index', columns=['Clusters','Silhouette', 'sil_stdv']).dropna()
    results = results.astype({"Clusters": int})
    results_sorted = results.sort_values(['Silhouette'], ascending=False)
    K_loop = results_sorted.index[0]  # Get max Sil K
    
    return results, int(K_loop)

def GMM_clustering_final(embeddings: np.array, settings, K: int= 3):
    
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
    n_init = settings.gmm['n_init']
    init_params = settings.gmm['init_params']
    covariance_type = settings.gmm['covariance_type']
    warm_start = settings.gmm['warm_start']
    random_state = settings.random_state

    logging.info(f'Running final clustering with K = {K}')

    GMM_final = GMM(K, n_init=n_init, init_params=init_params, warm_start=warm_start,
                    covariance_type=covariance_type, random_state=random_state, verbose=0)
    GMM_final.fit(embeddings)
    labels_final = GMM_final.predict(embeddings)

    if GMM_final.converged_:
        logging.info('GMM converged.')
    else:
        logging.warning('GMM did not converge. Please check you input configuration.')
       
    logging.info('Calculating CVIs')
    results_real = Calculate_CVIs(embeddings, labels=labels_final, Random = False)
    # create DataFrame from real results
    df_real = pd.DataFrame.from_dict(results_real, orient='columns')
    results_random = Calculate_CVIs(embeddings, labels=None, num_clusters=K, Random = True)
    # create DataFrame from random results
    df_random = pd.DataFrame.from_dict(results_random, orient='columns')
    # concatenate DataFrames along columns axis
    results_CVIs = pd.concat([df_real, df_random], axis=1)

    cluster_final = pd.DataFrame({'cluster': labels_final}, index=data.index)
    results_clustered = data.join(cluster_final)

    if 'mol' in results_clustered.columns:  # Check if mol column from standardization is present
        try:
            results_clustered['SMILES_standardized'] = results_clustered['mol'].apply(
                lambda x: Chem.MolToSmiles(x))
            results_clustered.drop(['mol'], axis=1, inplace=True)
        except Exception as e:
            logging.warning('Something went wrong converting standardized molecules back to SMILES code..: {e}')
            pass

    results_clustered.to_csv(f'results/{name}/{name}_Clustered.csv', index=True, header=True)
    results_CVIs.to_csv(f'results/{name}/{name}_CVIs.csv', index=True, header=True)

    return results_clustered, results_CVIs, GMM_final

def Elbow_plot(name, results_loop, optimal_K):
    """Draw the elbow plot of SIL score vs. K.

    Parameters:
    name (str): The name of the dataset or experiment.
    results_loop (pd.DataFrame): A DataFrame containing the Silhouette score and standard deviation for each number of clusters tested.
    optimal_K (int): The optimal number of clusters selected.
    """
    optimal_score = results_loop.loc[results_loop['Clusters'] == optimal_K, 'Silhouette'].values[0]
    optimal_stdv = results_loop.loc[results_loop['Clusters'] == optimal_K, 'sil_stdv'].values[0]
    optimal_label = f'Optimal K={optimal_K}\nSilhouette={optimal_score:.3f}±{optimal_stdv:.3f}'
    
    fig, ax1 = plt.subplots(figsize=(14, 6))

    sil = sns.lineplot(data=results_loop, x='Clusters', y="Silhouette", color='b', ci=None, estimator=np.median,
                       ax=ax1)
    
    sil_error = ax1.errorbar(x=results_loop['Clusters'], y=results_loop['Silhouette'], yerr=results_loop['sil_stdv'],
                             fmt='none', ecolor='b', capsize=4, elinewidth=1.5)
    
    plt.axvline(x=optimal_K, color='r', linestyle='--', label=optimal_label, linewidth=1.5)

    plt.legend(fancybox=True, framealpha=0.5, fontsize='15', loc='best', title_fontsize='30')
    plt.tick_params(labelsize=12)
    plt.title(f"Elbow plot - Sil vs. K", fontsize=20)
    plt.xlabel("Number of clusters (K)", fontsize=15)
    plt.ylabel("Silhouette", fontsize=15)
    plt.tight_layout()
    plt.savefig(f'results/{name}/{name}_Elbowplot.png')

def Distribution_plot(model, embedding):
    """Plot individual SIL scores each sample, agregated by cluster """
    
    labels_final = model.predict(embedding)
    sil_bysample = silhouette_samples(embedding, labels_final, metric='cosine')
    sil_svg = round(float(silhouette_score(embedding, labels_final, metric='cosine')),3)
    
    y_lower = 10
    y_tick_pos_ = []

    fig, (ax) = plt.subplots(1)
    fig.set_size_inches(8, 12)

    for i in range(model.n_components):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_SILs = sil_bysample[labels_final == i]
        ith_cluster_SILs.sort()

        size_cluster_i = ith_cluster_SILs.shape[0]

        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / model.n_components)
        # color = sns.color_palette("rocket", n_colors=model.n_components)#, as_cmap=True)

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_SILs,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        y_tick_pos_.append(y_lower + 0.5 * size_cluster_i)

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    # ax.axvline(x=sil_svg, color="red", linestyle="--", label=f"Avg SIL score: {sil_svg}")
    ax.axvline(x=sil_svg, color="red", linestyle="--")

    ax.set_title(f"Silhouette Plot for {model.n_components} clusters", fontsize=20)

    l_xlim = max(-1, min(-0.1, round(min(sil_bysample) - 0.1, 1)))
    u_xlim = min(1, round(max(sil_bysample) + 0.1, 1))
    ax.set_xlim([l_xlim, u_xlim])
    ax.set_ylim([0, embedding.shape[0] + (model.n_components + 1) * 10])
    ax.set_xlabel("Silhouette coefficient values", fontsize=20)
    ax.set_ylabel("Cluster label", fontsize=20)
    ax.set_yticks(y_tick_pos_)
    ax.set_yticklabels([str(i) for i in range(model.n_components)]) #,fontdict={'fontsize':15}
    # change the fontsize
    ax.tick_params(axis='both', labelsize=15)   
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    # ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f'results/{name}/SIL_bycluster.png')

####################################### SOMoC main ########################################
###########################################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='SOMoC', description='SOMoC is a clustering methodology based on the combination of molecular fingerprinting, dimensionality reduction by the Uniform Manifold Approximation and Projection (UMAP) algorithm and clustering with the Gaussian Mixture Model (GMM) algorithm.')
    parser.add_argument('-c','--config', help='Path to JSON config file', required=True)
    parser.add_argument('-i','--input', help='Input file is a .CSV file with one molecule per line in SMILES format. Molecules must be in the first column.', required=False)
    parser.add_argument('-l','--log-level', help='Choose the logging level to show', choices=['debug', 'info', 'warning', 'error', 'critical'], default='info', required=False)

    args = parser.parse_args()

    logging.basicConfig(
    level=args.log_level.upper(),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"SOMoC.log", mode='w'),
        logging.StreamHandler()
    ]
)
    start_time = time.monotonic()

    settings = Settings(args.config)

    print('='*100)

    # Get input data
    data_raw, name = Get_input_data(args.input)
    
    # Create output dir
    Make_dir(f'results/{name}')

    # Standardize molecules
    if settings.standardize_molec == True:
        data = Standardize_molecules(data_raw)
    else:
        logging.info('Skipping molecules standardization.')
        data = data_raw

    print('='*100)
    # Calculate Fingerprints
    X = Fingerprints_calculator(data)

    print('='*100)
    # Reduce feature space with UMAP
    embedding, n_components = UMAP_reduction(X, settings)
   
    print('='*100)
    if settings.optimal_K is not False:
        # Run the clustering and calculate all CVIs
        results_clustered, results_CVIs, results_model = GMM_clustering_final(embedding, settings, K=settings.optimal_K)
    else:
        # If optimal_K is not set, run the GMM clustering loop to get K
        results_loop, optimal_K = GMM_clustering_loop(embedding, settings)
        results_clustered, results_CVIs, results_model = GMM_clustering_final(embedding, settings, K=optimal_K)
        settings.optimal_K = optimal_K
        Elbow_plot(name, results_loop, optimal_K)
       
    print('='*100)
    logging.info('Generating plots.')
    Distribution_plot(results_model, embedding)

    logging.info('Saving run settings to JSON file.')
    # Write the settings JSON file
    settings.save_settings(name, df=results_CVIs)

    logging.info('ALL DONE !')
    logging.info(f'SOMoC run took {time.monotonic() - start_time:.3f} seconds')

    print('='*100)
    print("CVIs results")
    print(results_CVIs)
    print('='*100)
    



