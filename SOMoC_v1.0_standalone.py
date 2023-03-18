#!/usr/bin/env python
# coding: utf-8
"""
@author: LIDeB UNLP
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
from typing import List, Tuple, Union
from datetime import date
from pathlib import Path
import numpy as np
import json
import logging
from tqdm import tqdm
from datetime import datetime

import plotly.express as plx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, pairwise_distances
from validclust import dunn
import umap
from rdkit import Chem
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from molvs import Standardizer

###################################### CONFIGURATION ######################################
###########################################################################################

# Input file is a .CSV file with one molecule per line in SMILES format.
# Molecules must be in the first column.
input_file = None

# If you already know the number of clusters in your data, then set K to this value.
# Otherwise, set K=None to let SOMoC approaximate it by running a range of K values.
# Alternatively, you can use the generated elbowplot to find K yourself and re-reun SOMoC with a fixed K.
K = None            # Optimal number of clusters K

# Perform molecule standardization using the MolVS package
smiles_standardization = False       

### UMAP parameters ###
n_neighbors = 10    # The size of local neighborhood used for manifold approximation. Larger values result in more global views of the manifold, while smaller values result in more local data being preserved.
min_dist = 0.0      # The effective minimum distance between embedded points. Smaller values will result in a more clustered/clumped embedding where nearby points on the manifold are drawn closer together, while larger values will result on a more even dispersal of points.
random_state = 10   # Use a fixed seed for reproducibility.
metric = "jaccard"  # The metric to use to compute distances in high dimensional space.

### GMM parameters ###
max_K = 25                      # Max number of clusters to cosidering during the GMM loop
Kers = np.arange(2, max_K+1, 1) # Generate the range of K values to explore
iterations = 10                  # Iterations of GMM to run for each K
n_init = 5                      # Number of initializations on each GMM run, then just keep the best one.
init_params = 'kmeans'          # How to initialize. Can be random or K-means
covariance_type = 'full'        # Type of covariance to consider: "spherical", "diag", "tied", "full"
warm_start = False

#################################### Helper functions #####################################
###########################################################################################

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

def UMAP_reduction(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1,
                   metric: str = 'jaccard', random_state: int = 42) -> Tuple[np.ndarray, int]:

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
                        random_state=random_state).fit(X)
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

def GMM_clustering_loop(embeddings: np.ndarray, max_K: int = 10, iterations: int = 2, n_init: int = 2, init_params: str = 'kmeans', covariance_type: str = 'full', warm_start: bool = False) -> Tuple[pd.DataFrame, int]:
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
    logging.info("SOMoC will try to find the optimal K")

    temp = {i: [] for i in range(max_K)}  # pre-allocate the dictionary

    for n in tqdm(range(2, max_K), desc='Optimizing the number of cluters'):
        temp_sil = [None] * iterations # pre-allocate the list
        for x in range(iterations):
            gmm = GMM(n, n_init=n_init, init_params=init_params, covariance_type=covariance_type,
                      warm_start=warm_start, random_state=x, verbose=0).fit(embeddings)
            labels = gmm.predict(embeddings)
            temp_sil[x] = silhouette_score(
                embeddings, labels, metric='cosine')
        temp[n] = [n,np.mean(temp_sil), np.std(temp_sil)]

    results = pd.DataFrame.from_dict(
        temp, orient='index', columns=['Clusters','Silhouette', 'sil_stdv'])
    results_sorted = results.sort_values(['Silhouette'], ascending=False)
    K_loop = results_sorted.index[0]  # Get max Sil K
    
    return results, int(K_loop)

def GMM_clustering_final(embeddings: np.array, K: int=3, n_init: int=10, init_params: str ='kmeans', warm_start: bool=False, covariance_type:str='full', random_state=None):
    
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

    # logging.info("CLUSTERING")
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

    results_clustered.to_csv(f'results/{name}/{name}_Clustered_SOMoC.csv', index=True, header=True)
    results_CVIs.to_csv(f'results/{name}/{name}_Validation_SOMoC.csv', index=True, header=True)

    return results_clustered, results_CVIs

def Elbow_plot(results):
    """Draw the elbow plot of SIL score vs. K"""

    fig = make_subplots(specs=[[{"secondary_y": False}]])

    fig.add_trace(go.Scatter(x=results['Clusters'], y=results['Silhouette'],
                             mode='lines+markers', name='Silhouette',
                             error_y=dict(type='data', symmetric=True, array=results["sil_stdv"]),
                             hovertemplate="Clusters = %{x}<br>Silhouette = %{y}"),
                  secondary_y=False)

    fig.update_layout(title="Silhouette vs. K", title_x=0.5,
                      title_font=dict(size=28, family='Calibri', color='black'),
                      legend_title_text = "Metric",
                      legend_title_font = dict(size=18, family='Calibri', color='black'),
                      legend_font = dict(size=15, family='Calibri', color='black'))
    fig.update_xaxes(title_text='Number of clusters (K)', range=[2 - 0.5, max_K + 0.5],
                     tickfont=dict(family='Arial', size=16, color='black'),
                     title_font=dict(size=25, family='Calibri', color='black'))
    fig.update_yaxes(title_text='Silhouette score',
                     tickfont=dict(family='Arial', size=16, color='black'),
                     title_font=dict(size=25, family='Calibri', color='black'), secondary_y=False)

    fig.update_layout(margin=dict(t=60, r=20, b=20, l=20), autosize=True)

    fig.write_html(f'results/{name}/{name}_elbowplot_SOMoC.html')

def Distribution_plot(data_clustered):
    """Plot the clusters size distribution"""
    sizes = data_clustered["cluster"].value_counts().to_frame()
    sizes.index.names = ['Cluster']
    sizes.columns = ['Size']
    sizes.reset_index(drop=False, inplace=True)
    sizes = sizes.astype({'Cluster': str, 'Size': int})

    fig = plx.bar(sizes, x=sizes.Cluster, y=sizes.Size, color=sizes.Cluster)

    fig.update_layout(legend_title="Cluster", plot_bgcolor='rgb(256,256,256)',
                      legend_title_font = dict(size=18, family='Calibri', color='black'),
                      legend_font = dict(size=15, family='Calibri', color='black'))
    fig.update_xaxes(title_text='Cluster', showline=True, linecolor='black',
                     gridcolor='lightgrey', zerolinecolor='lightgrey',
                     tickfont=dict(family='Arial', size=16, color='black'),
                     title_font=dict(size=20, family='Calibri', color='black'))
    fig.update_yaxes(title_text='Size', showline=True, linecolor='black',
                     gridcolor='lightgrey', zerolinecolor='lightgrey',
                     tickfont=dict(family='Arial', size=16, color='black'),
                     title_font=dict(size=20, family='Calibri', color='black'))

    fig.write_html(f'results/{name}/{name}_size_distribution_SOMoC.html')

    sizes.to_csv(f'results/{name}/{name}_Size_distribution_SOMoC.csv', index=True, header=True)  # Write the .CSV file

    return

def Save_settings(results_CVIs: pd.DataFrame):
    """
    Create a dictionary with the current run settings, save it as a JSON file,
    and return it.
    """
    
    # Create a dictionary with the settings
    settings = {}
    settings["timestamp"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    settings["fingerprint_type"] = "EState1"
    settings["umap"] = {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "n_components": n_components,
        "random_state": random_state,
        "metric": metric
    }
    settings["gmm"] = {
        "max_n_clusters": max_K,
        "n_init": n_init,
        "iterations": iterations,
        "init_params": init_params,
        "covariance_type": covariance_type
    }
    settings["optimal_K"] = K
    settings["CVIs"] =  {
        "silhouette": results_CVIs.loc['silhouette']['SOMoC'],
        "calinski_harabasz": results_CVIs.loc['calinski_harabasz']['SOMoC'],
        "davies_bouldin": results_CVIs.loc['davies_bouldin']['SOMoC'],
        "dunn": results_CVIs.loc['dunn']['SOMoC']
    }

    # Save the settings as a JSON file
    file_path = f"results/{name}/{name}_{settings['timestamp']}.json"

    with open(file_path, "w") as json_file:
        json.dump(settings, json_file, indent="\t")

    return settings

####################################### SOMoC main ########################################
###########################################################################################

if __name__ == '__main__':

    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.filemode('w'),
        logging.FileHandler("SOMoC.log", mode='w'),
        logging.StreamHandler()
    ]
)
    start_time = time.monotonic()

    print('='*100)

    # Get input data
    data_raw, name = Get_input_data(input_file=input_file)
    
    # Create output dir
    Make_dir(f'results/{name}')

    # Standardize molecules
    if smiles_standardization == True:
        data = Standardize_molecules(data_raw)
    else:
        logging.info('Skipping molecules standardization.')
        data = data_raw

    print('='*100)
    # Calculate Fingerprints
    X = Fingerprints_calculator(data)

    print('='*100)
    # Reduce feature space with UMAP
    embedding, n_components = UMAP_reduction(X, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state)
   
    print('='*100)
    # If K is not set, run the GMM clustering loop to get K
    if K is None:
        results_loop, K = GMM_clustering_loop(embedding, max_K=max_K, iterations=iterations, n_init=n_init, init_params=init_params, covariance_type=covariance_type, warm_start=warm_start)
    
    # Run the final clustering and calculate all CVIs
    results_clustered, results_CVIs = GMM_clustering_final(embedding, K=K, n_init=n_init, init_params=init_params, covariance_type=covariance_type, warm_start=warm_start)
   
    print('='*100)
    logging.info('Generating plots.')
    Elbow_plot(results_loop)
    Distribution_plot(results_clustered)

    logging.info('Saving run settings to JSON file.')
    Save_settings(results_CVIs)    # Write the settings JSON file

    logging.info('ALL DONE !')
    logging.info(f'SOMoC run took {time.monotonic() - start_time:.3f} seconds')

    print('='*100)
    print("CVIs results")
    print(results_CVIs)
    print('='*100)



    



