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
smiles_standardization = True       

### UMAP parameters ###
n_neighbors = 10    # The size of local neighborhood used for manifold approximation. Larger values result in more global views of the manifold, while smaller values result in more local data being preserved.
min_dist = 0.0      # The effective minimum distance between embedded points. Smaller values will result in a more clustered/clumped embedding where nearby points on the manifold are drawn closer together, while larger values will result on a more even dispersal of points.
random_state = 10   # Use a fixed seed for reproducibility.
metric = "jaccard"  # The metric to use to compute distances in high dimensional space.

### GMM parameters ###
max_K = 10                      # Max number of clusters to cosidering during the GMM loop
Kers = np.arange(2, max_K+1, 1) # Generate the range of K values to explore
iterations = 3                  # Iterations of GMM to run for each K
n_init = 5                      # Number of initializations on each GMM run, then just keep the best one.
init_params = 'kmeans'          # How to initialize. Can be random or K-means
covariance_type = 'diag'        # Type of covariance to consider: "spherical", "diag", "tied", "full"
warm_start = False

#################################### Helper functions #####################################
###########################################################################################

def Get_name(archive):
    """strip path and extension to return the name of a file"""
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
        
    name = os.path.splitext(os.path.basename(file_path))[0]
    
    with open(file_path) as f:
        data = pd.read_csv(f, delimiter=',', header=None)
    
    return data, name

def Standardize_molecules(data: pd.DataFrame):
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

    time_start = time.monotonic()
    data_ = data.copy()

    list_of_smiles = data_.iloc[:, 0]
    standardized_mols = [np.nan] * len(list_of_smiles)

    for i, smiles in enumerate(list_of_smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            standardized_mol = Standardizer().standardize(mol)
            # standardized_mol = Standardizer().super_parent(mol)
            standardized_mols[i] = standardized_mol
        except Exception as e:
            print(f"Failed to process molecule {i+1}: ({e})")

    data_['mol'] = standardized_mols
    data_ = data_.dropna() # Drop failed molecules

    num_processed_mols = len(standardized_mols)
    num_failed_mols = len([m for m in standardized_mols if m is np.nan])

    print(f'{num_processed_mols-num_failed_mols} molecules processed in {time.monotonic() - time_start:.3f} seconds')
    
    if num_failed_mols:
        print(f"{num_failed_mols} molecules failed to be processed")

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
    
    print('=' * 50)
    # print("Calculating EState molecular fingerprints...")
    print("Encoding")

    time_start = time.monotonic()

    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    fps = [None] * len(mols)
    i=0
    for mol in mols:
        try:
            fp = FingerprintMol(mol)[0]  # EState fingerprint
            fps[i] = fp
            i += 1
        # except:
        except Exception as e:
            print(f"Failed to process molecule {i+1}: ({e})")
            # raise RuntimeError("Error in fingerprint calculation for molecule with SMILES: " + Chem.MolToSmiles(mol))

    fingerprints = np.stack(fps, axis=0)

    print(f'Fingerprints calculation took {time.monotonic() - time_start:.3f} seconds')
    print('=' * 50)

    return fingerprints

def UMAP_reduction(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1,
                   metric: str = 'euclidean', random_state: int = 42,
                   verbose: bool = True) -> Tuple[np.ndarray, int]:

    """
    Reduce feature space using the UMAP algorithm.

    Parameters:
        X (np.ndarray): Input data as a NumPy array.
        n_neighbors (int): Number of neighbors to use for the UMAP algorithm.
        min_dist (float): Minimum distance threshold for the UMAP algorithm.
        metric (str): Distance metric to use for the UMAP algorithm.
        random_state (int): Random seed for the UMAP algorithm.
        verbose (bool): Whether to print progress messages.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing the reduced feature space (as a NumPy array) and the number of components
        used for the reduction.

    Raises:
        ValueError: If the input is not a NumPy array or the number of neighbors is greater than the length of the input data.
    """

    if not isinstance(X, np.ndarray):
        raise ValueError("Input must be a NumPy array")
    
    if n_neighbors >= len(X):
        raise ValueError("The number of neighbors must be smaller than the number of molecules to cluster")

    if verbose:
        print('Reducing feature space with UMAP...')

    start_time = time.monotonic()

    # Set a lower bound to the number of components
    n_components = max(int(len(X) * 0.01), 3)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric,
                        random_state=random_state).fit(X)
    embedding = reducer.transform(X)

    if verbose:
        print(f'{embedding.shape[1]} features have been retained.')
        print(f'UMAP took {time.monotonic() - start_time:.3f} seconds')

    return embedding, n_components


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
    print("Clustering")
    print(f'Running GMM clustering for {max_K} iterations..')
    
    start_time = time.monotonic()

    temp = {i: [] for i in range(max_K)}  # pre-allocate the dictionary

    for n in range(2, max_K):
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
    
    print(f'GMM clustering loop took {time.monotonic() - start_time:.3f} seconds')
    print(' '*100)

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
    print('='*50)
    print("Final clustering")
    print(f'Running GMM with K = {K}')

    start_time = time.monotonic()

    GMM_final = GMM(K, n_init=n_init, init_params=init_params, warm_start=warm_start,
                    covariance_type=covariance_type, random_state=random_state, verbose=0)
    GMM_final.fit(embeddings)
    labels_final = GMM_final.predict(embeddings)

    if GMM_final.converged_:
        print('GMM converged.')
    else:
        print('GMM did not converge. Please check you input configuration.')

    sil_ok = silhouette_score(embeddings, labels_final, metric='cosine').round(4)
    db_score = davies_bouldin_score(embeddings, labels_final).round(4)
    ch_score = calinski_harabasz_score(embeddings, labels_final).round(4)
    dist_dunn = pairwise_distances(embeddings).round(4)
    dunn_score = dunn(dist_dunn, labels_final).round(4)

    valid_metrics = [sil_ok, db_score, ch_score, dunn_score]

    random_means,random_sds = Cluster_random(embeddings, num_iterations=500, num_clusters=K)

    table_metrics = pd.DataFrame([valid_metrics, random_means, random_sds]).T
    table_metrics = table_metrics.rename(index={0: 'Silhouette score', 1: "Davies Bouldin score",
                                         2: 'Calinski Harabasz score', 3: 'Dunn Index'}, columns={0: "Value", 1: "Mean Random", 2: "SD Random"})

    print(f'GMM clustering took {time.monotonic() - start_time:.3f} seconds')

    print('='*50)
    print("Validation metrics")
    print(table_metrics)

    cluster_final = pd.DataFrame({'cluster': labels_final}, index=data.index)
    data_clustered = data.join(cluster_final)

    if 'mol' in data_clustered.columns:  # Check if mol column from standardization is present
        try:
            data_clustered['SMILES_standardized'] = data_clustered['mol'].apply(
                lambda x: Chem.MolToSmiles(x))
            data_clustered.drop(['mol'], axis=1, inplace=True)
        except Exception as e:
            print('Something went wrong converting standardized molecules back to SMILES code..: {e}')
            pass

    data_clustered.to_csv(f'results_SOMoC_{name}/{name}_Clustered_SOMoC.csv', index=True, header=True)
    table_metrics.to_csv(f'results_SOMoC_{name}/{name}_Validation_SOMoC.csv', index=True, header=True)

    return data_clustered

def Cluster_random(embeddings: np.array, num_iterations: int = 500, num_clusters: int = 3):
    """
    Perform random clustering and calculate several CVIs.

    Args:
        embeddings (numpy.ndarray): An array of shape (n_samples, n_features) containing the data to be clustered.
        num_iterations (int): Number of random clusterings to perform. Default is 500.
        num_clusters (int): Number of clusters to generate randomly. Default is 3.

    Returns:
        Two lists containing CVIs means and stds (rounded to 4 decimal places):
        - random_means: Average of all CVIs.
        - random_sds: Standard deviation of all CVIs.
    """

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

    return random_means,random_sds

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

    fig.write_html(f'results_SOMoC_{name}/{name}_elbowplot_SOMoC.html')

    print('By dafault SOMoC uses the K which resulted in the highest Silhouette score.')
    print('However, you can check the Silhouette vs. K elbow plot to choose the optimal K, identifying an inflection point in the curve (elbow method)')
    print('Then, re-run SOMoC with a fixed K.')
    print("Note: Silhouette score is bounded [-1,1], the closer to 1 the better")


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

    fig.write_html(f'results_SOMoC_{name}/{name}_size_distribution_SOMoC.html')

    sizes.to_csv(f'results_SOMoC_{name}/{name}_Size_distribution_SOMoC.csv', index=True, header=True)  # Write the .CSV file

    return

def Setting_info():
    """Create a dataframe with current run setting"""
    today = date.today()
    fecha = today.strftime("%d/%m/%Y")
    settings = []
    settings.append(["Date: ", fecha])
    settings.append(["Setings:", ""])
    settings.append(["", ""])
    settings.append(["Fingerprint type:", "EState1"])
    settings.append(["", ""])
    settings.append(["UMAP", ""])
    settings.append(["n_neighbors:", str(n_neighbors)])
    settings.append(["min_dist:", str(min_dist)])
    settings.append(["n_components:", str(n_components)])
    settings.append(["random_state:", str(random_state)])
    settings.append(["metric:", str(metric)])
    settings.append(["", ""])
    settings.append(["GMM", ""])
    settings.append(["max Nº of clusters (K):", str(max_K)])
    settings.append(["Optimal K:", str(K)])
    settings.append(["iterations:", str(iterations)])
    settings.append(["n_init:", str(n_init)])
    settings.append(["init_params", str(init_params)])
    settings.append(["covariance_type", str(covariance_type)])
    settings.append(["", ""])
    settings.append(["Total running time : ", total_time])
    settings.append(["", ""])
    settings_df = pd.DataFrame(settings)
    settings_df.to_csv(f'results_SOMoC_{name}/{name}_Settings_SOMoC.csv', index=True, header=False)
    return

####################################### SOMoC main ########################################
###########################################################################################


if __name__ == '__main__':

    start = time.time()

    print('-'*50)

    # Get input data
    data_raw, name = Get_input_data(input_file=input_file)

    # Create output dir
    Make_dir(f'results_SOMoC_{name}')

    # Standardize molecules
    if smiles_standardization == True:
        data = Standardize_molecules(data_raw)
    else:
        print('Skipping molecules standardization..\n')
        data = data_raw

    # Calculate Fingerprints
    X = Fingerprints_calculator(data)

    # Reduce feature space with UMAP
    embedding, n_components = UMAP_reduction(X, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state, verbose=True)
    # If K is not set, run the GMM clustering loop to get K
    if K is None:
        results, K = GMM_clustering_loop(embedding, max_K=max_K, iterations=iterations, n_init=n_init, init_params=init_params, covariance_type=covariance_type, warm_start=warm_start)
        Elbow_plot(results)

    # Run the final clustering and calculate all CVIs
    data_clustered = GMM_clustering_final(embedding, K=K, n_init=n_init, init_params=init_params, covariance_type=covariance_type, warm_start=warm_start)

    # Generate distribution plot and .CSV file
    Distribution_plot(data_clustered)

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    total_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

    # Write the settings file
    settings = Setting_info()

    print('='*50)
    print('ALL DONE !')
    print(f'SOMoC run took {total_time}')
    print('='*50)


    



