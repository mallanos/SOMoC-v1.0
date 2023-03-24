#!/usr/bin/env python
# coding: utf-8
"""
@author: Manu Llanos
"""
##################################### Import packages ####################################
###########################################################################################

import pandas as pd
import time
from typing import List, Tuple, Union, Optional, Dict, Any
import numpy as np
import logging
from tqdm import tqdm
import argparse

import umap
# from rdkit import Chem
# from rdkit.Chem.EState.Fingerprinter import FingerprintMol

from modules.data import *
from modules.plotting import *
from modules.clustering import *
from modules.encoding import *

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
    n_neighbors = settings.reducing['n_neighbors']
    min_dist = settings.reducing['min_dist']
    init = settings.reducing['init']
    metric = settings.reducing['metric']
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
    
    # Load settings from JSON file
    settings = Settings(args.config)
    
    # Load data
    data_handler = LoadData(args.input) 

    # Get the smiles
    data_raw, name = data_handler.parse_smiles_csv() 

    plotter = ClusteringPlotter(name)
    
    # Create output dir 
    make_dir(f'results/{name}') 
    
    # Convert SMILES to RDKit molecule
    data = data_handler.smiles_to_mol(data=data_raw, standardize=settings.standardize_molec)
    
    encoder = Encoding(data, settings)

    # Calculate Fingerprints
    X = encoder.fingerprints_calculator() 

    # Reduce feature space with UMAP
    embedding, n_components = UMAP_reduction(X, settings)
   
    cluster = Clustering(name, embedding, settings)

    if settings.optimal_K is not False:
        # Run the clustering and calculate all CVIs
        results_clustered, results_CVIs, results_model = cluster.GMM_final(K=settings.optimal_K)
    else:
        # If optimal_K is not set, run the GMM clustering loop to get K
        results_loop, optimal_K = cluster.GMM_loop()
        results_clustered, results_CVIs, results_model = cluster.GMM_final(K=optimal_K)
        # Update the original JSON file
        settings.optimal_K = optimal_K
        # Generate the elbow plot   
        plotter.elbow_plot_SIL(results_loop, optimal_K)

    # Generate the distribution plot   
    plotter.distribution_plot(results_model, embedding)

    # Write the settings JSON file
    settings.save_settings(name, df=results_CVIs)

    logging.info('ALL DONE !')
    logging.info(f'SOMoC run took {time.monotonic() - start_time:.3f} seconds')

    print('='*100)
    print("CVIs results")
    print(results_CVIs)
    print('='*100)
    



