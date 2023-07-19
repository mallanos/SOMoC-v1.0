#!/usr/bin/env python
# coding: utf-8
"""
@author: Manu Llanos
"""
##################################### Import packages ####################################
###########################################################################################

import time
import logging
import argparse

from modules.data import *
from modules.plotting import *
from modules.clustering import *
from modules.encoding import *
from modules.reducing import *

####################################### SOMoC main ########################################
###########################################################################################
#TODO
# Support SDF molecules as input
# Support for multiple CVIs to be optimized ==> Almost done
# Allow scaffold decomposition before Clustering
# Get representative from cluster. Scaffold or MCC
# Save clustering model to be used later

def main():

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
    ])
    
    start_time = time.monotonic()
    
    # Load settings from JSON file
    settings = Settings(args.config)
    
    # Load data
    data = LoadData(args.input) 

    # Get the smiles
    data_raw, dataset_name = data.parse_smiles_csv() 
   
    # Create output dir 
    make_dir(f'results/{data.dataset_name}') 
    
    # Convert SMILES to RDKit molecule
    data_mols = data.smiles_to_mol(data=data_raw, standardize=settings.standardize_molec)

    # Calculate Fingerprints
    encoder = Encoding(data_mols, settings)
    X = encoder.fingerprints_calculator()

    # Reduce feature space
    reducer = Reducing(X, settings)
    embedding = reducer.reduce()
   
    clusterer = Clustering(data.dataset_name, embedding, settings)
    K, results_loop, labels, results_CVIs, clustering_model = clusterer.cluster()
    
    data_clustered = merge_data(data.dataset_name, data_mols, labels)
    
    plotter = Plotting(dataset_name)

    # Generate Elbow plot
    if results_loop is not None: plotter.elbow_plot(results_loop, K)

    # Generate the distribution plot   
    plotter.distribution_plot(clustering_model, embedding)
    
    # Generate MCS per cluster
    plotter.plot_cluster_mcs(data_clustered)

    # Write the settings JSON file
    settings.save_settings(data.dataset_name, df=results_CVIs)

    logging.info('ALL DONE !')
    logging.info(f'SOMoC run took {time.monotonic() - start_time:.3f} seconds')

    print('='*100)
    print("CVIs results")
    print(results_CVIs)
    print('='*100)
    
if __name__ == '__main__':
    main()

