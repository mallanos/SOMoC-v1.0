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

import logging
import pandas as pd
from pathlib import Path
from typing import Tuple

from rdkit import Chem
from molvs import Standardizer

def get_file_name(file_path: str) -> str:
    """Strip path and extension to return the name of a file"""
    return Path(file_path).stem

def make_dir(dirName: str):
    """Create a directory and not fail if it already exist"""
    try:
        os.makedirs(dirName)
    except FileExistsError:
        pass

class Settings:
    def __init__(self, config_file: str) -> None:
        self.load_settings(config_file)

    def load_settings(self, config_file: str) -> None:
        
        logging.info('Loading settings from JSON file.')

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
        logging.info('Saving run settings to JSON file.')

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

class LoadData:
    def __init__(self, _file: str) -> None:
        test_file="test/focal_adhesion.csv"
        self.file_path = _file or test_file

    def parse_smiles_csv(self) -> Tuple[pd.DataFrame, str]:
        """Get data from a CSV file"""

        file_path = Path(self.file_path)
        if not file_path.exists():
            raise ValueError(f'File not found: {file_path}')

        with file_path.open() as f:
            data = pd.read_csv(f, delimiter=',', header=None)

        name = get_file_name(file_path)
        logging.info(f'Loading {name} dataset')
        logging.info(f'{len(data)} Smiles loaded..')

        return data, name

    def smiles_to_mol(self, data: pd.DataFrame, standardize: bool=False) -> pd.DataFrame:
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
        df = data.copy()

        input_smiles = df.iloc[:, 0]
        output_smiles = [np.nan] * len(input_smiles)

        for i, smiles in tqdm(enumerate(input_smiles), total=len(input_smiles), desc='Converting SMILES to molecules'):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if standardize:
                    try:
                        standardized_mol = Standardizer().standardize(mol)
                        # standardized_mol = Standardizer().super_parent(mol)
                        output_smiles[i] = standardized_mol
                    except Exception as e:
                        logging.warning(f"Failed to standardize molecule {i+1}: ({e})")
                else:
                    output_smiles[i] = mol
            except Exception as e:
                logging.warning(f"Failed to process molecule {i+1}: ({e})")

        df['mol'] = output_smiles
        df.dropna(inplace=True) # Drop failed molecules

        num_processed_mols = len(output_smiles)
        num_failed_mols = len([m for m in output_smiles if m is np.nan])

        logging.info(f'{num_processed_mols-num_failed_mols} molecules processed')

        if num_failed_mols:
            logging.warning(f"{num_failed_mols} molecules failed to be standardized")

        return df