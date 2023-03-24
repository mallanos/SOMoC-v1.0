
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.EState.Fingerprinter import FingerprintMol

class Encoding():
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def fingerprints_calculator(self) -> np.ndarray:
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

        if 'smiles' in self.data.columns:
            smiles_list = self.data['smiles']
        else:
            smiles_list = self.data.iloc[:,0]
        
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