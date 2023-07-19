import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

from rdkit.Chem import AllChem, MolFromSmiles, MACCSkeys
from rdkit.Chem.EState.Fingerprinter import FingerprintMol

class Encoding():
    def __init__(self, data: pd.DataFrame, settings) -> None:
        self.data = data
        if 'smiles' in self.data.columns:
            self.smiles_list = self.data['smiles']
        else:
            self.smiles_list = self.data.iloc[:,0]
        self.fingerprints = None
        self.settings = settings

    def fingerprints_calculator(self) -> np.ndarray:
        """
        Calculate molecular fingerprints using the RDKit package.

        Parameters:
            fingerprint_type (str): The type of fingerprint to calculate. Must be 'estate', 'morgan' or 'maccs'.
            radius (int): The radius of the Morgan fingerprint. Default is 2.
            nbits (int): The number of bits in the Morgan fingerprint. Default is 2048.

        Returns:
            np.ndarray: The calculated molecular fingerprints as a NumPy array.

        Raises:
            ValueError: If the input fingerprint type is not 'estate', 'morgan' or 'maccs'.
            RuntimeError: If there is a problem with fingerprint calculation of some SMILES.
        """
        fingerprint_type = self.settings.encoding['fingerprint_type']
        radius = self.settings.encoding['radius']
        nbits = self.settings.encoding['nbits']

        if fingerprint_type not in ['estate', 'morgan', 'maccs']:
            raise ValueError("Invalid fingerprint type. Must be 'estate','morgan or 'maccs'.")

        logging.info(f"Calculating {fingerprint_type.upper()} molecular fingerprints")

        mols = [MolFromSmiles(smiles) for smiles in self.smiles_list]
        fps = [None] * len(mols)
        for i, mol in tqdm(enumerate(mols), total=len(mols), desc=f'Calculating {fingerprint_type.upper()} fingerprints'):
            try:
                if fingerprint_type == 'estate':
                    fp = FingerprintMol(mol)[0]  # EState fingerprint
                    # fp = AllChem.GetMorganFingerprint(mol, radius, useFeatures=True, nBits=nbits)  # EState fingerprint
                elif fingerprint_type == 'morgan':
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, useFeatures=False, nBits=nbits)  # Morgan fingerprint
                elif fingerprint_type == 'maccs':
                    fp = MACCSkeys.GenMACCSKeys(mol)  # MACCS fingerprints
                fps[i] = fp
            except Exception as e:
                logging.warning(f"Failed fingerprint calculation for molecule {i+1}: ({e})")

        fingerprints = np.stack(fps, axis=0)

        return fingerprints
