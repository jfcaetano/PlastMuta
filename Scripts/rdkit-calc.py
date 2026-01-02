#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:47:17 2025

@author: jfcaetano
"""



import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
import cirpy

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*'

# --- Load full data ---
csv_file = 'toxic-raw.csv'
smiles_column = 'canonical_smiles'
df = pd.read_csv(csv_file)

# --- Descriptor selection ---
descriptor_columns = [
    name for name in dir(Descriptors)
    if callable(getattr(Descriptors, name)) and (
        name in ['BalabanJ','BertzCT','Chi0','Chi0n','Chi0v','Chi1','Chi1n','Chi1v','Chi2n','Chi2v','Chi3n','Chi3v','Chi4n','Chi4v','EState_VSA1','EState_VSA10','EState_VSA11','EState_VSA2','EState_VSA3','EState_VSA4','EState_VSA5','EState_VSA6','EState_VSA7','EState_VSA8','EState_VSA9','HallKierAlpha','HeavyAtomCount','HeavyAtomMolWt','Kappa1','Kappa2','Kappa3','MaxAbsEStateIndex','MaxAbsPartialCharge','MaxEStateIndex','MaxPartialCharge','MinAbsEStateIndex','MinAbsPartialCharge','MinEStateIndex','MinPartialCharge','MolLogP','MolMR','MolWt','NHOHCount','NOCount','NumAliphaticCarbocycles','NumAliphaticHeterocycles','NumAliphaticRings','NumAromaticCarbocycles','NumAromaticHeterocycles','NumAromaticRings','NumHAcceptors','NumHDonors','NumHeteroatoms','NumRadicalElectrons','NumRotatableBonds','NumSaturatedCarbocycles','NumSaturatedHeterocycles','NumSaturatedRings','NumValenceElectrons','PEOE_VSA1','PEOE_VSA10','PEOE_VSA11','PEOE_VSA12','PEOE_VSA13','PEOE_VSA14','PEOE_VSA2','PEOE_VSA3','PEOE_VSA4','PEOE_VSA5','PEOE_VSA6','PEOE_VSA7','PEOE_VSA8','PEOE_VSA9','SMR_VSA1','SMR_VSA10','SMR_VSA2','SMR_VSA3','SMR_VSA4','SMR_VSA5','SMR_VSA6','SMR_VSA7','SMR_VSA8','SMR_VSA9','SlogP_VSA1','SlogP_VSA10','SlogP_VSA11','SlogP_VSA12','SlogP_VSA2','SlogP_VSA3','SlogP_VSA4','SlogP_VSA5','SlogP_VSA6','SlogP_VSA7','SlogP_VSA8','SlogP_VSA9','TPSA','VSA_EState1','VSA_EState10','VSA_EState2','VSA_EState3','VSA_EState4','VSA_EState5','VSA_EState6','VSA_EState7','VSA_EState8','VSA_EState9'])]

# --- Functions ---

def compute_descriptors_with_fallback(canonical_smiles, original_smiles):
    """Try descriptors on canonical SMILES, fallback to original."""
    for smiles in [canonical_smiles, original_smiles]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return [getattr(Descriptors, desc)(mol) for desc in descriptor_columns]
        except:
            continue
    return [None] * len(descriptor_columns)

def get_cas_from_smiles(smiles):
    """Use Cirpy to retrieve CAS number from SMILES."""
    if smiles:
        try:
            cas = cirpy.resolve(smiles, 'cas')
            return cas if cas else None
        except:
            return None
    return None

# --- Canonicalize SMILES ---
df['canonical_SMILES'] = df['canonical_smiles']

# --- Compute descriptors ---
descriptor_data = []
cas_data = []

for i in tqdm(df.index, desc="Computing descriptors & CAS"):
    canon = df.loc[i, 'canonical_SMILES']
    orig = df.loc[i, smiles_column]
    
    descriptor_data.append(compute_descriptors_with_fallback(canon, orig))
    

# --- Combine data ---
desc_df = pd.DataFrame(descriptor_data, columns=descriptor_columns)
final_df = pd.concat([df.reset_index(drop=True), desc_df], axis=1)

# --- Fill missing descriptor values with 0 ---
final_df[descriptor_columns] = final_df[descriptor_columns].fillna(0)

# --- Save to CSV ---
final_df.to_csv('toxic_raw_rdkit.csv', index=False)
print(f"\nFinal output saved to toxic_rdkit.csv with {len(final_df)} rows.")
