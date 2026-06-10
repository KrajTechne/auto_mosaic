"""
One-time data preparation for autoresearch experiments.
Creates:
    1. JAX Numpy array of the motif alpha-carbon coordinates
    2. JAX Numpy array of the motif beta-carbon coordinates
    3. Beta Coordinates to Distogram
    4. Evaluation Criteria of final designs

"""

import os
import sys
import time
import math
import argparse
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import biotite
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import sklearn.metrics
from sklearn.metrics.pairwise import pairwise_distances
from mosaic.structure_prediction import TargetChain
from mosaic.common import TOKENS

# ------------------------------------------------------------------------
# Constants (fixed, do not modify)
#-----------------------------------------------------------------------


#-------------------------------------------------------------------------
# Configuration 
#-------------------------------------------------------------------------
MAX_OPTIMIZER_STEPS = 100
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
PATH_INPUT_STRUCTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gopher_alpha_snake.pdb")
CHAIN_MOTIF = {
    'A' : {'pos_native' : [107,108,109,110,111,112] , 'seq' : "LTKWTN"}, # (Alternate A Motif: 61-67)
    #'B' : {'pos' : [96,97,98,99,100,101,102,103,104,105], 'seq' : "TVMVVKPDRI"},
    'D' : {'pos_native' : [8,9,10,11,12], 'seq' : "FPGER"}
}
SEQ_TARGET= "LIDVVVVCDESNSIYPWDAVKNFLEKFVQGLDIGPTKTQVGLIQYANNPRVVFNLNTYKTKEEMIVATSQTSQYGGDLTNTFGAIQYARKYAYSAASGGRRSATKVMVVVTDGESHDGSMLKAVIDQCNHDNILRFGIAVLGYLNRNALDTKNLIKEIKAIASIPTERYFFNVSDEAALLEKAGTLGEQIFSI"

# -----------------------------------------------------------------------
# Motif Coordinate Extraction
# ----------------------------------------------------------------------

def extract_atom_array(struc_file_path: str, ca_only = False):
    """ Extract atom array from either CIF File or PDB File"""
    if struc_file_path.endswith(".cif"):
        pdbx_file = pdbx.CIFFile.read(struc_file_path)
        atom_array = pdbx.get_structure(pdbx_file=pdbx_file, model = 1)
    elif struc_file_path.endswith(".pdb"):
        pdb_file = pdb.PDBFile.read(struc_file_path)
        atom_array = pdb_file.get_structure(model = 1)
    else:
        raise ValueError("File must be either a PDB or CIF file")
    if ca_only:
        atom_array = atom_array[atom_array.atom_name == "CA"]
    return atom_array

def extract_motif_cb(atom_array, chain_id: str):
    """
    Extracts the beta carbon coordinates of the motif from the atom array.
    Saves the beta carbon coordinates of the motif from the atom array as a jax numpy array.
    """
    atom_array_cb = atom_array[(atom_array.atom_name == "CB") | ((atom_array.res_name == "GLY") & (atom_array.atom_name == "CA"))]
    atom_array_cb_chain = atom_array_cb[atom_array_cb.chain_id == chain_id]
    motif_res_pos = CHAIN_MOTIF[chain_id]['pos_native']
    motif_cb_coords = atom_array_cb_chain[np.isin(atom_array_cb_chain.res_id, motif_res_pos)].coord
    
    # Save the motif_cb_coords as a jax numpy array
    path_save_cb_coords = os.path.join(DATA_DIR, f"motif_cb_coords_{chain_id}.npy")
    np.save(path_save_cb_coords, motif_cb_coords)
    return path_save_cb_coords

def extract_motif_ca(atom_array, chain_id: str):
    """
    Extracts the alpha carbon coordinates of the motif from the atom array.
    Saves the alpha carbon coordinates of the motif from the atom array as a jax numpy array.
    Return the save path to the jax numpy array
    """
    atom_array_ca = atom_array[atom_array.atom_name == "CA"]
    atom_array_ca_chain = atom_array_ca[atom_array_ca.chain_id == chain_id]
    motif_res_pos = CHAIN_MOTIF[chain_id]['pos_native']
    motif_ca_coords = atom_array_ca_chain[np.isin(atom_array_ca_chain.res_id, motif_res_pos)].coord
    
    # Save the motif_ca_coords as a jax numpy array
    path_save_ca_coords = os.path.join(DATA_DIR, f"motif_ca_coords_{chain_id}.npy")
    np.save(path_save_ca_coords, motif_ca_coords)
    return path_save_ca_coords
#------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------- Evaluate Designed Structure (DO NOT CHANGE THIS! - It is a fixed metric) -----------------------
#------------------------------------------------------------------------------------------------------------------------------------
def calculate_motif_rmsd(designed_array, motif_res_pos: list, motif_coords_native: np.ndarray):
    """
    Helper Function for calculating motif_rmsd
    Calculates the backbone CA RMSD between the design and native motif coords (Only focused on the motif)
    """
    # 1. Filter down to Alpha Carbons for standard backbone alignment
    design_ca = designed_array[((designed_array.atom_name == "CA") & (designed_array.chain_id == "A"))]
    motif_coords_designed = design_ca[np.isin(design_ca.res_id, motif_res_pos)].coord
    
    # Safety Check: Ensure atom counts match exactly
    if len(motif_coords_designed) != len(motif_coords_native):
        raise ValueError(f"Atom count mismatch! Design: {len(motif_coords_designed)}, Native: {len(motif_coords_native)}")
    
    # 2. Superimpose the design onto the native motif geometry
    # This finds the optimal rotation and translation matrix
    superimposed_coords_design, transformation = struc.superimpose(motif_coords_native, motif_coords_designed)
    
    # 3. Calculate the exact RMSD
    rmsd = struc.rmsd(motif_coords_native, superimposed_coords_design)

    return rmsd

def compute_harmonic_mean(metric_a, metric_b):
    """
    Compute harmonic mean between common metric outputted by 2 different protein structure prediction models
    """
    numerator = 2 * metric_a * metric_b
    denominator = metric_a + metric_b
    hmean = numerator / denominator
    return hmean

def compute_composite_score(motif_rmsd_a, motif_rmsd_d, structure_iptm: float, binder_plddt: float) -> float:
    """
    Single composite score for autoresearch optimization. Lower is better.
    RMSD is normalized via 1-exp(-rmsd/1.5) so all terms are on [0,1).
    """
    mean_rmsd = np.mean([motif_rmsd_a, motif_rmsd_d])
    rmsd_score = 1.0 - np.exp(-mean_rmsd / 1.5)
    composite_score =  (2.0 * rmsd_score + 1.0 * (1.0 - structure_iptm) + 0.5 * (1.0 - binder_plddt))
    print(f"Composite Score: {composite_score:.4f}  (lower is better; target < 1.5)")
    print("-" * 50)
    return composite_score

def evaluate_optimized_structure(struc_model, seq_pssm, motif_id_pos: dict, design_iteration: int, model_name:str, recycling_steps: int = 5, sampling_steps: int = 21):
    """
    Evaluates the optimized structure by calculating the following metrics:
    1. Motif RMSD for each motif
    2. Structure IPTM
    3. Structure PLDDT
    """
    # 1. Convert pssm into seq
    seq_tokenized = seq_pssm.argmax(-1)
    seq_binder = "".join(TOKENS[i] for i in seq_tokenized)
    # 2. Create features for Boltz Complex Structure Prediction
    struc_model_features, struc_model_writer = struc_model.target_only_features(
        chains = [
            TargetChain(sequence = seq_binder, use_msa = False),
            TargetChain(sequence = SEQ_TARGET, use_msa = True),
        ]
    )
    # 3. Predict Boltz Structure
    if model_name == "Boltz2":
        pred = struc_model.predict(PSSM = seq_pssm, features = struc_model_features, writer = struc_model_writer, key = jax.random.key(11))
    elif model_name == "ESMFold2":
        pred = struc_model.predict(PSSM = seq_pssm, features = struc_model_features, writer = struc_model_writer, key = jax.random.key(11),
                                   recycling_steps = recycling_steps, sampling_steps = sampling_steps)
    # 4. Save outputted gemmi structure to file and open up as a Biotite atom array
    designed_structure = pred.st
    path_designed_structure = os.path.join(DATA_DIR, f"{model_name}_structure_{design_iteration}.pdb")
    with open(path_designed_structure, "w") as f:
        f.write(designed_structure.make_pdb_string())
    designed_array = extract_atom_array(path_designed_structure)
    # 5. Calculate Motif RMSD for each motif
    motif_rmsd_dict = {}
    for motif_id, motif_res_pos in motif_id_pos.items():
        motif_coords_native_ca = np.load(os.path.join(DATA_DIR, f"motif_ca_coords_{motif_id}.npy"))
        motif_coords_designed_res_pos = motif_res_pos['pos_design']
        motif_rmsd = calculate_motif_rmsd(designed_array = designed_array, motif_res_pos = motif_coords_designed_res_pos,
                                          motif_coords_native= motif_coords_native_ca)
        motif_rmsd_dict[motif_id] = motif_rmsd
    # 6. Calculate Structure IPTM
    structure_iptm = pred.iptm
    # 7. Calculate Structure PLDDT
    binder_plddt = pred.plddt[:len(seq_binder)].mean()
    # 8. Display Metrics for Agent to interpret and decide next step:
    print(f"Validation conducted by {model_name}:")
    print(f"You have Designed Binder Sequence: {seq_binder}")
    print("Your selection of hyperparameters has resulted in: ")
    for motif_id, rmsd in motif_rmsd_dict.items():
        print(f"Motif From Chain: {motif_id} has an associated RMSD: {rmsd:.2f}")
        print("As always, smaller RMSD is better and ideal RMSD is < 1.5 Angstroms")
    print(f"{model_name} ipTM: {structure_iptm:.2f}")
    print(f"{model_name} pLDDT: {binder_plddt:.2f}")
    print(" ") # Add line of empty space after every print sequence for a given model
    # 9. Extract the Motif RMSDs
    motif_rmsd_a = motif_rmsd_dict['A']
    motif_rmsd_d = motif_rmsd_dict['D']

    return motif_rmsd_a, motif_rmsd_d, structure_iptm, binder_plddt

#--------------------------------------------------------------------------------------------------------------------
# Main
#--------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Create main directory to store data
    os.makedirs(DATA_DIR, exist_ok= True)
    # 1. Load in structure
    atom_array_complex = extract_atom_array(PATH_INPUT_STRUCTURE)

    # 2. Extract and Save respective motif Ca and Cb coordinates
    for chain_id, motif_dict in CHAIN_MOTIF.items():
        
        # Extract Alpha-Carbon Coordinates and generate path to numpy array containing them
        path_save_ca_coords = extract_motif_ca(atom_array = atom_array_complex, chain_id = chain_id)
        CHAIN_MOTIF[chain_id]['path_coords_ca'] = path_save_ca_coords

        # Extract Beta-Carbon Coordinates and generate path to numpy array containing them
        path_save_cb_coords = extract_motif_cb(atom_array = atom_array_complex, chain_id = chain_id)
        CHAIN_MOTIF[chain_id]['path_coords_cb'] = path_save_cb_coords
    
    print("Motif Coordinates Extracted and Saved! Ready for Gradient-Based Optimization!")




