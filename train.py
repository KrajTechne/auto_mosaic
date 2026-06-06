"""
Autoresearch pretraining script adapted to Motif Scaffolding. Single-GPU, single-file.
Consolidated from review of Escalante Bio's Mosaic Database & Sergey Ovchinikov's ColabDesign (AF2 Gradient Backpropagation Design)
Usage: uv run train.py
"""

import os
os.environ['PYTORCH_ALLOC_CONF'] = "expandable_segments:True"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"  # Use 95% of GPU (default 75%)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   # Allocate on-demand, not upfront

import gc
import math
import time
import numpy as np
import sklearn.metrics
from sklearn.metrics.pairwise import pairwise_distances

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
import mosaic
from mosaic.proteinmpnn.mpnn import load_mpnn_sol
from mosaic.models.boltz2 import Boltz2
from mosaic.common import LossTerm
from mosaic.structure_prediction import StructureModelOutput
import mosaic.losses.structure_prediction as sp
from mosaic.common import TOKENS
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
from mosaic.losses.transformations import NoCys, SetPositions, SoftClip
from mosaic.structure_prediction import TargetChain
from mosaic.optimizers import simplex_APGM

from prepare import SEQ_TARGET, CHAIN_MOTIF, MAX_OPTIMIZER_STEPS, DATA_DIR, calculate_motif_rmsd, evaluate_optimized_structure, compute_composite_score

#-----------------------------------------------------------------------------------------------------
# Helper Function for Loss Function
#-----------------------------------------------------------------------------------------------------
def coords_to_distogram(coords: np.ndarray):
    """ Converts coords into 1-hot encoded distogram of shape (N residues, N residues, 64 bins)
        64 bins are in line with AlphaFold2 and Boltz2 distogram bin ranges

        1. Compute all pairwise distances between residues CB distances
        2. Bin distances into 64 bins
        3. Convert to 1-hot encoding
    """
    # 1. Compute pairwise distances between residues CB coordinates
    distances = pairwise_distances(coords, metric = 'euclidean')
    
    # 2. Define the AlphaFold/Boltz 64 distance bins
    dgram_bins = jnp.append(0, jnp.linspace(2.3125, 21.6875, 63))
    
    # 3. Find which bin each distance falls into and subtract 1 to get the indices for each bin
    bin_indices = jnp.digitize(distances, dgram_bins) - 1
    
    # Cap indices at 63 (the final "no contact" bin for distances > 22A)
    bin_indices = jnp.clip(bin_indices, 0, 63)
    
    # 5. Convert to one-hot probabilities (M, M, 64)
    target_dgram_one_hot = jax.nn.one_hot(bin_indices, 64)

    return target_dgram_one_hot
#----------------------------------------------------------------------------------------------------
# Motif-Specific Custom Loss Functions Derived From Escalante Bio's Mosaic Github Repo
#----------------------------------------------------------------------------------------------------

class MotifDistogramCE(LossTerm):
    """
    Categorical Cross-Entropy for Motif Scaffolding.
    Evaluates the distogram error ONLY on the specified motif positions,
    ignoring the rest of the hallucinated scaffold.
    """
    f: Float[Array, "M M Bins"]        # Target distogram of just the motif
    motif_positions: Int[Array, "M"]   # Indices of where the motif is in the scaffold
    name: str = "motif_dgram"
    l: float = -np.inf
    u: float = np.inf

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        # 1. Extract the N x N x Bins distogram from the network
        logits = output.distogram_logits
        
        # 2. Slice out ONLY the M x M x Bins sub-matrix for the motif.
        # jnp.ix_ safely creates the meshgrid to extract the correct rows and columns.
        motif_logits = logits[jnp.ix_(self.motif_positions, self.motif_positions)]

        # 3. Expand dims to ensure broadcastability with the target 'f'
        f = jnp.expand_dims(self.f, [i for i in range(3 - self.f.ndim)])

        # 4. Calculate categorical cross-entropy exactly as before, but on the M x M slice
        ce = -jnp.fill_diagonal(
            (
                jax.nn.log_softmax(motif_logits) * f
            ).sum(-1),
            0,
            inplace=False,
        ).mean()

        return ce.clip(self.l, self.u), {self.name: ce}

class MotifRMSDLoss(LossTerm):
    """
    Penalizes absolute structural deviation of the designed motif 
    from the target backbone motif using the ColabDesign Kabsch alignment.
    """
    target_coords: Float[Array, "M 3"]  # Native Ca coordinates of the motif
    motif_positions: Int[Array, "M"]    # Sequence indices of the motif
    name: str = "motif_rmsd"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        # 1. Extract Predicted C-alpha coordinates for the motif (Index 1 is CA)
        pred_coords = output.backbone_coordinates[self.motif_positions, 1, :]
        
        P = pred_coords
        T = self.target_coords
        
        # 2. Center both sets of coordinates to the origin
        P_mu = P.mean(axis=0)
        T_mu = T.mean(axis=0)
        P_c = P - P_mu
        T_c = T - T_mu
        
        # 3. ColabDesign's Kabsch logic (pure JAX)
        ab = jnp.swapaxes(P_c, -1, -2) @ T_c
        u, s, vh = jnp.linalg.svd(ab, full_matrices=False)
        
        # Reflection check
        flip = jnp.linalg.det(u @ vh) < 0
        u_ = jnp.where(flip, -u[..., -1].T, u[..., -1].T).T
        u = u.at[..., -1].set(u_)
        
        # Alignment matrix
        R = u @ vh
        
        # 4. Apply alignment and compute RMSD
        P_aligned = (P_c @ R) + T_mu
        
        msd = jnp.mean(jnp.sum((P_aligned - T) ** 2, axis=-1))
        rmsd = jnp.sqrt(msd + 1e-8)
        
        return rmsd, {self.name: rmsd}
# -----------------------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# -----------------------------------------------------------------------------------------

# Define Lengths of Linkers between Motifs
MOTIF_CHAIN_ORDER = ['A', 'D'] # Chain IDs of the two motifs (MUST ALWAYS INCLUDE BOTH "A" & "D". Free to change order)
LINKER_LEN1 = 30 # Length of linker from start of protein (N-terminus) to first motif
LINKER_LEN2 = 30 # Length of linker between first and second motif
LINKER_LEN3 = 30 # Length of linker between second motif and end of protein (C-terminus)

# Define weights of the respective loss functions in the final total/composite loss function
# Final loss function = sum of all loss functions weighted by their respective weights
WEIGHT_BINDER_CONTACT_LOSS_FUNCTION = 0.5 # Weight of the binder contact loss function in the total/composite loss function
WEIGHT_WITHIN_BINDER_CONTACT_LOSS_FUNCTION = 0.5 # Weight of the within-binder contact loss function in the total/composite loss function
WEIGHT_INVERSE_FOLDING_SEQ_RECOVERY_LOSS_FUNCTION = 1.0 # Weight of the inverse folding sequence recovery loss function in the total/composite loss function
WEIGHT_TARGET_BINDER_PAE_LOSS_FUNCTION = 0.5 # Weight of the target to binder (directional PAE) PAE loss function in the total/composite loss function
WEIGHT_BINDER_TARGET_PAE_LOSS_FUNCTION = 0.5 # Weight of the binder to target (directional PAE) PAE loss function in the total/composite loss function
WEIGHT_WITHIN_BINDER_PAE_LOSS_FUNCTION = 0.5 # Weight of the within-binder PAE loss function in the total/composite loss function
WEIGHT_IPTM_LOSS_FUNCTION = 0.1 # Weight of the iptm loss function in the total/composite loss
WEIGHT_PTM_ENERGY_LOSS_FUNCTION = 0.1 # Weight of the ptm energy loss function in the total/composite loss
WEIGHT_PLDDT_LOSS_FUNCTION = 0.1 # Weight of the plddt loss function in the total/composite loss
WEIGHT_MOTIF_DISTOGRAM_LOSS_FUNCTION = 0.1 # Weight of the motif distogram loss function in the total/composite loss
WEIGHT_MOTIF_RMSD_LOSS_FUNCTION = 0.1 # Weight of the motif rmsd loss function in the total/composite loss

# Initial "soft" PSSM -> Try to sharpen PSSM into a discrete sequence (e.g. one-hot PSSM) -> Further sharpening with a "hard" PSSM
# Optimizer Parameters: 
soft_pssm_hyparams = {
    'n_steps' : 50,
    'stepsize' : 1.5,
    'momentum' : 0.3,
    'scale'    : 1.00,
    'logspace' : False,
}
sharp_pssm_hyparams = {
    'n_steps' : 30,
    'stepsize' : 3.5,
    'momentum' : 0.0,
    'scale'    : 1.25,
    'logspace' : True
}
hard_pssm_hyparams = {
    'n_steps' : 20,
    'stepsize' : 3.5,
    'momentum' : 0.0,
    'scale'    : 1.40,
    'logspace' : True
}

# If total number of pssm steps is > 100, raise error
if soft_pssm_hyparams['n_steps'] + sharp_pssm_hyparams['n_steps'] + hard_pssm_hyparams['n_steps'] > MAX_OPTIMIZER_STEPS:
    raise ValueError("Total number of PSSM steps is greater than 100. Please reduce the number of steps such that the total number of steps is less than or equal to 100.")
# ------------------------------------------------------------------------------------------------------------
# Setup: Initial Seq Design, Load in Models (Boltz & Soluble MPNN) Define Composite Loss Function
# -------------------------------------------------------------------------------------------------------------
# 1. Extract motif coordinates
motif_ca_coords_first = np.load(os.path.join(DATA_DIR, f"motif_ca_coords_{MOTIF_CHAIN_ORDER[0]}.npy"))
motif_cb_coords_first = np.load(os.path.join(DATA_DIR, f"motif_cb_coords_{MOTIF_CHAIN_ORDER[0]}.npy"))
motif_ca_coords_second = np.load(os.path.join(DATA_DIR, f"motif_ca_coords_{MOTIF_CHAIN_ORDER[1]}.npy"))
motif_cb_coords_second = np.load(os.path.join(DATA_DIR, f"motif_cb_coords_{MOTIF_CHAIN_ORDER[1]}.npy"))

# 2. Create motif distograms
motif_distogram_first = coords_to_distogram(motif_cb_coords_first)
motif_distogram_second = coords_to_distogram(motif_cb_coords_second)

# 3. Define Initial Binder Sequence
binder_seq = ("X" * LINKER_LEN1) + CHAIN_MOTIF[MOTIF_CHAIN_ORDER[0]]['seq'] + ("X" * LINKER_LEN2) + CHAIN_MOTIF[MOTIF_CHAIN_ORDER[1]]['seq'] + ("X" * LINKER_LEN3)
CHAIN_MOTIF[MOTIF_CHAIN_ORDER[0]]['pos_design'] = list(range(LINKER_LEN1, LINKER_LEN1 + len(CHAIN_MOTIF[MOTIF_CHAIN_ORDER[0]]['seq'])))
CHAIN_MOTIF[MOTIF_CHAIN_ORDER[1]]['pos_design'] = list(range(binder_seq.find(CHAIN_MOTIF[MOTIF_CHAIN_ORDER[1]]['seq']), 
                                                             binder_seq.find(CHAIN_MOTIF[MOTIF_CHAIN_ORDER[1]]['seq']) + len(CHAIN_MOTIF[MOTIF_CHAIN_ORDER[1]]['seq'])))
motif_first_indices = jnp.array(CHAIN_MOTIF[MOTIF_CHAIN_ORDER[0]]['pos_design'])
motif_second_indices = jnp.array(CHAIN_MOTIF[MOTIF_CHAIN_ORDER[1]]['pos_design'])
# 4. Load in Initial Models
model_boltz = Boltz2()
model_mpnn = load_mpnn_sol(0.05)

# 5. Define composite loss function
BINDER_LENGTH = len(binder_seq)
bias = (jnp.zeros((BINDER_LENGTH, 20)).at[:BINDER_LENGTH, TOKENS.index('C')].set(-1e6))

# 5.1, define loss function for structure prediction of entire binder-target complex
structure_prediction_loss = ((WEIGHT_BINDER_CONTACT_LOSS_FUNCTION * sp.BinderTargetContact()) 
                             + (WEIGHT_WITHIN_BINDER_CONTACT_LOSS_FUNCTION * sp.WithinBinderContact()) 
                             + (WEIGHT_INVERSE_FOLDING_SEQ_RECOVERY_LOSS_FUNCTION * sp.InverseFoldingSequenceRecovery(model_mpnn, temp = jnp.array(0.001), bias = bias)) 
                             + (WEIGHT_TARGET_BINDER_PAE_LOSS_FUNCTION * sp.TargetBinderPAE()) 
                             + (WEIGHT_BINDER_TARGET_PAE_LOSS_FUNCTION * sp.BinderTargetPAE()) 
                             + (WEIGHT_WITHIN_BINDER_PAE_LOSS_FUNCTION * sp.WithinBinderPAE()) 
                             + (WEIGHT_IPTM_LOSS_FUNCTION * sp.IPTMLoss()) 
                             + (WEIGHT_PTM_ENERGY_LOSS_FUNCTION * sp.pTMEnergy()) 
                             + (WEIGHT_PLDDT_LOSS_FUNCTION * sp.PLDDTLoss()))
# 5.2, define motif-specific loss functions for each motif
motif_first_loss = ((WEIGHT_MOTIF_DISTOGRAM_LOSS_FUNCTION * MotifDistogramCE(motif_distogram_first, motif_first_indices)) + (WEIGHT_MOTIF_RMSD_LOSS_FUNCTION * MotifRMSDLoss(motif_ca_coords_first, motif_first_indices)))
motif_second_loss = ((WEIGHT_MOTIF_DISTOGRAM_LOSS_FUNCTION * MotifDistogramCE(motif_distogram_second, motif_second_indices)) + (WEIGHT_MOTIF_RMSD_LOSS_FUNCTION * MotifRMSDLoss(motif_ca_coords_second, motif_second_indices)))

# 5.3, define composite loss function for entire binder-target complex and motifs
loss_fn = structure_prediction_loss + motif_first_loss + motif_second_loss

# 5.4, establish loss function derived from Boltz2 Model
features, _ = boltz_features, boltz_writer = model_boltz.binder_features(binder_length = BINDER_LENGTH, chains = [TargetChain(sequence = SEQ_TARGET, use_msa = True)])

loss_fn_boltz = model_boltz.build_multisample_loss(
    loss = loss_fn,
    features = features,
    recycling_steps = 1,
    num_samples = 4
)

# 5.5, Add Wrapper around the Boltz Loss Function such that gradients only flow through "X" residues or mutable residues
masked_loss = SetPositions.from_sequence(wildtype = binder_seq, loss = loss_fn_boltz)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Optimization: Define Initial PSSM and Optimize in 3 Stages: Soft -> Sharp -> Discrete
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. Create Initial PSSM
num_mutable_residues = len(masked_loss.variable_positions)
pssm_initial = np.random.uniform(low = 0.25, high = 0.75) * jax.random.gumbel(key = jax.random.key(np.random.randint(100000000)), shape = (num_mutable_residues, 20))

# 2. Generate an initial, "soft" (non-sparse) PSSM 
_, pssm = simplex_APGM(loss_function= masked_loss,
                       x=jax.nn.softmax(pssm_initial),
                       n_steps= soft_pssm_hyparams['n_steps'],
                       stepsize=soft_pssm_hyparams['stepsize'],
                       momentum=soft_pssm_hyparams['momentum'],
                       scale=soft_pssm_hyparams['scale'],
                       logspace= soft_pssm_hyparams['logspace'],
                       max_gradient_norm=1.0,
                       )

# 3. Sharpen the PSSM into a discrete sequence (e.g. a one-hot PSSM)
_, pssm = simplex_APGM(loss_function= masked_loss,
                       x=jnp.log(pssm + 1e-5),
                       n_steps= sharp_pssm_hyparams['n_steps'],
                       stepsize=sharp_pssm_hyparams['stepsize'],
                       momentum=sharp_pssm_hyparams['momentum'],
                       scale=sharp_pssm_hyparams['scale'],
                       logspace= sharp_pssm_hyparams['logspace'],
                       max_gradient_norm=1.0,
                       )
# 4. Further sharpen the PSSM into a discrete sequence (e.g. a one-hot PSSM)
_, pssm = simplex_APGM(loss_function= masked_loss,
                       x=jnp.log(pssm + 1e-5),
                       n_steps= hard_pssm_hyparams['n_steps'],
                       stepsize=hard_pssm_hyparams['stepsize'],
                       momentum=hard_pssm_hyparams['momentum'],
                       scale=hard_pssm_hyparams['scale'],
                       logspace= hard_pssm_hyparams['logspace'],
                       max_gradient_norm=1.0,
                       )

# 5. Add fixed residues back into the PSSM
pssm_with_fixed_residues = masked_loss.sequence(pssm)

# --------------------------------------------------------------------------------------------------------------
# Final Evaluation
# --------------------------------------------------------------------------------------------------------------
design_iteration = sum(".pdb" in x for x in os.listdir(DATA_DIR)) - 1 # Account for the initial PDB file used for the initial motifs
composite_score = evaluate_optimized_structure(model_boltz = model_boltz, seq_pssm = pssm_with_fixed_residues, motif_id_pos = CHAIN_MOTIF,
                                                                                          design_iteration = design_iteration)