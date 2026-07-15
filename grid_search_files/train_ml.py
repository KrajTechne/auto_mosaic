"""
Autoresearch pretraining script adapted to Motif Scaffolding. Single-GPU, single-file.
Consolidated from review of Escalante Bio's Mosaic Database & Sergey Ovchinikov's ColabDesign (AF2 Gradient Backpropagation Design)
Usage: uv run train.py
"""

import os
os.environ['PYTORCH_ALLOC_CONF'] = "expandable_segments:True"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"  # Use 95% of GPU (default 75%)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   # Allocate on-demand, not upfront

import argparse
import json
import gc
import math
import time
import numpy as np
import sklearn.metrics
from sklearn.metrics.pairwise import pairwise_distances

import biotite
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx

import jax
import jax.numpy as jnp
import torch
from jaxtyping import Array, Float, Int
import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

# 2. Wake up JAX and claim its share of the GPU
_ = jnp.zeros(1).block_until_ready()

# 3. Wake up PyTorch so Triton can find the driver context
torch.cuda.init()

import mosaic
from mosaic.proteinmpnn.mpnn import load_mpnn_sol
from mosaic.models.boltz2 import Boltz2
from mosaic.models.esmfold2 import ESMFold2Full
from mosaic.losses.esmc import ESMCPseudoPerplexity, load_esmc
from mosaic.common import LossTerm
from mosaic.structure_prediction import StructureModelOutput
import mosaic.losses.structure_prediction as sp
from mosaic.common import TOKENS
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
from mosaic.losses.transformations import NoCys, SetPositions, SoftClip
from mosaic.structure_prediction import TargetChain
from mosaic.optimizers import simplex_APGM

from prepare import SEQ_TARGET, CHAIN_MOTIF, MAX_OPTIMIZER_STEPS, DATA_DIR, PATH_INPUT_STRUCTURE, calculate_motif_rmsd, evaluate_optimized_structure, compute_composite_score, compute_harmonic_mean, generate_template_motif_annotation, extract_gemmi_chain, generate_template_target_annotation

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

class JointMotifRMSDLoss(LossTerm):
    """
    Penalizes structural deviation of one motif from its native backbone,
    using a single Kabsch alignment fit jointly over both motifs' Ca
    coordinates (rather than aligning each motif independently). This ties
    each motif's RMSD to the relative position/orientation of the two
    motifs, not just its own local fold.
    """
    target_coords_self: Float[Array, "M 3"]   # Native Ca coordinates of this motif
    motif_positions_self: Int[Array, "M"]     # Sequence indices of this motif
    target_coords_other: Float[Array, "K 3"]  # Native Ca coordinates of the other motif
    motif_positions_other: Int[Array, "K"]    # Sequence indices of the other motif
    name: str = "motif_rmsd"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        # 1. Extract Predicted C-alpha coordinates for both motifs (Index 1 is CA)
        pred_self = output.backbone_coordinates[self.motif_positions_self, 1, :]
        pred_other = output.backbone_coordinates[self.motif_positions_other, 1, :]

        P = jnp.concatenate([pred_self, pred_other], axis=0)
        T = jnp.concatenate([self.target_coords_self, self.target_coords_other], axis=0)

        # 2. Center both sets of coordinates on their joint centroid
        P_mu = P.mean(axis=0)
        T_mu = T.mean(axis=0)
        P_c = P - P_mu
        T_c = T - T_mu

        # 3. ColabDesign's Kabsch logic (pure JAX), fit jointly over both motifs
        ab = jnp.swapaxes(P_c, -1, -2) @ T_c
        u, s, vh = jnp.linalg.svd(ab, full_matrices=False)

        # Reflection check
        flip = jnp.linalg.det(u @ vh) < 0
        u_ = jnp.where(flip, -u[..., -1].T, u[..., -1].T).T
        u = u.at[..., -1].set(u_)

        # Alignment matrix
        R = u @ vh

        # 4. Apply the joint alignment to this motif's coordinates and compute RMSD
        pred_self_aligned = ((pred_self - P_mu) @ R) + T_mu

        msd = jnp.mean(jnp.sum((pred_self_aligned - self.target_coords_self) ** 2, axis=-1))
        rmsd = jnp.sqrt(msd + 1e-8)

        return rmsd, {self.name: rmsd}

class AntiHelixLoss(LossTerm):
    """
    Penalizes excess alpha-helical content in the binder scaffold to encourage
    a mix of secondary structures (e.g. beta strands), using mosaic's i,i+3
    contact-probability signature with the inequality reversed relative to
    mosaic's HelixLoss.
    """
    max_distance: float = 6.0
    target_value: float = 0.0
    name: str = "anti_helix"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        # 1. Restrict to the binder-binder block of the distogram
        binder_len = sequence.shape[0]

        # 2. Log-probability that residues i and i+3 are in contact (<= max_distance).
        #    ~0 => alpha-helix-like (i,i+3 almost always close); -> -inf => extended/beta-like
        log_contact = sp.contact_log_probability(
            output.distogram_logits[:binder_len, :binder_len],
            self.max_distance,
            bins=output.distogram_bins,
        )
        value = jnp.diagonal(log_contact, 3).mean()

        # 3. Penalize helical character (value near 0); bounded reward as value -> -inf
        loss = jax.nn.elu(self.target_value + value)

        return loss, {self.name: loss}
    
class SheetBiasLoss(LossTerm):
    """
    Rewards beta-sheet content in the binder scaffold by encouraging the
    strand-strand contact ladder pattern: if residues i,j are in contact,
    (i+2, j+2) should also be in contact (parallel strands) and/or
    (i+2, j-2) should be in contact (antiparallel strands).
    Minimizing this loss (which returns a negative sheet score) maximizes
    beta-sheet character. Adapted from bjing2016/switchcraft SheetBiasLoss.
    """
    max_distance: float = 6.0
    min_seq_sep: int = 5 # tried with 5 for all others
    name: str = "sheet_bias"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        shift = 2

        # Log contact probabilities for the binder-binder block — stay in log space
        # to avoid double-suppressed gradients from the probability product x[i,j]*x[i+2,j+2]
        log_x = sp.contact_log_probability(
            output.distogram_logits[:binder_len, :binder_len],
            self.max_distance,
            bins=output.distogram_bins,
        )  # (L, L), values in (-inf, 0]

        L = binder_len
        i_idx = jnp.arange(L)
        sep_mask = (jnp.abs(i_idx[:, None] - i_idx[None, :]) >= self.min_seq_sep).astype(jnp.float32)

        eps = 1e-8

        # log(joint contact prob) = log_x[i,j] + log_x[i+2,j+2] — no exp, no underflow
        log_par_ladder = log_x[:L - shift, :L - shift] + log_x[shift:, shift:]
        log_anti_ladder = log_x[:L - shift, shift:] + log_x[shift:, :L - shift]

        par_m = sep_mask[:L - shift, :L - shift]
        anti_m = sep_mask[:L - shift, shift:]

        # Numerically stable mean via logsumexp: log(mean joint contact prob)
        par_score = jax.nn.logsumexp(jnp.where(par_m > 0, log_par_ladder, -jnp.inf).ravel()) - jnp.log(par_m.sum() + eps)
        anti_score = jax.nn.logsumexp(jnp.where(anti_m > 0, log_anti_ladder, -jnp.inf).ravel()) - jnp.log(anti_m.sum() + eps)

        sheet_score = 0.5 * (par_score + anti_score)  # in (-inf, 0], higher = more sheets

        # Stay in log space: loss in [0, inf), lower = more sheets
        # Gradient is O(1) via logsumexp softmax — not suppressed by exp(sheet_score)
        loss = -sheet_score

        return loss, {self.name: loss}
# -----------------------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# -----------------------------------------------------------------------------------------

def initialize_loss_function(loss_fn_weights: dict, motif_chain_order: list = ['A', 'D']):
    """ Initialize the overall composite loss function with a set of pre-defined weights for each sub-component loss function """
    # Define Lengths of Linkers between Motifs
    LINKER_LEN1 = 50 # Length of linker from start of protein (N-terminus) to first motif
    LINKER_LEN2 = 75 # Length of linker between first and second motif
    LINKER_LEN3 = 50 # Length of linker between second motif and end of protein (C-terminus)
    MOTIF_CHAIN_ORDER = motif_chain_order

    # Define weights of the respective loss functions in the final total/composite loss function -------------------------------------
    # Final loss function = sum of all loss functions weighted by their respective weights -------------------------------------------
    
    # Weight of the binder contact loss function in the total/composite loss function
    WEIGHT_BINDER_CONTACT_LOSS_FUNCTION = loss_fn_weights['binder_contact'] 
    # Weight of the within-binder contact loss function in the total/composite loss function
    WEIGHT_WITHIN_BINDER_CONTACT_LOSS_FUNCTION = loss_fn_weights['within_binder_contact'] 
    # Weight of the inverse folding sequence recovery loss function in the total/composite loss function
    WEIGHT_INVERSE_FOLDING_SEQ_RECOVERY_LOSS_FUNCTION = loss_fn_weights['inverse_folding_seq_recovery']
    # Weight of the target to binder (directional PAE) PAE loss function in the total/composite loss function
    WEIGHT_TARGET_BINDER_PAE_LOSS_FUNCTION = loss_fn_weights['target_binder_pae']
    # Weight of the binder to target (directional PAE) PAE loss function in the total/composite loss function
    WEIGHT_BINDER_TARGET_PAE_LOSS_FUNCTION = loss_fn_weights['binder_target_pae']
    # Weight of the within-binder PAE loss function in the total/composite loss function
    WEIGHT_WITHIN_BINDER_PAE_LOSS_FUNCTION = loss_fn_weights['within_binder_pae'] 
    # Weight of the iptm loss function in the total/composite loss
    WEIGHT_IPTM_LOSS_FUNCTION = loss_fn_weights['iptm'] 
    # Weight of the ptm energy loss function in the total/composite loss
    WEIGHT_PTM_ENERGY_LOSS_FUNCTION = loss_fn_weights['ptm_energy']
    # Weight of the plddt loss function in the total/composite loss
    WEIGHT_PLDDT_LOSS_FUNCTION = loss_fn_weights['plddt']
    # Weight of the anti-helix loss function (penalizes excess alpha-helical content, encourages mixed secondary structure) in the total/composite loss
    WEIGHT_ANTI_HELIX_LOSS_FUNCTION = loss_fn_weights['anti_helix']
    # (Saw beta strand at 2,5,8) Weight of the bias-sheet loss function (encourages excess beta-sheet content in the total/composite loss
    WEIGHT_BIAS_SHEET_LOSS_FUNCTION = loss_fn_weights['bias_sheet'] 
    # Weight of the first motif distogram loss function in the total/composite loss
    WEIGHT_FIRST_MOTIF_DISTOGRAM_LOSS_FUNCTION = loss_fn_weights['first_motif_distogram']
    # Weight of the first motif rmsd loss function in the total/composite loss
    WEIGHT_FIRST_MOTIF_RMSD_LOSS_FUNCTION = loss_fn_weights['first_motif_rmsd'] 
    # Weight of the second motif distogram loss function in the total/composite loss
    WEIGHT_SECOND_MOTIF_DISTOGRAM_LOSS_FUNCTION = loss_fn_weights['second_motif_distogram']
    # Weight of the second motif rmsd loss function in the total/composite loss
    WEIGHT_SECOND_MOTIF_RMSD_LOSS_FUNCTION = loss_fn_weights['second_motif_rmsd']

    # Initial "soft" PSSM -> Try to sharpen PSSM into a discrete sequence (e.g. one-hot PSSM) -> Further sharpening with a "hard" PSSM
    # Mutable Seq Len:
    SEQ_LEN_MUTABLE = LINKER_LEN1 + LINKER_LEN2 + LINKER_LEN3
    # Optimizer Parameters: 
    soft_pssm_hyparams = {
        'n_steps' : 100,
        'stepsize' : 0.2 * np.sqrt(SEQ_LEN_MUTABLE),
        'momentum' : 0.3,
        'scale'    : 1.00,
        'logspace' : False,
        'max_gradient_norm' : 1.00
        }
    sharp_pssm_hyparams = {
        'n_steps' : 50,
        'stepsize' : 0.5 * np.sqrt(SEQ_LEN_MUTABLE),
        'momentum' : 0.0,
        'scale'    : 1.25,
        'logspace' : True,
        'max_gradient_norm' : 1.00
    }
    hard_pssm_hyparams = {
        'n_steps' : 15,
        'stepsize' : 0.5 * np.sqrt(SEQ_LEN_MUTABLE),
        'momentum' : 0.0,
        'scale'    : 1.40,
        'logspace' : True,
        'max_gradient_norm' : 1.00
    }

    # PSSM optimizer hyperparameters: Combine all optimizer hyperparameters into a single dictionary
    pssm_hyparams = {
        'soft' : soft_pssm_hyparams,
        'sharp' : sharp_pssm_hyparams,
        'hard' : hard_pssm_hyparams,
        'num_mutable_residues' : SEQ_LEN_MUTABLE
    }

    # If total number of pssm steps is > 100, raise error
    if soft_pssm_hyparams['n_steps'] + sharp_pssm_hyparams['n_steps'] + hard_pssm_hyparams['n_steps'] > MAX_OPTIMIZER_STEPS:
        raise ValueError(f"Total number of PSSM steps is greater than {MAX_OPTIMIZER_STEPS}. Please reduce the number of steps such that the total number of steps is less than or equal to {MAX_OPTIMIZER_STEPS}.")
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

    # 4. Create template structure for binder_seq & target
    list_atom_array_motifs = []
    for chain in MOTIF_CHAIN_ORDER:
        atom_array_motif = generate_template_motif_annotation(chain_motif = CHAIN_MOTIF, path_input_structure= PATH_INPUT_STRUCTURE, chain= chain)
        list_atom_array_motifs.append(atom_array_motif)

    atom_array_template_binder = struc.concatenate(list_atom_array_motifs)

    # Now for Target:
    atom_array_chain_target = generate_template_target_annotation(path_input_structure = PATH_INPUT_STRUCTURE, chain = "C")
    atom_array_template = struc.concatenate([atom_array_template_binder, atom_array_chain_target])
    gemmi_binder_chain = extract_gemmi_chain(atom_array_template= atom_array_template)
    gemmi_target_chain = extract_gemmi_chain(atom_array_template= atom_array_template, desired_chain = "B")

    # 5. Load in Initial Models
    model_boltz = Boltz2()
    model_mpnn = load_mpnn_sol(0.05)

    # 6. Define composite loss function
    BINDER_LENGTH = len(binder_seq)
    bias = (jnp.zeros((BINDER_LENGTH, 20)).at[:BINDER_LENGTH, TOKENS.index('C')].set(-1e6))

    # 6.1, define loss function for structure prediction of entire binder-target complex
    structure_prediction_loss = ((WEIGHT_BINDER_CONTACT_LOSS_FUNCTION * sp.BinderTargetContact()) 
                                 + (WEIGHT_WITHIN_BINDER_CONTACT_LOSS_FUNCTION * sp.WithinBinderContact()) 
                                 + (WEIGHT_INVERSE_FOLDING_SEQ_RECOVERY_LOSS_FUNCTION * InverseFoldingSequenceRecovery(model_mpnn, temp = jnp.array(0.001), 
                                                                bias = bias)) 
                                 + (WEIGHT_TARGET_BINDER_PAE_LOSS_FUNCTION * sp.TargetBinderPAE()) 
                                 + (WEIGHT_BINDER_TARGET_PAE_LOSS_FUNCTION * sp.BinderTargetPAE()) 
                                 + (WEIGHT_WITHIN_BINDER_PAE_LOSS_FUNCTION * sp.WithinBinderPAE()) 
                                 + (WEIGHT_IPTM_LOSS_FUNCTION * sp.IPTMLoss()) 
                                 + (WEIGHT_PTM_ENERGY_LOSS_FUNCTION * sp.pTMEnergy()) 
                                 + (WEIGHT_PLDDT_LOSS_FUNCTION * sp.PLDDTLoss())
                                 + (WEIGHT_ANTI_HELIX_LOSS_FUNCTION * AntiHelixLoss())
                                 + (WEIGHT_BIAS_SHEET_LOSS_FUNCTION * SheetBiasLoss()))
    # 6.2, define motif-specific loss functions for each motif
    motif_first_loss = ((WEIGHT_FIRST_MOTIF_DISTOGRAM_LOSS_FUNCTION * MotifDistogramCE(motif_distogram_first, motif_first_indices)) + (WEIGHT_FIRST_MOTIF_RMSD_LOSS_FUNCTION * JointMotifRMSDLoss(motif_ca_coords_first, motif_first_indices, motif_ca_coords_second, motif_second_indices)))
    motif_second_loss = ((WEIGHT_SECOND_MOTIF_DISTOGRAM_LOSS_FUNCTION * MotifDistogramCE(motif_distogram_second, motif_second_indices)) + (WEIGHT_SECOND_MOTIF_RMSD_LOSS_FUNCTION * JointMotifRMSDLoss(motif_ca_coords_second, motif_second_indices, motif_ca_coords_first, motif_first_indices)))
    # Add inter-cross motif distogram loss
    combined = jnp.concatenate([motif_first_indices, motif_second_indices])
    combined_target_dgram = coords_to_distogram(np.concatenate([motif_cb_coords_first, motif_cb_coords_second]))
    inter_loss = 1.0 * MotifDistogramCE(combined_target_dgram, combined)

    # 6.3, define composite loss function for entire binder-target complex and motifs
    loss_fn = structure_prediction_loss + motif_first_loss + motif_second_loss + inter_loss

    # 6.4, establish loss function derived from Boltz2 Model
    features, _ = model_boltz.target_only_features(
        chains = [TargetChain(sequence = binder_seq, use_msa = False, template_chain = gemmi_binder_chain), 
                  TargetChain(sequence = SEQ_TARGET, use_msa = True, template_chain = gemmi_target_chain)])

    loss_fn_boltz = model_boltz.build_multisample_loss(
        loss = loss_fn,
        features = features,
        sampling_steps = 5,
        recycling_steps = 1,
        num_samples = 4
    )

    # 6.45: Add in ESMC for another loss function component
    # mutable (X/linker) positions within the binder — not the fixed motif residues
    #designable = jnp.array([i for i, c in enumerate(binder_seq) if c == "X"])

    #pll = ESMCPseudoPerplexity(esm=load_esmc("esmc_300m"),
    #                           design_idx=designable,   # only mask/score the designed positions
    #                           num_samples=4,)

    design_loss = loss_fn_boltz

    # 6.5, Add Wrapper around the Boltz Loss Function such that gradients only flow through "X" residues or mutable residues
    masked_loss = SetPositions.from_sequence(wildtype = binder_seq, loss = design_loss)

    # 6.6 Add NoCys Wrapper around the Masked Positions & Boltz Loss Function to ensure "Cys" are not sampled as binder residues
    masked_cys_loss = NoCys(masked_loss)

    return masked_cys_loss, pssm_hyparams, model_boltz

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Optimization: Define Initial PSSM and Optimize in 3 Stages: Soft -> Sharp -> Discrete
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
def optimize_pssm(masked_cys_loss, pssm_hyparams, seed: int):
    """
    This function takes in a masked_cys_loss function and a pssm_hyparams dictionary as inputs. It then performs three stages of optimization on the PSSM to generate a final, discrete PSSM. The function returns the final, discrete PSSM.
    """

    # 1. Create Initial PSSM
    num_mutable_residues = pssm_hyparams['num_mutable_residues']
    # 2. Initialize Key from the fixed seed (same seed reused across grid configs == common random
    # numbers, so differences in downstream metrics reflect the weights, not the initialization)
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)
    # 19 potential residues since Cys are explicitly not included in the PSSM
    pssm_initial = rng.uniform(low = 0.25, high = 0.75) * jax.random.gumbel(key = key, shape = (num_mutable_residues, 19))

    # 2. Generate an initial, "soft" (non-sparse) PSSM 
    _, pssm = simplex_APGM(loss_function= masked_cys_loss,
                           x=jax.nn.softmax(pssm_initial),
                           n_steps= pssm_hyparams['soft']['n_steps'],
                           stepsize= pssm_hyparams['soft']['stepsize'],
                           momentum= pssm_hyparams['soft']['momentum'],
                           scale= pssm_hyparams['soft']['scale'],
                           logspace= pssm_hyparams['soft']['logspace'],
                           max_gradient_norm= pssm_hyparams['soft']['max_gradient_norm'],
                           key = key
                       )

    # 3. Sharpen the PSSM into a discrete sequence (e.g. a one-hot PSSM)
    pssm, _ = simplex_APGM(loss_function= masked_cys_loss,
                       x=jnp.log(pssm + 1e-5),
                       n_steps= pssm_hyparams['sharp']['n_steps'],
                       stepsize=pssm_hyparams['sharp']['stepsize'],
                       momentum=pssm_hyparams['sharp']['momentum'],
                       scale=pssm_hyparams['sharp']['scale'],
                       logspace= pssm_hyparams['sharp']['logspace'],
                       max_gradient_norm= pssm_hyparams['sharp']['max_gradient_norm'],
                       key = key
                       )
    # 4. Further sharpen the PSSM into a discrete sequence (e.g. a one-hot PSSM)
    pssm, _ = simplex_APGM(loss_function= masked_cys_loss,
                       x=jnp.log(pssm + 1e-5),
                       n_steps=  pssm_hyparams['hard']['n_steps'],
                       stepsize= pssm_hyparams['hard']['stepsize'],
                       momentum= pssm_hyparams['hard']['momentum'],
                       scale= pssm_hyparams['hard']['scale'],
                       logspace= pssm_hyparams['hard']['logspace'],
                       max_gradient_norm= pssm_hyparams['hard']['max_gradient_norm'],
                       key = key
                       )

    # 5. Add Cysteine back into the PSSM
    pssm_with_cysteine = masked_cys_loss.sequence(pssm)
    # 6. Add Fixed Residues back into PSSM
    pssm_with_fixed_residues = masked_cys_loss.loss.sequence(pssm_with_cysteine)

    return pssm_with_fixed_residues

# --------------------------------------------------------------------------------------------------------------
# Final Evaluation
# --------------------------------------------------------------------------------------------------------------
def evaluate_optimized_pssm(pssm_optimized, model_boltz, CHAIN_MOTIF, DATA_DIR):
    """
    This function takes in an optimized PSSM, a model_boltz object, a CHAIN_MOTIF dictionary, and a DATA_DIR string. It then evaluates the optimized PSSM by predicting the structure of the optimized PSSM using the model_boltz object. The function returns the RMSD between the predicted structure and the original structure, the RMSD between the predicted structure and the design target, the predicted interface score, and the predicted PDB file path.
    """

    design_iteration = sum(".pdb" in x for x in os.listdir(DATA_DIR)) # Initial PDB not stored in DATA_DIR
    # 0. Start Prequel Print Statements:
    print("-" * 50)
    print(f"For Design Iteration: {design_iteration}")
    print(" ")  # Add empty space after output header and is also used to separate individual struc_model outputs
    # 1. Predict Structure via Boltz2
    rmsd_a_boltz, rmsd_d_boltz, iptm_boltz, plddt_boltz = evaluate_optimized_structure(struc_model = model_boltz, seq_pssm= pssm_optimized, motif_id_pos= CHAIN_MOTIF, design_iteration = design_iteration, model_name = "Boltz2")
    # 2. Predict Structure via ESMFold2
    model_esm2 = ESMFold2Full()
    rmsd_a_esmfold, rmsd_d_esmfold, iptm_esmfold, plddt_esmfold = evaluate_optimized_structure(struc_model= model_esm2,
                                                                                               seq_pssm= pssm_optimized,
                                                                                               motif_id_pos= CHAIN_MOTIF,
                                                                                               design_iteration = design_iteration,
                                                                                               model_name= "ESMFold2")
    # 3. Compute harmonic means for each model's: Boltz2 and ESMFold2 metrics: (rmsd_a, rmsd_d, iptm, plddt)
    struc_model_metrics = {'rmsd_a' : [rmsd_a_boltz, rmsd_a_esmfold],
                           'rmsd_d' : [rmsd_d_boltz, rmsd_d_esmfold],
                           'iptm' : [iptm_boltz, iptm_esmfold],
                           'plddt' : [plddt_boltz, plddt_esmfold]}
    hmean_dict = {}
    for metric, metric_pair in struc_model_metrics.items():
        metric_boltz, metric_esmfold = metric_pair
        hmean_dict[metric] = compute_harmonic_mean(metric_a = metric_boltz, metric_b = metric_esmfold)
    # 4. Display harmonic means (Boltz2 <-> ESMFold2 agreement) for the Agent to interpret
    print("Harmonic Means (Boltz2 <-> ESMFold2 agreement):")
    print(f"hmean_rmsd_A: {hmean_dict['rmsd_a']:.2f}")
    print(f"hmean_rmsd_D: {hmean_dict['rmsd_d']:.2f}")
    print(f"hmean_ipTM: {hmean_dict['iptm']:.2f}")
    print(f"hmean_pLDDT: {hmean_dict['plddt']:.2f}")
    print(" ")
    composite_score = compute_composite_score(motif_rmsd_a= hmean_dict['rmsd_a'], motif_rmsd_d = hmean_dict['rmsd_d'], 
                                              structure_iptm= hmean_dict['iptm'], binder_plddt= hmean_dict['plddt'])

    # 5. Return all of the metrics
    metrics = {'rmsd_a_boltz' : rmsd_a_boltz,
               'rmsd_d_boltz' : rmsd_d_boltz,
               'iptm_boltz' : iptm_boltz, 
               'plddt_boltz' : plddt_boltz,
               'rmsd_a_esmfold' : rmsd_a_esmfold,
               'rmsd_d_esmfold' : rmsd_d_esmfold,
               'iptm_esmfold' : iptm_esmfold,
               'plddt_esmfold' : plddt_esmfold,
               'hmean_rmsd_a' : hmean_dict['rmsd_a'],
               'hmean_rmsd_d' : hmean_dict['rmsd_d'],
               'hmean_iptm' : hmean_dict['iptm'],
               'hmean_plddt' : hmean_dict['plddt'],
               'composite_score' : composite_score,
               'design_iteration' : design_iteration}
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('loss_fn_weights', type = json.loads, help = "Dictionary with sub-composite loss function weights")
    parser.add_argument('--seed', type = int, default = 0, help = "Seed for PSSM initialization; reuse the same seed across a grid sweep for common-random-numbers comparability")
    parser.add_argument('--parent_run_id', type = str, default = None, help = "MLflow run id of the parent sweep run to nest this trial's run under")
    parser.add_argument('--result_path', type = str, default = None, help = "Path to write this trial's metrics as JSON, so a driver (e.g. an Optuna objective) can read back a score")
    args = parser.parse_args()
    loss_fn_weights = args.loss_fn_weights
    print(f"loss_fn_weights: {loss_fn_weights}")

    run_tags = {}
    if args.parent_run_id is not None:
        parent_run = mlflow.get_run(args.parent_run_id)
        mlflow.set_experiment(experiment_id = parent_run.info.experiment_id)
        run_tags[MLFLOW_PARENT_RUN_ID] = args.parent_run_id

    with mlflow.start_run(tags = run_tags):
        mlflow.log_params(loss_fn_weights)
        mlflow.log_param('seed', args.seed)

        # 1. Initialize Loss Function, Define optimizer hyparams, and initialize the Boltz model
        loss_fn, pssm_hyparams, model_boltz = initialize_loss_function(loss_fn_weights= loss_fn_weights)
        # 2. Using the defined loss function -> optimize pssm
        pssm_optimized = optimize_pssm(masked_cys_loss= loss_fn, pssm_hyparams= pssm_hyparams, seed= args.seed)
        # 3. Evaluate Final Optimized PSSM
        metrics = evaluate_optimized_pssm(pssm_optimized= pssm_optimized, model_boltz = model_boltz, CHAIN_MOTIF= CHAIN_MOTIF, DATA_DIR= DATA_DIR)

        # 4. Log metrics and the predicted structures for this trial
        design_iteration = metrics.pop('design_iteration')
        mlflow.log_param('design_iteration', design_iteration)
        # Cast to plain floats: iptm/plddt-derived values are jax Array scalars, and mlflow's
        # metric store expects real floats (a logged jax Array silently serializes as garbage).
        metrics = {name: float(value) for name, value in metrics.items()}
        mlflow.log_metrics(metrics)
        for model_name in ("Boltz2", "ESMFold2"):
            pdb_path = os.path.join(DATA_DIR, f"{model_name}_structure_{design_iteration}.pdb")
            if os.path.exists(pdb_path):
                mlflow.log_artifact(pdb_path)

        # 5. Hand the score back to whatever process launched this trial (e.g. an Optuna objective)
        if args.result_path is not None:
            with open(args.result_path, 'w') as f:
                json.dump(metrics, f)

if __name__ == "__main__":
    main()



