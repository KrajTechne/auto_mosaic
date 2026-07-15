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

from prepare_single import SEQ_TARGET, CHAIN_MOTIF, MAX_OPTIMIZER_STEPS, DATA_DIR, PATH_INPUT_STRUCTURE, calculate_motif_rmsd, evaluate_optimized_structure, compute_composite_score, compute_harmonic_mean, generate_template_motif_annotation, extract_gemmi_chain, generate_template_target_annotation

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


def derive_interface_register(
    motif_cb_coords: np.ndarray,   # [M, 3] native CB of motif residues, in motif_binder_indices order
    target_cb_coords: np.ndarray,  # [T, 3] native CB of target residues, in SEQ_TARGET order
    motif_binder_indices: np.ndarray,  # [M] binder-relative position of each motif CB above
    contact_threshold: float = 8.0,
):
    """Derive a register of specific motif<->target contacts from the NATIVE complex.

    Returns (motif_idx, target_idx) arrays of equal length K, matched k-by-k:
      motif_idx[k]  : BINDER-relative position of a motif residue
      target_idx[k] : 0-based position WITHIN THE TARGET chain of its native partner
    A pair is included when the native CB-CB distance is < contact_threshold.

    IMPORTANT: this is only correct if `target_cb_coords` is ordered identically
    to SEQ_TARGET (i.e. target token t == row t here). Verify that against your
    `generate_template_target_annotation` / SEQ_TARGET before trusting it; a
    mis-ordered target silently optimizes the wrong contacts.
    """
    d = pairwise_distances(motif_cb_coords, target_cb_coords, metric="euclidean")  # [M, T]
    m_local, t_local = np.where(d < contact_threshold)  # native contacting pairs
    motif_idx = np.asarray(motif_binder_indices)[m_local].astype(np.int32)
    target_idx = t_local.astype(np.int32)  # already 0-based within target
    return jnp.array(motif_idx), jnp.array(target_idx)

def build_contact_groups(motif_idx, target_idx):
    """Convert a flat k-by-k register into grouped, padded arrays for Option 3.

    Groups the (motif, target) pairs by unique motif residue. Returns:
      motif_group_idx : [G]      unique motif (binder) positions
      target_groups   : [G, Tmax] target (target-relative) candidates, zero-padded
      mask            : [G, Tmax] 1 for real candidates, 0 for padding
    Build once at construction time (numpy); pass the arrays into the LossTerm.
    """
    motif_idx = np.asarray(motif_idx).tolist()
    target_idx = np.asarray(target_idx).tolist()
    groups: dict[int, list[int]] = {}
    for m, t in zip(motif_idx, target_idx):
        groups.setdefault(m, []).append(t)
    motif_group_idx = sorted(groups)
    Tmax = max(len(groups[m]) for m in motif_group_idx)
    G = len(motif_group_idx)
    target_groups = np.zeros((G, Tmax), dtype=np.int32)
    mask = np.zeros((G, Tmax), dtype=np.float32)
    for g, m in enumerate(motif_group_idx):
        ts = groups[m]
        target_groups[g, : len(ts)] = ts
        mask[g, : len(ts)] = 1.0
    return (
        jnp.array(motif_group_idx, dtype=jnp.int32),
        jnp.array(target_groups),
        jnp.array(mask),
    )
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

class MotifRMSDLoss(LossTerm):
    """
    Single-motif backbone RMSD via a standard Kabsch fit on ONE motif's Ca
    coordinates. Use this for single-motif scaffolding (no second motif to tie
    the relative placement to, so the joint fit of JointMotifRMSDLoss is not
    applicable). Identical Kabsch math, fit over just this motif.
    """
    target_coords: Float[Array, "M 3"]   # Native Ca coordinates of the motif
    motif_positions: Int[Array, "M"]     # Sequence indices of the motif in the binder
    name: str = "motif_rmsd"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        # 1. Predicted C-alpha coordinates for the motif (index 1 is CA)
        pred = output.backbone_coordinates[self.motif_positions, 1, :]
        T = self.target_coords

        # 2. Center on each set's centroid
        P_mu = pred.mean(axis=0)
        T_mu = T.mean(axis=0)
        P_c = pred - P_mu
        T_c = T - T_mu

        # 3. Kabsch (pure JAX), fit over this single motif
        ab = jnp.swapaxes(P_c, -1, -2) @ T_c
        u, s, vh = jnp.linalg.svd(ab, full_matrices=False)
        flip = jnp.linalg.det(u @ vh) < 0
        u_ = jnp.where(flip, -u[..., -1].T, u[..., -1].T).T
        u = u.at[..., -1].set(u_)
        R = u @ vh

        # 4. Apply alignment and compute RMSD
        pred_aligned = (P_c @ R) + T_mu
        msd = jnp.mean(jnp.sum((pred_aligned - T) ** 2, axis=-1))
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

class SpecificContactLoss(LossTerm):
    """
    Maximize log P(d < contact_distance) for SPECIFIC motif<->target residue
    pairs, read from the DISTOGRAM head. The distogram is trunk-derived
    (computed before the diffusion sample split), so this term is deterministic
    given the trunk, low-variance, and identical across the num_samples diffusion
    draws -- i.e. the smooth, noise-immune interface driver. Weight it heavily
    early.

    Index convention (matches the cross-chain terms in this codebase, which
    slice [:binder_len, binder_len:]):
      motif_idx  : 0-based positions WITHIN THE BINDER (subset of the motif
                   index arrays already used above).
      target_idx : 0-based positions WITHIN THE TARGET chain; offset by
                   binder_len internally. Pairs are matched k-by-k (register-
                   specific), NOT "any contact in a patch".

    Loss = -mean_k log P( d(motif_k, target_k) < contact_distance ).
    """
    motif_idx: Int[Array, "K"]
    target_idx: Int[Array, "K"]
    contact_distance: float = 8.0
    name: str = "specific_contact"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        tgt = self.target_idx + binder_len  # binder-relative target -> absolute token

        # [N, N] log-probability of contact for every token pair (full complex).
        log_p = sp.contact_log_probability(
            output.distogram_logits,
            self.contact_distance,
            bins=output.distogram_bins,
        )
        log_p_pairs = log_p[self.motif_idx, tgt]  # [K] register-specific pairs

        loss = -log_p_pairs.mean()
        return loss, {
            f"{self.name}_logp": log_p_pairs.mean(),
            f"{self.name}_p": jnp.exp(log_p_pairs).mean(),
            f"{self.name}_n_sat": (log_p_pairs > jnp.log(0.5)).sum(),
        }

class SpecificContactAnyPerRes(LossTerm):
    """-mean_g logsumexp_t( log P_{g,t} ).

    For each motif residue g, logsumexp over its candidate targets is a smooth
    max of the candidate contact log-probs -> "is this residue's BEST available
    contact made?" (OR within the group). Averaging -logsumexp across residues
    ANDs across residues -> "is EVERY residue anchored to at least one of its
    candidates?". This is the right form when a residue has several candidate
    partners and can physically reach only some of them.

    `temperature` sharpens the soft-max: logsumexp(lp / T) * T. T->0 approaches a
    hard max (reward only the single best candidate); T=1 is plain logsumexp
    (more gradient spread across candidates). Build the grouped arrays with
    `build_contact_groups`."""
    motif_group_idx: Int[Array, "G"]
    target_groups: Int[Array, "G T"]
    mask: Float[Array, "G T"]
    contact_distance: float = 8.0
    temperature: float = 1.0
    name: str = "contact_any"

    def __call__(self, sequence: Float[Array, "N 20"], output: StructureModelOutput, key):
        binder_len = sequence.shape[0]
        log_p = sp.contact_log_probability(
            output.distogram_logits, self.contact_distance, bins=output.distogram_bins
        )
        tgt = self.target_groups + binder_len                       # [G, T]
        lp = log_p[self.motif_group_idx[:, None], tgt]              # [G, T]
        # mask padding to -inf so it never contributes to the soft-max
        lp = jnp.where(self.mask > 0, lp, -jnp.inf)
        # soft-OR within each residue's candidate set
        per_group = self.temperature * jax.nn.logsumexp(lp / self.temperature, axis=-1)  # [G]
        loss = -per_group.mean()                                    # AND across residues
        # diagnostics: best-candidate contact prob per residue, and how many
        # residues have ANY candidate satisfied
        best_lp = lp.max(axis=-1)                                   # [G]
        return loss, {
            f"{self.name}_best_logp": best_lp.mean(),
            f"{self.name}_best_p": jnp.exp(best_lp).mean(),
            f"{self.name}_n_res_sat": (best_lp > jnp.log(0.5)).sum(),
            f"{self.name}_n_res": jnp.array(self.mask.shape[0]),
        }

class SpecificPAELoss(LossTerm):
    """
    Minimize expected PAE over SPECIFIC motif<->target residue pairs (both
    directions, symmetrized). PAE comes from the confidence head, which takes
    the diffusion-sampled coordinates as input, so this term is noisy across
    samples (like TargetBinderPAE / BinderTargetPAE). Use as confidence
    refinement at lower weight / later; the num_samples average smooths it.

    Same index convention as SpecificContactLoss.

    Loss = mean_k 0.5 * ( PAE(motif_k -> target_k) + PAE(target_k -> motif_k) ).
    """
    motif_idx: Int[Array, "K"]
    target_idx: Int[Array, "K"]
    name: str = "specific_pae"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        tgt = self.target_idx + binder_len

        # Expected PAE (Å) per token pair: sum_b softmax(logits)_b * bin_center_b
        # (same convention as interaction_prediction_score in the codebase).
        pae = jnp.sum(
            jax.nn.softmax(output.pae_logits, axis=-1) * output.pae_bins, axis=-1
        )  # [N, N]

        mt = pae[self.motif_idx, tgt]   # motif -> target direction
        tm = pae[tgt, self.motif_idx]   # target -> motif direction
        pae_pairs = 0.5 * (mt + tm)     # [K], symmetrized

        loss = pae_pairs.mean()
        return loss, {
            f"{self.name}": pae_pairs.mean(),
            f"{self.name}_max": pae_pairs.max(),
        }

class MotifScaffoldContactLoss(LossTerm):
    """Each motif residue should contact >=1 DISTAL scaffold residue (burial into
    the body), so the motif packs into the scaffold instead of hanging off it.
    Distogram-based (trunk-derived, smooth). Excludes near-neighbors so it's not
    trivially satisfied by the backbone."""
    motif_positions: Int[Array, "M"]        # binder-relative motif indices
    scaffold_positions: Int[Array, "S"]     # binder scaffold indices, EXCLUDING a
                                            # window around the motif (the distal core)
    contact_distance: float = 8.0
    name: str = "motif_scaffold_contact"

    def __call__(self, sequence, output, key):
        log_p = sp.contact_log_probability(
            output.distogram_logits, self.contact_distance, bins=output.distogram_bins)
        # motif (rows) x distal-scaffold (cols), both in the binder block
        lp = log_p[self.motif_positions[:, None], self.scaffold_positions[None, :]]  # [M, S]
        # each motif residue: reward its BEST scaffold contact (any-of), then
        # require every motif residue to have one -> mean of -logsumexp
        per_res = jax.nn.logsumexp(lp, axis=-1)   # [M]  soft-OR over scaffold
        loss = -per_res.mean()                    # AND across motif residues
        return loss, {f"{self.name}_best_p": jnp.exp(lp.max(-1)).mean(),
                      f"{self.name}_n_res_sat": (lp.max(-1) > jnp.log(0.5)).sum()}
# -----------------------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# -----------------------------------------------------------------------------------------

def initialize_loss_function(loss_fn_weights: dict, motif_chain_order: list = ['D']):
    """ Initialize the overall composite loss function with a set of pre-defined weights for each sub-component loss function.

    Single-motif scaffolding: MOTIF_CHAIN_ORDER holds ONE chain. The binder is
    [N-linker] + motif + [C-linker]; there is no second motif, so the inter-motif
    distogram and the joint-Kabsch RMSD are not used (single-motif RMSD instead).
    """
    # Define Lengths of Linkers around the single motif
    LINKER_LEN_N = 50  # N-terminal linker (start of protein -> motif)
    LINKER_LEN_C = 50  # C-terminal linker (motif -> end of protein)
    MOTIF_CHAIN_ORDER = motif_chain_order
    assert len(MOTIF_CHAIN_ORDER) == 1, "Single-motif scaffolding expects exactly one motif chain."

    # Define weights of the respective loss functions in the final total/composite loss function -------------------------------------
    # Final loss function = sum of all loss functions weighted by their respective weights -------------------------------------------
    
    # Weight of the binder contact loss function in the total/composite loss function
    WEIGHT_BINDER_TARGET_CONTACT_LOSS_FUNCTION = loss_fn_weights['binder_target_contact'] 
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
    # Second-motif weights are unused in single-motif scaffolding; `.get` so weight
    # dicts that still carry them don't error (and dicts without them also run).
    WEIGHT_SECOND_MOTIF_DISTOGRAM_LOSS_FUNCTION = loss_fn_weights.get('second_motif_distogram', 0.0)
    WEIGHT_SECOND_MOTIF_RMSD_LOSS_FUNCTION = loss_fn_weights.get('second_motif_rmsd', 0.0)
    # Weight of the specific motif<->target contact loss (distogram-based; smooth, trunk-derived).
    # `.get` so existing weight dicts without these keys still run (default 0 => disabled).
    WEIGHT_SPECIFIC_CONTACT_LOSS_FUNCTION = loss_fn_weights.get('specific_contact', 0.0)
    # Weight of the specific motif<->target PAE loss (confidence head; noisy, lower weight / later).
    WEIGHT_SPECIFIC_PAE_LOSS_FUNCTION = loss_fn_weights.get('specific_pae', 0.0)
    # Weight of the specific motif <-> scaffold contact loss (distogram-based; smooth, trunk-derived)
    WEIGHT_MOTIF_SCAFFOLD_CONTACT_LOSS_FUNCTION = loss_fn_weights.get('motif_scaffold_contact', 2.0)
    # Weight of the Radius of Gyration Loss Function: Keep designed protein globular or centered around COM
    WEIGHT_DISTOGRAM_RADIUS_GYRATION_LOSS_FUNCTION = loss_fn_weights.get('radius_gyration', 1.5)

    # Initial "soft" PSSM -> Try to sharpen PSSM into a discrete sequence (e.g. one-hot PSSM) -> Further sharpening with a "hard" PSSM
    # Mutable Seq Len: N-linker + C-linker (the motif residues are fixed, not mutable)
    SEQ_LEN_MUTABLE = LINKER_LEN_N + LINKER_LEN_C
    # Optimizer Parameters: 
    soft_pssm_hyparams = {
        'n_steps' : 100, # Change to 100 in morning
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
    # 1. Extract motif coordinates (single motif)
    motif_ca_coords_first = np.load(os.path.join(DATA_DIR, f"motif_ca_coords_{MOTIF_CHAIN_ORDER[0]}.npy"))
    motif_cb_coords_first = np.load(os.path.join(DATA_DIR, f"motif_cb_coords_{MOTIF_CHAIN_ORDER[0]}.npy"))

    # 2. Create motif distogram
    motif_distogram_first = coords_to_distogram(motif_cb_coords_first)

    # 3. Define Initial Binder Sequence: [N-linker] + motif + [C-linker]
    binder_seq = ("X" * LINKER_LEN_N) + CHAIN_MOTIF[MOTIF_CHAIN_ORDER[0]]['seq'] + ("X" * LINKER_LEN_C)
    CHAIN_MOTIF[MOTIF_CHAIN_ORDER[0]]['pos_design'] = list(range(LINKER_LEN_N, LINKER_LEN_N + len(CHAIN_MOTIF[MOTIF_CHAIN_ORDER[0]]['seq'])))
    motif_first_indices = jnp.array(CHAIN_MOTIF[MOTIF_CHAIN_ORDER[0]]['pos_design'])

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
    # Epitope Residues: Special
    epitope_idx = [8,9,10,11,12,13,45,46,47,69,71,72,73,74,75,76,77,78,111,112,113,115,142]
    motif = set(motif_first_indices.tolist())
    exclude = {i for m in motif for i in range(m-2, m+3)}   # Define non-motif residues as +-2 away from motif residues
    paratope_idx = jnp.array(sorted(set(range(BINDER_LENGTH)) - exclude), dtype = jnp.int32) # Even tho, class accepts lists. jax only allows indexing via jax numpy arrays
    # ---
    structure_prediction_loss = ((WEIGHT_BINDER_TARGET_CONTACT_LOSS_FUNCTION * sp.BinderTargetContact(epitope_idx = epitope_idx, paratope_idx = paratope_idx)) 
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
                                 + (WEIGHT_BIAS_SHEET_LOSS_FUNCTION * SheetBiasLoss())
                                 + (WEIGHT_DISTOGRAM_RADIUS_GYRATION_LOSS_FUNCTION * sp.DistogramRadiusOfGyration(target_radius = 20)))
    # 6.2, define motif-specific loss for the single motif (distogram + single-motif Kabsch RMSD)
    motif_first_loss = (
        (WEIGHT_FIRST_MOTIF_DISTOGRAM_LOSS_FUNCTION * MotifDistogramCE(motif_distogram_first, motif_first_indices))
        + (WEIGHT_FIRST_MOTIF_RMSD_LOSS_FUNCTION * MotifRMSDLoss(motif_ca_coords_first, motif_first_indices))
    )
    # NOTE: single motif -> no second motif and no inter-motif relative-placement term.
    # The motif<->TARGET geometry is what matters now; set the specific-interface
    # register below (and weight specific_contact / specific_pae) to drive it.

    # 6.25, Specific motif<->target interface register (which target residue each
    # motif residue should contact). Pairs are matched k-by-k:
    #   motif_contact_idx  : BINDER-relative positions (subset of motif_first_indices)
    #   target_contact_idx : 0-based positions WITHIN THE TARGET chain
    # Leave both empty (default) to DISABLE the specific-interface terms.
    #
    # OPTION A (manual): set the pairs explicitly from your native complex, e.g.
    #   motif_contact_idx  = jnp.array([51, 53, 90], dtype=jnp.int32)
    #   target_contact_idx = jnp.array([12, 14, 88], dtype=jnp.int32)
    #
    # OPTION B (derive from native, RECOMMENDED once verified): uncomment below.
    # `motif_cb_coords_first` is already loaded; you need target CB coords ordered
    # like SEQ_TARGET (e.g. a saved target_cb_coords.npy). Confirm ordering first.
    #   target_cb_coords = np.load(os.path.join(DATA_DIR, "target_cb_coords.npy"))
    #   motif_contact_idx, target_contact_idx = derive_interface_register(
    #       motif_cb_coords=motif_cb_coords_first,
    #       target_cb_coords=target_cb_coords,
    #       motif_binder_indices=np.asarray(motif_first_indices),
    #       contact_threshold=8.0,
    #   )

    # Listing out all contacts for each residue in motif
    #motif_contact_idx = jnp.array([50,50,50,50,51,51,51,51,51,51,51,51,51,51,51,51,51,52,52,52,52,52,52,52,53,53,53,53,53,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,55,55,55,55,55,55,55], dtype=jnp.int32)
    #target_contact_idx = jnp.array([11,12,13,14,9,10,11,12,13,14,70,71,72,73,74,75,76,10,11,12,74,75,76,77,11,12,75,76,77,8,9,10,11,12,13,74,75,76,77,78,79,110,111,112,113,114,115,11,75,76,77,78,79,115], dtype=jnp.int32) # 0-indexed

    motif_contact_idx = jnp.array([50,50,50,51,51,51,51,51,52,52,52,52,52,53,53,53,53,54,54,54,54,54,54,54,55,55,55,55,55])
    target_contact_idx = jnp.array([11,14,13,72,73,11,14,13,11,75,76,10,13,142,11,76,75,76,11,10,111,78,112,12,77,76,115,46,75]) # 0-indexed
    g_idx, t_groups, g_mask = build_contact_groups(motif_contact_idx, target_contact_idx)
    contact = SpecificContactAnyPerRes(motif_group_idx=g_idx, target_groups=t_groups, mask=g_mask, contact_distance=8.0, temperature=1.0)

    motif_set = set(np.asarray(motif_first_indices).tolist())
    exclude = set(i for m in motif_set for i in range(m-5, m+6))   # ±5 window
    scaffold_positions = jnp.array([i for i in range(BINDER_LENGTH) if i not in exclude])

    # 6.3, composite loss = complex structure terms + single-motif terms
    loss_fn = structure_prediction_loss + motif_first_loss

    # 6.35, add specific motif<->target interface terms only when a register is set.
    # (Python-level guard at construction time; avoids nan from indexing empty pairs.)
    if motif_contact_idx.shape[0] > 0:
        specific_contact = WEIGHT_SPECIFIC_CONTACT_LOSS_FUNCTION * SpecificContactAnyPerRes(motif_group_idx=g_idx, target_groups=t_groups, 
                                                                                            mask=g_mask, contact_distance=8.0, temperature=1.0)
        specific_pae = WEIGHT_SPECIFIC_PAE_LOSS_FUNCTION * SpecificPAELoss(
            motif_idx=motif_contact_idx,
            target_idx=target_contact_idx,
            name="specific_pae",
        )
        motif_scaffold_contact = (WEIGHT_MOTIF_SCAFFOLD_CONTACT_LOSS_FUNCTION * MotifScaffoldContactLoss(motif_positions = motif_first_indices,
                                                                                                         scaffold_positions = scaffold_positions ))
        loss_fn = loss_fn + specific_contact + specific_pae + motif_scaffold_contact

    # 6.4, establish loss function derived from Boltz2 Model
    features, _ = model_boltz.target_only_features(
        chains = [TargetChain(sequence = binder_seq, use_msa = False, template_chain = gemmi_binder_chain), 
                  TargetChain(sequence = SEQ_TARGET, use_msa = True, template_chain = gemmi_target_chain)])

    loss_fn_boltz = model_boltz.build_multisample_loss(
        loss = loss_fn,
        features = features,
        sampling_steps = 25,
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
    rmsd_d_boltz, iptm_boltz, plddt_boltz = evaluate_optimized_structure(struc_model = model_boltz, seq_pssm= pssm_optimized, motif_id_pos= CHAIN_MOTIF, design_iteration = design_iteration, model_name = "Boltz2")
    # 2. Predict Structure via ESMFold2
    model_esm2 = ESMFold2Full()
    rmsd_d_esmfold, iptm_esmfold, plddt_esmfold = evaluate_optimized_structure(struc_model= model_esm2,
                                                                                               seq_pssm= pssm_optimized,
                                                                                               motif_id_pos= CHAIN_MOTIF,
                                                                                               design_iteration = design_iteration,
                                                                                               model_name= "ESMFold2")
    # 3. Compute harmonic means for each model's: Boltz2 and ESMFold2 metrics: (rmsd_d, iptm, plddt)
    struc_model_metrics = {'rmsd_d' : [rmsd_d_boltz, rmsd_d_esmfold],
                           'iptm' : [iptm_boltz, iptm_esmfold],
                           'plddt' : [plddt_boltz, plddt_esmfold]}
    hmean_dict = {}
    for metric, metric_pair in struc_model_metrics.items():
        metric_boltz, metric_esmfold = metric_pair
        hmean_dict[metric] = compute_harmonic_mean(metric_a = metric_boltz, metric_b = metric_esmfold)
    # 4. Display harmonic means (Boltz2 <-> ESMFold2 agreement) for the Agent to interpret
    print("Harmonic Means (Boltz2 <-> ESMFold2 agreement):")
    print(f"hmean_rmsd_D: {hmean_dict['rmsd_d']:.2f}")
    print(f"hmean_ipTM: {hmean_dict['iptm']:.2f}")
    print(f"hmean_pLDDT: {hmean_dict['plddt']:.2f}")
    print(" ")
    composite_score = compute_composite_score(motif_rmsd = hmean_dict['rmsd_d'], 
                                              structure_iptm= hmean_dict['iptm'], binder_plddt= hmean_dict['plddt'])

    # 5. Return all of the metrics
    metrics = {'rmsd_d_boltz' : rmsd_d_boltz,
               'iptm_boltz' : iptm_boltz, 
               'plddt_boltz' : plddt_boltz,
               'rmsd_d_esmfold' : rmsd_d_esmfold,
               'iptm_esmfold' : iptm_esmfold,
               'plddt_esmfold' : plddt_esmfold,
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



