"""Three ways to aggregate specific motif<->target contacts into one loss.

All three read log P(d_ij < cutoff) from the DISTOGRAM head (trunk-derived,
sampling-independent, low-variance) via mosaic's `contact_log_probability`, and
differ ONLY in how they combine the per-pair contact log-probabilities:

  Option 1  SpecificContactAND        -mean_k log P_k
            "satisfy EVERY pair"      hard AND; worst pair dominates the gradient.

  Option 2  SpecificContactMass       -log( mean_k P_k )
            "make contact mass high"  flat soft-OR; a few easy pairs can satisfy
                                      it, no per-residue coverage guarantee.

  Option 3  SpecificContactAnyPerRes  -mean_g logsumexp_t( log P_{g,t} )
            "each motif residue        OR within each residue's candidate set,
             touches >=1 of its        AND across residues. The principled choice
             candidate targets"        when a residue has several candidate
                                       partners and can only reach some of them.

Index convention (same as the other cross-chain terms): motif indices are
BINDER-relative; target indices are 0-based WITHIN THE TARGET chain (offset by
binder_len internally); pairs/candidates are matched explicitly.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from mosaic.common import LossTerm
from mosaic.losses.structure_prediction import StructureModelOutput
import mosaic.losses.structure_prediction as sp


# ---------------------------------------------------------------------------
# Option 1: AND  (== your current SpecificContactLoss; here for comparison)
# ---------------------------------------------------------------------------
class SpecificContactAND(LossTerm):
    """-mean_k log P(d_k < cutoff). Demands ALL pairs simultaneously.

    loss = -(1/K) Σ_k log P_k = -(1/K) log Π_k P_k  -> maximizes the JOINT
    probability that every pair is in contact. Because log -> -inf as P -> 0,
    the worst pair dominates: the gradient is ~ -1/P_k, huge for failing pairs,
    tiny for satisfied ones. Use ONLY when every listed pair is real and
    simultaneously satisfiable. Over-specified registers (a residue listed
    against many targets) make this permanently stuck (the impossible pairs
    pin the mean)."""
    motif_idx: Int[Array, "K"]
    target_idx: Int[Array, "K"]
    contact_distance: float = 8.0
    name: str = "contact_and"

    def __call__(self, sequence: Float[Array, "N 20"], output: StructureModelOutput, key):
        binder_len = sequence.shape[0]
        log_p = sp.contact_log_probability(
            output.distogram_logits, self.contact_distance, bins=output.distogram_bins
        )
        lp = log_p[self.motif_idx, self.target_idx + binder_len]   # [K]
        loss = -lp.mean()
        return loss, {
            f"{self.name}_logp": lp.mean(),
            f"{self.name}_p": jnp.exp(lp).mean(),
            f"{self.name}_n_sat": (lp > jnp.log(0.5)).sum(),
        }


# ---------------------------------------------------------------------------
# Option 2: MASS  (flat soft-OR over all pairs)
# ---------------------------------------------------------------------------
class SpecificContactMass(LossTerm):
    """-log( mean_k P(d_k < cutoff) ). Rewards total contact MASS.

    The log is OUTSIDE the average, so no single failing pair sends the loss to
    -inf; satisfying any subset lifts it. Equivalent to -logsumexp(lp) + log(K).
    Downside: it's a flat OR over the WHOLE register -- it can be satisfied by
    making a handful of easy pairs very high while ignoring the rest, so it does
    NOT guarantee that each motif residue is anchored. Rarely what you want for
    a structured interface; included for contrast."""
    motif_idx: Int[Array, "K"]
    target_idx: Int[Array, "K"]
    contact_distance: float = 8.0
    eps: float = 1e-6
    name: str = "contact_mass"

    def __call__(self, sequence: Float[Array, "N 20"], output: StructureModelOutput, key):
        binder_len = sequence.shape[0]
        log_p = sp.contact_log_probability(
            output.distogram_logits, self.contact_distance, bins=output.distogram_bins
        )
        lp = log_p[self.motif_idx, self.target_idx + binder_len]   # [K]
        mean_p = jnp.exp(lp).mean()
        loss = -jnp.log(mean_p + self.eps)
        return loss, {
            f"{self.name}_meanp": mean_p,
            f"{self.name}_n_sat": (lp > jnp.log(0.5)).sum(),
        }


# ---------------------------------------------------------------------------
# Option 3: ANY-PER-RESIDUE  (OR within each residue's candidates, AND across)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
# You already have the flat register:
#   motif_contact_idx  = jnp.array([50,50,50,50,51,51, ...])   # binder-relative, repeats
#   target_contact_idx = jnp.array([11,12,13,14, 9,10, ...])   # target-relative
#
# Option 1 (current behavior):
#   contact = SpecificContactAND(motif_contact_idx, target_contact_idx, 8.0)
#
# Option 2 (flat mass):
#   contact = SpecificContactMass(motif_contact_idx, target_contact_idx, 8.0)
#
# Option 3 (recommended for multi-partner residues):
#   g_idx, t_groups, g_mask = build_contact_groups(motif_contact_idx, target_contact_idx)
#   contact = SpecificContactAnyPerRes(
#       motif_group_idx=g_idx, target_groups=t_groups, mask=g_mask,
#       contact_distance=8.0, temperature=1.0)
#
# Then, as before, add to the structure-output loss (outside nothing special):
#   loss_fn = ... + W_CONTACT * contact
