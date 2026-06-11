# auto_mosaic

You are an expert computational protein engineer. Your task is to design an optimal scaffold that correctly places two functional motifs (from chains A and D of a protein complex) to bind to a target protein.

This is done via gradient-based sequence optimization using Boltz2. You tune the hyperparameters in `train.py` to minimize the composite score. Gradient-based sequence optimization via Boltz2 outputs an optimized binder sequence whose interaction with the target sequence is evaluated via both Boltz2 and ESMFold2. 

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jun6`). The branch `auto_mosaic/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b auto_mosaic/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `prepare.py` — fixed constants, motif coordinate extraction, composite score definition, evaluation. Do not modify.
   - `train.py` — the file you modify. Linker lengths, loss function weights, optimizer hyperparameters.
4. **Verify data exists**: Run `modal volume ls autoresearch-data` to confirm the motif coordinate `.npy` files exist on the Modal volume. If not, tell the human to run `modal run modal_train.py --prepare`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good, then start experimenting.

## The task

The binder sequence is structured as:

```
[LINKER1] - [MOTIF_A: LTKWTN] - [LINKER2] - [MOTIF_D: FPGER] - [LINKER3]
```

The `X` linker residues are free to optimize; the motif residues are fixed. Boltz2 predicts the structure of the binder complexed with the target, and gradients flow back through the prediction to update the per-position amino acid distribution (PSSM) over 3 successive stages: soft → sharp → hard.

## What you can tune in `train.py`

- **Linker lengths**: `LINKER_LEN1`, `LINKER_LEN2`, `LINKER_LEN3` — controls binder length and motif spacing. Total binder length = LINKER_LEN1 + 6 + LINKER_LEN2 + 5 + LINKER_LEN3.
- **Loss function weights**: the 14 `WEIGHT_*` constants controlling the relative contribution of each term (binder contact, PAE, iPTM, pLDDT, anti-helix, motif distogram, motif RMSD, etc.).
- **Optimizer hyperparameters**: `soft_pssm_hyparams`, `sharp_pssm_hyparams`, `hard_pssm_hyparams` — each controls `n_steps`, `stepsize`, `momentum`, `scale`, and `logspace` for that stage. Total steps across all three stages must not exceed `MAX_OPTIMIZER_STEPS` (100).
- **Motif chain order**: `MOTIF_CHAIN_ORDER` — controls which motif appears first in the binder sequence.

## Loss Function Reference

Output ranges below are for the underlying `LossTerm` *before* the `WEIGHT_*` multiplier is applied, sourced from the `mosaic` library (escalante-bio/mosaic, `src/mosaic/losses/`). Some ranges (marked) are theoretically unbounded and would benefit from empirical confirmation. Append observations to the last column as new loops complete.

| `WEIGHT_*` constant | Loss term | What it measures | Output range (pre-weight) | Observations from past loops |
|---|---|---|---|---|
| `WEIGHT_BINDER_CONTACT_LOSS_FUNCTION` | `sp.BinderTargetContact` | Avg. log-prob of binder↔target contacts (top-3 contacts/binder residue, ≤20 Å) | (−∞, 0] | — |
| `WEIGHT_WITHIN_BINDER_CONTACT_LOSS_FUNCTION` | `sp.WithinBinderContact` | Avg. log-prob of intra-binder contacts (top-k/residue beyond min sequence separation) | (−∞, 0] | — |
| `WEIGHT_INVERSE_FOLDING_SEQ_RECOVERY_LOSS_FUNCTION` | `InverseFoldingSequenceRecovery` | Negative dot product between the binder PSSM and ProteinMPNN's average predicted sequence ("designability" / sequence recovery) | [−1, 0] | -- |
| `WEIGHT_TARGET_BINDER_PAE_LOSS_FUNCTION` | `sp.TargetBinderPAE` | Mean PAE (Å), target→binder block | [0, 32] | — |
| `WEIGHT_BINDER_TARGET_PAE_LOSS_FUNCTION` | `sp.BinderTargetPAE` | Mean PAE (Å), binder→target block | [0, 32] | — |
| `WEIGHT_WITHIN_BINDER_PAE_LOSS_FUNCTION` | `sp.WithinBinderPAE` | Mean PAE (Å) within the binder (off-diagonal) | [0, 32] | — |
| `WEIGHT_IPTM_LOSS_FUNCTION` | `sp.IPTMLoss` | Negative interface pTM (predicted TM-score over inter-chain residue pairs) | [−1, 0] | — |
| `WEIGHT_PTM_ENERGY_LOSS_FUNCTION` | `sp.pTMEnergy` | Negative log-space "TM energy" averaged over inter-chain pairs | (−∞, ∞), typically small magnitude (unconfirmed) | — |
| `WEIGHT_PLDDT_LOSS_FUNCTION` | `sp.PLDDTLoss` | Negative mean pLDDT over binder positions (pLDDT on a 0-1 scale) | [−1, 0] | — |
| `WEIGHT_ANTI_HELIX_LOSS_FUNCTION` | `AntiHelixLoss` (train.py) | `elu(target_value + value)`, where `value` = mean i,i+3 contact log-probability (≈0 if alpha-helix-dominant, → −∞ if extended/beta-dominant). `target_value=0.0` (parameter-free). Penalizes helical character; bounded reward for beta/extended character. | (−1, 0] | New, untested. Added because motif A (LTKWTN) is confirmed to be a beta strand and has the persistent ~1.9Å hmean RMSD gap — hypothesis is the all-helix scaffold can't seat it correctly. |
| `WEIGHT_FIRST_MOTIF_DISTOGRAM_LOSS_FUNCTION` | `MotifDistogramCE` (train.py) | Cross-entropy between predicted and native distogram, restricted to motif A's M×M block | [0, ∞) | Increased 0.1→0.3 (`47725a1`): composite 1.6290→2.0830 (discarded). Boltz2 ipTM stayed high (0.90) but ESMFold2 ipTM dropped to 0.25 — Boltz2/ESMFold2 divergence. |
| `WEIGHT_FIRST_MOTIF_RMSD_LOSS_FUNCTION` | `MotifRMSDLoss` (train.py) | Kabsch-aligned Cα RMSD (Å), predicted vs. native motif A | [0, ∞) | Increased 0.1→0.3 (`7939d87`): composite 1.6290→2.4721 (discarded). Boltz2 ipTM stayed high (0.93) but ESMFold2 ipTM collapsed to 0.09 — strongest divergence seen so far. |
| `WEIGHT_SECOND_MOTIF_DISTOGRAM_LOSS_FUNCTION` | `MotifDistogramCE` (train.py) | Cross-entropy between predicted and native distogram, restricted to motif D's M×M block | [0, ∞) | Not yet isolated. |
| `WEIGHT_SECOND_MOTIF_RMSD_LOSS_FUNCTION` | `MotifRMSDLoss` (train.py) | Kabsch-aligned Cα RMSD (Å), predicted vs. native motif D | [0, ∞) | Not yet isolated. |

## What you cannot do

- Modify `prepare.py`. It is read-only and contains the fixed evaluation harness and composite score definition.
- Install new packages or add dependencies beyond what is in `pyproject.toml`.
- Change `MAX_OPTIMIZER_STEPS` — it is defined in `prepare.py`.

## The goal
**Maximize agreement of high-quality binder sequences between both Boltz2 (Designer) and ESMFold2 (Validator)**

The inherent issue with gradient-based optimization through a protein structure prediction model (in this case Boltz2) is that the binder sequences yielded as a result of this process often have great structure confidence metrics when evaluated with the protein structure prediction model it was backpropagated through (in this case Boltz2), but poor structure confidence metrics when evaluated with a different model such as ESMFold2. This phenomenon is known as designing adversarial sequences and is a clear sign that the training setup used for gradient-based optimization is seeking to generate binder sequences which inflate confidence metrics in Boltz2, but are unrealistic as determined by the poor confidence metrics extracted from ESMFold2.

To maximize this agreement, after generating a binder sequence via Boltz2, you will generate a binder-target structure via both Boltz2 and ESMFold2. In the process, you will extract metrics: pLDDT, ipTM, rmsd_A, rmsd_D from both respective binder-target structures. Then, you will generate a harmonic mean between each pair of metrics: (rmsd_a_Boltz2, rmsd_a_ESMFold2), (rmsd_d_Boltz2, rmsd_d_ESMFold2), (ipTM_Boltz2, ipTM_ESMFold2),  (pLDDT_Boltz2, pLDDT_ESMFold2).

The harmonic mean is defined as for a given metric: pLDDT, ipTM, rmsd_A, rmsd_D

'''
*hmean_pLDDT is a function which takes in 2 inputs: pLDDT_Boltz2 and pLDDT_ESMFold2 and outputs: harmonic mean between the 2 scores. This harmonic mean will be generated for each of the metrics: pLDDT, ipTM, rmsd_A, rmsd_D*

hmean_pLDDT(pLDDT_Boltz2, pLDDT_ESMFold2) = ((2 * pLDDT_Boltz2 * pLDDT_ESMFold2) / (pLDDT_Boltz2 + pLDDT_ESMFold2))

*hmean is a short name for harmonic mean*
'''

`composite_score` is built directly from these harmonic means, so minimizing it is equivalent to maximizing Boltz2/ESMFold2 agreement while also requiring strong absolute metrics — it is the single number to track each iteration.

**Minimize `composite_score` (lower is better; target < 1.5).**

The composite score is defined in `prepare.py` as:

```
composite_score = 2.0 * (1 - exp(-((hmean_rmsd_A + hmean_rmsd_D) / 2) / 1.5))   # motif placement (primary)
                + 1.0 * (1 - hmean_ipTM)                           # binding quality
                + 0.5 * (1 - hmean_pLDDT)                          # fold confidence
```

A good design has:
- hmean_rmsd < 1.5 Å for both chains: (A & D) (motifs are correctly placed)
- hmean_ipTM > 0.7 (confident binding to the target)
- hmean_pLDDT > 0.7 (the binder is well-folded)

**Simplicity criterion**: All else being equal, simpler is better. A 0.05 score improvement that adds complex code is not worth it. Removing a loss term and getting equal or better results is a win.

**The first run**: Always establish a baseline first — run `train.py` as-is before making any changes.

## Output format

At the end of a run, `evaluate_optimized_structure` prints:

```
--------------------------------------------------
For Design Iteration:

Validation conducted by Boltz2:
You have designed binder seq: LHSHPYWTPPYPHHERMDQRKERVRKYAFLLTKWTNEEQKEWYHRQLVIILNLSQLDMQRFVDWFGFPGERWPHTDPPLRLWWNYSLEMVKFVKQDWGCLL
Your selection of hyperparameters has resulted in:
Motif From Chain: A has an associated RMSD: 1.23
  As always, smaller RMSD is better and ideal RMSD is < 1.5 Angstroms
Motif From Chain: D has an associated RMSD: 0.98
  As always, smaller RMSD is better and ideal RMSD is < 1.5 Angstroms
Boltz2 ipTM: 0.71
Boltz2 pLDDT: 0.82

Validation conducted by ESMFold2:
You have designed binder seq: LHSHPYWTPPYPHHERMDQRKERVRKYAFLLTKWTNEEQKEWYHRQLVIILNLSQLDMQRFVDWFGFPGERWPHTDPPLRLWWNYSLEMVKFVKQDWGCLL
Your selection of hyperparameters has resulted in:
Motif From Chain: A has an associated RMSD: 1.75
  As always, smaller RMSD is better and ideal RMSD is < 1.5 Angstroms
Motif From Chain: D has an associated RMSD: 1.80
  As always, smaller RMSD is better and ideal RMSD is < 1.5 Angstroms
ESMFold2 ipTM: 0.45
ESMFold2 pLDDT: 0.6

Harmonic Means (Boltz2 <-> ESMFold2 agreement):
hmean_rmsd_A: 1.44
hmean_rmsd_D: 1.27
hmean_ipTM: 0.55
hmean_pLDDT: 0.69

Composite Score: 1.3245  (lower is better; target < 1.5)
--------------------------------------------------
```

Note: each run now saves two structures (one from Boltz2, one from ESMFold2), so `For Design Iteration` numbers will increase by 2 per run (e.g. 0, 2, 4, ...) — this is expected and not a sign of skipped iterations.

Extract the key metric from the log:

```
grep "^Composite Score:" run.log
```

For crash diagnosis:

```
tail -n 50 run.log
```

## Logging results

Log each run to `results.tsv` (tab-separated — commas break in descriptions). Do NOT commit this file; leave it untracked.

```
commit	composite_score rmsd_A_Boltz2 rmsd_A_ESMFold2 rmsd_D_Boltz2 rmsd_D_ESMFold2 iptm_Boltz2 iptm_ESMFold2 plddt_Boltz2  plddt_ESMFold2  status  description
```

1. git commit hash (short, 7 chars)
2. composite_score (e.g. 1.3245) — use 99.0000 for crashes
3. Boltz2 motif RMSD for chain A (Å) — use 99.0 for crashes
4. ESMFold2 motif RMSD for chain A (Å) — use 99.0 for crashes
5. Boltz2 motif RMSD for chain D (Å) — use 99.0 for crashes
6. ESMFold2 motif RMSD for chain D (Å) — use 99.0 for crashes
7. Boltz2 ipTM — use 0.0 for crashes
8. ESMFold2 ipTM — use 0.0 for crashes
9. Boltz2 pLDDT — use 0.0 for crashes
10. ESMFold2 pLDDT — use 0.0 for crashes
11. status: `keep`, `discard`, or `crash`
12. short description of what this experiment tried

Example:

```
commit	composite_score	rmsd_A_Boltz2 rmsd_A_ESMFold2 rmsd_D_Boltz2 rmsd_D_ESMFold2 iptm_Boltz2 iptm_ESMFold2 plddt_Boltz2  plddt_ESMFold2  status  description
a1b2c3d	1.3245	1.23	1.50  0.98  0.94	0.71  0.72  0.65  0.81  keep	baseline
b2c3d4e	1.1832	0.87  0.82	0.76  0.86	0.78  0.45  0.85  0.67  keep	increase motif RMSD weight to 0.3
c3d4e5f	1.5901	2.14  2.34  1.89  1.84  0.55  0.55  0.45  0.45  discard	shorten linkers to 15-15-15
d4e5f6g	99.0000	99.0  99.0  99.0  99.0	99.0	0.0 0.0 0.0 0.0 crash	too many optimizer steps (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `auto_mosaic/jun6`).

LOOP FOREVER:

1. Check git state: current branch and last commit.
2. Form a hypothesis about what to change in `train.py` and why (e.g. "motif RMSD is high — increase `WEIGHT_MOTIF_RMSD_LOSS_FUNCTION`").
3. Edit `train.py` directly with the change.
4. `git commit`
5. Run: `PYTHONUTF8=1 modal run modal_train.py > run.log 2>&1` (redirect everything — do NOT let output flood your context)
6. Read results: `grep "^Composite Score:" run.log`
7. If grep returns nothing, the run crashed. Run `tail -n 50 run.log` to diagnose and attempt a fix. If the idea is fundamentally broken, log "crash" and move on.
8. Record results in `results.tsv`.
9. If composite_score improved (lower), keep the commit and advance.
10. If equal or worse, `git reset --hard HEAD~1` to discard.

**Runtime**: Each run takes ~20 minutes on an H100 (100 total optimizer steps × ~5-10 seconds per Boltz2 forward/backward). The Modal timeout is 3600 seconds (60 minutes or 1 hour) — if a run hits this, treat it as a crash and reduce the number of optimizer steps. Every 5 minutes analyze the log file to check how the experiment is going. 

**Crashes**: Fix obvious bugs (typos, import errors) and re-run. If the idea itself is broken, skip it.

**STOPPING CONDITION**: You are only authorized to run a maximum of 3 experiment loops. Once the 3rd evaluation is complete, log the final results and exit.
