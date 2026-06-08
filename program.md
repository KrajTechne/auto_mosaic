# auto_mosaic

You are an expert computational protein engineer. Your task is to design an optimal scaffold that correctly places two functional motifs (from chains A and D of the gopher alpha-snake toxin) and presents them to a target protein for binding.

This is done via gradient-based sequence optimization using Boltz2. You tune the hyperparameters in `train.py` to minimize the composite score.

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
- **Loss function weights**: the 11 `WEIGHT_*` constants controlling the relative contribution of each term (binder contact, PAE, iPTM, pLDDT, motif distogram, motif RMSD, etc.).
- **Optimizer hyperparameters**: `soft_pssm_hyparams`, `sharp_pssm_hyparams`, `hard_pssm_hyparams` — each controls `n_steps`, `stepsize`, `momentum`, `scale`, and `logspace` for that stage. Total steps across all three stages must not exceed `MAX_OPTIMIZER_STEPS` (100).
- **Motif chain order**: `MOTIF_CHAIN_ORDER` — controls which motif appears first in the binder sequence.

## What you cannot do

- Modify `prepare.py`. It is read-only and contains the fixed evaluation harness and composite score definition.
- Install new packages or add dependencies beyond what is in `pyproject.toml`.
- Change `MAX_OPTIMIZER_STEPS` — it is defined in `prepare.py`.

## The goal

**Minimize `composite_score` (lower is better; target < 1.5).**

The composite score is defined in `prepare.py` as:

```
composite_score = 2.0 * (1 - exp(-mean_motif_rmsd / 1.5))   # motif placement (primary)
                + 1.0 * (1 - iPTM)                           # binding quality
                + 0.5 * (1 - pLDDT)                          # fold confidence
```

A good design has:
- Motif RMSD < 1.5 Å for both chains (motifs are correctly placed)
- iPTM > 0.8 (confident binding to the target)
- pLDDT > 0.7 (the binder is well-folded)

**Simplicity criterion**: All else being equal, simpler is better. A 0.05 score improvement that adds complex code is not worth it. Removing a loss term and getting equal or better results is a win.

**The first run**: Always establish a baseline first — run `train.py` as-is before making any changes.

## Output format

At the end of a run, `evaluate_optimized_structure` prints:

```
--------------------------------------------------
For Design Iteration:
You have designed binder seq: LHSHPYWTPPYPHHERMDQRKERVRKYAFLLTKWTNEEQKEWYHRQLVIILNLSQLDMQRFVDWFGFPGERWPHTDPPLRLWWNYSLEMVKFVKQDWGCLL
Your selection of hyperparameters has resulted in:
Motif From Chain: A has an associated RMSD: 1.23
  As always, smaller RMSD is better and ideal RMSD is < 1.5 Angstroms
Motif From Chain: D has an associated RMSD: 0.98
  As always, smaller RMSD is better and ideal RMSD is < 1.5 Angstroms
Structure IPTM: 0.71
Structure PLDDT: 0.82
Composite Score: 1.3245  (lower is better; target < 1.5)
--------------------------------------------------
```

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
commit	composite_score	rmsd_A	rmsd_D	iptm	status	description
```

1. git commit hash (short, 7 chars)
2. composite_score (e.g. 1.3245) — use 99.0000 for crashes
3. motif RMSD for chain A (Å) — use 99.0 for crashes
4. motif RMSD for chain D (Å) — use 99.0 for crashes
5. iPTM — use 0.0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short description of what this experiment tried

Example:

```
commit	composite_score	rmsd_A	rmsd_D	iptm	status	description
a1b2c3d	1.3245	1.23	0.98	0.71	keep	baseline
b2c3d4e	1.1832	0.87	0.76	0.78	keep	increase motif RMSD weight to 0.3
c3d4e5f	1.5901	2.14	1.89	0.55	discard	shorten linkers to 15-15-15
d4e5f6g	99.0000	99.0	99.0	0.0	crash	too many optimizer steps (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `auto_mosaic/jun6`).

LOOP FOREVER:

1. Check git state: current branch and last commit.
2. Form a hypothesis about what to change in `train.py` and why (e.g. "motif RMSD is high — increase `WEIGHT_MOTIF_RMSD_LOSS_FUNCTION`").
3. Edit `train.py` directly with the change.
4. `git commit`
5. Run: `modal run modal_train.py > run.log 2>&1` (redirect everything — do NOT let output flood your context)
6. Read results: `grep "^Composite Score:" run.log`
7. If grep returns nothing, the run crashed. Run `tail -n 50 run.log` to diagnose and attempt a fix. If the idea is fundamentally broken, log "crash" and move on.
8. Record results in `results.tsv`.
9. If composite_score improved (lower), keep the commit and advance.
10. If equal or worse, `git reset --hard HEAD~1` to discard.

**Runtime**: Each run takes ~15 minutes on an H100 (100 total optimizer steps × ~5-10 seconds per Boltz2 forward/backward). The Modal timeout is 1800 seconds (30 minutes or 1/2 hour) — if a run hits this, treat it as a crash and reduce the number of optimizer steps. Feel free to check in every 5 minutes to see how the experiment is going by analyzing the log file.

**Crashes**: Fix obvious bugs (typos, import errors) and re-run. If the idea itself is broken, skip it.

**STOPPING CONDITION**: You are only authorized to run a maximum of 3 experiment loops. Once the 3rd evaluation is complete, log the final results and exit.
