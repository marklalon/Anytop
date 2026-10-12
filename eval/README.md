# Evaluation

## Checkpoint battery

```
python eval/eval_checkpoint.py --task_config eval/eval_tasks.json
python eval/eval_checkpoint.py --task_config eval/eval_tasks.json --model_path save/<run>/model000250000.pt
```

Runs every task in `eval/eval_tasks.json` (`batch_size=4`, fp32), scores each
task's clips with the motion quality scorer and writes
`outputs/eval_checkpoint/<RUN>/<MODEL>/eval_report.html`. Runs are incremental:
existing task outputs are re-scored, only new or changed tasks are generated
(`--overwrite` regenerates everything). See the module docstring for the
config format.

## Quality score

Every clip gets two scores in `[0, 1]`; its quality is their geometric mean and
a task's quality is the median over its clips. The report lists both
dimensions, every clip's quality and the task's weakest sub-scores.

Both dimensions score a statistic by where it falls among real clips
(every dataset in `dataset/datasets.jsonl`, each species weighted equally):
up to the real median it scores 1, at the real p75 / p90 / p99 it scores
0.9 / 0.75 / 0.45, and past p99 it halves for every doubling. Body parts come
from `joint_parts.prefill_joint_parts`; lengths are in units of the skeleton's
rest span. The positions are the RIC channels -- what the BVH preview shows.

**Physics** (`eval/motion_quality/physics.py`) -- label-free.

- *rigidity*: bone-length-weighted mean |L/L0 - 1|, and the same over the
  longest third of the bones (a stretched toe tip is not visible);
- *smoothness*: per body part, 3rd and 4th frame differences over the velocity,
  and the spectral flatness of the velocity;
- *continuity*: the largest single-frame acceleration over the median
  acceleration of the low-passed motion (a skipped frame or a popping limb).

Tolerances come from real clips of the query's object_subset (rigidity: the
subset, shrunk toward all species when it holds few; smoothness: the wider of
subset and all species, since an insect twitches more than an elephant;
continuity: all species).

**Prior** (`eval/motion_quality/prior.py`) -- does the clip move like real
clips of its action? The reference is every corpus clip sharing a head word
with the task's label, never the query's own species; modifiers only weigh.
Only features that move toward the real distribution as a model trains are
scored:

- *symmetry*: |log| ratio of the left and right joints' motion amplitude in each
  sided body part;
- *coordination*: for the legs (foot chains, fore-aft swing) and the wings
  (tip height): do the limbs share one dominant frequency, how phase-locked they
  are, the left/right relative phase, and whether the root's travel matches
  stride times step frequency;
- *tempo*: the centroid frequency of each part's low-passed velocity.

A feature's deviation is its robust z (weighted median / MAD) against the
reference layer it fits best -- all species or the query's object_subset --
counted on the bad side only for symmetry, frequency agreement and lock. The
deviation is scored against the deviations of real clips of the same head
word, each measured against its own reference without its own species, so a
typical real clip scores 1 and a clip more typical than that gains nothing.
Amplitude, speed, posture and root travel are not scored: generated clips tend
to the average motion and are *more* typical than real clips on them.

Corpus statistics are cached per dataset source in
`<processed>/cache/physics_stats_v*.npz` and `prior_features_v*.npz` and
rebuilt when a clip changes.

Score a set of clips outside the battery:

```
python eval/evaluate_motion_quality.py --motions "outputs/x/*.npy" --object-type Buffalo --action-label "run, forward"
```

## Limits

The prior measures how the limbs move together, not what the motion means: two
actions with similar coordination (attack vs bite, walk forward vs walk left)
are not told apart. The label only selects the reference.

Foot chains and wing tips come from the joint-part prefill; a rig whose limbs
it mislabels (a bird's wings named as arms) is compared on the wrong parts.
