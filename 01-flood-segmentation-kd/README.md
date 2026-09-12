# Self-Supervised Cross-Modal Knowledge Distillation for Flood Segmentation

**Master's thesis (MAXU 5214), UTeM · July 2026 - present · individual work**
Supervised research project, Faculty of Artificial Intelligence and Cyber Security.

## The problem
Floods and clouds arrive together. Optical satellites (Sentinel-2) map water accurately but
see nothing through cloud; radar (Sentinel-1 SAR) sees through cloud but is much harder to
segment. Hand-labelling SAR is slow, so labelled SAR data stays scarce.

## The idea
Train the optical model as a **teacher** and distil what it knows into a **SAR-only
student**, so that at inference time only Sentinel-1 is needed - the sensor that still works
when the flood is happening.

## Data
[Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11): paired Sentinel-1 / Sentinel-2
chips with hand labels. Labels are `-1` no-data, `0` land, `1` water. Official split CSVs are
used, never random splits - random splits would put chips from the same flood event in both
train and test and inflate every number.

## Method
1. `train --modality s1` - SAR-only baseline (what the framework must beat).
2. `train --modality s2` - optical model: both the optical baseline and the teacher.
3. `distill --teacher-ckpt ...` - SAR student trained with the frozen optical teacher;
   response KD, pair-wise KD and confidence gating are switchable in the config.
4. `eval` on the test split and on the held-out Bolivia event.

## Results so far
| Run | Validation water IoU |
|---|---|
| Sentinel-1 SAR baseline | **0.6244** |
| Sentinel-2 optical teacher | **0.8261** |
| Modality gap to close | **0.2017** |
| Distillation (alpha 1.0, beta 1.0, no gate) | 0.6266 |
| Distillation (beta 0.1 + confidence gate 0.7) | 0.6206 |

## Engineering decisions worth defending
- **The ignore mask lives in one place.** If `-1` pixels leak into the loss or the metric,
  every reported number is quietly wrong (in a small test, a true IoU of 0.40 reads as 0.25).
- **The teacher is frozen**, so any gain is attributable to the transfer.
- **Inference takes one modality.** `TeacherStudent.predict()` accepts S1 only, which makes
  the deployment claim structurally true instead of a promise in the text.
- **Config-driven, seeded, logged.** Every run writes `config.json` and `log.csv` next to its
  checkpoint; `best.pt` is selected on validation water IoU, not on loss.

## Still open
Self-supervised pre-training of the teacher (the actual novelty), data augmentation, and the
larger weakly-labelled split.

## Files
```
run.py                 one entry point: explore | download | stats | train | distill | eval
configs/default.yaml   every number lives here - no magic numbers in code
src/config.py          loads YAML, applies CLI overrides, rejects typos
src/data.py            chip download, PyTorch Dataset, normalisation
src/models.py          U-Net and the TeacherStudent wrapper
src/losses.py          segmentation losses + the three distillation losses
src/metrics.py         IoU / F1 / precision / recall, ignore mask applied once
src/engine.py          train and eval loops, checkpoints, CSV logging, seeding
docs/original-README.md  the working README from the research repo
```

## Run it
```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
python run.py explore
python run.py download --split train,val,test
python run.py train --modality s1 --out-dir runs/base_s1
python run.py train --modality s2 --out-dir runs/teacher_s2
python run.py distill --teacher-ckpt runs/teacher_s2/best.pt --out-dir runs/distill
```
Full research repo: [github.com/imanfirdaus27/floodkd](https://github.com/imanfirdaus27/floodkd)
