# Industrial Defect ML System (MVTec AD)

**Applied Machine Learning (MAXD 5143), UTeM · June 2026 · group of 4**
Group 3: Syafiqah Amira binti Abdul Aziz, Muhammad Iman Firdaus bin Md Rostan,
Zulkifli bin Md Nasir, Hanis binti Hussin. Instructors: Dr. Khaled Cesar Al-Saih,
Ts. Dr. Muhammad Noorazlan Shah bin Zainudin.

An end-to-end "factory" pipeline on the MVTec AD dataset (`metal_nut` and `screw`
categories): distributed preprocessing, a CNN defect classifier, a reinforcement-learning
navigation agent, and model compression for edge deployment.

## Section A - Distributed data pipeline (PySpark)
- **`DriftCorrector`**, a custom `pyspark.ml.Transformer`: converts pixels to float,
  standardises per image, then re-anchors to a fixed reference distribution
  (mu = 0.45, sigma = 0.22) so illumination drift between batches stops confusing the model.
- **Salted-hash partitioning**: `CRC32(category + defect_type) mod 8`, then
  `partitionBy('category')` to Parquet, so downstream `groupBy('category')` needs no shuffle
  and no single partition becomes a hot spot.
- **Scalability benchmark** on 815 images: `local[1]` 8.45s, `local[4]` 4.28s - a **1.97x**
  speed-up against an ideal of 4x, with parallel efficiency dropping to 0.38 at 8 workers.
  That gap is Amdahl's law plus scheduling overhead, not a bug.

## Section B - CNN defect classifier (PyTorch)
`DefectCNN`, a four-block convolutional network (128x128 input, 3x3 kernels, batch norm,
dropout, global average pooling) driven by a `Config` dictionary so every experiment is one
line.
- **Regularisation study**: L2 only 66.3%, dropout only 69.9%, combined 68.7% validation
  accuracy - combined gave the lowest validation loss and the smallest overfitting gap.
- **Optimiser study**: Adam 68.7% (fastest, smoothest), SGD+momentum 68.1%, RMSprop 59.5%.
- **Honest weakness**: overall accuracy read 83%, but defect-class recall was only 0.28.
  The classifier finds good parts easily and misses many defective ones - the thing that
  actually matters on a production line. More defect samples and class weighting are next.

## Section C - Autonomous path optimisation (DQN)
A 15x15 `WarehouseEnv` with obstacles injected at defect coordinates, a `QNet` Q-network and
a `DQNAgent` with epsilon-greedy decay. Reward: +100 goal, -50 collision, -1 per step. The
greedy rollout showed the agent is **not** a usable navigator yet; the epsilon-decay study
documents what to change.

## Section D - Edge deployment
Dynamic INT8 quantisation on `nn.Linear` and `nn.Conv2d`, plus 30% L1-unstructured pruning
with the mask baked in via `prune.remove()`. INT8 was fastest with a 0.9 percentage-point
accuracy cost; at 1.57 MB the model was already too small for quantisation to shrink further.
The write-up argues that on a line running 100,000 items a day, 0.7 pp is ~700 extra
misclassifications - compression needs version documentation and monitoring, not just a
benchmark.

## Files
```
notebook.ipynb            the submitted notebook (code + outputs)
code/mvtec_pipeline.py    code cells extracted, markdown kept as comments
```

## Data
[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) - `metal_nut` and
`screw` categories. Not included here (hundreds of MB); the notebook downloads and extracts
them into `data/mvtec/`.

## Figures

The 12 images below are the real figures from the MAXD 5143 project report, extracted from the submitted PDF.

![Sample gallery from the MVTec AD categories used.](figures/01-mvtec-sample-gallery.png)

*Sample gallery from the MVTec AD categories used.*

![End-to-end pipeline: ingest → drift correction → Spark features → CNN → edge deployment.](figures/02-pipeline-flow.png)

*End-to-end pipeline: ingest → drift correction → Spark features → CNN → edge deployment.*

![DriftCorrector before/after — correcting exposure drift stopped the model learning brightness instead of defects.](figures/03-driftcorrector-before-after.png)

*DriftCorrector before/after — correcting exposure drift stopped the model learning brightness instead of defects.*

![Spark scalability: runtime, speed-up, throughput and efficiency vs executor count.](figures/04-spark-scalability-4panel.png)

*Spark scalability: runtime, speed-up, throughput and efficiency vs executor count.*

![Regularisation study (dropout, weight decay) vs validation accuracy.](figures/05-regularisation-study.png)

*Regularisation study (dropout, weight decay) vs validation accuracy.*

![Optimiser comparison and training curves.](figures/06-optimiser-study.png)

*Optimiser comparison and training curves.*

![Confusion matrix — this is where the weak defect recall (0.28) shows up.](figures/07-confusion-matrix.png)

*Confusion matrix — this is where the weak defect recall (0.28) shows up.*

![Sample predictions, predicted vs true label.](figures/08-sample-predictions.png)

*Sample predictions, predicted vs true label.*

![DQN diagnostics: reward, loss and Q-value curves never converge cleanly.](figures/09-dqn-training-diagnostics.png)

*DQN diagnostics: reward, loss and Q-value curves never converge cleanly.*

![Simulated warehouse layout for the RL rollout.](figures/10-warehouse-rollout.png)

*Simulated warehouse layout for the RL rollout.*

![Epsilon-decay study for the exploration schedule.](figures/11-epsilon-decay-study.png)

*Epsilon-decay study for the exploration schedule.*

![Edge trade-offs: FP32 50.04 ms · INT8 48.13 ms · PRUNE+INT8 49.00 ms, all 1.57 MB, accuracy 0.942 → 0.933.](figures/12-edge-deployment-tradeoffs.png)

*Edge trade-offs: FP32 50.04 ms · INT8 48.13 ms · PRUNE+INT8 49.00 ms, all 1.57 MB, accuracy 0.942 → 0.933.*
