# Future Work

## Stratified train/eval split by region

Now that region tracking is in place, the next step is stratified sampling during training — where the held-out eval set is sampled proportionally from each region instead of randomly. This ensures that per-region regression detection has consistent statistical power across regions.

This is a training logic change that touches `training/train.py` and needs its own plan. It should be tested in isolation rather than bundled with tracking features.

Optionally, also consider per-region loss weighting during training so under-represented regions get the same effective weight as over-represented ones.
