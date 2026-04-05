# Model Weights

The pipeline requires pre-trained model weights that are too large for git.

## Required files

| File | Size | Purpose |
|------|------|---------|
| `best_model_fpn.pth` | ~40 MB | Ridge likelihood model (row detection) |
| `mobile_sam.pt` | ~40 MB | Segment Anything (map annotator AI assist) |

## Download

Run the download script:

```bash
python models/download_models.py
```

Or download manually from the [GitHub Releases](https://github.com/neonnature0/CVRowDetection/releases) page and place files in this directory.

## Block detection checkpoints

Block detection model weights live in `block_detection/checkpoints/` and are managed separately (they are small enough to be checked in or are regenerated during training).
