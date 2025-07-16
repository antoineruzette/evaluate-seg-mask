# evaluate-seg-mask

A Python package for evaluating instance segmentation masks against ground truth.

## Installation

You can install the package directly from the repository:

```bash
pip install git+https://github.com/yourusername/evaluate-seg-mask.git
```

Or install in development mode from the local directory:

```bash
git clone https://github.com/yourusername/evaluate-seg-mask.git
cd evaluate-seg-mask
pip install -e .
```

## Usage

### Command Line Interface

```bash
evaluate-seg-mask <prediction_mask_path> [ground_truth_mask_path]
```

If ground truth path is not provided, it defaults to "../_static/images/student_group/001_masks.png".

### Python API

```python
from evaluate_seg_mask import evaluate_instance_segmentation, post_to_leaderboard

# Evaluate masks
results = evaluate_instance_segmentation(
    pred_path="path/to/prediction.png",
    gt_path="path/to/ground_truth.png",
    iou_threshold=0.5
)

# Optionally post to leaderboard
post_to_leaderboard("your_name", results)
```

## Metrics

The package computes the following metrics:

- Pixel Accuracy
- Pixel F1 Score
- Instance F1 Score
- Mean IoU
- Average Absolute Area Error
- Average Relative Area Error
- True Positive Instances
- False Positive Instances
- False Negative Instances

## License

MIT License