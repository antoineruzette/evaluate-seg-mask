import sys
from .evaluator import evaluate_instance_segmentation, post_to_leaderboard

def main():
    """Command-line interface for evaluate-seg-mask."""
    if len(sys.argv) < 2:
        print("Usage: evaluate-seg-mask <prediction_mask_path> [ground_truth_mask_path]")
        sys.exit(1)

    pred_path = sys.argv[1]
    gt_path = sys.argv[2] if len(sys.argv) > 2 else "../_static/images/student_group/001_masks.png"
    
    try:
        results = evaluate_instance_segmentation(pred_path, gt_path)
        name = input("Enter your name/team ID for the leaderboard: ")
        post_to_leaderboard(name, results)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 