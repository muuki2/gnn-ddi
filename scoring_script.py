"""
Scoring Script for GNN Molecular Graph Classification Challenge
================================================================

This script evaluates a participant's submission against the ground truth
test labels and computes the macro F1 score.

Usage:
    python scoring_script.py <submission_file>
    
Example:
    python scoring_script.py submissions/participant_submission.csv
    
The submission file should have two columns:
    - id: The molecule ID (matching those in test.csv)
    - target: The predicted label (0 or 1)

Note: The ground truth labels are stored securely and not publicly available.
This script is used by GitHub Actions for automated evaluation.
"""

import sys
import os
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


def validate_submission(submission_df, truth_df):
    """
    Validate that the submission file has the correct format.
    
    Args:
        submission_df: DataFrame with participant predictions
        truth_df: DataFrame with ground truth labels
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check required columns
    if 'id' not in submission_df.columns:
        return False, "Submission is missing 'id' column"
    
    if 'target' not in submission_df.columns:
        return False, "Submission is missing 'target' column"
    
    # Check number of rows
    if len(submission_df) != len(truth_df):
        return False, f"Submission has {len(submission_df)} rows, expected {len(truth_df)}"
    
    # Check that all IDs are present
    submission_ids = set(submission_df['id'].tolist())
    truth_ids = set(truth_df['id'].tolist())
    
    missing_ids = truth_ids - submission_ids
    if missing_ids:
        return False, f"Submission is missing {len(missing_ids)} IDs: {list(missing_ids)[:5]}..."
    
    extra_ids = submission_ids - truth_ids
    if extra_ids:
        return False, f"Submission has {len(extra_ids)} extra IDs: {list(extra_ids)[:5]}..."
    
    # Check that target values are valid (0 or 1)
    invalid_targets = submission_df[~submission_df['target'].isin([0, 1])]
    if len(invalid_targets) > 0:
        return False, f"Found {len(invalid_targets)} invalid target values (should be 0 or 1)"
    
    return True, ""


def compute_score(submission_df, truth_df):
    """
    Compute the macro F1 score for the submission.
    
    Args:
        submission_df: DataFrame with participant predictions
        truth_df: DataFrame with ground truth labels
        
    Returns:
        dict: Dictionary with various metrics
    """
    # Merge on ID to ensure alignment
    merged = truth_df.merge(submission_df, on='id', suffixes=('_true', '_pred'))
    
    y_true = merged['target_true'].values
    y_pred = merged['target_pred'].values
    
    # Compute metrics
    metrics = {
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_class_0': f1_score(y_true, y_pred, pos_label=0, average='binary'),
        'f1_class_1': f1_score(y_true, y_pred, pos_label=1, average='binary'),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def main():
    if len(sys.argv) < 2:
        print("Usage: python scoring_script.py <submission_file>")
        print("Example: python scoring_script.py submissions/my_submission.csv")
        sys.exit(1)
    
    submission_file = sys.argv[1]
    
    # Check if submission file exists
    if not os.path.exists(submission_file):
        print(f"Error: Submission file not found: {submission_file}")
        sys.exit(1)
    
    # Load ground truth labels
    # In production, this file is stored securely and populated by GitHub Actions
    truth_file = os.path.join(os.path.dirname(__file__), 'data', 'test_labels.csv')
    
    if not os.path.exists(truth_file):
        print(f"Error: Ground truth file not found: {truth_file}")
        print("Note: This file is only available in the evaluation environment.")
        sys.exit(1)
    
    print("="*60)
    print("GNN Molecular Graph Classification Challenge - Scoring")
    print("="*60)
    
    # Load files
    print(f"\nLoading submission: {submission_file}")
    submission_df = pd.read_csv(submission_file)
    
    print(f"Loading ground truth: {truth_file}")
    truth_df = pd.read_csv(truth_file)
    
    # Validate submission
    print("\nValidating submission format...")
    is_valid, error_msg = validate_submission(submission_df, truth_df)
    
    if not is_valid:
        print(f"❌ Validation failed: {error_msg}")
        sys.exit(1)
    
    print("✅ Submission format is valid")
    
    # Compute score
    print("\nComputing metrics...")
    metrics = compute_score(submission_df, truth_df)
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\n🎯 Macro F1 Score: {metrics['macro_f1']:.4f}")
    print(f"\nAdditional metrics:")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  - Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"  - F1 (class 0): {metrics['f1_class_0']:.4f}")
    print(f"  - F1 (class 1): {metrics['f1_class_1']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  Predicted:    0      1")
    print(f"  Actual 0:  {cm[0][0]:4d}   {cm[0][1]:4d}")
    print(f"  Actual 1:  {cm[1][0]:4d}   {cm[1][1]:4d}")
    
    print("\n" + "="*60)
    
    # Output the main score for GitHub Actions to capture
    # This line is parsed by update_leaderboard.py
    print(f"SCORE:{metrics['macro_f1']:.6f}")
    
    return metrics['macro_f1']


if __name__ == "__main__":
    score = main()
