"""
Leaderboard Update Script for GNN Molecular Graph Classification Challenge
==========================================================================

This script updates the leaderboard.md file with new submission scores.
It is called by the GitHub Actions workflow after scoring a submission.

Usage:
    python update_leaderboard.py <submission_file>
    
Environment Variables:
    ACTOR: GitHub username of the participant (set by GitHub Actions)
    
The script:
1. Computes the score for the submission
2. Updates the participant's entry in leaderboard.md (keeping best score)
3. Sorts the leaderboard by score (descending)
"""

import os
import sys
import re
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score


def load_leaderboard(leaderboard_path):
    """
    Load the current leaderboard from markdown file.
    
    Returns:
        list: List of tuples (rank, participant, score, date)
    """
    entries = []
    
    if not os.path.exists(leaderboard_path):
        return entries
    
    with open(leaderboard_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines (title, description, table header)
    in_table = False
    for line in lines:
        line = line.strip()
        
        # Detect table start
        if line.startswith('| Rank'):
            in_table = True
            continue
        
        # Skip separator line
        if line.startswith('|---') or line.startswith('| ---'):
            continue
        
        # Parse table rows
        if in_table and line.startswith('|'):
            parts = [p.strip() for p in line.strip('|').split('|')]
            if len(parts) >= 3:
                try:
                    rank = parts[0].strip()
                    participant = parts[1].strip()
                    score_str = parts[2].strip()
                    date = parts[3].strip() if len(parts) > 3 else ""
                    
                    # Extract numeric score
                    score = float(score_str)
                    entries.append({
                        'participant': participant,
                        'score': score,
                        'date': date
                    })
                except (ValueError, IndexError):
                    continue
    
    return entries


def save_leaderboard(leaderboard_path, entries):
    """
    Save the leaderboard to markdown file.
    
    Args:
        leaderboard_path: Path to leaderboard.md
        entries: List of entry dictionaries
    """
    # Sort by score (descending)
    entries.sort(key=lambda x: x['score'], reverse=True)
    
    with open(leaderboard_path, 'w') as f:
        # Header
        f.write("# 🏆 Leaderboard\n\n")
        f.write("Competition: **GNN Molecular Graph Classification Challenge**\n\n")
        f.write("Metric: **Macro F1 Score** (higher is better)\n\n")
        f.write("---\n\n")
        
        # Table header
        f.write("| Rank | Participant | Macro-F1 Score | Last Updated |\n")
        f.write("|------|-------------|----------------|---------------|\n")
        
        # Table rows
        for i, entry in enumerate(entries, 1):
            rank = i
            # Add medal emojis for top 3
            if i == 1:
                rank_str = "🥇 1"
            elif i == 2:
                rank_str = "🥈 2"
            elif i == 3:
                rank_str = "🥉 3"
            else:
                rank_str = str(i)
            
            # Mark baseline entries
            participant = entry['participant']
            if 'baseline' in participant.lower():
                participant = f"*{participant}*"
            
            f.write(f"| {rank_str} | {participant} | {entry['score']:.4f} | {entry['date']} |\n")
        
        # Footer
        f.write("\n---\n\n")
        f.write("*Italic entries are baseline models provided by organizers.*\n\n")
        f.write(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*\n")


def compute_submission_score(submission_file, truth_file):
    """
    Compute the macro F1 score for a submission.
    
    Args:
        submission_file: Path to submission CSV
        truth_file: Path to ground truth CSV
        
    Returns:
        float: Macro F1 score
    """
    submission_df = pd.read_csv(submission_file)
    truth_df = pd.read_csv(truth_file)
    
    # Merge on ID
    merged = truth_df.merge(submission_df, on='id', suffixes=('_true', '_pred'))
    
    y_true = merged['target_true'].values
    y_pred = merged['target_pred'].values
    
    return f1_score(y_true, y_pred, average='macro')


def main():
    if len(sys.argv) < 2:
        print("Usage: python update_leaderboard.py <submission_file>")
        sys.exit(1)
    
    submission_file = sys.argv[1]
    
    # Get participant name from environment or filename
    participant = os.environ.get('ACTOR')
    if not participant:
        # Extract from filename (e.g., "alice.csv" -> "alice")
        participant = os.path.basename(submission_file).replace('.csv', '')
    
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    leaderboard_path = os.path.join(script_dir, 'leaderboard.md')
    truth_file = os.path.join(script_dir, 'data', 'test_labels.csv')
    
    print(f"Updating leaderboard for participant: {participant}")
    print(f"Submission file: {submission_file}")
    
    # Check if ground truth exists
    if not os.path.exists(truth_file):
        print(f"Error: Ground truth file not found: {truth_file}")
        sys.exit(1)
    
    # Compute score
    try:
        score = compute_submission_score(submission_file, truth_file)
        print(f"Computed score: {score:.4f}")
    except Exception as e:
        print(f"Error computing score: {e}")
        sys.exit(1)
    
    # Load current leaderboard
    entries = load_leaderboard(leaderboard_path)
    print(f"Current leaderboard has {len(entries)} entries")
    
    # Check if participant already exists
    existing_idx = None
    for i, entry in enumerate(entries):
        if entry['participant'].lower() == participant.lower():
            existing_idx = i
            break
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    if existing_idx is not None:
        # Update existing entry if new score is better
        old_score = entries[existing_idx]['score']
        if score > old_score:
            print(f"Updating score: {old_score:.4f} -> {score:.4f}")
            entries[existing_idx]['score'] = score
            entries[existing_idx]['date'] = current_date
        else:
            print(f"Keeping existing score: {old_score:.4f} (new score: {score:.4f})")
    else:
        # Add new entry
        print(f"Adding new entry for {participant}")
        entries.append({
            'participant': participant,
            'score': score,
            'date': current_date
        })
    
    # Save updated leaderboard
    save_leaderboard(leaderboard_path, entries)
    print(f"Leaderboard updated successfully!")
    
    # Print current top 5
    entries.sort(key=lambda x: x['score'], reverse=True)
    print("\nTop 5 on leaderboard:")
    for i, entry in enumerate(entries[:5], 1):
        print(f"  {i}. {entry['participant']}: {entry['score']:.4f}")


if __name__ == "__main__":
    main()
