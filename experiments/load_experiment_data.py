"""Utility functions to parse experiment logs"""

import os
import pickle
import subprocess
import re
import json

def parse_submitit_output(submitit_output_path: str) -> dict:

    """Extracts val and test metrics from submitit output text file
    
    Args:
        submitit_output_path (str): path to submitit output file <id>_log.out
    
    Returns:
        metrics (dict): metrics for training and validation
    """

    output = subprocess.check_output(['tail', '-n', '50', submitit_output_path]) # Read last 50 lines
    output_str = " ".join(output.decode('utf-8').split('\n'))

    metrics = {} # 

    for metric_type in ['val', 'test']:

        metrics[metric_type] = {}

        re_pattern = re.compile(f"'{metric_type}/\w+':\s+\d+\.\d+")
        matches = re.findall(re_pattern, output_str) # Locate metric print-out in output string

        for match in matches: # Parse each match and add to results dict
            name, val = match.split(' ')
            metrics[metric_type][name.split('/')[1].replace("':",'')] = float(val)

    return metrics

def parse_and_save_metrics(submitit_output_path: str, save_path: str = './metrics.json'):

    # Load metrics from log file
    metrics = parse_submitit_output(submitit_output_path)

    # Save as JSON
    json.dump(metrics, open(save_path, 'w'), indent=2)

def parse_and_save_personalized_model_metrics(submitit_root_path: str, save_root: str = '.'):

    for i, sub_dir in enumerate(sorted(os.listdir(submitit_root_path))):
        
        # Get path to out file
        user_dir = os.path.join(submitit_root_path, sub_dir)
        try:
            out_file = [ckpt for ckpt in os.listdir(user_dir) if 'out' in ckpt][0]
        except IndexError: # No log directory found
            continue

        if '_' in sub_dir: # Personalized model eval
            user_id = sub_dir.split('_')[-1]
        else:
            user_id = i

        # Parse metrics for this user
        parse_and_save_metrics(
            submitit_output_path=os.path.join(user_dir, out_file),
            save_path=os.path.join(save_root, f"user{user_id}_metrics.json")
        )
        


if __name__ == "__main__":

    import sys

    parse_and_save_personalized_model_metrics(
        "/private/home/nmehlman/emg2qwerty/logs/2024-07-04/02/submitit_logs",
        "/private/home/nmehlman/emg2qwerty/experiments/2024070402/metrics"
    )