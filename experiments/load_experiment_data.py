"""Utility functions to parse experiment logs"""

import os
import pickle
import subprocess
import re
import json
import yaml

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

def get_config_from_log(submitit_output_path: str) -> dict:

    output = subprocess.check_output(['head', '-n', '250', submitit_output_path]) # Read last 50 lines
    output_lines = output.decode('utf-8').split('\n')

    config_start_line = [i for i, line in enumerate(output_lines) if line.startswith('Config')][0]
    config_end_line = [i for i, line in enumerate(output_lines) if 'verbose' in line][0] + 1

    output_str = "\n".join(output_lines[config_start_line:config_end_line])

    config = yaml.safe_load(output_str)

    return config

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
        