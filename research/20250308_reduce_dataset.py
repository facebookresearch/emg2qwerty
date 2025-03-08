# %%
from pathlib import Path

import start_research
import yaml


# %%
def reduce_dataset(yaml_path: Path, output_path: Path):
    """
    Reads a YAML file and removes every other entry from train, validation, and test sets.
    """
    # Read the YAML file
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    sessions = data["dataset"]
    out_sessions = {"user": f"{data['user']}_reduced", "dataset": {}}

    # Process each section (train, validation, test) if they exist
    for section in ["train", "val", "test"]:
        if section in sessions and isinstance(sessions[section], list):
            # Keep only every other entry
            out_sessions["dataset"][section] = sessions[section][::2]  # type: ignore
    # Write the modified data back to the file
    with open(output_path, "w") as file:
        yaml.dump(out_sessions, file, default_flow_style=False)

    print(f"Reduced dataset in {output_path}")


# %%
config_path = Path("config/user/generic.yaml")
output_path = Path("config/user/generic_reduced.yaml")
# %%
reduce_dataset(config_path, output_path)
# %%
