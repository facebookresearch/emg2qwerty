# %%
from pathlib import Path

import start_research
import yaml


# %%
def reduce_dataset(yaml_path: Path, output_path: Path, ratio: int = 2):
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
            if section == "train":
                out_sessions["dataset"][section] = sessions[section][::ratio]  # type: ignore
            else:
                out_sessions["dataset"][section] = sessions[section][:: (ratio // 5)]  # type: ignore
    # Write the modified data back to the file
    with open(output_path, "w") as file:
        yaml.dump(out_sessions, file, default_flow_style=False)

    print(f"Reduced dataset in {output_path}")


# %%
config_path = Path("config/user/generic.yaml")
output_path = Path("config/user/generic_reduced.yaml")
# %%
reduce_dataset(config_path, output_path, ratio=30)


# %%
def join_user_yaml_files(output_path: Path):
    """
    Finds all YAML files in config/user that begin with 'user' and joins them into a single YAML file.
    All files are expected to have the same structure.

    Args:
        output_path: Path to save the joined YAML file
    """
    # Find all user*.yaml files in config/user directory
    user_files = list(Path("config/user").glob("user*.yaml"))

    if not user_files:
        print("No user*.yaml files found in config/user directory")
        return

    # Initialize with the first file's data
    with open(user_files[0], "r") as file:
        combined_data = yaml.safe_load(file)

    # Create a combined dataset structure
    if "dataset" not in combined_data:
        combined_data["dataset"] = {}

    # Process each section (train, validation, test)
    for section in ["train", "val", "test"]:
        if section not in combined_data["dataset"]:
            combined_data["dataset"][section] = []

    # Join data from the rest of the files
    for file_path in user_files[1:]:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)

        if "dataset" in data:
            for section in ["train", "val", "test"]:
                if section in data["dataset"] and isinstance(data["dataset"][section], list):
                    combined_data["dataset"][section].extend(data["dataset"][section])

    # Write the combined data to the output file
    with open(output_path, "w") as file:
        yaml.dump(combined_data, file, default_flow_style=False)

    print(f"Joined {len(user_files)} user YAML files into {output_path}")


# %%

# Example usage of the new function
joined_output_path = Path("config/user/joined_users.yaml")
join_user_yaml_files(joined_output_path)
# %%
