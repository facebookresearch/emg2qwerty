# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to print statistics about the emg2qwerty dataset by loading its
associated metadata.csv file.
"""

from pathlib import Path

import click
import pandas as pd


def print_dataset_stats(metadata_df: pd.DataFrame):
    df = metadata_df.copy()

    df["count"] = 1
    df["duration_hours"] = df["duration_mins"] / 60.0

    keys = ["count", "duration_hours", "num_keystrokes"]

    with pd.option_context("display.max_rows", None):
        group_df = df.groupby(["user"]).sum()[keys]
        print(group_df.sort_values(by="count", ascending=False))

    num_users = len(set(df["user"]))
    num_sessions = df["count"].sum()
    total_duration = df["duration_hours"].sum()
    total_keystrokes = df["num_keystrokes"].sum()
    total_prompts = df["num_prompts"].sum()

    print("\n---------------\n")
    print("Dataset Summary:")
    print(f"Num users = {num_users}")
    print(f"Num sessions = {num_sessions}")
    print(f"Total duration = {total_duration:.2f} hours")
    print(f"Total keystrokes = {total_keystrokes}")
    print(f"Total prompts = {total_prompts}")
    print("\n---------------\n")


@click.command()
@click.option(
    "--dataset-root",
    type=str,
    default=Path(__file__).parents[1].joinpath("data"),
    help="Dataset root directory",
)
def main(dataset_root: str):
    metadata_df = pd.read_csv(Path(dataset_root).joinpath("metadata.csv"))
    print_dataset_stats(metadata_df)


if __name__ == "__main__":
    main()
