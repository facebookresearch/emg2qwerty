# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to aggregate and print experimental results as reported in Table 2 of the paper.
"""

import numpy as np
import pandas as pd

EXPERIMENTAL_RESULTS = [
    {
        "Model benchmark": "Generic (no personalization)",
        "LM": "No LM",
        "Metric": "Val CER",
        "CER": [60.07, 55.59, 47.38, 59.03, 58.93, 56.01, 58.08, 49.45],
    },
    {
        "Model benchmark": "Generic (no personalization)",
        "LM": "No LM",
        "Metric": "Test CER",
        "CER": [61.48, 59.96, 48.00, 54.69, 58.24, 53.86, 54.66, 52.17],
    },
    {
        "Model benchmark": "Generic (no personalization)",
        "LM": "LM",
        "Metric": "Val CER",
        "CER": [57.67, 52.94, 40.89, 56.06, 54.96, 52.90, 56.14, 45.23],
    },
    {
        "Model benchmark": "Generic (no personalization)",
        "LM": "LM",
        "Metric": "Test CER",
        "CER": [58.40, 56.66, 42.90, 51.63, 54.23, 49.55, 52.59, 48.32],
    },
    {
        "Model benchmark": "Personalized (random-init)",
        "LM": "No LM",
        "Metric": "Val CER",
        "CER": [24.13, 11.31, 10.05, 14.61, 10.64, 9.363, 22.2, 22.91],
    },
    {
        "Model benchmark": "Personalized (random-init)",
        "LM": "No LM",
        "Metric": "Test CER",
        "CER": [26.6, 14.15, 11.12, 13.25, 10.51, 7.648, 18.82, 20.94],
    },
    {
        "Model benchmark": "Personalized (random-init)",
        "LM": "LM",
        "Metric": "Val CER",
        "CER": [18.45, 7.612, 7.183, 9.928, 6.985, 6.869, 15.2, 16.01],
    },
    {
        "Model benchmark": "Personalized (random-init)",
        "LM": "LM",
        "Metric": "Test CER",
        "CER": [20.72, 7.959, 6.778, 7.29, 4.851, 3.824, 11.66, 13.28],
    },
    {
        "Model benchmark": "Personalized (finetuned)",
        "LM": "No LM",
        "Metric": "Val CER",
        "CER": [17.96, 8.392, 8.165, 9.543, 7.575, 7.148, 17.17, 15.19],
    },
    {
        "Model benchmark": "Personalized (finetuned)",
        "LM": "No LM",
        "Metric": "Test CER",
        "CER": [20.57, 10.32, 8.409, 8.928, 7.907, 5.811, 14.21, 14.06],
    },
    {
        "Model benchmark": "Personalized (finetuned)",
        "LM": "LM",
        "Metric": "Val CER",
        "CER": [13.72, 6.115, 6.224, 6.501, 5.236, 5.515, 12.35, 10.83],
    },
    {
        "Model benchmark": "Personalized (finetuned)",
        "LM": "LM",
        "Metric": "Test CER",
        "CER": [15.04, 6.183, 5.083, 4.75, 3.908, 3.162, 8.832, 8.68],
    },
]


def main():
    df = pd.DataFrame.from_records(EXPERIMENTAL_RESULTS)
    assert (df["CER"].map(len) == 8).all()

    df["CER_mean"] = df["CER"].map(np.mean)
    df["CER_std"] = df["CER"].map(np.std)
    df["CER_min"] = df["CER"].map(np.min)
    df["CER_max"] = df["CER"].map(np.max)

    df["CER (mean +/- std)"] = df.apply(
        lambda row: f"{row['CER_mean']:.2f} +/- {row['CER_std']:.2f}",
        axis=1,
    )
    df["CER (min, max)"] = df.apply(
        lambda row: f"[{row['CER_min']:.2f}, {row['CER_max']:.2f}]",
        axis=1,
    )

    df = pd.pivot(
        df,
        index="Model benchmark",
        columns=["Metric", "LM"],
        values=["CER (mean +/- std)", "CER (min, max)"],
    )
    with pd.option_context("display.precision", 2):
        print(df)


if __name__ == "__main__":
    main()
