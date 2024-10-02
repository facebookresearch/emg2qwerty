# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to generate/refresh hydra yaml config files for train, val and test
dataset splits for generic (user-agnostic) and personalized (user-specific)
modeling benchmarks.

Essentially, this script does the following:
1. Samples N users to be held out for personalization by giving precedence
   to those with the most number of sessions.
2. Generates N `user{n}.yaml` configs with train, val and test splits
   corresponding to each of the N users held out in (1).
   These can be used to benchmark per-user personalization.
3. Generates a single `generic.yaml` config such that the train and val splits
   are from all but the N users held out in (1), and the test split is the
   union of the test splits of the N held out `user{n}.yaml` configs.
   This can be used to benchmark user generalization.
"""

import logging
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import yaml


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def filter_users(df: pd.DataFrame, min_sessions: int) -> pd.Series:
    """Return a `pd.Series` consisting of users who have at least
    `min_sessions` sessions."""
    sessions_per_user = df.groupby("user")["session"].count()
    sessions_per_user = sessions_per_user[sessions_per_user >= min_sessions]
    users = sessions_per_user.index.to_series()
    return users


def sample_users(
    df: pd.DataFrame, n: int, min_sessions: int, seed: int | None = None
) -> pd.Series:
    """Sample `n` users from the given dataset who have at least
    `min_sessions` sessions."""
    users = filter_users(df, min_sessions=min_sessions)
    return users.sample(n=n, random_state=seed)


def sample_test_users(df: pd.DataFrame, n: int, seed: int | None = None) -> pd.Series:
    """Sample `n` users for personalization by giving precedence to those
    with the most number of sessions."""
    # Exclude uses with quality check warnings
    users_with_qc_tags = set(df[df.quality_check_tags.map(len) > 0].user)
    test_candidate_df = df[~df.user.isin(users_with_qc_tags)]

    # Compute session counts per user
    sessions_per_user = test_candidate_df.groupby("user")["session"].count()
    unique_counts = np.unique(sessions_per_user.values)

    # Pick users with the most number of sessions first until we have `n` users
    test_users: list[Any] = []
    for num_sessions in unique_counts[::-1]:
        n_remaining = n - len(test_users)
        if n_remaining <= 0:
            break

        users = sessions_per_user[sessions_per_user == num_sessions]
        if len(users) > n_remaining:
            users = users.sample(n=n_remaining, random_state=seed)
        test_users.extend(users.index)

    test_users = pd.Series(test_users)
    return test_users


def stratified_sample(
    df: pd.DataFrame, n: int, seed: int | None = None
) -> pd.DataFrame:
    """Sample `n` rows per user from `df`."""
    random_state = np.random.RandomState(seed)
    return df.groupby("user", group_keys=False).apply(
        lambda x: x.sample(n=n, random_state=random_state)
    )


def generate_split(
    df: pd.DataFrame,
    min_train_sessions_per_user: int,
    n_val_sessions_per_user: int,
    n_test_sessions_per_user: int,
    seed: int | None = None,
):
    """Split `df` into train, val and test partitions satisfying the
    provided per-user constraints."""
    # Filter out users with too few sessions to satisfy constraints
    min_sessions = (
        min_train_sessions_per_user + n_val_sessions_per_user + n_test_sessions_per_user
    )
    users = filter_users(df, min_sessions=min_sessions)
    df = df[df.user.isin(users)]

    # Sample test sessions
    test = stratified_sample(df, n=n_test_sessions_per_user, seed=seed)

    # Sample val sessions leaving out test
    train_val = df[~df.index.isin(test.index)]
    val = stratified_sample(train_val, n=n_val_sessions_per_user, seed=seed)

    # Rest is train
    train = train_val[~train_val.index.isin(val.index)]

    return train, val, test


def dump_split(
    user: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> None:
    config_dir = Path(__file__).parents[1].joinpath("config")
    path = config_dir.joinpath(f"user/{user}.yaml")

    def _format_split(split: dict[str, pd.DataFrame]) -> dict[str, Any]:
        fields = ["user", "session"]
        return {k: df[fields].to_dict("records") for k, df in split.items()}

    config = {
        "user": user,
        "dataset": _format_split(
            {
                "train": train,
                "val": val,
                "test": test,
            }
        ),
    }
    with open(path, "w") as f:
        f.write("# @package _global_\n")
        yaml.safe_dump(config, f, sort_keys=False)

    log.info(
        f"Dumped split for {user=} to {path}: "
        f"{len(train)} train sessions, "
        f"{len(val)} val sessions, "
        f"{len(test)} test sessions"
    )


@click.command()
@click.option(
    "--dataset-root",
    type=str,
    default=Path(__file__).parents[1].joinpath("data"),
    help="Dataset root directory",
)
@click.option(
    "--n-test-users",
    type=int,
    default=8,
    help="Number of users to be held out for test/personalization",
)
@click.option(
    "--min-train-sessions-per-user",
    type=int,
    default=2,
    help="Drop users for whom at least these many training sessions "
    "cannot be satisfied after considering val/test",
)
@click.option(
    "--n-val-sessions-per-user",
    type=int,
    default=2,
    help="Number of validation sessions per user",
)
@click.option(
    "--n-test-sessions-per-user",
    type=int,
    default=2,
    help="Number of test sessions per held-out/personalization user",
)
@click.option(
    "--seed",
    type=int,
    default=1501,
    help="Random seed for deterministic train/val/test splits",
)
def main(
    dataset_root: str,
    n_test_users: int,
    min_train_sessions_per_user: int,
    n_val_sessions_per_user: int,
    n_test_sessions_per_user: int,
    seed: int,
):
    df = pd.read_csv(Path(dataset_root).joinpath("metadata.csv"))
    df.quality_check_tags = df.quality_check_tags.apply(yaml.safe_load)

    # Sample users to be held-out for personalization and split the dataset
    # into two - one with sessions excluding these users to benchmark generic
    # (user-agnostic) modeling, and another with sessions belonging only to
    # these users for cross-user (out-of-domain) evaluation of generic models
    # as well as for benchmarking personalized (user-specific) models.
    test_users = sample_test_users(df, n=n_test_users, seed=seed)
    personalized_user_df = df[df.user.isin(test_users)]
    generic_user_df = df[~df.user.isin(test_users)]

    # Train/val/test splits for held-out users
    personalized_train, personalized_val, personalized_test = generate_split(
        df=personalized_user_df,
        min_train_sessions_per_user=min_train_sessions_per_user,
        n_val_sessions_per_user=n_val_sessions_per_user,
        n_test_sessions_per_user=n_test_sessions_per_user,
        seed=seed,
    )

    # Train and val split for generic user with sessions excluding those
    # from held-out users. Testing will be on sessions sampled from
    # held-out users (i.e., `personalized_test` split).
    generic_train, generic_val, _ = generate_split(
        df=generic_user_df,
        min_train_sessions_per_user=min_train_sessions_per_user,
        n_val_sessions_per_user=n_val_sessions_per_user,
        n_test_sessions_per_user=0,
        seed=seed,
    )

    # Dump split for generic user benchmark
    dump_split(
        user="generic",
        train=generic_train,
        val=generic_val,
        test=personalized_test,
    )

    # Dump `n_test_users` splits for per-user personalization benchmarks
    for i, user in enumerate(test_users):
        dump_split(
            user=f"user{i}",
            train=personalized_train[personalized_train["user"] == user],
            val=personalized_val[personalized_val["user"] == user],
            test=personalized_test[personalized_test["user"] == user],
        )


if __name__ == "__main__":
    main()
