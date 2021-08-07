# Script to generate/refresh config files for train, val and test dataset
# splits for generic (user-agnostic) and personalized (user-specific)
# modeling benchmarks.
#
# Essentially, this script does the following:
# 1. Samples N users to be held-out for personalization by giving precendence
#    to those with the most number of sessions.
# 2. Generates N `user{n}.yaml` configs with train, val and test data from
#    each of the N corresponding held-out users.
# 3. Generates a single `generic.yaml` config such that the train and val
#    data are from all but the N held-out users, and the test data is the
#    combined test data from each of the N `user{n}.yaml` configs.

import logging
from pathlib import Path
from typing import Any, Dict, Optional

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
    df: pd.DataFrame, n: int, min_sessions: int, seed: Optional[int] = None
) -> pd.Series:
    """Sample `n` users from the given dataset who have at least
    `min_sessions` sessions."""
    users = filter_users(df, min_sessions=min_sessions)
    return users.sample(n, random_state=seed)


def sample_test_users(
    df: pd.DataFrame, n: int, seed: Optional[int] = None
) -> pd.Series:
    """Sample `n` users for personalization by giving precedence to those
    with the most number of sessions."""
    users_with_qc_tags = set(df[df.quality_check_tags.map(len) > 0].user)
    test_candidate_df = df[~df.user.isin(users_with_qc_tags)]

    # Compute session counts per user
    sessions_per_user = test_candidate_df.groupby("user")["session"].count()
    unique_counts = np.unique(sessions_per_user.values)

    # Pick users with the most number of sessions first until we have `n` users
    test_users = pd.Series(dtype=object)
    for num_sessions in unique_counts[::-1]:
        remaining = n - len(test_users)
        if remaining <= 0:
            break

        users = sessions_per_user[sessions_per_user == num_sessions]
        if len(users) > remaining:
            users = users.sample(remaining, random_state=seed)
        test_users = test_users.append(users.index.to_series())

    return test_users


def stratified_sample(
    df: pd.DataFrame, n: int, seed: Optional[int] = None
) -> pd.DataFrame:
    """Sample `n` rows per user from `df`."""
    random_state = np.random.RandomState(seed)
    return df.groupby("user", group_keys=False).apply(
        lambda x: x.sample(n, random_state=random_state)
    )


def split_dataset(
    df: pd.DataFrame,
    min_train_sessions_per_user: int,
    n_val_sessions_per_user: int,
    n_test_sessions_per_user: int,
    seed: Optional[int] = None,
):
    """Split `df` into train, val and test partitions satisfying the
    provided per-user constraints."""
    # Filter out users with too few sessions to satisfy constraints
    min_sessions = (
        min_train_sessions_per_user + n_val_sessions_per_user + n_test_sessions_per_user
    )
    users = filter_users(df, min_sessions=min_sessions)

    # Sample test sessions
    all = df[df.user.isin(users)]
    test = stratified_sample(all, n=n_test_sessions_per_user, seed=seed)

    # Sample val sessions leaving out test
    train_val = all[~all.index.isin(test.index)]
    val = stratified_sample(train_val, n=n_val_sessions_per_user, seed=seed)

    # Rest is train
    train = train_val[~train_val.index.isin(val.index)]

    return train, val, test


def dump_split(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, path: Path
) -> None:
    log.info(
        f"Config: {path}, train: {len(train)} sessions, "
        f"val: {len(val)} sessions, test: {len(test)} sessions"
    )

    def _format_split(split: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        fields = ["user", "session"]
        return {k: df[fields].to_dict("records") for k, df in split.items()}

    split = {"train": train, "val": val, "test": test}
    with open(path, "w") as f:
        f.write("# @package _global_\n")
        yaml.safe_dump({"dataset": _format_split(split)}, f, sort_keys=False)


@click.command()
@click.option(
    "--dataset-root",
    type=str,
    required=True,
    help="Dataset root directory",
)
@click.option(
    "--n-test-users",
    type=int,
    default=8,
    help="Number of users to be held-out for test/personalization",
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
    df = pd.read_csv(Path(dataset_root).joinpath("summary.csv"))
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
    personalized_train, personalized_val, personalized_test = split_dataset(
        personalized_user_df,
        min_train_sessions_per_user=min_train_sessions_per_user,
        n_val_sessions_per_user=n_val_sessions_per_user,
        n_test_sessions_per_user=n_test_sessions_per_user,
        seed=seed,
    )

    # Train and val splits for generic model with sessions excluding those
    # from held-out users. Testing will be on sessions sampled from
    # held-out users (i.e., `personalized_test` split).
    generic_train, generic_val, _ = split_dataset(
        generic_user_df,
        min_train_sessions_per_user=min_train_sessions_per_user,
        n_val_sessions_per_user=n_val_sessions_per_user,
        n_test_sessions_per_user=0,
        seed=seed,
    )

    config_dir = Path(__file__).parents[1].joinpath("config")

    # Dump split for generic benchmark
    dump_split(
        generic_train,
        generic_val,
        personalized_test,
        path=config_dir.joinpath("user/generic.yaml"),
    )

    # Dump `n_test_users` splits for per-user personalization benchmarks
    for i, user in enumerate(test_users):
        dump_split(
            personalized_train[personalized_train["user"] == user],
            personalized_val[personalized_val["user"] == user],
            personalized_test[personalized_test["user"] == user],
            path=config_dir.joinpath(f"user/user{i}.yaml"),
        )


if __name__ == "__main__":
    main()
