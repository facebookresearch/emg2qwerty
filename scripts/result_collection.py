import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import typer


def collect_results(experiment_dir):
    """Collect metrics from experiment runs."""
    results = []

    # Find all experiment directories
    for model_dir in Path(experiment_dir).glob("*"):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # Find the metrics file
        metrics_files = list(model_dir.glob("**/test_metrics.json"))
        if not metrics_files:
            print(f"No metrics found for {model_name}")
            continue

        # Load the metrics
        with open(metrics_files[0], "r") as f:
            metrics = json.load(f)

        metrics["model"] = model_name
        results.append(metrics)

    return pd.DataFrame(results)


def plot_comparison(results_df, metric_name="test/CER"):
    """Plot comparison of a specific metric across models."""
    plt.figure(figsize=(10, 6))
    results_df.sort_values(metric_name).plot.bar(x="model", y=metric_name)
    plt.title(f"Comparison of {metric_name} across models")
    plt.ylabel(metric_name)
    plt.tight_layout()
    plt.savefig(f"{metric_name.replace('/', '_')}_comparison.png")


def main(experiment_dir: str = typer.Argument(..., help="Directory containing experiment results")):
    """Analyze experiment results."""
    results = collect_results(experiment_dir)
    print(results)

    if not results.empty:
        plot_comparison(results)
        plot_comparison(results, "test/WER")


if __name__ == "__main__":
    typer.run(main)
