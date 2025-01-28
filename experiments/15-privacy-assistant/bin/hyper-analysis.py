import json
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data(json_file: str) -> pd.DataFrame:
    """
    Load JSON data and extract relevant fields into a DataFrame.

    Parameters:
    - json_file: Path to the aggregated_data.json file.

    Returns:
    - DataFrame containing extracted fields.
    """
    with open(json_file, "r") as file:
        data = json.load(file)

    records = []
    for entry in data:
        try:
            record = {
                "dataset_name": entry.get("configuration", {}).get("dataset_name"),
                "epsilon": entry.get("hyperparameters", {}).get("target_epsilon"),
                "learning_rate": entry.get("best_params", {}).get("learning_rate"),
                "batch_size": entry.get("best_params", {}).get("batch_size"),
                "max_grad_norm": entry.get("best_params", {}).get("max_grad_norm"),
                "MulticlassAccuracy": entry.get("final_metrics", {}).get(
                    "MulticlassAccuracy"
                ),
            }
            records.append(record)
        except Exception as e:
            print(f"Error processing entry: {e}")

    df = pd.DataFrame(records)
    print("Data loaded successfully. Number of records:", len(df))

    return df


def clean_and_select_epsilons(df: pd.DataFrame, every_nth_epsilon: int) -> pd.DataFrame:
    """
    Clean the DataFrame by removing records with missing essential fields and select every_nth_epsilon
    that are common across all datasets by selecting every Nth epsilon and including the last epsilon.

    Parameters:
    - df: Original DataFrame.
    - every_nth_epsilon: Interval to select epsilon values for analysis.

    Returns:
    - Cleaned DataFrame with selected epsilon values.
    """
    essential_columns = [
        "epsilon",
        "learning_rate",
        "batch_size",
        "max_grad_norm",
        "MulticlassAccuracy",
        "dataset_name",
    ]
    df_clean = df.dropna(subset=essential_columns).reset_index(drop=True)
    print("After dropping missing values, number of records:", len(df_clean))

    datasets = df_clean["dataset_name"].unique()
    print("Datasets found:", datasets)

    # Find common epsilon values across all datasets
    epsilon_sets = {
        dataset: set(df_clean[df_clean["dataset_name"] == dataset]["epsilon"].unique())
        for dataset in datasets
    }
    common_epsilons = sorted(set.intersection(*epsilon_sets.values()))
    print(
        f"Number of common epsilon values across all datasets: {len(common_epsilons)}"
    )

    if not common_epsilons:
        raise ValueError("No common epsilon values found across all datasets.")

    # Select every Nth epsilon and include the last epsilon
    selected_epsilons = common_epsilons[::every_nth_epsilon]
    if common_epsilons[-1] not in selected_epsilons:
        selected_epsilons.append(common_epsilons[-1])

    print(f"Selected epsilon values for analysis: {selected_epsilons}")

    # Filter to include only selected epsilons
    df_selected = df_clean[df_clean["epsilon"].isin(selected_epsilons)].reset_index(
        drop=True
    )
    print("Number of records after selecting epsilons:", len(df_selected))

    return df_selected


def plot_density(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str = None,
    title: str = "",
    save_path: str = "",
):
    """
    Create a density plot and save the plot.

    Parameters:
    - df: DataFrame containing the data.
    - x: Column name for the x-axis.
    - y: Column name for the y-axis.
    - hue: (Optional) Column name for color encoding.
    - title: Title of the plot.
    - save_path: Path to save the plot image.
    """
    plt.figure(figsize=(10, 8))
    sns.kdeplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        fill=True,
        cmap="viridis",
        alpha=0.5,
        levels=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999],
    )
    plt.title(title)
    plt.xlabel(x.capitalize())
    plt.ylabel(y.capitalize())
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved density plot to {save_path}")

    plt.close()


def plot_pairplot(
    df: pd.DataFrame,
    variables: List[str],
    hue: str = None,
    title: str = "",
    save_path: str = "",
):
    """
    Create a pair plot for the specified variables and save the plot.

    Parameters:
    - df: DataFrame containing the data.
    - variables: List of column names to include in the pair plot.
    - hue: (Optional) Column name for color encoding.
    - title: Title of the plot.
    - save_path: Path to save the plot image.
    """
    g = sns.pairplot(df, vars=variables, hue=hue, diag_kind="kde", corner=True)
    g.figure.suptitle(title, y=1.02)
    plt.tight_layout()

    if save_path:
        g.savefig(save_path, bbox_inches="tight")
        print(f"Saved pair plot to {save_path}")

    plt.close()


def plot_correlation_heatmap(
    df: pd.DataFrame, variables: List[str], title: str = "", save_path: str = ""
):
    """
    Create a heatmap of the correlation matrix for the specified variables and save the plot.

    Parameters:
    - df: DataFrame containing the data.
    - variables: List of column names to include in the heatmap.
    - title: Title of the plot.
    - save_path: Path to save the plot image.
    """
    corr_matrix = df[variables].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved correlation heatmap to {save_path}")

    plt.close()


def plot_epsilon_vs_learning_rate(
    df: pd.DataFrame, title: str = "", save_path: str = ""
):
    """
    Create a line plot with epsilons on the x-axis, learning rates on the y-axis,
    and different colored lines for each dataset. Save the plot.

    Parameters:
    - df: DataFrame containing the data.
    - title: Title of the plot.
    - save_path: Path to save the plot image.
    """
    # Aggregate data
    df_agg = (
        df.groupby(["dataset_short_name", "epsilon"])["learning_rate"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_agg,
        x="epsilon",
        y="learning_rate",
        hue="dataset_short_name",
        marker="o",
    )
    plt.title(title)
    plt.xlabel("Epsilon")
    plt.ylabel("Mean Learning Rate")
    plt.xscale("log")

    unique_epsilons = sorted(df_agg["epsilon"].unique())
    plt.xticks(
        ticks=unique_epsilons, labels=[f"{e:.2f}" for e in unique_epsilons], rotation=45
    )
    plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved epsilon vs. learning rate plot to {save_path}")

    plt.close()


def plot_epsilon_vs_max_grad_norm(
    df: pd.DataFrame, title: str = "", save_path: str = ""
):
    """
    Create a line plot with epsilons on the x-axis, max_grad_norm on the y-axis,
    and different colored lines for each dataset. Save the plot.

    Parameters:
    - df: DataFrame containing the data.
    - title: Title of the plot.
    - save_path: Path to save the plot image.
    """
    # Aggregate data
    df_agg = (
        df.groupby(["dataset_short_name", "epsilon"])["max_grad_norm"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_agg,
        x="epsilon",
        y="max_grad_norm",
        hue="dataset_short_name",
        marker="o",
    )
    plt.title(title)
    plt.xlabel("Epsilon")
    plt.ylabel("Mean Max Grad Norm")
    plt.xscale("log")

    unique_epsilons = sorted(df_agg["epsilon"].unique())
    plt.xticks(
        ticks=unique_epsilons, labels=[f"{e:.2f}" for e in unique_epsilons], rotation=45
    )
    plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved epsilon vs. max grad norm plot to {save_path}")

    plt.close()


def plot_epsilon_vs_batch_size(df: pd.DataFrame, title: str = "", save_path: str = ""):
    """
    Create a line plot with epsilons on the x-axis, batch sizes on the y-axis,
    and different colored lines for each dataset. Save the plot.

    Parameters:
    - df: DataFrame containing the data.
    - title: Title of the plot.
    - save_path: Path to save the plot image.
    """
    # Aggregate data
    df_agg = (
        df.groupby(["dataset_short_name", "epsilon"])["batch_size"].mean().reset_index()
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_agg, x="epsilon", y="batch_size", hue="dataset_short_name", marker="o"
    )
    plt.title(title)
    plt.xlabel("Epsilon")
    plt.ylabel("Mean Batch Size")
    plt.xscale("log")

    unique_epsilons = sorted(df_agg["epsilon"].unique())
    plt.xticks(
        ticks=unique_epsilons, labels=[f"{e:.2f}" for e in unique_epsilons], rotation=45
    )
    plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved epsilon vs. batch size plot to {save_path}")

    plt.close()


def plot_epsilon_vs_accuracy(df: pd.DataFrame, title: str = "", save_path: str = ""):
    """
    Create a line plot with epsilons on the x-axis, MulticlassAccuracy on the y-axis,
    and different colored lines for each dataset. Save the plot.

    Parameters:
    - df: DataFrame containing the data.
    - title: Title of the plot.
    - save_path: Path to save the plot image.
    """
    # Aggregate data: Compute mean accuracy for each dataset and epsilon
    df_agg = (
        df.groupby(["dataset_short_name", "epsilon"])["MulticlassAccuracy"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_agg,
        x="epsilon",
        y="MulticlassAccuracy",
        hue="dataset_short_name",
        marker="o",
    )
    plt.title(title)
    plt.xlabel("Epsilon")
    plt.ylabel("Mean Accuracy")
    plt.xscale("log")  # Use log scale for epsilon

    # Set x-ticks to actual epsilon values rounded to two decimals
    unique_epsilons = sorted(df_agg["epsilon"].unique())
    plt.xticks(
        ticks=unique_epsilons, labels=[f"{e:.2f}" for e in unique_epsilons], rotation=45
    )
    plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved epsilon vs. accuracy plot to {save_path}")

    plt.close()


def main(
    json_file: str,
    every_nth_epsilon: int,
    selected_datasets: List[str],
    output_dir: str,
):
    """
    Main function to execute the analysis workflow.

    Parameters:
    - json_file: Path to the aggregated_data.json file.
    - every_nth_epsilon: Interval to select epsilon values for analysis.
    - selected_datasets: List of dataset short names to include in the analysis.
    - output_dir: Directory to save all output images and reports.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Subdirectories for different plot types
    plot_subdirs = {
        "density_plots": "density_plots",
        "pair_plots": "pair_plots",
        "correlation_heatmaps": "correlation_heatmaps",
        "line_plots": "line_plots",
    }

    for subdir in plot_subdirs.values():
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    # Line plot subdirectories
    line_plot_types = {
        "learning_rate": "learning_rate",
        "max_grad_norm": "max_grad_norm",
        "batch_size": "batch_size",
        "accuracy": "accuracy",
    }

    for plot_type in line_plot_types.values():
        os.makedirs(
            os.path.join(output_dir, plot_subdirs["line_plots"], plot_type),
            exist_ok=True,
        )

    df = load_data(json_file)
    df_selected = clean_and_select_epsilons(df, every_nth_epsilon)

    # Clean dataset names
    df_selected["dataset_short_name"] = df_selected["dataset_name"].apply(
        lambda x: x.split("/")[-1] if "/" in x else x
    )

    # Filter by selected datasets
    df_selected = df_selected[
        df_selected["dataset_short_name"].isin(selected_datasets)
    ].reset_index(drop=True)
    print(f"Filtered datasets. Number of records after filtering: {len(df_selected)}")

    # Do the actual plotting
    plot_epsilon_vs_accuracy(
        df=df_selected,
        title="Epsilon vs. Mean Accuracy by Dataset",
        save_path=os.path.join(
            output_dir,
            plot_subdirs["line_plots"],
            line_plot_types["accuracy"],
            "epsilon_vs_accuracy.png",
        ),
    )

    pairplot_vars = [
        "learning_rate",
        "batch_size",
        "max_grad_norm",
        "epsilon",
        "MulticlassAccuracy",
    ]
    plot_pairplot(
        df=df_selected,
        variables=pairplot_vars,
        hue="dataset_short_name",
        title="Pair Plot of Hyperparameters, Epsilon, and Accuracy by Dataset",
        save_path=os.path.join(
            output_dir, plot_subdirs["pair_plots"], "pairplot_combined_datasets.png"
        ),
    )

    correlation_vars = [
        "learning_rate",
        "batch_size",
        "max_grad_norm",
        "epsilon",
        "MulticlassAccuracy",
    ]
    plot_correlation_heatmap(
        df=df_selected,
        variables=correlation_vars,
        title="Correlation Heatmap of Hyperparameters, Epsilon, and Accuracy",
        save_path=os.path.join(
            output_dir,
            plot_subdirs["correlation_heatmaps"],
            "correlation_heatmap_combined.png",
        ),
    )

    plot_epsilon_vs_learning_rate(
        df=df_selected,
        title="Epsilon vs. Mean Learning Rate by Dataset",
        save_path=os.path.join(
            output_dir,
            plot_subdirs["line_plots"],
            line_plot_types["learning_rate"],
            "epsilon_vs_learning_rate.png",
        ),
    )

    plot_epsilon_vs_max_grad_norm(
        df=df_selected,
        title="Epsilon vs. Mean Max Grad Norm by Dataset",
        save_path=os.path.join(
            output_dir,
            plot_subdirs["line_plots"],
            line_plot_types["max_grad_norm"],
            "epsilon_vs_max_grad_norm.png",
        ),
    )

    plot_epsilon_vs_batch_size(
        df=df_selected,
        title="Epsilon vs. Mean Batch Size by Dataset",
        save_path=os.path.join(
            output_dir,
            plot_subdirs["line_plots"],
            line_plot_types["batch_size"],
            "epsilon_vs_batch_size.png",
        ),
    )

    for dataset in selected_datasets:
        df_dataset = df_selected[df_selected["dataset_short_name"] == dataset]
        print(f"Generating plots for dataset: {dataset}")

        plot_epsilon_vs_accuracy(
            df=df_dataset,
            title=f"Epsilon vs. Mean Accuracy for {dataset}",
            save_path=os.path.join(
                output_dir,
                plot_subdirs["line_plots"],
                line_plot_types["accuracy"],
                f"epsilon_vs_accuracy_{dataset}.png",
            ),
        )

        plot_density(
            df=df_dataset,
            x="learning_rate",
            y="max_grad_norm",
            hue=None,
            title=f"Density Plot of Learning Rate and Max Grad Norm for {dataset}",
            save_path=os.path.join(
                output_dir,
                plot_subdirs["density_plots"],
                f"density_learning_rate_vs_max_grad_norm_{dataset}.png",
            ),
        )

        plot_pairplot(
            df=df_dataset,
            variables=pairplot_vars,
            hue="epsilon",
            title=f"Pair Plot of Hyperparameters, Epsilon, and Accuracy for {dataset}",
            save_path=os.path.join(
                output_dir, plot_subdirs["pair_plots"], f"pairplot_{dataset}.png"
            ),
        )

        plot_correlation_heatmap(
            df=df_dataset,
            variables=correlation_vars,
            title=f"Correlation Heatmap for {dataset}",
            save_path=os.path.join(
                output_dir,
                plot_subdirs["correlation_heatmaps"],
                f"correlation_heatmap_{dataset}.png",
            ),
        )

        plot_epsilon_vs_learning_rate(
            df=df_dataset,
            title=f"Epsilon vs. Mean Learning Rate for {dataset}",
            save_path=os.path.join(
                output_dir,
                plot_subdirs["line_plots"],
                line_plot_types["learning_rate"],
                f"epsilon_vs_learning_rate_{dataset}.png",
            ),
        )

        plot_epsilon_vs_max_grad_norm(
            df=df_dataset,
            title=f"Epsilon vs. Mean Max Grad Norm for {dataset}",
            save_path=os.path.join(
                output_dir,
                plot_subdirs["line_plots"],
                line_plot_types["max_grad_norm"],
                f"epsilon_vs_max_grad_norm_{dataset}.png",
            ),
        )

        plot_epsilon_vs_batch_size(
            df=df_dataset,
            title=f"Epsilon vs. Mean Batch Size for {dataset}",
            save_path=os.path.join(
                output_dir,
                plot_subdirs["line_plots"],
                line_plot_types["batch_size"],
                f"epsilon_vs_batch_size_{dataset}.png",
            ),
        )

    print("All plots have been generated and saved successfully.")


if __name__ == "__main__":
    json_file_path = "aggregated_data.json"
    every_nth_epsilon = 1
    output_directory = "plots"

    selected_datasets = [
        "sun397",
        "eurosat",
        "plant_village",
        "oxford_iiit_pet",
        "colorectal_histology",
        "caltech_birds2011",
        "cassava",
    ]

    main(json_file_path, every_nth_epsilon, selected_datasets, output_directory)
