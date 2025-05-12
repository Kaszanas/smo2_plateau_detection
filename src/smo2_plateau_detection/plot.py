import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from spm1d.plot import plot_mean_sd


def plot_data(
    plot_dir: Path,
    df: pd.DataFrame,
    sheet_name: str,
    column_name: str,
) -> None:
    """
    Plots the raw data for the given column.

    Parameters
    ----------
    plot_dir : Path
        Directory where the plots will be saved.
    df : pd.DataFrame
        Dataframe containing the data.
    sheet_name : str
        Name of the Excel sheet.
    column_name : str
        Name of the column to plot.
    """

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=df.index, y=column_name)
    plt.xlabel("Time (seconds)")
    plt.ylabel(column_name)
    plt.title(f"{sheet_name} {column_name} vs Time")

    plot_file = plot_dir / f"{sheet_name}_{column_name}_vs_time.png"

    y_min, y_max = df[column_name].min(), df[column_name].max()
    plt.yticks(np.arange(y_min, y_max + 1, 2))

    x_min, x_max = df.index.min(), df.index.max()
    plt.xticks(np.arange(x_min, x_max + 1, 10).tolist() + [x_min, x_max])

    # Plot vertical black lines at the start and end of the x-axis
    plt.axvline(x=x_min, color="black", linestyle="-")
    plt.axvline(x=x_max, color="black", linestyle="-")

    plt.savefig(plot_file)
    plt.close()


def plot_results_absolute(
    plot_dir: Path,
    sheet_name: str,
    criterion: str,
    df: pd.DataFrame,
    column_name: str,
    start_end_list: List[Tuple[int, int]],
    threshold: int | None = None,
    display_threshold: bool = False,
):
    """
    Plotting function for the absolute value criterion.

    Parameters
    ----------
    plot_dir : Path
        Directory where the plots will be saved.
    sheet_name : str
        Name of the Excel sheet.
    criterion : str
        Name of the criterion.
    df : pd.DataFrame
        Dataframe containing the data.
    column_name : str
        Name of the column to plot.
    start_end_list : List[Tuple[int, int]]
        List of tuples containing the start and end indices of the plateaus.
    threshold : int | None, optional
        threshold to draw, by default None
    display_threshold : bool, optional
        Boolean if the the threshold should be display, by default False
    """

    if not start_end_list:
        logging.info(f"No plateaus detected for {sheet_name} - {criterion}")
        return

    for start_idx, end_idx in start_end_list:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x=df.index, y=column_name)

        # Highlight the plateau
        plt.axvspan(start_idx, end_idx, color="red", alpha=0.3, label="Plateau")

        plt.axvline(x=start_idx, color="green", linestyle="-", ymin=0)
        plt.axvline(x=end_idx, color="green", linestyle="-", ymin=0)

        # Plot ayline at the threshold value

        y_start = df[column_name].iloc[start_idx]
        y_end = df[column_name].iloc[end_idx]
        if threshold and display_threshold:
            plt.axhline(y=y_start + threshold, color="blue", linestyle="--", xmin=0)
            plt.axhline(y=y_start - threshold, color="blue", linestyle="--", xmin=0)

            # Threshold text labels:
            plt.text(
                start_idx + 1,
                y_start + threshold,
                f"{y_start + threshold}",
                color="blue",
                verticalalignment="bottom",
                fontsize=8,
            )
            plt.text(
                start_idx + 1,
                y_start - threshold,
                f"{y_start - threshold}",
                color="blue",
                verticalalignment="bottom",
                fontsize=8,
            )

        # Value at start:
        plt.text(
            start_idx + 1,
            y_start,
            f"{y_start}",
            color="red",
            verticalalignment="bottom",
            fontsize=8,
        )

        # Value at end:
        plt.text(
            end_idx + 1,
            y_end,
            f"{y_end}",
            color="red",
            verticalalignment="bottom",
            fontsize=8,
        )

        # Add labels and title
        plt.xlabel("Time (seconds)")
        plt.ylabel("SmO2 Averaged")
        plt.title(f"{sheet_name} - {criterion}, {start_idx} - {end_idx}")
        plt.legend()

        y_min, y_max = 0, 100
        plt.yticks(np.arange(y_min, y_max, 5))

        x_min, x_max = df.index.min(), df.index.max()
        plt.xticks(np.arange(x_min, x_max + 1, 10).tolist() + [x_min, x_max])

        # Plot vertical black lines at the start and end of the x-axis
        plt.axvline(x=x_min, color="black", linestyle="--")
        plt.axvline(x=x_max, color="black", linestyle="--")

        # Add text labels for start_idx and end_idx
        plt.text(
            start_idx + 1,
            y_max - 10,
            f"{start_idx}",
            color="green",
            verticalalignment="bottom",
            fontsize=14,
        )
        plt.text(
            end_idx + 1,
            y_max - 10,
            f"{end_idx}",
            color="green",
            verticalalignment="bottom",
            fontsize=14,
        )

        # Save the plot
        plot_file = plot_dir / f"{sheet_name}_{criterion}_{start_idx}_{end_idx}.png"
        plt.savefig(plot_file)
        plt.close()


def plot_mean_sd_intervals(plot_dir: Path, data: List[np.ndarray], title: str) -> None:
    numpy_data = np.array(data)

    plot_mean_sd(Y=numpy_data)

    plt.xlabel("Time [s]")  # Replace with your desired label
    plt.ylabel("SmO2 [%] Averaged")  # Replace with your desired label

    plot_path_pdf = (plot_dir / f"{title}.pdf").resolve()
    plot_path_png = (plot_dir / f"{title}.png").resolve()

    plt.savefig(plot_path_pdf, dpi=300)
    plt.savefig(plot_path_png, dpi=300)

    plt.close()
