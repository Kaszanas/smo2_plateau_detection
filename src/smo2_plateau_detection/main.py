import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from smo2_plateau_detection.calculate_criteria import (
    detect_criterion_a,
    detect_criterion_b,
    detect_criterion_c,
    detect_criterion_d,
)
from smo2_plateau_detection.plot import plot_data, plot_results_absolute


def pre_process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Responsible for selecting the selected column and interpolating the values
    to have a one second interval.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be pre-processed.

    Returns
    -------
    pd.DataFrame
        The pre-processed dataframe
    """

    # Select only the important columns:
    # Important column: SmO2 Averaged
    # All samples are 2 seconds apart, no other columns are important
    df = df.loc[:, ["SmO2 Averaged"]]
    df = df.loc[df["SmO2 Averaged"].notnull()]
    initial_length = len(df)

    logging.info(df.head())

    # Old indices for interpolation:
    old_indices = range(0, len(df))

    # Creating new indices for interpolation:
    new_indices = (pd.Index(range(0, len(df) * 2)) / 2).to_numpy()

    # Interpolating the values:
    interpolated_values = np.interp(new_indices, old_indices, df["SmO2 Averaged"])

    # Creating a new dataframe with the interpolated values:
    interpolated_df = pd.DataFrame(interpolated_values, columns=["SmO2 Averaged"])

    if initial_length * 2 != len(interpolated_df):
        logging.warning("Interpolation issue!")

    return interpolated_df


def cut_out_test_data(
    df: pd.DataFrame,
    omit_start_rows: int,
    omit_end_rows: int,
) -> pd.DataFrame:
    """
    Cuts out the data and returns the dataframe without the first and last rows,
    as specified by the input arguments.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be cut.
    omit_start_rows : int
        How many rows to omit from the start.
    omit_end_rows : int
        How many rows to omit from the end.

    Returns
    -------
    pd.DataFrame
        The dataframe without the omitted rows.
    """

    without_test_data = df.iloc[omit_start_rows:-omit_end_rows]
    without_test_data.reset_index(drop=True, inplace=True)

    if len(without_test_data) != len(df) - omit_start_rows - omit_end_rows:
        logging.warning("Test data cutting issue!")

    return without_test_data


def get_last_n_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Returns the last n rows of the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to get the last n rows from.
    n : int
        How many rows should be returned.

    Returns
    -------
    pd.DataFrame
        The last n rows of the dataframe
    """

    last_n_rows = df.iloc[-n:]
    last_n_rows.reset_index(drop=True, inplace=True)

    if len(last_n_rows) != n:
        logging.warning(f"Expected {n} rows, got {len(last_n_rows)} rows!")

    return last_n_rows


def get_average_time_plateau(start_end_list: List[Tuple[int, int]]) -> float:
    """
    Gets the average time of the plateaus from a list that contains start times
    and end times of the plateaus.

    Parameters
    ----------
    start_end_list : list
        A list of tuples containing the start and end times of the plateaus.

    Returns
    -------
    float
        The average time of the plateaus.
    """

    total_time = 0
    for start, end in start_end_list:
        total_time += end - start

    return total_time / len(start_end_list)


def get_max_time_plateau(start_end_list: list) -> float:
    """
    Acquires the maximum time of the plateaus from a list that contains start times
    and end times of the plateaus.

    Parameters
    ----------
    start_end_list : list
        A list of tuples containing the start and end times of the plateaus.

    Returns
    -------
    float
        The maximum time of the plateaus.
    """

    max_time = 0
    for start, end in start_end_list:
        time = end - start
        if time > max_time:
            max_time = time

    return max_time


def get_plateau_results(input_path: Path, plot_dir: Path, result_dir: Path) -> dict:
    """
    Gets results for all of the criteria specified in the Methods section.

    Parameters
    ----------
    input_path : Path
        The path to the directory containing the input data.
    plot_dir : Path
        The path to the directory where the plots should be saved.
    result_dir : Path
        The path to the directory where the results should be saved.

    Returns
    -------
    dict
        A dictionary containing the results.
    """

    results = defaultdict(lambda: defaultdict(int))

    criteria = {
        "absolute difference +-5": (detect_criterion_a, 5, True),
        "absolute difference +-10": (detect_criterion_b, 10, True),
        "percent change +-5%": (detect_criterion_c, 5, False),
        "percent change +-10%": (detect_criterion_d, 10, False),
    }

    for excel_file in input_path.glob("*.xlsx"):
        file_name = excel_file.stem
        results[file_name]["detected_list"] = []

        logging.info(f"Processing {file_name}")

        # Load all excel sheets, skip first 3 rows (these are empty:
        sheets = pd.read_excel(
            excel_file,
            sheet_name=None,
            header=0,
            skiprows=3,
        )

        column_name = "SmO2 Averaged"

        total_sheets = len(sheets)
        results[file_name]["total_sheets"] = total_sheets
        for sheet_name, sheet_df in sheets.items():
            sheet_df = pre_process_data(sheet_df)

            # Cut out the test data:
            only_max_test_data = cut_out_test_data(
                df=sheet_df,
                omit_start_rows=32,
                omit_end_rows=32,
            )

            # Select only the last 45 rows:
            only_max_test_data = get_last_n_rows(df=only_max_test_data, n=45)

            plot_data(
                plot_dir=plot_dir,
                df=only_max_test_data,
                sheet_name=sheet_name,
                column_name="SmO2 Averaged",
            )

            for criterion_name, criterion_args in criteria.items():
                criterion_func, threshold, display_threshold = criterion_args

                criterion_result, start_end_list = criterion_func(df=only_max_test_data)
                if criterion_result:
                    results[file_name][f"detected_n_{criterion_name}"] += 1
                    results[file_name]["detected_list"].append(
                        (sheet_name, criterion_name)
                    )

                    plot_results_absolute(
                        plot_dir=plot_dir,
                        sheet_name=sheet_name,
                        criterion=criterion_name,
                        df=only_max_test_data,
                        column_name=column_name,
                        start_end_list=start_end_list,
                        threshold=threshold,
                        display_threshold=display_threshold,
                    )

    return results


def save_results(results_dir: Path, results: dict) -> None:
    """
    Saves a dictionary to a JSON file with the results.

    Parameters
    ----------
    results_dir : Path
        Path to the directory where the results should be saved.
    results : dict
        the dictionary containing the results.
    """

    json_file = results_dir / "results.json"
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def main():
    logging.basicConfig(level=logging.INFO)

    DATA_DIR = Path("./data").resolve()
    if not DATA_DIR.exists():
        logging.info(f"Data directory {str(DATA_DIR)} does not exist! Creating!")
        DATA_DIR.mkdir(parents=True)

        logging.info(f"Please add data to {str(DATA_DIR)} and run the script again!")
        return

    PLOT_DIR = Path("./plots").resolve()
    if not PLOT_DIR.exists():
        logging.info(f"Plot directory {str(PLOT_DIR)} does not exist! Creating!")
        PLOT_DIR.mkdir(parents=True)

    RESULTS_DIR = Path("./results").resolve()
    if not RESULTS_DIR.exists():
        logging.info(f"Results directory {str(RESULTS_DIR)} does not exist! Creating!")
        RESULTS_DIR.mkdir(parents=True)

    results = get_plateau_results(
        input_path=DATA_DIR,
        plot_dir=PLOT_DIR,
        result_dir=RESULTS_DIR,
    )

    save_results(results_dir=RESULTS_DIR, results=results)


if __name__ == "__main__":
    # Set the backend to Agg
    plt.switch_backend("Agg")

    main()
