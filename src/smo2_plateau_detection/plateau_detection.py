from typing import List, Tuple
import pandas as pd


def index_shorter_than_df(i: int, len_df: int) -> bool:
    """
    Check if the index is shorter than the dataframe length.

    Parameters
    ----------
    i : int
        The index to check.
    len_df : int
        The length of the dataframe.

    Returns
    -------
    bool
        True if the index is shorter than the dataframe length, False otherwise.
    """

    index = i + 1

    return index < len_df


def absolute_diff_lower_than_threshold(
    absolute_diff: int | float,
    threshold: int | float,
) -> bool:
    """
    Check if the absolute difference between two values is lower than a threshold.

    Parameters
    ----------
    absolute_diff : int | float
        The absolute difference between two values.
    threshold : int | float
        The threshold for the absolute difference.

    Returns
    -------
    bool
        True if the absolute difference is lower than the threshold, False otherwise.
    """

    lower_than_threshold = absolute_diff < threshold
    return lower_than_threshold


def diff(current_value: float, future_value: float) -> float:
    """
    Calculate the difference between two values. Useful for debugging.

    Parameters
    ----------
    current_value : float
        Current value.
    future_value : float
        Future value for which the difference is calculated.

    Returns
    -------
    float
        The difference between the two values.
    """

    value_diff = current_value - future_value

    return value_diff


def find_plateaus_absolute_value(
    df: pd.DataFrame,
    column_name: str,
    threshold: int,
    window_size: int,
) -> List[Tuple[int, int]]:
    """
    Find plateaus based on the absolute difference between consecutive values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data.
    column_name : str
        Name of the column to analyze.
    threshold : int
        The threshold for the difference between consecutive values.
    window_size : int
        The minimum size of the plateau

    Returns
    -------
    List[Tuple[int, int]]
        A list of tuples containing the start and end indices of the plateaus.
    """

    plateaus = []
    i = 0  # Start index

    while i < len(df):
        # Find start of plateau
        start_index = i

        # Move forward in consecutive stable values
        while index_shorter_than_df(
            i=i,
            len_df=len(df),
        ) and absolute_diff_lower_than_threshold(
            absolute_diff=abs(
                diff(
                    current_value=df[column_name].iloc[start_index],
                    future_value=df[column_name].iloc[i + 1],
                )
            ),
            threshold=threshold,
        ):
            i += 1

        # The last valid index before the threshold is exceeded
        end_index = i

        # If plateau length is smaller than window_size, ignore it
        plateau_size = end_index - start_index + 1
        if plateau_size >= window_size:
            plateaus.append((start_index, end_index))

        # Move to the next potential plateau
        i += 1

    return plateaus


def percentage_diff(current_value: float, future_value: float) -> float:
    """
    Calculate the percentage difference between two values.

    Parameters
    ----------
    current_value : float
        Current value.
    future_value : float
        Future value for which the percentage difference is calculated.

    Returns
    -------
    float
        The percentage difference between the two values.
    """

    # Avoid division by zero:
    if current_value == 0:
        current_value = 0.0001

    return abs(future_value - current_value) / current_value * 100


def find_plateaus_percentage_difference(
    df: pd.DataFrame,
    column_name: str,
    threshold_percentage: float,
    window_size: int,
) -> List[Tuple[int, int]]:
    """
    Find plateaus based on the percentage difference between consecutive values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data.
    column_name : str
        Name of the column to analyze.
    threshold_percentage : float
        The threshold percentage for the difference between consecutive values.
    window_size : int
        The minimum size of the plateau.

    Returns
    -------
    List[Tuple[int, int]]
        A list of tuples containing the start and end indices of the plateaus.
    """

    plateaus = []
    i = 0  # Start index

    while i < len(df):
        # Find start of plateau
        start_index = i
        start_value = df[column_name].iloc[start_index]

        # Move forward in consecutive stable values
        while index_shorter_than_df(
            i=i,
            len_df=len(df),
        ) and absolute_diff_lower_than_threshold(
            absolute_diff=percentage_diff(
                current_value=start_value,
                future_value=df[column_name].iloc[i + 1],
            ),
            threshold=threshold_percentage,
        ):
            i += 1

        # The last valid index before the threshold is exceeded
        end_index = i

        # If plateau length is smaller than window_size, ignore it
        plateau_size = end_index - start_index + 1
        if plateau_size >= window_size:
            plateaus.append((start_index, end_index))

        # Move to the next potential plateau
        i += 1

    return plateaus
