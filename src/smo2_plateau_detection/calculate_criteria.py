from typing import List, Tuple
from smo2_plateau_detection.plateau_detection import (
    find_plateaus_absolute_value,
    find_plateaus_percentage_difference,
)

import pandas as pd


def detect_criterion_a(df: pd.DataFrame) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Criterion A: SMO2 window of 30 seconds with change less than +- 5 points
    from the start of the plateau to any value before the end of the plateau.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data.

    Returns
    -------
    Tuple[bool, List[Tuple[int, int]]]
        A tuple containing a boolean value indicating if a plateau was detected
        and a list of tuples containing the start and end indices of the plateaus.
    """

    # Criterion A: SMO2 window of 30 seconds with change less than +- 5 points
    # means plateau:
    plateaus = find_plateaus_absolute_value(
        df=df,
        column_name="SmO2 Averaged",
        threshold=5,
        window_size=30,
    )

    detected_plateau = False
    if plateaus:
        detected_plateau = True

    return detected_plateau, plateaus


def detect_criterion_b(df: pd.DataFrame) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Criterion B: SMO2 window of 30 seconds with change less than +- 10 points
    from the start of the plateau to any value before the end of the plateau.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data.

    Returns
    -------
    Tuple[bool, List[Tuple[int, int]]]
        A tuple containing a boolean value indicating if a plateau was detected
        and a list of tuples containing the start and end indices of the plateaus.
    """

    # Criterion B: SMO2 window of 30 seconds with change less than += 10 points
    # means plateau:
    plateaus = find_plateaus_absolute_value(
        df=df,
        column_name="SmO2 Averaged",
        threshold=10,
        window_size=30,
    )

    detected_plateau = False
    if plateaus:
        detected_plateau = True

    return detected_plateau, plateaus


def detect_criterion_c(df: pd.DataFrame) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Criterion C: SMO2 window of 30 seconds with change less than +- 5% from the
    start of the plateau to any value before the end of the plateau.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data.

    Returns
    -------
    Tuple[bool, List[Tuple[int, int]]]
        A tuple containing a boolean value indicating if a plateau was detected
        and a list of tuples containing the start and end indices of the plateaus.
    """

    plateaus = find_plateaus_percentage_difference(
        df=df,
        column_name="SmO2 Averaged",
        threshold_percentage=5,
        window_size=30,
    )

    detected_plateau = False
    if plateaus:
        detected_plateau = True

    return detected_plateau, plateaus


def detect_criterion_d(df: pd.DataFrame) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Criterion D: SMO2 window of 30 seconds with change less than +- 10% from the
    start of the plateau to any value before the end of the plateau.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data.

    Returns
    -------
    Tuple[bool, List[Tuple[int, int]]]
        A tuple containing a boolean value indicating if a plateau was detected
        and a list of tuples containing the start and end indices of the plateaus.
    """

    plateaus = find_plateaus_percentage_difference(
        df=df,
        column_name="SmO2 Averaged",
        threshold_percentage=10,
        window_size=30,
    )

    detected_plateau = False
    if plateaus:
        detected_plateau = True

    return detected_plateau, plateaus
