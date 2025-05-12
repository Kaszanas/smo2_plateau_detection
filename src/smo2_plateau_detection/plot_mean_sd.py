import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from smo2_plateau_detection.main import cut_out_test_data, pre_process_data
from smo2_plateau_detection.plot import plot_mean_sd_intervals


def get_data(input_path: Path) -> Dict[str, List[np.ndarray]]:
    matrices_for_plotting = {
        "Quadriceps Normoxia": [],
        "Quadriceps Hypoxia": [],
        "Triceps Normoxia": [],
        "Triceps Hypoxia": [],
    }

    for excel_file in input_path.glob("*.xlsx"):
        file_name = excel_file.stem

        logging.info(f"Processing {file_name}")

        # Load all excel sheets, skip first 3 rows (these are empty:
        sheets = pd.read_excel(
            excel_file,
            sheet_name=None,
            header=0,
            skiprows=3,
        )

        for _, sheet_df in sheets.items():
            sheet_df = pre_process_data(sheet_df)

            # Cut out the test data:
            only_max_test_data = cut_out_test_data(
                df=sheet_df,
                omit_start_rows=32,
                omit_end_rows=32,
            )

            add_to_matrix(
                file_name=file_name,
                only_max_test_data=only_max_test_data,
                matrices_for_plotting=matrices_for_plotting,
            )

    return matrices_for_plotting


def add_to_matrix(
    file_name: str,
    only_max_test_data: pd.DataFrame,
    matrices_for_plotting: Dict[str, List[np.ndarray]],
) -> None:
    column_name = "SmO2 Averaged"

    test_data_array = only_max_test_data[column_name].to_numpy()
    if "hipo" in file_name.lower() and "quad" in file_name.lower():
        matrices_for_plotting["Quadriceps Hypoxia"].append(test_data_array)
        return
    if "normo" in file_name.lower() and "quad" in file_name.lower():
        matrices_for_plotting["Quadriceps Normoxia"].append(test_data_array)
        return
    if "hipo" in file_name.lower() and "tricep" in file_name.lower():
        matrices_for_plotting["Triceps Hypoxia"].append(test_data_array)
        return
    if "normo" in file_name.lower() and "tricep" in file_name.lower():
        matrices_for_plotting["Triceps Normoxia"].append(test_data_array)
        return

    logging.warning(
        f"Could not match file name to any of the predicaments: {str(file_name)}"
    )


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

    results_for_plotting = get_data(
        input_path=DATA_DIR,
    )

    for key, value in results_for_plotting.items():
        # Verify data length
        all_lengths = [len(data) for data in value]
        # Find the inconsistent length:
        inconsistent_length = Counter(all_lengths)
        if len(inconsistent_length) > 1:
            logging.warning(
                f"Data lengths are inconsistent for {key}: {inconsistent_length}"
            )

        # Cut out the array with the inconsistent length to become the same length:
        min_length = min(all_lengths)
        for i in range(len(value)):
            if len(value[i]) != min_length:
                logging.warning(
                    f"Cutting out {len(value[i]) - min_length} values from {key}"
                )
                value[i] = value[i][:min_length]

        plot_mean_sd_intervals(
            plot_dir=PLOT_DIR,
            data=value,
            title=key,
        )


if __name__ == "__main__":
    plt.switch_backend("Agg")

    main()
