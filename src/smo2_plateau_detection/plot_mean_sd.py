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
        "Quadriceps Normoxia Test": [],
        "Quadriceps Hypoxia Test": [],
        "Triceps Normoxia Test": [],
        "Triceps Hypoxia Test": [],
        "Quadriceps Normoxia All": [],
        "Quadriceps Hypoxia All": [],
        "Triceps Normoxia All": [],
        "Triceps Hypoxia All": [],
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

        for sheet_name, sheet_df in sheets.items():
            sheet_df = pre_process_data(sheet_df)

            path_to_data_export = Path(f"./data/export/").resolve()
            if not path_to_data_export.exists():
                logging.info(
                    f"Data export directory {str(path_to_data_export)} does not exist! Creating!"
                )
                path_to_data_export.mkdir(parents=True)

            path_to_exported_file = (
                path_to_data_export / f"{file_name}_{sheet_name}.csv"
            ).resolve()
            sheet_df.to_csv(path_or_buf=path_to_exported_file)

            # Cut out the test data:
            only_max_test_data = cut_out_test_data(
                df=sheet_df,
                omit_start_rows=32,
                omit_end_rows=32,
            )

            # Add to the matrix entire data:
            add_to_matrix(
                file_name=file_name,
                only_max_test_data=sheet_df,
                matrices_for_plotting=matrices_for_plotting,
                suffix=" All",
            )

            # Add to the matrix only test data:
            add_to_matrix(
                file_name=file_name,
                only_max_test_data=only_max_test_data,
                matrices_for_plotting=matrices_for_plotting,
                suffix=" Test",
            )

    return matrices_for_plotting


def add_to_matrix(
    file_name: str,
    only_max_test_data: pd.DataFrame,
    matrices_for_plotting: Dict[str, List[np.ndarray]],
    suffix: str,
) -> None:
    column_name = "SmO2 Averaged"

    test_data_array = only_max_test_data[column_name].to_numpy()
    if "hipo" in file_name.lower() and "quad" in file_name.lower():
        matrices_for_plotting[f"Quadriceps Hypoxia{suffix}"].append(test_data_array)
        return
    if "normo" in file_name.lower() and "quad" in file_name.lower():
        matrices_for_plotting[f"Quadriceps Normoxia{suffix}"].append(test_data_array)
        return
    if "hipo" in file_name.lower() and "tricep" in file_name.lower():
        matrices_for_plotting[f"Triceps Hypoxia{suffix}"].append(test_data_array)
        return
    if "normo" in file_name.lower() and "tricep" in file_name.lower():
        matrices_for_plotting[f"Triceps Normoxia{suffix}"].append(test_data_array)
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

        place_vertical = False
        if "All" in key:
            place_vertical = True

        plot_mean_sd_intervals(
            plot_dir=PLOT_DIR,
            data=value,
            title=key,
            place_vertical=place_vertical,
        )


if __name__ == "__main__":
    plt.switch_backend("Agg")

    main()
