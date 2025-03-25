## Code for Muscle Oxygen Saturation Plateau Detection

This repository contains the code for criteria designed to detect plateaus in muscle oxygen saturation (SmO2) data. This code makes certain assumptions about the input and output data formats as explaind in the sections below.

### Input Data Format

Input data in the form of Excel files should be placed in the `input` directory. The plateau detection algorithms run for each of the sheets in the file. When loading the data, first three rows are skipped, and it is assumed that the `SmO2 Averaged` column is available.

### Output Data Format

The results output is placed in the `results` directory and is saved as a `.json` file with the required statistics for further analysis.

### Plotting Output

The code generates plots for each of the sheets and each of the detected plateaus. The plots are saved in the `plots` directory.

## Running the Code

To run the code make sure that you have `uv` installed. Run the following commands to get started:

1. `uv sync` to install the dependencies.
2. Activate the virtual environment.
3. Run the code with: `python src/smo2_plateau_detection/main.py`.