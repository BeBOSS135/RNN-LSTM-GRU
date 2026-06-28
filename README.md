# RNN-LSTM-GRU

This project compares basic RNN architectures (RNN, LSTM, GRU) for time-series forecasting tasks using TensorFlow and Keras.

## Tech Stack

- Python 3.8+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Conda (optional)

## Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RNN-LSTM-GRU.git
   cd RNN-LSTM-GRU
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install required packages:
   ```bash
   pip install tensorflow pandas numpy matplotlib
   ```

## Usage Example

To run the example, execute the following command:

```bash
python ComparingBasicRNNArchitectures.py
```

This script will load the necessary data and train RNN, LSTM, and GRU models on it.

## Folder Structure Overview

- `ComparingBasicRNNArchitectures.py`: Main script for comparing RNN architectures.
- `figures/`: Directory to store generated figures and plots.
- `data/`: Placeholder directory for dataset files (to be provided by the user).

## Environment

- Python 3.8+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib

You can create a conda environment with these packages using:

```bash
conda create -n rnn-env python=3.8 tensorflow pandas numpy matplotlib
conda activate rnn-env
```

## Hardware

Minimum: A GPU with ~4GB VRAM; runs on CPU for small datasets.

## Dataset

The code consumes the following data files:
- `Input_Data.txt`
- `Output_Data.txt`
- `CoffeeMachinemaxAgg.txt`
- `CoffeeMachinemaxApp.txt`
- `summary_results.csv`

Ensure these files are placed in a directory specified by `DATA_DIR` (default is the current directory).
