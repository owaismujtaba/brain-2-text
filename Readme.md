# Brain-2-Text

A Python project for loading, processing, and managing neural data stored in HDF5 files, with reusable logging and dataset utilities.

## Project Structure

```
src/
  dataset/
    dataset.py      # Classes for reading and managing HDF5 datasets
    utils.py        # Utility functions (e.g., recursive file loading)
  logging/
    log.py          # Reusable logger setup
```

## Features

- **HDF5 Data Loading:** Read neural features and metadata from HDF5 files.
- **Custom Dataset Class:** Easily integrate with PyTorch DataLoader.
- **Recursive File Search:** Find all relevant files in a directory tree.
- **Reusable Logging:** Unified logging to file and console.

## Installation

1. Clone the repository:
    ```
    git clone <your-repo-url>
    cd brain-2-text
    ```

2. Install dependencies:
    ```
    pip install torch h5py
    ```

## Usage

### Logging

Import and set up a logger:
```python
from src.logging.log import setup_logger
logger = setup_logger(name="brain2text", log_file="logs/train.log", level="INFO")
```

### Dataset Loading

```python
from src.dataset.dataset import H5pyDataset, DatasetLoader

# Initialize dataset loader
loader = DatasetLoader(data_dir="path/to/data", logger=logger)
dataset = loader.get_dataloader(kind='train')
```

### Utilities

```python
from src.dataset.utils import get_all_files

files = get_all_files("path/to/data", extensions=('.hdf5',))
```

## Folder Structure

- `src/dataset/`: Data loading and utility functions.
- `src/logging/`: Logger configuration.

## Requirements

- Python 3.8+
- torch
- h5py

## License

MIT License

## Author

Your Name Here