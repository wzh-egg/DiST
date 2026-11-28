# DiST

DiST forecasts air quality by combining station observations with gridded air quality fields. In the AIR experiment, the model uses the past 12 hours to predict hourly AQI at 1,332 monitoring stations in China for the next 12 hours. Station and grid representations exchange information over four rounds.

## Code

| File | Purpose |
| --- | --- |
| `model.py` | Defines the DiST model and its horizon-specific fusion weights. |
| `train.py` | Trains the model and saves the best checkpoint. |
| `evaluate.py` | Loads a checkpoint and evaluates it on the test set. |
| `preprocess.py` | Builds the region, station-neighbor, and grid-patch indices. |
| `configs/air.json` | AIR model and training settings. |
| `src/` | Model layers, data loading, and training utilities. |

## Installation

The AIR configuration uses Linux, Python 3.10, and PyTorch 2.5.1 with CUDA 12.4.

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install 'numpy>=1.23,<2'
```

## Data

The AIR experiment covers 2016–2019 and uses two hourly data sources.

| Input | Source |
| --- | --- |
| Station AQI | Download the nationwide station-level records and select the AQI measurements. [China historical air quality archive](https://quotsoft.net/air/) |
| Gridded AQI | The experiment derives gridded AQI from pollutant concentrations in the [China Air Quality Reanalysis dataset (CAQRA)](https://doi.org/10.11922/sciencedb.00053). |

### Prepare the inputs

The commands below assume you have assembled the station observations, forecast window indices, and spatial metadata. Prepare the grid cache separately by deriving hourly AQI from CAQRA, extracting and normalizing the 5 × 5 patches, and retaining their validity masks. The released preprocessing scripts cover the station series and spatial indices.

```bash
python src/prepare_stations.py ./raw/stations ./splits ./data/stations
bash run.sh preprocess ./raw/coordinates_lon_lat.npy ./raw/candidates.npz ./data/geometry.npz
```

- `./raw/stations/` contains `targets.npy` (hour × station, with NaN for missing observations), `station_ids.npy`, and hourly `timestamps.npy` in Beijing time.
- `./splits/` contains `train_origins.npy`, `val_origins.npy`, and `test_origins.npy`. Each index identifies the first forecast hour of a window.
- `./data/grid/` contains the normalized grid values (`normalized.npy`) and their boolean validity mask (`valid.npy`). Both arrays have shape `[hours, patches, 5, 5]`. The patch axis follows the order of `unique_ids` in `candidates.npz`.

## Training and evaluation

Run from the repository root with the prepared station directory, grid directory, spatial indices, and output directory. The AIR model and training settings are in `configs/air.json`.

```bash
bash run.sh train ./data/stations ./data/grid ./data/geometry.npz ./runs/air
bash run.sh test  ./data/stations ./data/grid ./data/geometry.npz ./runs/air
```

Training saves the checkpoint with the lowest validation MAE to `runs/air/best.pt`. Evaluation loads this checkpoint and writes the test metrics to `runs/air/test_metrics.json`.
