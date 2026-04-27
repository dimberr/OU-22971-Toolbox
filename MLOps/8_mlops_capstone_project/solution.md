setup conda environment (linux/macos) in MLOps/8_mlops_capstone_project dir:
1. make sure you have conda/miniconda installed
2. conda env create -f environment.yml
3. conda activate 22971-mlflow


This notebook assumes the existence of the files:
- `TLC_data/green_tripdata_2020-01.parquet`
- `TLC_data/green_tripdata_2020-04.parquet`
- `TLC_data/green_tripdata_2020-08.parquet`
- `TLC_data/taxi_zone_lookup.csv`

Download them here:

https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page


Optional, for map overlay:
- `TLC_data/NYC_Taxi_Zones.geojson`

Download `NYC_Taxi_Zones_YYYYMMDD.geojson` here:

https://data.cityofnewyork.us/Transportation/NYC-Taxi-Zones/8meu-9t5y/about_data

and rename to `NYC_Taxi_Zones.geojson`.


start in separate shell mlflow server for experiments tracking:
mlflow server --workers 1 --port 5000 --backend-store-uri sqlite:///mlflow_tracking/mlflow.db --default-artifact-root mlflow_tracking/mlruns