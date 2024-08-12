from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import tqdm
import xarray as xr
from shapely.geometry import Point


def to_float32(df):
    return df.astype({c: np.float32 for c in df.columns if df[c].dtype == np.float64})


def rename_cols(df, i):
    BAND_MAPPING = {
        "B02": f"OPTICAL-B02-ts{i}-10m",
        "B03": f"OPTICAL-B03-ts{i}-10m",
        "B04": f"OPTICAL-B04-ts{i}-10m",
        "B05": f"OPTICAL-B05-ts{i}-20m",
        "B06": f"OPTICAL-B06-ts{i}-20m",
        "B07": f"OPTICAL-B07-ts{i}-20m",
        "B08": f"OPTICAL-B08-ts{i}-10m",
        "B8A": f"OPTICAL-B8A-ts{i}-20m",
        "B11": f"OPTICAL-B11-ts{i}-20m",
        "B12": f"OPTICAL-B12-ts{i}-20m",
        "VH": f"SAR-VH-ts{i}-20m",
        "VV": f"SAR-VV-ts{i}-20m",
        "precipitation-flux": f"METEO-precipitation_flux-ts{i}-100m",
        "temperature-mean": f"METEO-temperature_mean-ts{i}-100m",
    }
    return df.rename(columns=BAND_MAPPING)


def xr_to_df(netcdf_file):
    """
    Convert the NetCDF dataset to a pandas DataFrame.

    Parameters:
    netcdf_file (str): The path to the NetCDF file extracted from OpenEO.

    Returns:
    pandas.DataFrame: The data stored in netcdf arranged in a DataFrame.
    """

    netcdf = xr.load_dataset(netcdf_file)
    df = pd.DataFrame()
    tps = list(netcdf["t"])
    for i in range(len(tps)):
        tp = (
            netcdf.sel(t=tps[i])
            .to_dataframe()
            .drop(columns=["DEM", "feature_names", "t", "lat", "lon"])
        )
        tp = rename_cols(tp, i)
        df = pd.concat([df, tp], axis=1)

    # add static columns
    df["start_date"] = np.datetime_as_string(tps[0].data, unit="D")
    df["end_date"] = np.datetime_as_string(tps[-1].data, unit="D")
    df["lat"] = netcdf["lat"].data
    df["lon"] = netcdf["lon"].data
    df["DEM-alt-20m"] = netcdf.sel(t=tps[0])["DEM"].data
    df["DEM-slo-20m"] = 0
    return df


def add_labels(df, labels_file):
    """
    Adds labels to a DataFrame based on a GeoDataFrame of labels.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to add labels to.
    - gdf_labels (geopandas.GeoDataFrame): The GeoDataFrame containing the labels. this dataframe
    was used to collect datapoints from OpenEO so it contains the geometries which will be intersected with the
    df to add the labels.

    Returns:
    - pandas.DataFrame: The DataFrame with labels added.

    """
    label_gdf = gpd.read_file(labels_file)
    df["geometry"] = [Point(r.lon, r.lat) for r in df.itertuples()]
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    df_labeled = gpd.sjoin(gdf, label_gdf, how="left", op="within")
    df_labeled = df_labeled.drop(columns=["geometry", "index_right"])
    df_labeled = df_labeled.drop_duplicates()
    df_labeled = df_labeled.replace(np.nan, 65535)
    return to_float32(df_labeled)


def filter_ts(df_to_filter, window_of_interest, no_data_value=65535, num_ts=36):
    """
    Filters the time series data in the given DataFrame based on the specified window of interest.

    Args:
        df_to_filter (pandas.DataFrame): The DataFrame containing the time series data to be filtered.
        window_of_interest (tuple): A tuple representing the window of interest, where the first element is the start month and the second element is the end month.
        no_data_value (int, optional): The value to replace the filtered time series data with. Defaults to 65535.
        num_ts (int, optional): The number of time series in the DataFrame. Defaults to 36.

    Returns:
        pandas.DataFrame: The filtered DataFrame with the specified time series data replaced by the no_data_value.
    """
    if len(df_to_filter["start_date"].unique()) == len(
        df_to_filter["end_date"].unique()
    ):
        ref_row = df_to_filter.iloc[0]
        months = get_month_array(num_ts, ref_row)
        ts_to_filter = [
            t
            for t in range(12)
            if t not in np.arange(window_of_interest[0] - 1, window_of_interest[1])
        ]
        ts_cols = [
            ts for ts in df_to_filter.columns for t in ts_to_filter if f"-ts{t}-" in ts
        ]
        df_to_filter.loc[:, ts_cols] = np.float32(no_data_value)
    else:
        for i, row in tqdm.tqdm(df_to_filter.iterrows()):
            row = df_to_filter.iloc[i]
            months = get_month_array(num_ts, row)
            ts_to_filter = [
                t
                for t in range(12)
                if t not in np.arange(window_of_interest[0] - 1, window_of_interest[1])
            ]
            ts_cols = [
                ts
                for ts in df_to_filter.columns
                for t in ts_to_filter
                if f"-ts{t}-" in ts
            ]
            df_to_filter.loc[i, ts_cols] = np.float32(no_data_value)
    return df_to_filter


def get_month_array(num_timesteps: int, row: pd.Series) -> np.ndarray:
    start_date, end_date = datetime.strptime(
        row.start_date, "%Y-%m-%d"
    ), datetime.strptime(row.end_date, "%Y-%m-%d")

    # Calculate the step size for 10-day intervals and create a list of dates
    step = int((end_date - start_date).days / (num_timesteps - 1))
    date_vector = [start_date + timedelta(days=i * step) for i in range(num_timesteps)]

    # Ensure last date is not beyond the end date
    if date_vector[-1] > end_date:
        date_vector[-1] = end_date

    return np.array([d.month - 1 for d in date_vector])
