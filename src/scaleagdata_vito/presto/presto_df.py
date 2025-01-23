import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def prep_dataframe(
    df: pd.DataFrame,
    filter_function: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    dekadal: bool = False,
):
    """Duplication from eval.py but otherwise we would need catboost during
    presto inference on OpenEO.
    """
    # SAR cannot equal 0.0 since we take the log of it
    cols = [
        f"SAR-{s}-ts{t}-20m" for s in ["VV", "VH"] for t in range(36 if dekadal else 12)
    ]

    df = df.drop_duplicates(subset=["sample_id", "lat", "lon", "start_date"])
    df = df[~pd.isna(df).any(axis=1)]
    df = df[~(df.loc[:, cols] == 0.0).any(axis=1)]
    df = df.set_index("sample_id")
    if filter_function is not None:
        df = filter_function(df)
    return df


def process_parquet(
    df: pd.DataFrame,
    use_valid_time: bool = False,
    num_timesteps: int = 36,
    min_edge_buffer: int = 2,
    no_data_value: int = 65535,
) -> pd.DataFrame:
    """
    This function takes in a DataFrame with S1, S2 and ERA5 observations and their respective dates
    in long format and returns it in wide format.

    Each row of the input DataFrame should represent a unique combination
    of sample_id and timestamp, where timestamp is the date of the observation.

    This function performs the following operations:
    - initializing the start_date and end_date as the first and last available observation;
    - computing relative position of the timestamp (timestamp_ind variable) in the timeseries;
    - checking for missing timesteps in the middle of the timeseries
      and filling them with NODATAVALUE
    - pivoting the DataFrame to wide format with columns for each feature column
      and timesteps as suffixes
    - assigning the correct suffixes to the band names
    - computing the number of available timesteps in the timeseries;
      it represents the absolute number of timesteps for which observations are
      available; it cannot be less than num_timesteps; if this is the case,
      sample is considered faulty and is removed from the dataset
    - post-processing with prep_dataframe function

    Args:
        df (pd.DataFrame): Input dataframe containing EO data and the following required attributes:
            ["sample_id", "timestamp"].
        use_valid_time (bool): If True, the function will use the valid_time column to check
            if valid_time lies within the range of available observations,
            with min_edge_buffer buffer.
            Samples where this is not the case are removed from the dataset.
            If False, the function will not use the valid_time column
            and will not perform this check.

    Returns
    -------
    pd.DataFrame
        pivoted DataFrame with columns for each band and timesteps as suffixes

    Raises
    ------
    AttributeError
        error is raised if DataFrame does not contain the required columns
    ValueError
        error is raised if pivot results in an empty DataFrame
    """
    df.rename(
        columns={
            "S1-SIGMA0-VV": "SAR-VV",
            "S1-SIGMA0-VH": "SAR-VH",
            "S2-L2A-B02": "OPTICAL-B02",
            "S2-L2A-B03": "OPTICAL-B03",
            "S2-L2A-B04": "OPTICAL-B04",
            "S2-L2A-B05": "OPTICAL-B05",
            "S2-L2A-B06": "OPTICAL-B06",
            "S2-L2A-B07": "OPTICAL-B07",
            "S2-L2A-B08": "OPTICAL-B08",
            "S2-L2A-B8A": "OPTICAL-B8A",
            "S2-L2A-B11": "OPTICAL-B11",
            "S2-L2A-B12": "OPTICAL-B12",
            "AGERA5-PRECIP": "METEO-precipitation_flux",
            "AGERA5-TMEAN": "METEO-temperature_mean",
            "slope": "DEM-slo-20m",
            "elevation": "DEM-alt-20m",
            # since the openEO output has the attribute "valid_time",
            # # we need the following line for compatibility with earlier datasets
            "valid_date": "valid_time",
            "date": "timestamp",
        },
        inplace=True,
    )

    static_features = ["DEM-alt-20m", "DEM-slo-20m", "lat", "lon"]
    required_columns = ["sample_id", "timestamp"] + static_features
    if not all([col in df.columns for col in required_columns]):
        missing_columns = [col for col in required_columns if col not in df.columns]
        raise AttributeError(
            f"DataFrame must contain the following columns: {missing_columns}"
        )

    bands10m = ["OPTICAL-B02", "OPTICAL-B03", "OPTICAL-B04", "OPTICAL-B08"]
    bands20m = [
        "SAR-VH",
        "SAR-VV",
        "OPTICAL-B05",
        "OPTICAL-B06",
        "OPTICAL-B07",
        "OPTICAL-B11",
        "OPTICAL-B12",
        "OPTICAL-B8A",
    ]
    bands100m = ["METEO-precipitation_flux", "METEO-temperature_mean"]

    feature_columns = bands10m + bands20m + bands100m
    # for index columns we need to include all columns that are not feature columns
    index_columns = [col for col in df.columns if col not in feature_columns]
    index_columns.remove("timestamp")

    # check that all feature columns are present in the DataFrame
    # or initialize them with NODATAVALUE
    for feature_col in feature_columns:
        if feature_col not in df.columns:
            df[feature_col] = no_data_value

    df["timestamp_ind"] = df.groupby("sample_id")["timestamp"].rank().astype(int) - 1

    # Assign start_date and end_date as the minimum and maximum available timestamp
    df["start_date"] = df["sample_id"].map(df.groupby(["sample_id"])["timestamp"].min())
    df["end_date"] = df["sample_id"].map(df.groupby(["sample_id"])["timestamp"].max())
    index_columns.extend(["start_date", "end_date"])

    if use_valid_time:
        df["valid_time_ts_diff_days"] = (
            df["valid_time"] - df["timestamp"]
        ).dt.days.abs()
        valid_position = (
            df.set_index("timestamp_ind")
            .groupby("sample_id")["valid_time_ts_diff_days"]
            .idxmin()
        )
        df["valid_position"] = df["sample_id"].map(valid_position)
        index_columns.append("valid_position")

        df["valid_position_diff"] = df["timestamp_ind"] - df["valid_position"]

        # define samples where valid_time is outside the range of the actual extractions
        # and remove them from the dataset
        latest_obs_position = df.groupby(["sample_id"])[
            ["valid_position", "timestamp_ind", "valid_position_diff"]
        ].max()
        df["is_last_available_ts"] = (
            df["sample_id"].map(latest_obs_position["timestamp_ind"])
            == df["timestamp_ind"]
        )
        samples_after_end_date = latest_obs_position[
            (
                latest_obs_position["valid_position"]
                > latest_obs_position["timestamp_ind"]
            )
        ].index
        samples_before_start_date = latest_obs_position[
            (latest_obs_position["valid_position"] < 0)
        ].index

        if len(samples_after_end_date) > 0 or len(samples_before_start_date) > 0:
            logger.warning(
                f"""\
    Removing {len(samples_after_end_date)} \
    samples with valid_time after the end_date \
    and {len(samples_before_start_date)} samples with valid_time before the start_date"""
            )
            df = df[~df["sample_id"].isin(samples_before_start_date)]
            df = df[~df["sample_id"].isin(samples_after_end_date)]

        # compute average distance between observations
        # and use it as an approximation for frequency
        obs_timestamps = pd.Series(df["timestamp"].unique()).sort_values()
        avg_distance = int(obs_timestamps.diff().abs().dt.days.mean())

        # add timesteps before the start_date where needed
        intermediate_dummy_df = pd.DataFrame()
        for n_ts_to_add in range(1, min_edge_buffer + 1):
            samples_to_add_ts_before_start = latest_obs_position[
                (min_edge_buffer - latest_obs_position["valid_position"])
                >= -n_ts_to_add
            ].index
            dummy_df = df[
                (df["sample_id"].isin(samples_to_add_ts_before_start))
                & (df["timestamp_ind"] == 0)
            ].copy()
            dummy_df["timestamp"] = dummy_df["timestamp"] - pd.DateOffset(
                days=(n_ts_to_add * avg_distance)
            )  # type: ignore
            dummy_df[feature_columns] = no_data_value
            intermediate_dummy_df = pd.concat([intermediate_dummy_df, dummy_df])
        df = pd.concat([df, intermediate_dummy_df])

        # add timesteps after the end_date where needed
        intermediate_dummy_df = pd.DataFrame()
        for n_ts_to_add in range(1, min_edge_buffer + 1):
            samples_to_add_ts_after_end = latest_obs_position[
                (min_edge_buffer - latest_obs_position["valid_position_diff"])
                >= n_ts_to_add
            ].index
            dummy_df = df[
                (df["sample_id"].isin(samples_to_add_ts_after_end))
                & (df["is_last_available_ts"])
            ].copy()
            dummy_df["timestamp"] = dummy_df["timestamp"] + pd.DateOffset(
                months=(n_ts_to_add * avg_distance)
            )  # type: ignore
            dummy_df[feature_columns] = no_data_value
            intermediate_dummy_df = pd.concat([intermediate_dummy_df, dummy_df])
        df = pd.concat([df, intermediate_dummy_df])

        # reinitialize timestep_ind
        df["timestamp_ind"] = (
            df.groupby("sample_id")["timestamp"].rank().astype(int) - 1
        )

    df["available_timesteps"] = df["sample_id"].map(
        df.groupby("sample_id")["timestamp"].nunique().astype(int)
    )
    index_columns.append("available_timesteps")

    # finally pivot the dataframe
    index_columns = list(set(index_columns))
    df_pivot = df.pivot(
        index=index_columns, columns="timestamp_ind", values=feature_columns
    )
    df_pivot = df_pivot.fillna(no_data_value)

    if df_pivot.empty:
        raise ValueError("Left with an empty DataFrame!")

    df_pivot.reset_index(inplace=True)
    df_pivot.columns = [
        f"{xx[0]}-ts{xx[1]}" if isinstance(xx[1], int) else xx[0]
        for xx in df_pivot.columns.to_flat_index()
    ]  # type: ignore
    df_pivot.columns = [
        f"{xx}-10m" if any(band in xx for band in bands10m) else xx
        for xx in df_pivot.columns
    ]  # type: ignore
    df_pivot.columns = [
        f"{xx}-20m" if any(band in xx for band in bands20m) else xx
        for xx in df_pivot.columns
    ]  # type: ignore
    df_pivot.columns = [
        f"{xx}-100m" if any(band in xx for band in bands100m) else xx
        for xx in df_pivot.columns
    ]  # type: ignore

    if use_valid_time:
        df_pivot["year"] = df_pivot["valid_time"].dt.year
        df_pivot["valid_time"] = df_pivot["valid_time"].dt.date.astype(str)

        min_center_point = np.maximum(
            num_timesteps // 2,
            df_pivot["valid_position"] + min_edge_buffer - num_timesteps // 2,
        )
        max_center_point = np.minimum(
            df_pivot["available_timesteps"] - num_timesteps // 2,
            df_pivot["valid_position"] - min_edge_buffer + num_timesteps // 2,
        )

        faulty_samples = min_center_point > max_center_point
        if faulty_samples.sum() > 0:
            logger.warning(f"Dropping {faulty_samples.sum()} faulty samples.")
        df_pivot = df_pivot[~faulty_samples]

    samples_with_too_few_ts = df_pivot["available_timesteps"] < num_timesteps
    if samples_with_too_few_ts.sum() > 0:
        logger.warning(
            f"Dropping {samples_with_too_few_ts.sum()} samples with \
number of available timesteps less than {num_timesteps}."
        )
        df_pivot = df_pivot[~samples_with_too_few_ts]

    df_pivot["start_date"] = df_pivot["start_date"].dt.date.astype(str)
    df_pivot["end_date"] = df_pivot["end_date"].dt.date.astype(str)

    df_pivot = prep_dataframe(df_pivot)

    return df_pivot


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
            for t in range(num_ts)
            if months[t]
            not in np.arange(window_of_interest[0] - 1, window_of_interest[1])
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
                for t in range(num_ts)
                if months[t]
                not in np.arange(window_of_interest[0] - 1, window_of_interest[1])
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


def load_dataset(files_root_dir, num_timesteps=36, no_data_value=65535):
    files = list(Path(files_root_dir).glob("*/*/*.geoparquet"))
    df_list = []
    corrupted = []
    for f in tqdm(files):
        _data = pd.read_parquet(f, engine="fastparquet")
        if not all(
            item in _data.columns for item in ["lat", "lon", "start_date", "end_date"]
        ):
            corrupted.append(f)
            continue
        _ref_id = str(f).split("/")[-2].split("=")[-1]
        _data["ref_id"] = _ref_id
        _data_pivot = process_parquet(_data, num_timesteps=num_timesteps)
        _data_pivot.reset_index(inplace=True)
        df_list.append(_data_pivot)
    df = pd.concat(df_list)
    df = df.fillna(no_data_value)
    del df_list
    return df
