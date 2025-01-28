import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Literal, Optional

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
    num_timesteps: int = 12,
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
            "original_date": "valid_time",
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
        df["valid_time"] = pd.to_datetime(df["valid_time"])
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


def _get_correct_date(
    dt_in: pd.Timestamp, compositing_window: Literal["dekad", "monthly"] = "dekad"
):
    if compositing_window == "dekad":
        if dt_in.day <= 10:
            correct_date = datetime(dt_in.year, dt_in.month, 1, 0, 0, 0)
        if dt_in.day >= 11 and dt_in.day <= 20:
            correct_date = datetime(dt_in.year, dt_in.month, 11, 0, 0, 0)
        if dt_in.day >= 21:
            correct_date = datetime(dt_in.year, dt_in.month, 21, 0, 0, 0)
    elif compositing_window == "monthly":
        correct_date = datetime(dt_in.year, dt_in.month, 1, 0, 0, 0)
    else:
        raise ValueError(f"Unknown compositing window: {compositing_window}")

    return correct_date


def get_buffered_window_of_interest(
    window_of_interest: List[str],
    buffer: int = 3,
    compositing_window: Literal["dekad", "monthly"] = "dekad",
):
    start_date = _get_correct_date(
        pd.to_datetime(window_of_interest[0]), compositing_window=compositing_window
    )
    end_date = _get_correct_date(
        pd.to_datetime(window_of_interest[1]), compositing_window=compositing_window
    )
    buffer_days = buffer * 10 if compositing_window == "dekad" else buffer * 30

    start_buffer_date = start_date - pd.Timedelta(days=buffer_days + 1)
    end_buffer_date = end_date + pd.Timedelta(days=buffer_days + 1)

    return [
        start_buffer_date.strftime("%Y-%m-%d"),
        end_buffer_date.strftime("%Y-%m-%d"),
    ]


def window_of_interest_from_valid_date(
    valid_dates: List[str],
    buffer: int = 3,
    compositing_window: Literal["dekad", "monthly"] = "dekad",
):
    start_dates, end_dates = [], []
    for date in valid_dates:
        curr_window_of_interest = [date, date]
        start_date, end_date = get_buffered_window_of_interest(
            curr_window_of_interest,
            buffer=buffer,
            compositing_window=compositing_window,
        )
        start_dates.append(start_date)
        end_dates.append(end_date)
    return [min(start_dates), max(end_dates)]


def out_window_to_nodata(
    df: pd.DataFrame, window_of_interest: List[str], no_data_value: int = 65535
):
    bands = [
        "S1-SIGMA0-VV",
        "S1-SIGMA0-VH",
        "S2-L2A-B02",
        "S2-L2A-B03",
        "S2-L2A-B04",
        "S2-L2A-B05",
        "S2-L2A-B06",
        "S2-L2A-B07",
        "S2-L2A-B08",
        "S2-L2A-B8A",
        "S2-L2A-B11",
        "S2-L2A-B12",
        "AGERA5-PRECIP",
        "AGERA5-TMEAN",
        "slope",
        "elevation",
    ]
    # check that bands are present in the dataframe
    existing_bands = [b for b in bands if b in df.columns]

    cutoff_start_date = pd.to_datetime(window_of_interest[0])
    cutoff_end_date = pd.to_datetime(window_of_interest[1])
    # Identify rows whose date is outside the window of interest
    df["timestamp_datetime"] = pd.to_datetime(df.timestamp.dt.strftime("%Y-%m-%d"))
    outside_range = ~df["timestamp_datetime"].between(
        cutoff_start_date, cutoff_end_date
    )
    # Assign the fixed value to the band columns for rows outside the range
    df.loc[outside_range, existing_bands] = no_data_value
    return df


def extract_window_of_interest(
    df: pd.DataFrame, window_of_interest: List[str], buffer: Optional[int] = None
):
    df_filtered = df.copy()
    df_filtered["timestamp_datetime"] = pd.to_datetime(
        df_filtered.timestamp.dt.strftime("%Y-%m-%d")
    )
    if buffer is not None:
        window_of_interest = get_buffered_window_of_interest(
            window_of_interest, buffer=buffer
        )
    cutoff_start_date = pd.to_datetime(window_of_interest[0])
    cutoff_end_date = pd.to_datetime(window_of_interest[1])
    df_filtered = df_filtered[
        (df_filtered.timestamp_datetime >= cutoff_start_date)
        & (df_filtered.timestamp_datetime <= cutoff_end_date)
    ]

    # replace start and end date with the cutoff dates and convert timestamp to string
    df_filtered.drop(columns=["timestamp_datetime"], inplace=True)
    df_filtered["start_date"] = window_of_interest[0]
    df_filtered["end_date"] = window_of_interest[1]

    return df_filtered


def load_dataset(
    files_root_dir: str,
    window_of_interest: List[str] = [""],
    use_valid_time: bool = False,
    num_ts: int = 36,
    mask_out_of_window: bool = False,
    buffer_window: int = 0,
    no_data_value: int = 65535,
    composite_window: Literal["dekad", "monthly"] = "dekad",
):

    files = list(Path(files_root_dir).glob("*/*/*.geoparquet"))
    df_list = []

    for f in tqdm(files):
        _data = pd.read_parquet(f, engine="fastparquet")
        _ref_id = str(f).split("/")[-2].split("=")[-1]
        _data["ref_id"] = _ref_id
        # if no window of interest is provided, use the original date column to create a window of interest
        if (
            (window_of_interest == [""])
            and ("original_date" in _data.columns)
            and use_valid_time
        ):
            window_of_interest = window_of_interest_from_valid_date(
                _data["original_date"].unique().tolist(),
                buffer=num_ts // 2,
                compositing_window=composite_window,
            )
        if window_of_interest != [""]:
            # whether to mask out of window data with no_data_value
            if mask_out_of_window:
                _data = out_window_to_nodata(
                    _data, window_of_interest, no_data_value=no_data_value
                )
            # extend the window of interest by buffering it with a a number of buffer dates before and after indicated by buffer_window
            if buffer_window > 0:
                window_of_interest_buffered = get_buffered_window_of_interest(
                    window_of_interest,
                    buffer=buffer_window,
                    compositing_window=composite_window,
                )
                # filter data to only include data within the window of interest
                _data = extract_window_of_interest(_data, window_of_interest_buffered)
            else:
                _data = extract_window_of_interest(_data, window_of_interest)
        _data_pivot = process_parquet(_data)
        _data_pivot.reset_index(inplace=True)
        df_list.append(_data_pivot)
    df = pd.concat(df_list)
    df = df.fillna(no_data_value)
    del df_list
    return df
