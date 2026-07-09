import logging
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("__main__")

STATIC_FEATURES = ["DEM-alt-20m", "DEM-slo-20m", "lat", "lon"]
REQUIRED_COLUMNS = ["sample_id", "timestamp"] + STATIC_FEATURES

BAND_MAPPINGS = {
    "10m": ["OPTICAL-B02", "OPTICAL-B03", "OPTICAL-B04", "OPTICAL-B08"],
    "20m": [
        "SAR-VH",
        "SAR-VV",
        "OPTICAL-B05",
        "OPTICAL-B06",
        "OPTICAL-B07",
        "OPTICAL-B11",
        "OPTICAL-B12",
        "OPTICAL-B8A",
    ],
    "100m": ["METEO-precipitation_flux", "METEO-temperature_mean"],
}

FEATURE_COLUMNS = BAND_MAPPINGS["10m"] + BAND_MAPPINGS["20m"] + BAND_MAPPINGS["100m"]

COLUMN_RENAMES = {
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
}

EXPECTED_DISTANCES = {"month": 31, "dekad": 10}
NODATAVALUE = 65535


class DataFrameValidator:
    @staticmethod
    def validate_required_columns(df_long: pd.DataFrame) -> None:
        missing_columns = [
            col for col in REQUIRED_COLUMNS if col not in df_long.columns
        ]
        if missing_columns:
            raise AttributeError(
                f"DataFrame must contain the following columns: {missing_columns}"
            )

    @staticmethod
    def validate_timestamps(df_long: pd.DataFrame, freq: str = "month") -> None:
        if freq == "month":
            if not df_long["timestamp"].dt.is_month_start.all():
                bad_dates = df_long[~df_long["timestamp"].dt.is_month_start][
                    "timestamp"
                ].unique()
                raise ValueError(
                    f"All monthly timestamps must be at month start. Found: {bad_dates}"
                )
        elif freq == "dekad":
            if not df_long["timestamp"].dt.day.isin([1, 11, 21]).all():
                raise ValueError(
                    "All dekad timestamps must be at the 1st, 11th, or 21st of the month"
                )
        else:
            raise NotImplementedError(f"Frequency {freq} not supported")

    @staticmethod
    def check_faulty_samples(
        df_wide: pd.DataFrame, min_edge_buffer: int
    ) -> pd.DataFrame:
        min_center_point = np.maximum(
            df_wide["available_timesteps"] // 2,
            df_wide["valid_position"]
            + min_edge_buffer
            - df_wide["available_timesteps"] // 2,
        )
        max_center_point = np.minimum(
            df_wide["available_timesteps"] - df_wide["available_timesteps"] // 2,
            df_wide["valid_position"]
            - min_edge_buffer
            + df_wide["available_timesteps"] // 2,
        )

        faulty_samples = min_center_point > max_center_point
        if faulty_samples.sum() > 0:
            logger.warning(f"Dropping {faulty_samples.sum()} faulty sample(s).")

        return df_wide[~faulty_samples]

    @staticmethod
    def check_min_timesteps(
        df_wide: pd.DataFrame, required_min_timesteps: int
    ) -> pd.DataFrame:
        samples_with_too_few_ts = (
            df_wide["available_timesteps"] < required_min_timesteps // 2
        )
        if samples_with_too_few_ts.sum() > 0:
            logger.warning(
                f"Dropping {samples_with_too_few_ts.sum()} sample(s) with \
number of available timesteps less than {required_min_timesteps}."
            )
        df_wide = df_wide[~samples_with_too_few_ts]
        if len(df_wide) == 0:
            raise ValueError(
                f"Left with an empty DataFrame! \
All samples have fewer timesteps than required ({required_min_timesteps})."
            )
        else:
            return df_wide

    @staticmethod
    def check_median_distance(df_long: pd.DataFrame, freq: str) -> pd.DataFrame:
        # compute median distance between observations
        # and use it as an approximation for frequency

        # Pre-calculate daily differences without repeated .unique()
        ts_subset_df = (
            df_long[["sample_id", "timestamp"]]
            .drop_duplicates()
            .sort_values(by=["sample_id", "timestamp"])
        )
        ts_subset_df["timestamp_diff_days"] = (
            ts_subset_df.groupby("sample_id")["timestamp"].diff().dt.days.abs()
        )
        median_distance = (
            ts_subset_df.groupby("sample_id")["timestamp_diff_days"]
            .median()
            .fillna(0)
            .astype(int)
        )

        if freq == "month":
            samples_with_mismatching_distance = median_distance[
                median_distance != EXPECTED_DISTANCES["month"]
            ]
        elif freq == "dekad":
            samples_with_mismatching_distance = median_distance[
                median_distance != EXPECTED_DISTANCES["dekad"]
            ]
        else:
            raise NotImplementedError(f"Frequency {freq} not supported")

        if len(samples_with_mismatching_distance) > 0:
            logger.warning(
                f"Found {len(samples_with_mismatching_distance)} samples with median distance \
between observations not corresponding to {freq}. \
Removing them from the dataset."
            )
            df_long = df_long[
                ~df_long["sample_id"].isin(samples_with_mismatching_distance.index)
            ]
            if len(df_long) == 0:
                raise ValueError(
                    f"Left with an empty DataFrame! All samples have median distance between \
observations not corresponding to {freq}."
                )
        else:
            logger.info(
                f"Expected observations frequency: {freq}; \
Median observed distance between observations: {median_distance.unique()} days"
            )

        return df_long


class TimeSeriesProcessor:

    @staticmethod
    def calculate_valid_position(df_long: pd.DataFrame) -> pd.DataFrame:
        df_long["valid_time_ts_diff_days"] = (
            df_long["valid_time"] - df_long["timestamp"]
        ).dt.days.abs()
        valid_position = (
            df_long.set_index("timestamp_ind")
            .groupby("sample_id")["valid_time_ts_diff_days"]
            .idxmin()
        )
        df_long["valid_position"] = df_long["sample_id"].map(valid_position)
        return df_long

    @staticmethod
    def get_expected_dates(start_date, end_date, freq):
        start_date = _get_correct_date(start_date, compositing_window=freq)
        end_date = _get_correct_date(end_date, compositing_window=freq)
        if freq == "dekad":
            date_array = _get_dekadal_dates(start_date, end_date)
        elif freq == "month":
            date_array = _get_monthly_dates(start_date, end_date)
        else:
            raise NotImplementedError(f"Frequency {freq} not supported")
        return pd.DatetimeIndex(date_array)

    @staticmethod
    def fill_missing_dates(
        df_long: pd.DataFrame, freq: str, index_columns: List[str]
    ) -> pd.DataFrame:

        def fill_sample(sample_df):
            expected_dates = TimeSeriesProcessor.get_expected_dates(
                sample_df["start_date"].iloc[0], sample_df["end_date"].iloc[0], freq
            )
            missing_dates = expected_dates.difference(sample_df["timestamp"])
            if not missing_dates.empty:
                static_cols = sample_df.iloc[0][index_columns].to_dict()
                for date in missing_dates:
                    new_row = {**static_cols, "timestamp": date}
                    for col in FEATURE_COLUMNS:
                        new_row[col] = NODATAVALUE
                    sample_df.loc[-1] = new_row
                    sample_df.reset_index(drop=True, inplace=True)
            return sample_df

        unique_date_pairs = df_long[["start_date", "end_date"]].drop_duplicates()
        unique_date_pairs["expected_n_observations"] = unique_date_pairs.apply(
            lambda xx: len(
                TimeSeriesProcessor.get_expected_dates(
                    xx["start_date"], xx["end_date"], freq
                )
            ),
            axis=1,
        )
        unique_date_pairs.set_index(["start_date", "end_date"], inplace=True)

        expected_observations_s = df_long.groupby(
            ["sample_id", "start_date", "end_date"]
        )[["sample_id", "start_date", "end_date", "timestamp"]].apply(
            lambda xx: xx["timestamp"].nunique()
        )
        expected_observations_s.name = "actual_n_observations"
        expected_observations_df = expected_observations_s.reset_index()
        expected_observations_df.set_index(["start_date", "end_date"], inplace=True)

        expected_observations_df["expected_n_observations"] = (
            expected_observations_df.index.map(
                unique_date_pairs["expected_n_observations"]
            )
        )
        expected_observations_df.reset_index(drop=False, inplace=True)

        samples_to_fill = expected_observations_df[
            expected_observations_df["actual_n_observations"]
            != expected_observations_df["expected_n_observations"]
        ]["sample_id"].unique()

        if samples_to_fill.size == 0:
            logger.info("All samples have the expected number of observations.")
            return df_long
        else:
            logger.warning(
                f"{len(samples_to_fill)} samples have missing observations. \
Filling them with NODATAVALUE."
            )
            df_subset = df_long[df_long["sample_id"].isin(samples_to_fill)]
            df_long = df_long[~df_long["sample_id"].isin(samples_to_fill)]
            df_subset = (
                df_subset.groupby("sample_id")[
                    [*index_columns, *FEATURE_COLUMNS, "timestamp"]
                ]
                .apply(fill_sample)
                .reset_index(drop=True)
            )
            return pd.concat([df_long, df_subset], ignore_index=True)

    @staticmethod
    def check_vt_closeness(
        df_long: pd.DataFrame, min_edge_buffer: int, freq: str
    ) -> pd.DataFrame:
        """
        Check valid_time closeness to the edges of the time series.
        Essential for downstream processing at the Dataset level.
        Samples that fail this check will be removed.

        Parameters
        ----------
        df_long : pd.DataFrame
            Long-format DataFrame containing time series data with columns including 'sample_id',
            'timestamp', 'valid_position', 'timestamp_ind', and feature columns.
        min_edge_buffer : int
            Minimum number of timestamps required as buffer before the first valid observation
            and after the last valid observation.
            Must be consistent with what is used at the Dataset level.
        freq : str
            Frequency of time series data, either 'month' or 'dekad' (10-day period).

        Returns
        -------
        pd.DataFrame
            DataFrame with only those samples that satisfy the minimum edge buffer
            requirement.
        """

        if df_long.empty:
            return df_long

        # Summarise per sample to evaluate distance from the valid time to window edges.
        summary = (
            df_long.groupby("sample_id")
            .agg(
                start_date=("start_date", "first"),
                end_date=("end_date", "first"),
                valid_time=("valid_time", "first"),
                first_timestamp_ind=("timestamp_ind", "min"),
                last_timestamp_ind=("timestamp_ind", "max"),
                valid_position=("valid_position", "first"),
            )
            .copy()
        )

        summary["distance_to_start"] = (
            summary["valid_position"] - summary["first_timestamp_ind"]
        )
        summary["distance_to_end"] = (
            summary["last_timestamp_ind"] - summary["valid_position"]
        )

        faulty_end = summary[summary["distance_to_end"] < min_edge_buffer]

        # best effort to identify the dataset being processed, purely for logging
        ref_id = "_".join(df_long["sample_id"].iloc[0].split("_")[:-1])
        if not faulty_end.empty:
            logger.warning(
                f"{ref_id}: Dropping {len(faulty_end)} samples with valid_time too close to the end of the time series. \n"
                f"Reason: Minimum edge buffer of {min_edge_buffer} not satisfied. Samples with the following date ranges are affected:\n"
                f"{faulty_end[['start_date', 'valid_time', 'end_date']].drop_duplicates().to_string(index=False)}"
            )

        remaining_summary = summary.drop(index=faulty_end.index, errors="ignore")
        faulty_start = remaining_summary[
            remaining_summary["distance_to_start"] < min_edge_buffer
        ]
        if not faulty_start.empty:
            logger.warning(
                f"{ref_id}: Dropping {len(faulty_start)} samples with valid_time too close to the start of the time series. \n"
                f"Reason: Minimum edge buffer of {min_edge_buffer} not satisfied. Samples with the following date ranges are affected:\n"
                f"{faulty_start[['start_date', 'valid_time', 'end_date']].drop_duplicates().to_string(index=False)}"
            )

        to_drop = faulty_end.index.union(faulty_start.index)
        if to_drop.empty:
            logger.info(
                f"{ref_id}: All samples' valid_time satisfy the min_edge_buffer requirement."
            )
            return df_long

        return df_long[~df_long["sample_id"].isin(to_drop)]


class ColumnProcessor:
    @staticmethod
    def rename_columns(df_long: pd.DataFrame) -> pd.DataFrame:
        return df_long.rename(columns=COLUMN_RENAMES)

    @staticmethod
    def add_band_suffix(df_wide: pd.DataFrame) -> pd.DataFrame:
        df_wide.columns = [
            f"{xx[0]}-ts{xx[1]}" if isinstance(xx[1], int) else xx[0]
            for xx in df_wide.columns.to_flat_index()
        ]  # type: ignore
        df_wide.columns = [
            f"{xx}-10m" if any(band in xx for band in BAND_MAPPINGS["10m"]) else xx
            for xx in df_wide.columns
        ]  # type: ignore
        df_wide.columns = [
            f"{xx}-20m" if any(band in xx for band in BAND_MAPPINGS["20m"]) else xx
            for xx in df_wide.columns
        ]  # type: ignore
        df_wide.columns = [
            (f"{xx}-100m" if any(band in xx for band in BAND_MAPPINGS["100m"]) else xx)
            for xx in df_wide.columns
        ]  # type: ignore
        return df_wide

    @staticmethod
    def construct_index(df_long: pd.DataFrame) -> List[str]:
        # for index columns we need to include all columns that are not feature columns
        index_columns = [col for col in df_long.columns if col not in FEATURE_COLUMNS]
        index_columns.remove("timestamp")
        return index_columns

    @staticmethod
    def check_feature_columns(df_long: pd.DataFrame) -> pd.DataFrame:
        # check that all feature columns are present in the DataFrame
        # or initialize them with NODATAVALUE
        missing_features = [
            col for col in FEATURE_COLUMNS if col not in df_long.columns
        ]
        if len(missing_features) > 0:
            df_long[missing_features] = NODATAVALUE
            logger.warning(
                f"The following features are missing and are filled \
with NODATAVALUE: {missing_features}"
            )
        return df_long

    @staticmethod
    def check_sar_columns(df_long: pd.DataFrame) -> pd.DataFrame:
        # SAR cannot equal 0.0 since we take the log of it
        # TO DO: need to check the behavior of presto itself in this case
        sar_cols = ["SAR-VV", "SAR-VH"]
        faulty_sar_observations = (df_long[sar_cols] == 0.0).sum().sum()
        if faulty_sar_observations > 0:
            affected_samples = df_long[(df_long[sar_cols] == 0.0).any(axis=1)][
                "sample_id"
            ].nunique()
            logger.warning(
                f"Found {faulty_sar_observations} SAR observation(s) \
equal to 0 across {affected_samples} sample(s). \
Replacing them with NODATAVALUE."
            )
            df_long[sar_cols] = df_long[sar_cols].replace(0.0, NODATAVALUE)

        return df_long


def process_parquet(
    df: pd.DataFrame,
    freq: Literal["month", "dekad"] = "month",
    use_valid_time: bool = False,
    # required_min_timesteps: Optional[int] = None,
    min_edge_buffer: int = 2,
    return_after_fill: bool = False,
    required_min_timesteps: Optional[int] = None,
) -> pd.DataFrame:

    if df.empty:
        raise ValueError("Input DataFrame is empty!")

    # Determine required minimum timesteps based on frequency
    if required_min_timesteps is None:
        if freq == "dekad":
            required_min_timesteps = 36
        if freq == "month":
            required_min_timesteps = 12

    # `feature_index` is an openEO spefic column we should remove to avoid
    # it being treated as unique values which is not true after merging
    # multiple parquet files
    if "feature_index" in df.columns:
        df = df.drop("feature_index", axis=1)

    # Process columns
    df = (
        df.pipe(ColumnProcessor.rename_columns)
        .pipe(ColumnProcessor.check_feature_columns)
        .pipe(ColumnProcessor.check_sar_columns)
    )

    # Validate input
    validator = DataFrameValidator()
    validator.validate_required_columns(df)
    validator.validate_timestamps(df, freq)
    df = validator.check_median_distance(df, freq)

    index_columns = ColumnProcessor.construct_index(df)

    # Assign start_date and end_date as the minimum and maximum available timestamp
    df["start_date"] = df["sample_id"].map(df.groupby(["sample_id"])["timestamp"].min())
    df["end_date"] = df["sample_id"].map(df.groupby(["sample_id"])["timestamp"].max())
    index_columns.extend(["start_date", "end_date"])
    index_columns = list(set(index_columns))

    # Perform check on number of unique values for each index column
    nsamples = df["sample_id"].nunique()
    to_drop = []
    for col in index_columns:
        if df[col].nunique() > nsamples:
            # best effort to identify the dataset being processed, purely for logging
            ref_id = "_".join(df["sample_id"].iloc[0].split("_")[:-1])
            df = df.drop(col, axis=1)
            to_drop.append(col)
            logger.warning(
                f"{ref_id}: Column {col} has more unique values than samples. This may cause issues, column has been dropped!"
            )
    index_columns = [col for col in index_columns if col not in to_drop]

    # Process time series
    processor = TimeSeriesProcessor()
    df = processor.fill_missing_dates(df, freq, index_columns)
    if return_after_fill:
        return df

    # Initialize timestep_ind
    df["timestamp_ind"] = df.groupby("sample_id")["timestamp"].rank().astype(int) - 1

    if use_valid_time:
        df = processor.calculate_valid_position(df)
        index_columns.append("valid_position")
        df["valid_position_diff"] = df["timestamp_ind"] - df["valid_position"]
        df = processor.check_vt_closeness(df, min_edge_buffer, freq)

    df["available_timesteps"] = df["sample_id"].map(
        df.groupby("sample_id")["timestamp"].nunique().astype(int)
    )
    index_columns.append("available_timesteps")
    index_columns = list(set(index_columns))

    # Transform to wide format
    df_pivot = df.pivot(
        index=index_columns,
        columns="timestamp_ind",
        values=FEATURE_COLUMNS,
    )
    df_pivot = df_pivot.fillna(NODATAVALUE)
    if df_pivot.empty:
        raise ValueError("Left with an empty DataFrame!")

    df_pivot.reset_index(inplace=True)
    df_pivot = ColumnProcessor.add_band_suffix(df_pivot)

    if use_valid_time:
        df_pivot["valid_time"] = df_pivot["valid_time"].astype("datetime64[ns]")
        df_pivot["year"] = df_pivot["valid_time"].dt.year
        df_pivot["valid_time"] = df_pivot["valid_time"].dt.date.astype(str)
        df_pivot = validator.check_faulty_samples(df_pivot, min_edge_buffer)

    if required_min_timesteps:
        df_pivot = validator.check_min_timesteps(df_pivot, required_min_timesteps)

    df_pivot["start_date"] = df_pivot["start_date"].dt.date.astype(str)
    df_pivot["end_date"] = df_pivot["end_date"].dt.date.astype(str)
    df_pivot = df_pivot.set_index("sample_id")

    return df_pivot


def _get_dekadal_dates(start_date: str, end_date: str):

    start_date_ = np.datetime64(start_date, "D")
    end_date_ = np.datetime64(end_date, "D")

    # Extract year, month, and day
    year = start_date_.astype("object").year
    month = start_date_.astype("object").month
    day = start_date_.astype("object").day

    date_vector = [start_date_]
    while date_vector[-1] != end_date_:
        if day < 21:
            day += 10
            date_vector.append(np.datetime64(f"{year}-{month:02d}-{day}"))
        else:
            month = month + 1 if month < 12 else 1
            year = year + 1 if month == 1 else year
            day = 1
            date_vector.append(np.datetime64(f"{year}-{month:02d}-01"))
    return date_vector


def _get_monthly_dates(start_date: str, end_date: str):
    # truncate to month precision
    start_month = np.datetime64(start_date, "M")
    end_month = np.datetime64(end_date, "M")  # Truncate to month start
    date_vector = np.arange(start_month, end_month + 1, dtype="datetime64[M]").astype(
        "datetime64[D]"
    )
    return date_vector


def _get_correct_date(
    dt_in: str, compositing_window: Literal["dekad", "month"] = "dekad"
) -> np.datetime64:
    """
    Determine the correct date based on the input date and compositing window.
    """

    # Extract year, month, and day
    year = np.datetime64(dt_in, "D").astype("object").year
    month = np.datetime64(dt_in, "D").astype("object").month
    day = np.datetime64(dt_in, "D").astype("object").day

    if compositing_window == "dekad":
        if day <= 10:
            correct_date = np.datetime64(f"{year}-{month:02d}-01")
        elif 11 <= day <= 20:
            correct_date = np.datetime64(f"{year}-{month:02d}-11")
        else:
            correct_date = np.datetime64(f"{year}-{month:02d}-21")
    elif compositing_window == "month":
        correct_date = np.datetime64(f"{year}-{month:02d}-01")
    else:
        raise ValueError(f"Unknown compositing window: {compositing_window}")

    return correct_date


def get_buffered_window_of_interest(
    window_of_interest: List[str],
    buffer: int = 3,
    compositing_window: Literal["dekad", "month"] = "dekad",
) -> List[str]:
    start_date = _get_correct_date(
        window_of_interest[0], compositing_window=compositing_window
    )
    end_date = _get_correct_date(
        window_of_interest[1], compositing_window=compositing_window
    )
    buffer_days = buffer * 10 if compositing_window == "dekad" else buffer * 30

    start_buffer_date = start_date - np.timedelta64(buffer_days + 1, "D")
    end_buffer_date = end_date + np.timedelta64(buffer_days + 1, "D")

    return [
        np.datetime_as_string(start_buffer_date, unit="D"),
        np.datetime_as_string(end_buffer_date, unit="D"),
    ]


def window_of_interest_from_valid_date(
    valid_dates: List[str],
    buffer: int,
    compositing_window: Literal["dekad", "month"] = "dekad",
) -> List[str]:
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
    df: pd.DataFrame,
    expected_dates: pd.DatetimeIndex,
    window_of_interest: Optional[List[str]],
    no_data_value: int = 65535,
):
    # bands = [
    #     "S1-SIGMA0-VV",
    #     "S1-SIGMA0-VH",
    #     "S2-L2A-B02",
    #     "S2-L2A-B03",
    #     "S2-L2A-B04",
    #     "S2-L2A-B05",
    #     "S2-L2A-B06",
    #     "S2-L2A-B07",
    #     "S2-L2A-B08",
    #     "S2-L2A-B8A",
    #     "S2-L2A-B11",
    #     "S2-L2A-B12",
    #     "AGERA5-PRECIP",
    #     "AGERA5-TMEAN",
    #     "slope",
    #     "elevation",
    # ]
    bands = [
        "METEO-temperature_mean",
        "METEO-precipitation_flux",
        "OPTICAL-B02",
        "OPTICAL-B03",
        "OPTICAL-B04",
        "OPTICAL-B05",
        "OPTICAL-B06",
        "OPTICAL-B07",
        "OPTICAL-B08",
        "OPTICAL-B8A",
        "OPTICAL-B11",
        "OPTICAL-B12",
        "SAR-VV",
        "SAR-VH",
        "DEM-slo-20m",
        "DEM-alt-20m",
    ]
    # check that bands are present in the dataframe
    # existing_bands = [b for b in bands if b in df.columns]

    # Convert timestamps and window_of_interest to np.datetime64
    if window_of_interest is not None:
        cutoff_start_date = np.datetime64(window_of_interest[0], "D")
        cutoff_end_date = np.datetime64(window_of_interest[1], "D")

        # # Identify rows outside the window of interest
        # outside_range = (df["timestamp"] < cutoff_start_date) | (
        #     df["timestamp"] > cutoff_end_date
        # )
        outside_range_indices = [
            i
            for i, ts in enumerate(expected_dates)
            if (ts < cutoff_start_date) or (ts > cutoff_end_date)
        ]
        # outside_range_indices = expected_dates[
        #     (expected_dates < cutoff_start_date) | (expected_dates > cutoff_end_date)
        # ]

        time_steps_to_mask = [
            f"{b}-ts{ts}" for ts in outside_range_indices for b in bands
        ]
        # Assign no_data_value to the relevant bands for rows outside the range
        df.loc[:, time_steps_to_mask] = no_data_value

        return df
    else:
        raise ValueError(
            "window_of_interest must be provided for out_window_to_nodata."
        )


def extract_window_of_interest(
    df: pd.DataFrame,
    window_of_interest: List[str],
    buffer: Optional[int] = None,
):

    if buffer is not None:
        window_of_interest = get_buffered_window_of_interest(
            window_of_interest, buffer=buffer
        )
    else:
        # Convert timestamps and window_of_interest to np.datetime64
        cutoff_start_date = np.datetime64(window_of_interest[0], "D")
        cutoff_end_date = np.datetime64(window_of_interest[1], "D")

        # Filter rows within the window of interest
        df_filtered = df[
            (df["timestamp"] >= cutoff_start_date)
            & (df["timestamp"] <= cutoff_end_date)
        ]

    # Assign start and end dates
    df_filtered["start_date"] = window_of_interest[0]
    df_filtered["end_date"] = window_of_interest[1]
    return df_filtered


def process_data_with_window(
    _data: pd.DataFrame,
    window_of_interest: Optional[List[str]],
    required_min_timesteps: int,
    use_valid_time: bool,
    buffer_window: int,
    composite_window: Literal["dekad", "month"],
) -> pd.DataFrame:
    if (
        (window_of_interest is None)
        and ("original_date" in _data.columns)
        and use_valid_time
    ):
        assert (
            required_min_timesteps > 0
        ), "required_min_timesteps must be > 0 when window_of_interest is not provided"
        window_of_interest = window_of_interest_from_valid_date(
            _data["original_date"].unique().tolist(),
            buffer=required_min_timesteps // 2,
            compositing_window=composite_window,
        )
    if window_of_interest is not None:
        if buffer_window > 0:
            window_of_interest = get_buffered_window_of_interest(
                window_of_interest,
                buffer=buffer_window,
                compositing_window=composite_window,
            )
        _data = extract_window_of_interest(_data, window_of_interest)
    _data_pivot = process_parquet(
        _data, freq=composite_window, use_valid_time=use_valid_time, required_min_timesteps=required_min_timesteps
    )
    _data_pivot.reset_index(inplace=True)
    return _data_pivot


def load_dataset(
    files_root_dir: str,
    window_of_interest: Optional[List[str]] = None,
    use_valid_time: bool = False,
    required_min_timesteps: int = 36,
    buffer_window: int = 0,
    no_data_value: int = 65535,
    composite_window: Literal["dekad", "month"] = "dekad",
    out_of_window_to_nodata: bool = False,
):

    files = list(Path(files_root_dir).rglob("**/*parquet"))
    df_list = []

    for f in tqdm(files):
        _data = pd.read_parquet(f, engine="fastparquet")
        _ref_id = str(f).split("/")[-2].split("=")[-1]
        _data["ref_id"] = _ref_id
        _data["timestamp"] = _data["timestamp"].dt.tz_localize(None)  # .dt.floor("D")
        # if no window of interest is provided, use the original date column to create a window of interest
        if isinstance(window_of_interest, dict):
            woi = None
            for k in window_of_interest.keys():
                if k in str(f):
                    woi = window_of_interest[k]
                    break
            assert (
                woi is not None
            ), f"No window of interest info found for the dataset {str(f)}"
        else:
            woi = window_of_interest if not out_of_window_to_nodata else None
        _data_pivot = process_data_with_window(
            _data,
            woi,
            required_min_timesteps,
            use_valid_time,
            buffer_window,
            composite_window,
        )
        df_list.append(_data_pivot)
    df = pd.concat(df_list).reset_index(drop=True)
    df = df.fillna(no_data_value)
    del df_list
    if out_of_window_to_nodata:
        expected_dates = TimeSeriesProcessor.get_expected_dates(
            df["start_date"].min(), df["end_date"].max(), composite_window
        )
        df = out_window_to_nodata(
            df,
            expected_dates=expected_dates,
            window_of_interest=window_of_interest,
            no_data_value=no_data_value,
        )
    return df
