from datetime import datetime, timedelta
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd
from presto.dataops import (
    BANDS,
    BANDS_GROUPS_IDX,
    NORMED_BANDS,
    S1_S2_ERA5_SRTM,
    DynamicWorld2020_2021,
)
from presto.masking import BAND_EXPANSION
from torch.utils.data import Dataset

IDX_TO_BAND_GROUPS = {}
for band_group_idx, (key, val) in enumerate(BANDS_GROUPS_IDX.items()):
    for idx in val:
        IDX_TO_BAND_GROUPS[NORMED_BANDS[idx]] = band_group_idx


class ScaleAgBase(Dataset):
    _NODATAVALUE = 65535
    NUM_TIMESTEPS = 12
    BAND_MAPPING = {
        "OPTICAL-B02-ts{}-10m": "B2",
        "OPTICAL-B03-ts{}-10m": "B3",
        "OPTICAL-B04-ts{}-10m": "B4",
        "OPTICAL-B05-ts{}-20m": "B5",
        "OPTICAL-B06-ts{}-20m": "B6",
        "OPTICAL-B07-ts{}-20m": "B7",
        "OPTICAL-B08-ts{}-10m": "B8",
        "OPTICAL-B8A-ts{}-20m": "B8A",
        "OPTICAL-B11-ts{}-20m": "B11",
        "OPTICAL-B12-ts{}-20m": "B12",
        "SAR-VH-ts{}-20m": "VH",
        "SAR-VV-ts{}-20m": "VV",
        "METEO-precipitation_flux-ts{}-100m": "total_precipitation",
        "METEO-temperature_mean-ts{}-100m": "temperature_2m",
    }
    STATIC_BAND_MAPPING = {"DEM-alt-20m": "elevation", "DEM-slo-20m": "slope"}

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_name: str,
        task: Literal["regression", "binary", "multiclass"],
    ):
        self.df = dataframe.replace({np.nan: self._NODATAVALUE})
        self.target_name = target_name
        self.task = task

        if self.task == "multiclass":
            self.class_to_index = {
                label: idx for idx, label in enumerate(dataframe[target_name].unique())
            }
            self.index_to_class = {
                idx: label for idx, label in enumerate(dataframe[target_name].unique())
            }

    def __len__(self):
        return self.df.shape[0]

    @staticmethod
    def get_target(row_d: Dict, target_name: str) -> int:
        return int(row_d[target_name])

    @classmethod
    def row_to_arrays(
        cls, row: pd.Series, target_name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        # https://stackoverflow.com/questions/45783891/is-there-a-way-to-speed-up-the-pandas-getitem-getitem-axis-and-get-label
        # This is faster than indexing the series every time!
        row_d = pd.Series.to_dict(row)

        latlon = np.array([row_d["lat"], row_d["lon"]], dtype=np.float32)
        month = datetime.strptime(row_d["start_date"], "%Y-%m-%d").month - 1

        eo_data = np.zeros((cls.NUM_TIMESTEPS, len(BANDS)))
        # an assumption we make here is that all timesteps for a token
        # have the same masking
        mask = np.zeros((cls.NUM_TIMESTEPS, len(BANDS_GROUPS_IDX)))
        for df_val, presto_val in cls.BAND_MAPPING.items():
            # retrieve ts for each band column df_val
            values = np.array(
                [float(row_d[df_val.format(t)]) for t in range(cls.NUM_TIMESTEPS)]
            )
            # this occurs for the DEM values in one point in Fiji
            values = np.nan_to_num(values, nan=cls._NODATAVALUE)
            idx_valid = values != cls._NODATAVALUE
            if presto_val in ["VV", "VH"]:
                # convert to dB
                idx_valid = idx_valid & (values > 0)
                values[idx_valid] = 20 * np.log10(values[idx_valid]) - 83
            elif presto_val == "total_precipitation":
                # scaling, and AgERA5 is in mm, Presto expects m
                values[idx_valid] = values[idx_valid] / (100 * 1000.0)
            elif presto_val == "temperature_2m":
                # remove scaling. conversion to celsius is done in the normalization
                values[idx_valid] = values[idx_valid] / 100
            mask[:, IDX_TO_BAND_GROUPS[presto_val]] += ~idx_valid
            # add values to eo at specifix index. the order followed is the one suggested by BANDS
            eo_data[:, BANDS.index(presto_val)] = values
        for df_val, presto_val in cls.STATIC_BAND_MAPPING.items():
            # this occurs for the DEM values in one point in Fiji
            values = np.nan_to_num(row_d[df_val], nan=cls._NODATAVALUE)
            idx_valid = values != cls._NODATAVALUE
            eo_data[:, BANDS.index(presto_val)] = values
            mask[:, IDX_TO_BAND_GROUPS[presto_val]] += ~idx_valid

        return (
            cls.check(eo_data),
            mask.astype(bool),
            latlon,
            month,
            row_d[target_name],
        )

    def __getitem__(self, idx):
        raise NotImplementedError

    @classmethod
    def normalize_and_mask(cls, eo: np.ndarray):
        # TODO: this can be removed
        keep_indices = [idx for idx, val in enumerate(BANDS) if val != "B9"]
        normed_eo = S1_S2_ERA5_SRTM.normalize(eo)  # this adds NDVI and normalizes
        # TODO: fix this. For now, we replicate the previous behaviour
        # only keeps the bands present in the data after normalization and sets to 0 the no_data locations
        normed_eo = np.where(eo[:, keep_indices] != cls._NODATAVALUE, normed_eo, 0)
        return normed_eo

    @staticmethod
    def check(array: np.ndarray) -> np.ndarray:
        assert not np.isnan(array).any()
        return array


class ScaleAGDataset(ScaleAgBase):

    UPPER_BOUND = 120000
    LOWER_BOUND = 10000

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_name: str,
        task: Literal["regression", "binary", "multiclass"],
    ):
        # bound label values to valid range
        if task == "regression":
            dataframe[target_name] = dataframe[target_name].clip(
                lower=self.LOWER_BOUND, upper=self.UPPER_BOUND
            )
            self.mean = np.mean(dataframe[target_name])
            self.std = np.std(dataframe[target_name])

        super().__init__(dataframe, target_name, task)

    def __getitem__(self, idx):
        # Get the sample
        row = self.df.iloc[idx, :]
        eo, mask_per_token, latlon, month, target = self.row_to_arrays(
            row, self.target_name
        )
        mask_per_variable = np.repeat(mask_per_token, BAND_EXPANSION, axis=1)
        if self.task == "regression":
            target = self.normalize_target(target)
        # convert classes to indices for multiclass
        elif self.task == "multiclass":
            target = self.class_to_index[target]
        return (
            self.normalize_and_mask(eo),
            target,
            np.ones(self.NUM_TIMESTEPS) * (DynamicWorld2020_2021.class_amount),
            latlon,
            month,
            mask_per_variable,
        )

    def normalize_target(self, target):
        return (target - self.LOWER_BOUND) / (self.UPPER_BOUND - self.LOWER_BOUND)

    def z_scaling(self, x):
        return (x - self.mean) / self.std

    def revert_to_original_units(self, target_norm):
        return target_norm * (self.UPPER_BOUND - self.LOWER_BOUND) + self.LOWER_BOUND


class ScaleAG10DDataset(ScaleAGDataset):

    NUM_TIMESTEPS = 36

    def __getitem__(self, idx):
        # Get the sample
        row = self.df.iloc[idx, :]
        eo, mask_per_token, latlon, _, target = self.row_to_arrays(
            row, self.target_name
        )
        mask_per_variable = np.repeat(mask_per_token, BAND_EXPANSION, axis=1)
        # normalize target for regression
        if self.task == "regression":
            target = self.normalize_target(target)
        # convert classes to indices for multiclass
        elif self.task == "multiclass":
            target = self.class_to_index[target]
        return (
            self.normalize_and_mask(eo),
            target,
            np.ones(self.NUM_TIMESTEPS) * (DynamicWorld2020_2021.class_amount),
            latlon,
            self.get_month_array(row),
            mask_per_variable,
        )

    @classmethod
    def get_month_array(cls, row: pd.Series) -> np.ndarray:
        start_date, end_date = datetime.strptime(
            row.start_date, "%Y-%m-%d"
        ), datetime.strptime(row.end_date, "%Y-%m-%d")

        # Calculate the step size for 10-day intervals and create a list of dates
        step = int((end_date - start_date).days / (cls.NUM_TIMESTEPS - 1))
        date_vector = [
            start_date + timedelta(days=i * step) for i in range(cls.NUM_TIMESTEPS)
        ]

        # Ensure last date is not beyond the end date
        if date_vector[-1] > end_date:
            date_vector[-1] = end_date

        return np.array([d.month - 1 for d in date_vector])
