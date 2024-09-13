from typing import Union

import pandas as pd

from scaleagdata_vito.presto.presto_utils import get_feature_list, normalize_target


def prepare_data_for_cb(
    df: pd.DataFrame,
    target_name: str,
    lower_bound: Union[int, float, None] = None,
    upper_bound: Union[int, float, None] = None,
    num_time_steps: int = 36,
):
    feature_list = get_feature_list(num_time_steps)
    df_cx = df.loc[:, feature_list]
    if upper_bound is not None and lower_bound is not None:
        df_cy = normalize_target(
            df[target_name],
            upper_bound=upper_bound,
            lower_bound=lower_bound,
        )
    else:
        df_cy = df[target_name]
    # replace no_data value with nans for catboost to handle them differntly
    df_cx.replace(65535, None, inplace=True)
    return df_cx, df_cy


def prepare_cropland_data_for_presto(wc_df, sample_frac=0.005):
    def filter_remove_noncrops(df: pd.DataFrame) -> pd.DataFrame:
        crop_labels = [10, 11, 12, 13]
        df = df.loc[df.LANDCOVER_LABEL.isin(crop_labels)]
        return df

    wc_dataset = filter_remove_noncrops(wc_df).sample(frac=sample_frac)

    # convert the target variable to binary
    wc_dataset["LANDCOVER_LABEL"] = (wc_dataset["LANDCOVER_LABEL"] == 11).astype(int)
    return wc_dataset
