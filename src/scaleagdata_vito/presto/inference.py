from pathlib import Path
from typing import Literal, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import xarray as xr
from einops import rearrange
from matplotlib.ticker import ScalarFormatter
from prometheo.models.presto.wrapper import PretrainedPrestoWrapper
from prometheo.predictors import collate_fn
from rasterio.transform import from_bounds
from torch.utils.data import DataLoader

from scaleagdata_vito.presto.datasets import (
    InferenceDataset,
    ScaleAgInferenceDataset,
)


class PrestoPredictor:
    def __init__(
        self,
        model: PretrainedPrestoWrapper,
        batch_size: int = 8192,
        task_type: Literal["regression", "binary", "multiclass"] = "regression",
        composite_window: Literal["dekad", "month"] = "dekad",
    ):
        """
        Initialize the PrestoFeatureExtractor with a Presto model.

        Args:
            model (Presto): The Presto model used for feature extraction.
            batch_size (int): Batch size for dataloader.
        """
        self.model = model  # .to(device="cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.task_type = task_type
        self.composite_window = composite_window

    def predict(
        self,
        path_to_file: Path,
        target_mean: Union[float, None] = None,
        target_std: Union[float, None] = None,
        mask_path: Union[str, Path, None] = None,
        coords: Union[None, tuple] = None,
    ) -> np.ndarray:
        cl = ScaleAgInferenceDataset(composite_window=self.composite_window)
        s1_cube, s2_cube, meteo_cube, dem_cube, latlon_cube, timestamps_cube = (
            cl.nc_to_array(path_to_file, coords=coords)
        )
        ds = InferenceDataset(
            s1_cube, s2_cube, meteo_cube, dem_cube, latlon_cube, timestamps_cube
        )
        dl = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn
        )

        all_probs = []
        self.model.eval()
        for batch in dl:
            with torch.no_grad():
                output = self.model(batch)
                # binary classification
                if self.task_type == "binary":
                    probs = torch.sigmoid(output).cpu().numpy()
                # multiclass classification
                elif self.task_type == "multiclass":
                    probs = torch.softmax(output, dim=-1).cpu().numpy()
                elif self.task_type == "regression":
                    probs = output.cpu().numpy()
                    if target_mean is not None and target_std is not None:
                        probs = self.revert_to_original_units(
                            probs, target_mean, target_std
                        )
                    # probs = np.expm1(probs)
                    else:
                        raise ValueError(
                            "target_mean and target_std used during training"
                            "must be provided for converting results to origininal units"
                        )
                else:
                    raise ValueError(
                        "task_type must be either 'binary', 'multiclass' or 'regression'"
                    )
                all_probs.append(probs.flatten())
        all_probs = np.concatenate(all_probs)
        return all_probs

    def get_predictions(
        self, probs: np.ndarray, threshold: float = 0.5
    ) -> xr.DataArray:
        if self.task_type == "binary":
            preds = probs > threshold
        elif self.task_type == "multiclass":
            preds = probs.argmax(axis=-1)
        else:
            raise ValueError("task_type must be either 'binary' or 'multiclass'")
        return preds

    def revert_to_original_units(self, target_norm, target_mean, target_std):
        return target_norm * target_std + target_mean


def reshape_result(
    result: np.ndarray,
    path_to_input_file: Path,
    out_path=None,
    epsg_code_utm=None,
    coords: Union[None, tuple] = None,
    mask_path: Union[str, Path, None] = None,
    epsg: Union[int, None] = None,
) -> np.ndarray:
    input_arr = xr.load_dataset(path_to_input_file)
    if coords is not None:
        input_arr = input_arr.isel(
            x=slice(coords[0], coords[2]), y=slice(coords[1], coords[3])
        )

    x_coords = input_arr.x.values
    y_coords = input_arr.y.values
    if result.shape[0] != len(x_coords) * len(y_coords):
        class_dim = result.shape[0] // (len(x_coords) * len(y_coords))
        reshaped_result = rearrange(
            result, "(c y x) -> y x c", y=len(y_coords), x=len(x_coords), c=class_dim
        )
    else:
        reshaped_result = rearrange(
            result, "(y x) -> y x", y=len(y_coords), x=len(x_coords)
        )
    if mask_path is not None:
        with rasterio.open(mask_path) as src:
            src_mask = src.read(1).astype(np.uint8)

            x_vals = input_arr.x.values
            y_vals = input_arr.y.values
            x_res = float(np.abs(x_vals[1] - x_vals[0])) if len(x_vals) > 1 else 1.0
            y_res = float(np.abs(y_vals[1] - y_vals[0])) if len(y_vals) > 1 else 1.0

            x_min = float(np.min(x_vals)) - x_res / 2
            x_max = float(np.max(x_vals)) + x_res / 2
            y_min = float(np.min(y_vals)) - y_res / 2
            y_max = float(np.max(y_vals)) + y_res / 2

            dst_mask = np.zeros((len(y_vals), len(x_vals)), dtype=np.uint8)
            dst_transform = from_bounds(
                x_min,
                y_min,
                x_max,
                y_max,
                len(x_vals),
                len(y_vals),
            )

            rasterio.warp.reproject(
                source=src_mask,
                destination=dst_mask,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=src.crs if epsg_code_utm is None else f"EPSG:{epsg_code_utm}",
                resampling=rasterio.enums.Resampling.nearest,
            )
            dst_mask = dst_mask.astype(bool)
        reshaped_result = np.where(dst_mask, reshaped_result, 0)

    if out_path is not None:
        # Get bounds
        west = x_coords.min()
        east = x_coords.max()
        south = y_coords.min()
        north = y_coords.max()

        # Create geotransform
        transform = from_bounds(
            west, south, east, north, reshaped_result.shape[1], reshaped_result.shape[0]
        )

        # Save as GeoTIFF
        coords_str = (
            f"x{coords[0]}-{coords[2]}_y{coords[1]}-{coords[3]}"
            if coords is not None
            else ""
        )
        output_path = out_path / f"{path_to_input_file.stem}_{coords_str}_map.tif"

        if epsg_code_utm is None:
            raise ValueError(
                "epsg_code_utm must be provided to save the predictions map as GeoTIFF"
            )
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=reshaped_result.shape[0],
            width=reshaped_result.shape[1],
            count=1,
            dtype=reshaped_result.dtype,
            crs=f"EPSG:{epsg_code_utm}",
            transform=transform,
        ) as dst:
            dst.write(reshaped_result, 1)
    return reshaped_result


def min_max_normalize(image, gamma=1.0):
    # normalize image and set NaNs to NODATA value
    image = np.nan_to_num(image, 65535).astype("uint16")
    image = (image - image.min()) / (image.max() - image.min()) * gamma
    image = np.clip(
        image, 0, 1
    )  # Ensure values are between 0 and 1 after applying gamma
    return image


def plot_results(
    path_to_input_file,
    task,
    prob_map=None,
    pred_map=None,
    bin_th=0.5,
    ts_index=0,
    coords=None,
    rgb_gamma=1.0,
):
    rgb = xr.load_dataset(path_to_input_file)
    if coords is not None:
        rgb = rgb.isel(x=slice(coords[0], coords[2]), y=slice(coords[1], coords[3]))
    bands = ["S2-L2A-B04", "S2-L2A-B03", "S2-L2A-B02"]
    rgb = np.stack([rgb[band].values for band in bands], axis=-1)[ts_index]
    if task == "binary":
        fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.1)

        axs = [fig.add_subplot(gs[i]) for i in range(3)]
        cax = fig.add_subplot(gs[3])  # dedicated colorbar axis

        axs[0].imshow(min_max_normalize(rgb, gamma=rgb_gamma))
        axs[0].set_title("RGB")
        axs[0].axis("off")

        pred_map = prob_map > bin_th
        axs[1].imshow(pred_map, cmap="gray")
        axs[1].set_title(f"Prediction Map > {bin_th}")
        axs[1].axis("off")

        im = axs[2].imshow(prob_map, cmap="magma", vmin=0, vmax=1)
        axs[2].set_title("Probability Map")
        axs[2].axis("off")

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_ticks(np.arange(0, 1.1, 0.1))
        plt.show()
    elif task == "multiclass":
        if pred_map is None:
            raise ValueError("pred_map must be provided for multiclass classification")
        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)

        axs = [fig.add_subplot(gs[i]) for i in range(2)]
        axs[0].imshow(min_max_normalize(rgb))
        axs[0].set_title("RGB")
        axs[0].axis("off")

        axs[1].imshow(pred_map, cmap="nipy_spectral")
        axs[1].set_title("Prediction Map")
        axs[1].axis("off")

        plt.show()
    else:
        if prob_map is None:
            raise ValueError("prob_map must be provided for regression")
        fig = plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.1)

        axs = [fig.add_subplot(gs[i]) for i in range(2)]
        cax = fig.add_subplot(gs[2])  # separate axis for colorbar

        axs[0].imshow(min_max_normalize(rgb, gamma=rgb_gamma))
        axs[0].set_title("RGB")
        axs[0].axis("off")

        im = axs[1].imshow(prob_map, cmap="magma", vmin=0, vmax=prob_map.max())
        axs[1].set_title("Prediction Map")
        axs[1].axis("off")

        cbar = fig.colorbar(im, cax=cax)
        # Use scientific (power of 10) format
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits(
            (-2, 3)
        )  # show scientific notation for large/small values
        cbar.ax.yaxis.set_major_formatter(formatter)
        plt.show()
