from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from einops import rearrange
from prometheo.models.presto.wrapper import PretrainedPrestoWrapper
from prometheo.predictors import collate_fn
from torch.utils.data import DataLoader

from scaleagdata_vito.presto.datasets_prometheo import (
    InferenceDataset,
    ScaleAgInferenceDataset,
)


class PrestoPredictor:
    def __init__(
        self, 
        model: PretrainedPrestoWrapper,
        batch_size: int = 8192,
        task_type: Literal["regression", "binary", "multiclass"] = "regression",
        compositing_window: Literal["dekad", "month"] = "dekad"
        ):
        """
        Initialize the PrestoFeatureExtractor with a Presto model.

        Args:
            model (Presto): The Presto model used for feature extraction.
            batch_size (int): Batch size for dataloader.
        """
        self.model = model #.to(device="cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.task_type = task_type
        self.compositing_window = compositing_window

    def predict(self, path_to_file: Path) -> np.ndarray:
        cl = ScaleAgInferenceDataset(compositing_window=self.compositing_window)
        s1_cube, s2_cube, meteo_cube, dem_cube, latlon_cube, timestamps_cube = cl.nc_to_array(path_to_file)
        ds = InferenceDataset(s1_cube,
                              s2_cube,
                              meteo_cube,
                              dem_cube,
                              latlon_cube,
                              timestamps_cube)
        dl = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        all_probs = []
        self.model.eval()
        for batch in dl:
            with torch.no_grad():
                output = self.model(batch, inference=True)
                # binary classification
                if self.task_type == "binary":
                    probs = torch.sigmoid(output).cpu().numpy()                
                # multiclass classification
                elif self.task_type == "multiclass":
                    probs = torch.softmax(output, dim=-1).cpu().numpy()
                all_probs.append(probs.flatten())
        all_probs = np.concatenate(all_probs)
        return all_probs
        
    def get_predictions(self, probs: np.ndarray, threshold: int = 0.5) -> xr.DataArray:
        if self.task_type == "binary":
            preds = probs > threshold
        elif self.task_type == "multiclass":
            preds = probs.argmax(axis=-1)
        else:
            raise ValueError("task_type must be either 'binary' or 'multiclass'")
        return preds


def reshape_result(result: np.ndarray, path_to_input_file: Path):
    input_arr = xr.load_dataset(path_to_input_file)
    x_coords = input_arr.x.values
    y_coords = input_arr.y.values
    reshaped_result = rearrange(result, '(y x) -> y x', y=len(y_coords), x=len(x_coords))
    return reshaped_result

def min_max_normalize(image):
    # Ensure the input is a numpy array
    if not isinstance(image, np.ndarray):
        image = np.array(image, dtype=np.float32)
    return (image - image.min()) / (image.max() - image.min())

def plot_results(prob_map, path_to_input_file, task, pred_map=None, ts_index=0):
    rgb = xr.load_dataset(path_to_input_file)
    bands = ['S2-L2A-B04', 'S2-L2A-B03', 'S2-L2A-B02']
    rgb = np.stack([rgb[band].values for band in bands], axis=-1)
    rgb = min_max_normalize(rgb)
    if task!= "regression":
        fig, axs = plt.subplots(1, 3, figsize=(15, 8))

        # Plot the RGB bands
        axs[0].imshow(min_max_normalize(rgb[ts_index]))
        axs[0].set_title('RGB')
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        # Plot the preds_map
        axs[1].imshow(pred_map, cmap='viridis')
        axs[1].set_title('Prediction Map')
        axs[1].set_xticks([])
        axs[1].set_yticks([])

        # Plot the prob_map
        axs[2].imshow(prob_map, cmap='viridis', vmin=0, vmax=1)
        axs[2].set_title('Probability Map')
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        cbar = fig.colorbar(axs[2].images[0], ax=axs[2], fraction=0.046, pad=0.04)
        cbar.set_ticks([0, 1])
        cbar.set_label('Probability')

        plt.show()
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 8))

        # Plot the RGB bands
        axs[0].imshow(min_max_normalize(rgb[ts_index]))
        axs[0].set_title('RGB')
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        # Plot the preds_map
        axs[1].imshow(prob_map, cmap='viridis')
        axs[1].set_title('Prediction Map')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        
        cbar = fig.colorbar(axs[2].images[0], ax=axs[2], fraction=0.046, pad=0.04)
        cbar.set_label('Probability')
        plt.show()