"""Private methods for cropland/croptype mapping. The public functions that
are interracting with the methods here are defined in the `worldcereal.job`
sub-module.
"""

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr
from openeo import DataCube
from openeo.udf import XarrayDataCube
from openeo_gfmap import TemporalContext
from openeo_gfmap.features.feature_extractor import (
    PatchFeatureExtractor,
    apply_feature_extractor,
)
from openeo_gfmap.inference.model_inference import ModelInference, apply_model_inference
from openeo_gfmap.preprocessing.scaling import compress_uint8, compress_uint16
from prometheo.models.presto.wrapper import PretrainedPrestoWrapper
from prometheo.predictors import collate_fn
from pydantic import BaseModel
from torch.utils.data import DataLoader

from scaleagdata_vito.presto.datasets_prometheo import ScaleAgInferenceDataset


class ScaleAgRegressor(ModelInference):
    import numpy as np

    def __init__(self):
        super().__init__()

        self.onnx_session = None

    def dependencies(self) -> list:
        return []  # Disable the dependencies from PIP install

    def output_labels(self) -> list:
        """
        Returns the output labels for the regression
        """

        return ["prediction", "probability"] 
        
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predicts labels using the provided features array.

        LUT needs to be explicitly sorted here as openEO does
        not guarantee the order of a json object being preserved when decoding
        a process graph in the backend.
        """
        import numpy as np


        if self.onnx_session is None:
            raise ValueError("Model has not been loaded. Please load a model first.")

        # Prepare input data for ONNX model
        outputs = self.onnx_session.run(None, {"inputs": inputs})

        # extract outputs 
        probabilities = np.zeros((len(outputs[0]),), dtype=np.uint8)
        predictions = outputs[1].astype(np.float32)

        # round probabilities 
        for i in range(len(outputs[0])):
            probabilities[i] = int(round(outputs[0][i] * 100))

        return np.hstack(
            [predictions[:, np.newaxis], probabilities[:, np.newaxis]]
        ).transpose()

    def execute(self, inarr: xr.DataArray) -> xr.DataArray:

        if "classifier_url" not in self._parameters:
            raise ValueError('Missing required parameter "classifier_url"')
        classifier_url = self._parameters.get("classifier_url")
        self.logger.info(f'Loading classifier model from "{classifier_url}"')

        # shape and indices for output ("xy", "bands")
        x_coords, y_coords = inarr.x.values, inarr.y.values
        inarr = inarr.transpose("bands", "x", "y").stack(xy=["x", "y"]).transpose()

        self.onnx_session = self.load_ort_session(classifier_url)

        # Run catboost classification
        self.logger.info("Catboost classification with input shape: %s", inarr.shape)
        regression = self.predict(inarr.values)
        self.logger.info("Classification done with shape: %s", inarr.shape)
        output_labels = self.output_labels()    

        regression_da = xr.DataArray(
            regression.reshape((len(output_labels), len(x_coords), len(y_coords))),
            dims=["bands", "x", "y"],
            coords={
                "bands": output_labels,
                "x": x_coords,
                "y": y_coords,
            },
        )

        return regression_da
    


class FeaturesParameters(BaseModel):
    """Parameters for the feature extraction UDFs. Types are enforced by
    Pydantic.

    Attributes
    ----------
    rescale_s1 : bool (default=False)
        Whether to rescale Sentinel-1 bands before feature extraction. Should be
        left to False, as this is done in the Presto UDF itself.
    presto_model_url : str
        Public URL to the Presto model used for feature extraction. The file
        should be a PyTorch serialized model.
    compile_presto : bool (default=False)
        Whether to compile the Presto encoder for speeding up large-scale inference.
    """

    rescale_s1: bool
    presto_model_url: str
    compile_presto: bool
    

class ScaleAgBinaryClassifier(PatchFeatureExtractor):
    """Feature extractor to use Presto model to compute per-pixel embeddings.
    This will generate a datacube with 128 bands, each band representing a
    feature from the Presto model.

    Interesting UDF parameters:
    - presto_url: A public URL to the Presto model file. A default Presto
      version is provided if the parameter is left undefined.
    - rescale_s1: Is specifically disabled by default, as the presto
      dependencies already take care of the backscatter decompression. If
      specified, should be set as `False`.
    """

    import functools

    PRESTO_WHL_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/dependencies/presto_worldcereal-0.1.6-py3-none-any.whl" #####
    BASE_URL = "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies"  # NOQA
    DEPENDENCY_NAME = "worldcereal_deps.zip" #########

    GFMAP_BAND_MAPPING = {
        "S2-L2A-B02": "B2",
        "S2-L2A-B03": "B3",
        "S2-L2A-B04": "B4",
        "S2-L2A-B05": "B5",
        "S2-L2A-B06": "B6",
        "S2-L2A-B07": "B7",
        "S2-L2A-B08": "B8",
        "S2-L2A-B8A": "B8A",
        "S2-L2A-B11": "B11",
        "S2-L2A-B12": "B12",
        "S1-SIGMA0-VH": "VH",
        "S1-SIGMA0-VV": "VV",
        "AGERA5-TMEAN": "temperature",
        "AGERA5-PRECIP": "precipitation",
    }

    @functools.lru_cache(maxsize=6)
    def unpack_presto_wheel(self, wheel_url: str, destination_dir: str) -> str:
        import urllib.request
        import zipfile
        from pathlib import Path

        # Downloads the wheel file
        modelfile, _ = urllib.request.urlretrieve(
            wheel_url, filename=Path.cwd() / Path(wheel_url).name
        )
        with zipfile.ZipFile(modelfile, "r") as zip_ref:
            zip_ref.extractall(destination_dir)
        return destination_dir

    def output_labels(self) -> list:
        """Returns the output labels from this UDF, which is the output labels
        of the presto embeddings"""
        return ["predictions", "probabilities"]

    def evaluate_resolution(self, inarr: xr.DataArray) -> int:
        """Helper function to get the resolution in meters for
        the input array.

        Parameters
        ----------
        inarr : xr.DataArray
            input array to determine resolution for.

        Returns
        -------
        int
            resolution in meters.
        """

        if self.epsg == 4326:
            from pyproj import Transformer
            from shapely.geometry import Point
            from shapely.ops import transform

            self.logger.info(
                "Converting WGS84 coordinates to EPSG:3857 to determine resolution."
            )

            transformer = Transformer.from_crs(self.epsg, 3857, always_xy=True)
            points = [Point(x, y) for x, y in zip(inarr.x.values, inarr.y.values)]
            points = [transform(transformer.transform, point) for point in points]

            resolution = abs(points[1].x - points[0].x)

        else:
            resolution = abs(inarr.x[1].values - inarr.x[0].values)

        self.logger.info(f"Resolution for computing slope: {resolution}")

        return resolution

    def compute_slope(self, inarr: xr.DataArray, resolution: int) -> xr.DataArray:
        """Computes the slope using the scipy library. The input array should
        have the following bands: 'elevation' And no time dimension. Returns a
        new DataArray containing the new `slope` band.

        Parameters
        ----------
        inarr : xr.DataArray
            input array containing a band 'elevation'.
        resolution : int
            resolution of the input array in meters.

        Returns
        -------
        xr.DataArray
            output array containing 'slope' band in degrees.
        """

        import random  # pylint: disable=import-outside-toplevel

        import numpy as np  # pylint: disable=import-outside-toplevel
        from scipy.ndimage import (  # pylint: disable=import-outside-toplevel
            convolve,
            zoom,
        )

        def _rolling_fill(darr, max_iter=2):
            """Helper function that also reflects values inside
            a patch with NaNs."""
            if max_iter == 0:
                return darr
            else:
                max_iter -= 1
            # arr of shape (rows, cols)
            mask = np.isnan(darr)

            if ~np.any(mask):
                return darr

            roll_params = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(roll_params)

            for roll_param in roll_params:
                rolled = np.roll(darr, roll_param, axis=(0, 1))
                darr[mask] = rolled[mask]

            return _rolling_fill(darr, max_iter=max_iter)

        def _downsample(arr: np.ndarray, factor: int) -> np.ndarray:
            """Downsamples a 2D NumPy array by a given factor with average resampling and reflect padding.

            Parameters
            ----------
            arr : np.ndarray
                The 2D input array.
            factor : int
                The factor by which to downsample. For example, factor=2 downsamples by 2x.

            Returns
            -------
            np.ndarray
                Downsampled array.
            """

            # Get the original shape of the array
            X, Y = arr.shape

            # Calculate how much padding is needed for each dimension
            pad_X = (
                factor - (X % factor)
            ) % factor  # Ensures padding is only applied if needed
            pad_Y = (
                factor - (Y % factor)
            ) % factor  # Ensures padding is only applied if needed

            # Pad the array using 'reflect' mode
            padded = np.pad(arr, ((0, pad_X), (0, pad_Y)), mode="reflect")

            # Reshape the array to form blocks of size 'factor' x 'factor'
            reshaped = padded.reshape(
                (X + pad_X) // factor, factor, (Y + pad_Y) // factor, factor
            )

            # Take the mean over the factor-sized blocks
            downsampled = np.nanmean(reshaped, axis=(1, 3))

            return downsampled

        dem = inarr.sel(bands="elevation").values
        dem_arr = dem.astype(np.float32)

        # Invalid to NaN and keep track of these pixels
        dem_arr[dem_arr == 65535] = np.nan
        idx_invalid = np.isnan(dem_arr)

        # Fill NaNs with rolling fill
        dem_arr = _rolling_fill(dem_arr)

        # We make sure DEM is at 20m for slope computation
        # compatible with global slope collection
        factor = int(20 / resolution)
        if factor < 1 or factor % 2 != 0:
            raise NotImplementedError(
                f"Unsupported resolution for slope computation: {resolution}"
            )
        dem_arr_downsampled = _downsample(dem_arr, factor)
        x_odd, y_odd = dem_arr.shape[0] % 2 != 0, dem_arr.shape[1] % 2 != 0

        # Mask NaN values in the DEM data
        dem_masked = np.ma.masked_invalid(dem_arr_downsampled)

        # Define convolution kernels for x and y gradients (simple finite difference approximation)
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / (
            8.0 * 20  # array is now at 20m resolution
        )  # x-derivative kernel

        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / (
            8.0 * 20  # array is now at 20m resolution
        )  # y-derivative kernel

        # Apply convolution to compute gradients
        dx = convolve(dem_masked, kernel_x)  # Gradient in the x-direction
        dy = convolve(dem_masked, kernel_y)  # Gradient in the y-direction

        # Reapply the mask to the gradients
        dx = np.ma.masked_where(dem_masked.mask, dx)
        dy = np.ma.masked_where(dem_masked.mask, dy)

        # Calculate the magnitude of the gradient (rise/run)
        gradient_magnitude = np.ma.sqrt(dx**2 + dy**2)

        # Convert gradient magnitude to slope (in degrees)
        slope = np.ma.arctan(gradient_magnitude) * (180 / np.pi)

        # Upsample to original resolution with bilinear interpolation
        mask = slope.mask
        mask = zoom(mask, zoom=factor, order=0)
        slope = zoom(slope, zoom=factor, order=1)
        slope[mask] = 65535

        # Strip one row or column if original array was odd in that dimension
        if x_odd:
            slope = slope[:-1, :]
        if y_odd:
            slope = slope[:, :-1]

        # Fill slope values where the original DEM had NaNs
        slope[idx_invalid] = 65535
        slope[np.isnan(slope)] = 65535
        slope = slope.astype(np.uint16)

        return xr.DataArray(
            slope[None, :, :],
            dims=("bands", "y", "x"),
            coords={
                "bands": ["slope"],
                "y": inarr.y,
                "x": inarr.x,
            },
        )

    def execute(self, inarr: xr.DataArray) -> xr.DataArray:
        import sys

        if self.epsg is None:
            raise ValueError(
                "EPSG code is required for Presto feature extraction, but was "
                "not correctly initialized."
            )
        if "presto_model_url" not in self._parameters:
            raise ValueError('Missing required parameter "presto_model_url"')
        presto_model_url = self._parameters.get("presto_model_url")
        self.logger.info(f'Loading Presto model from "{presto_model_url}"')
        presto_wheel_url = self._parameters.get("presto_wheel_url", self.PRESTO_WHL_URL)
        self.logger.info(f'Loading Presto wheel from "{presto_wheel_url}"')

        ignore_dependencies = self._parameters.get("ignore_dependencies", False)
        if ignore_dependencies:
            self.logger.info(
                "`ignore_dependencies` flag is set to True. Make sure that "
                "Presto and its dependencies are available on the runtime "
                "environment"
            )

        # The below is required to avoid flipping of the result
        # when running on OpenEO backend!
        inarr = inarr.transpose("bands", "t", "x", "y")

        # Change the band names
        new_band_names = [
            self.GFMAP_BAND_MAPPING.get(b.item(), b.item()) for b in inarr.bands
        ]
        inarr = inarr.assign_coords(bands=new_band_names)

        # Handle NaN values in Presto compatible way
        inarr = inarr.fillna(65535)
        
        # Unzip de dependencies on the backend
        if not ignore_dependencies:
            self.logger.info("Unzipping dependencies")
            deps_dir = self.extract_dependencies(self.BASE_URL, self.DEPENDENCY_NAME)
            self.logger.info("Unpacking presto wheel")
            deps_dir = self.unpack_presto_wheel(presto_wheel_url, deps_dir)

            self.logger.info("Appending dependencies")
            sys.path.append(str(deps_dir))

        if "slope" not in inarr.bands:
            # If 'slope' is not present we need to compute it here
            self.logger.warning("`slope` not found in input array. Computing ...")
            resolution = self.evaluate_resolution(inarr.isel(t=0))
            slope = self.compute_slope(inarr.isel(t=0), resolution)
            slope = slope.expand_dims({"t": inarr.t}, axis=0).astype("float32")

            inarr = xr.concat([inarr.astype("float32"), slope], dim="bands")

        self.logger.info("Predicting with Fine-tuned Presto")
        
        # implement similar predictor to `predict_with_head` in the
        # scaleagdata_vito.presto.utils.py file
        
        output = predict_with_ft_presto(
            inarr,
            presto_url=presto_model_url,
            task_type=self._parameters.get("task_type", "regression"),
            num_outputs=self._parameters.get("num_outputs", None),
            compositing_window=self._parameters.get("compositing_window", "dekad"),
        )
        return output

    def _execute(self, cube: XarrayDataCube, parameters: dict) -> XarrayDataCube:
        # Disable S1 rescaling (decompression) by default
        if parameters.get("rescale_s1", None) is None:
            parameters.update({"rescale_s1": False})
        return super()._execute(cube, parameters)

    def dependencies(self) -> list:
        # We are just overriding the parent method to suppress the warning
        return []


def predict_with_ft_presto(
    inarr: xr.DataArray,
    presto_url: str,
    task_type: Literal[ "regression", "binary", "multiclass"],
    num_outputs: Optional[int] = None,
    compositing_window: Literal["dekad", "month"] = "dekad",
    batch_size: int = 8192,
) -> Union[np.ndarray, xr.DataArray]:

    # Load the model based on the task type
    from prometheo.models.presto.wrapper import PretrainedPrestoWrapper, load_pretrained
    
    if task_type == "regression":
        finetuned_model = PretrainedPrestoWrapper(num_outputs=1, regression=True)
    elif task_type == "binary":
        finetuned_model = PretrainedPrestoWrapper(num_outputs=1, regression=False)
    elif task_type == "multiclass":
        finetuned_model = PretrainedPrestoWrapper(num_outputs=num_outputs, regression=False)
       
    finetuned_model = load_pretrained(
        finetuned_model,
        fine_tuned_model_url=presto_url,
    )

    presto_model = PrestoPredictor(
        model=finetuned_model,
        batch_size=batch_size,
        task_type=task_type,
        compositing_window=compositing_window,
    )
    # assert isinstance(inarr, xr.DataArray):
    return presto_model.predict(inarr)
    # else:
    #     raise ValueError("Input data must be either xr.DataArray or pd.DataFrame")
    

class PrestoPredictor:
    def __init__(self,
                 model: PretrainedPrestoWrapper, 
                 batch_size: int = 8192,
                 task_type: Literal["regression", "binary", "multiclass"] = "regression",
                 compositing_window: Literal["dekad", "month"] = "dekad"):
        """
        Initialize the PrestoFeatureExtractor with a Presto model.

        Args:
            model (Presto): The Presto model used for feature extraction.
            batch_size (int): Batch size for dataloader.
        """
        self.model = model
        self.batch_size = batch_size
        self.task_type = task_type
        self.compositing_window = compositing_window

    def predict(self, inarr: xr.DataArray) -> xr.DataArray:
        dataset = ScaleAgInferenceDataset(inarr, compositing_window=self.compositing_window)
        dl = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        all_preds, all_probs = [], []
        self.model.eval()
        for batch in dl:
            with torch.no_grad():
                probs = self.model(batch)
                # binary classification
                if self.task_type == "binary":
                    preds = nn.functional.sigmoid(probs)
                # multiclass classification
                elif self.task_type == "multiclass":
                    preds = nn.functional.softmax(probs, dim=-1)
                preds = preds.cpu().numpy().flatten()

                all_preds.append(preds)
                all_probs.append(probs.cpu().numpy().flatten())
        
        # need to be reshaped 
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)
        return all_preds, all_probs
