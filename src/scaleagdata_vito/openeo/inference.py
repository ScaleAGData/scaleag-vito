"""Private methods for cropland/croptype mapping. The public functions that
are interracting with the methods here are defined in the `worldcereal.job`
sub-module.
"""

from typing import Optional

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
from pydantic import BaseModel

from scaleagdata_vito.openeo.parameters import PostprocessParameters


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
    
    
class ScaleAgBinaryClassifier(ModelInference):
    import numpy as np

    def __init__(self):
        super().__init__()

        self.onnx_session = None

    def dependencies(self) -> list:
        return []  # Disable the dependencies from PIP install

    def output_labels(self) -> list:
        """
        Returns the output labels for the classification.

        LUT needs to be explicitly sorted here as openEO does
        not guarantee the order of a json object being preserved when decoding
        a process graph in the backend.
        """
        lut = self._parameters["lookup_table"]
        lut_sorted = {k: v for k, v in sorted(lut.items(), key=lambda item: item[1])}
        class_names = lut_sorted.keys()

        return ["prediction", "probability"] + [
            f"probability_{name}" for name in class_names
        ]

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predicts labels using the provided features array.

        LUT needs to be explicitly sorted here as openEO does
        not guarantee the order of a json object being preserved when decoding
        a process graph in the backend.
        """
        import numpy as np

        # Classes names to codes
        lookup_table = self._parameters.get("lookup_table", None)

        if lookup_table is None:
            raise ValueError(
                "Lookup table is not defined. Please provide lookup_table in the UDFs parameters."
            )

        lut_sorted = {
            k: v for k, v in sorted(lookup_table.items(), key=lambda item: item[1])
        }

        if self.onnx_session is None:
            raise ValueError("Model has not been loaded. Please load a model first.")

        # Prepare input data for ONNX model
        outputs = self.onnx_session.run(None, {"inputs": inputs})

        # Extract classes as INTs and probability of winning class values
        labels = np.zeros((len(outputs[0]),), dtype=np.uint16)
        probabilities = np.zeros((len(outputs[0]),), dtype=np.uint8)
        for i, (label, prob) in enumerate(zip(outputs[0], outputs[1])):
            labels[i] = lut_sorted[label]
            probabilities[i] = int(round(prob[label] * 100))

        # Extract per class probabilities
        output_probabilities = []
        for output_px in outputs[1]:
            output_probabilities.append(
                [output_px[label] for label in lut_sorted.keys()]
            )

        output_probabilities = (
            (np.array(output_probabilities) * 100).round().astype(np.uint8)
        )

        return np.hstack(
            [labels[:, np.newaxis], probabilities[:, np.newaxis], output_probabilities]
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
        classification = self.predict(inarr.values)
        self.logger.info("Classification done with shape: %s", inarr.shape)

        output_labels = self.output_labels()

        classification_da = xr.DataArray(
            classification.reshape((len(output_labels), len(x_coords), len(y_coords))),
            dims=["bands", "x", "y"],
            coords={
                "bands": output_labels,
                "x": x_coords,
                "y": y_coords,
            },
        )

        return classification_da

class ScaleAgMulticlassClassifier(ModelInference):
    import numpy as np

    def __init__(self):
        super().__init__()

        self.onnx_session = None

    def dependencies(self) -> list:
        return []  # Disable the dependencies from PIP install

    def output_labels(self) -> list:
        """
        Returns the output labels for the classification.

        LUT needs to be explicitly sorted here as openEO does
        not guarantee the order of a json object being preserved when decoding
        a process graph in the backend.
        """
        lut = self._parameters["lookup_table"]
        lut_sorted = {k: v for k, v in sorted(lut.items(), key=lambda item: item[1])}
        class_names = lut_sorted.keys()

        return ["prediction", "probability"] + [
            f"probability_{name}" for name in class_names
        ]

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predicts labels using the provided features array.

        LUT needs to be explicitly sorted here as openEO does
        not guarantee the order of a json object being preserved when decoding
        a process graph in the backend.
        """
        import numpy as np

        # Classes names to codes
        lookup_table = self._parameters.get("lookup_table", None)

        if lookup_table is None:
            raise ValueError(
                "Lookup table is not defined. Please provide lookup_table in the UDFs parameters."
            )

        lut_sorted = {
            k: v for k, v in sorted(lookup_table.items(), key=lambda item: item[1])
        }

        if self.onnx_session is None:
            raise ValueError("Model has not been loaded. Please load a model first.")

        # Prepare input data for ONNX model
        outputs = self.onnx_session.run(None, {"inputs": inputs})

        # Extract classes as INTs and probability of winning class values
        labels = np.zeros((len(outputs[0]),), dtype=np.uint16)
        probabilities = np.zeros((len(outputs[0]),), dtype=np.uint8)
        for i, (label, prob) in enumerate(zip(outputs[0], outputs[1])):
            labels[i] = lut_sorted[label]
            probabilities[i] = int(round(prob[label] * 100))

        # Extract per class probabilities
        output_probabilities = []
        for output_px in outputs[1]:
            output_probabilities.append(
                [output_px[label] for label in lut_sorted.keys()]
            )

        output_probabilities = (
            (np.array(output_probabilities) * 100).round().astype(np.uint8)
        )

        return np.hstack(
            [labels[:, np.newaxis], probabilities[:, np.newaxis], output_probabilities]
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
        classification = self.predict(inarr.values)
        self.logger.info("Classification done with shape: %s", inarr.shape)

        output_labels = self.output_labels()

        classification_da = xr.DataArray(
            classification.reshape((len(output_labels), len(x_coords), len(y_coords))),
            dims=["bands", "x", "y"],
            coords={
                "bands": output_labels,
                "x": x_coords,
                "y": y_coords,
            },
        )

        return classification_da


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
    

class PrestoPredictor(PatchFeatureExtractor):
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
        "AGERA5-TMEAN": "temperature_2m",
        "AGERA5-PRECIP": "total_precipitation",
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
        return [f"presto_ft_{i}" for i in range(128)]

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

        # # Add valid_date attribute to the input array if we need it and
        # # it's not there. For now we take center timestamp in this case.
        # use_valid_date_token = self._parameters.get("use_valid_date_token", False)
        # if "valid_date" not in inarr.attrs:
        #     if use_valid_date_token:
        #         # Only log warning if we will use the valid_date token
        #         self.logger.warning(
        #             "No `valid_date` attribute found in input array. Taking center timestamp."
        #         )
        #     inarr.attrs["valid_date"] = inarr.t.values[6]

        # Unzip de dependencies on the backend
        if not ignore_dependencies:
            self.logger.info("Unzipping dependencies")
            deps_dir = self.extract_dependencies(self.BASE_URL, self.DEPENDENCY_NAME)
            self.logger.info("Unpacking presto wheel")
            deps_dir = self.unpack_presto_wheel(presto_wheel_url, deps_dir)

            self.logger.info("Appending dependencies")
            sys.path.append(str(deps_dir))

        from scaleagdata_vito.presto.utils import predict_with_head

        if "slope" not in inarr.bands:
            # If 'slope' is not present we need to compute it here
            self.logger.warning("`slope` not found in input array. Computing ...")
            resolution = self.evaluate_resolution(inarr.isel(t=0))
            slope = self.compute_slope(inarr.isel(t=0), resolution)
            slope = slope.expand_dims({"t": inarr.t}, axis=0).astype("float32")

            inarr = xr.concat([inarr.astype("float32"), slope], dim="bands")

        batch_size = self._parameters.get("batch_size", 256)
        compile_presto = self._parameters.get("compile_presto", False)
        self.logger.info(f"Compile presto: {compile_presto}")

        self.logger.info("Predicting with Fine-tuned Presto")
        predictions = predict_with_head(
            inarr,
            batch_size=batch_size,
            model_url=presto_model_url,
            compile_presto=compile_presto,
        )
        return predictions

    def _execute(self, cube: XarrayDataCube, parameters: dict) -> XarrayDataCube:
        # Disable S1 rescaling (decompression) by default
        if parameters.get("rescale_s1", None) is None:
            parameters.update({"rescale_s1": False})
        return super()._execute(cube, parameters)

    def dependencies(self) -> list:
        # We are just overriding the parent method to suppress the warning
        return []


def _scaleag_map(
    inputs: DataCube,
    temporal_extent: TemporalContext,
    parameters: FeaturesParameters,
    postprocess_parameters: PostprocessParameters,
) -> DataCube:
    """Method to produce cropland map from preprocessed inputs, using
    a Presto feature extractor and a CatBoost classifier.

    Parameters
    ----------
    inputs : DataCube
        preprocessed input cube
    temporal_extent : TemporalContext
        temporal extent of the input cube
    cropland_parameters: CropLandParameters
        Parameters for the cropland product inference pipeline
    postprocess_parameters: PostprocessParameters
        Parameters for the postprocessing
    Returns
    -------
    DataCube
        binary labels and probability
    """
    
    # Run model inference on inputs
    classes = apply_model_inference(
        model_inference_class=cropland_parameters.classifier,
        cube=inputs,
        parameters=parameters,
        size=[
            {"dimension": "x", "unit": "px", "value": 128},
            {"dimension": "y", "unit": "px", "value": 128},
            {"dimension": "t", "value": "P1D"},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 0},
            {"dimension": "y", "unit": "px", "value": 0},
        ],
    )

    # Get rid of temporal dimension
    classes = classes.reduce_dimension(dimension="t", reducer="mean")

    # Postprocess
    if postprocess_parameters.enable:
        if postprocess_parameters.save_intermediate:
            classes = classes.save_result(
                format="GTiff",
                options=dict(
                    filename_prefix=f"{WorldCerealProductType.CROPLAND.value}-raw_{temporal_extent.start_date}_{temporal_extent.end_date}"
                ),
            )
        classes = _postprocess(classes, postprocess_parameters, lookup_table)

    # Cast to uint8
    classes = compress_uint8(classes)

    return classes