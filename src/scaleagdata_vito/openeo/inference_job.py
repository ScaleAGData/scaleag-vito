"""Executing inference jobs on the OpenEO backend."""

import json
from pathlib import Path
from typing import Dict, Optional, Union

from openeo_gfmap import Backend, BackendContext, BoundingBoxExtent, TemporalContext
from openeo_gfmap.backend import BACKEND_CONNECTIONS
from pydantic import BaseModel
from typing_extensions import TypedDict

from scaleagdata_vito.openeo.mapping import (
    _binary_map,
    _multiclass_map,
    _regression_map,
)
from scaleagdata_vito.openeo.parameters import (
    PostprocessParameters,
    ScaleAgParametersBinary,
    ScaleAgParametersMultiClass,
    ScaleAgParametersRegression,
    ScaleAgProductType,
)
from scaleagdata_vito.openeo.preprocessing import scaleag_preprocessed_inputs

ONNX_DEPS_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/openeo/onnx_dependencies_1.16.3.zip"


class ScaleAgProduct(TypedDict):
    """Dataclass representing a ScaleAgData inference product.

    Attributes
    ----------
    url: str
        URL to the product.
    type: ScaleAgProductType
        Type of the product..
    temporal_extent: TemporalContext
        Period of time for which the product has been generated (binary, regression, multiclass).
    path: Optional[Path]
        Path to the downloaded product.
    lut: Optional[Dict]
        Look-up table for the product.
    """

    url: str
    type: ScaleAgProductType
    temporal_extent: TemporalContext
    path: Optional[Path]
    lut: Optional[Dict]


class InferenceResults(BaseModel):
    """Dataclass to store the results of the ScaleAgData job.

    Attributes
    ----------
    job_id : str
        Job ID of the finished OpenEO job.
    products: Dict[str, ScaleAgProduct]
        Dictionary with the different products.
    metadata: Optional[Path]
        Path to metadata file, if it was downloaded locally.
    """

    job_id: str
    products: Dict[str, ScaleAgProduct]
    metadata: Optional[Path]


def generate_map(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    product_type: ScaleAgProductType,
    output_dir: Optional[Union[Path, str]] = None,
    regression_parameters: ScaleAgParametersRegression = ScaleAgParametersRegression(),
    multiclass_parameters: ScaleAgParametersMultiClass = ScaleAgParametersMultiClass(),
    binary_parameters: ScaleAgParametersBinary = ScaleAgParametersBinary(),
    postprocess_parameters: PostprocessParameters = PostprocessParameters(),
    out_format: str = "GTiff",
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    tile_size: Optional[int] = 128,
    job_options: Optional[dict] = None,
    luts: Optional[Dict[str, Dict]] = None,
) -> InferenceResults:
    """Main function to generate a ScaleAgData product.

    Parameters
    ----------
    spatial_extent : BoundingBoxExtent
        spatial extent of the map
    temporal_extent : TemporalContext
        temporal range to consider
    product_type : ScaleAgProductType
        Type of the product to generate (binary, regression, multiclass).
    output_dir : Optional[Union[Path, str]], optional
        output path to download the product to, by default None
    regression_parameters : ScaleAgParametersRegression, optional
        Parameters for the regression model, by default ScaleAgParametersRegression()
    multiclass_parameters : ScaleAgParametersMultiClass, optional
        Parameters for the multiclass model, by default ScaleAgParametersMultiClass()
    binary_parameters : ScaleAgParametersBinary, optional
        Parameters for the binary model, by default ScaleAgParametersBinary()
    postprocess_parameters : PostprocessParameters, optional
        Parameters for the postprocessing, by default PostprocessParameters()
    out_format : str, optional
        Output format of the product, by default "GTiff"
    backend_context : BackendContext, optional
        backend to run the job on, by default CDSE
    tile_size: int, optional
        Tile size to use for the data loading in OpenEO, by default 128
        so it uses the OpenEO default setting.
    job_options: dict, optional
        Additional job options to pass to the OpenEO backend, by default None
    luts: Optional[Dict[str, Dict]], optional
        Look-up tables for the products, by default None

    Returns
    -------
    InferenceResults
        Results of the finished ScaleAgData job.

    Raises
    ------
    ValueError
        if the product is not supported
    ValueError
        if the out_format is not supported
    """


    if out_format not in ["GTiff", "NetCDF"]:
        raise ValueError(f"Format {format} not supported.")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Make a connection to the OpenEO backend
    connection = BACKEND_CONNECTIONS[backend_context.backend]()

    # Preparing the input cube for inference
    inputs = scaleag_preprocessed_inputs(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        tile_size=tile_size,
    )

    # Explicit filtering again for bbox because of METEO low
    # resolution causing issues
    inputs = inputs.filter_bbox(dict(spatial_extent))

    # Construct the feature extraction and model inference pipeline
    if product_type == ScaleAgProductType.REGRESSION:
        classes = _regression_map(
            inputs,
            temporal_extent,
            parameters=regression_parameters,
            postprocess_parameters=postprocess_parameters,
        )

    elif product_type == ScaleAgProductType.MULTICLASS:
        classes = _multiclass_map(
            inputs,
            temporal_extent,
            parameters=multiclass_parameters,
            postprocess_parameters=postprocess_parameters,
        )
        
    elif product_type == ScaleAgProductType.BINARY:
        classes = _binary_map(
            inputs,
            temporal_extent,
            parameters=binary_parameters,
            postprocess_parameters=postprocess_parameters,
        )
    else:
        raise ValueError(
            f"Product type {product_type} not supported. "
            f"Please use one of {list(ScaleAgProductType)}"
        )

    # Submit the job
    JOB_OPTIONS = {
        "driver-memory": "4g",
        "executor-memory": "2g",
        "executor-memoryOverhead": "1g",
        "python-memory": "3g",
        "soft-errors": "true",
        "udf-dependency-archives": [f"{ONNX_DEPS_URL}#onnx_deps"],
    }
    if job_options is not None:
        JOB_OPTIONS.update(job_options)

    # Execute the job
    job = classes.execute_batch(
        out_format=out_format,
        job_options=JOB_OPTIONS,
        title=f"ScaleAgData [{product_type.value}] job",
        description="Job that performs end-to-end ScaleAgData inference",
        filename_prefix=f"{product_type.value}_{temporal_extent.start_date}_{temporal_extent.end_date}",
    )


    # Get job results
    job_result = job.get_results()

    # Get the products
    assets = job_result.get_assets()
    products = {}
    for asset in assets:
        asset_name = asset.name.split(".")[0].split("_")[0]
        asset_type = asset_name.split("-")[0]
        asset_type = getattr(ScaleAgProductType, asset_type.upper())
        if output_dir is not None:
            filepath = asset.download(target=output_dir)
        else:
            filepath = None

        products[asset_name] = {
            "url": asset.href,
            "type": asset_type,
            "temporal_extent": temporal_extent,
            "path": filepath,
            "lut": luts if luts is not None else luts[asset_type.value],
        }

    # Download job metadata if output path is provided
    if output_dir is not None:
        metadata_file = output_dir / "job-results.json"
        metadata_file.write_text(json.dumps(job_result.get_metadata()))
    else:
        metadata_file = None

    # Compile InferenceResults and return
    return InferenceResults(
        job_id=job.job_id, products=products, metadata=metadata_file
    )


def collect_inputs(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    output_path: Union[Path, str],
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    tile_size: Optional[int] = 128,
    job_options: Optional[dict] = None,
):
    """Function to retrieve preprocessed inputs that are being
    used in the generation of ScaleAgData products.

    Parameters
    ----------
    spatial_extent : BoundingBoxExtent
        spatial extent of the map
    temporal_extent : TemporalContext
        temporal range to consider
    output_path : Union[Path, str]
        output path to download the product to
    backend_context : BackendContext
        backend to run the job on, by default CDSE
    tile_size: int, optional
        Tile size to use for the data loading in OpenEO, by default 128
        so it uses the OpenEO default setting.
    job_options: dict, optional
        Additional job options to pass to the OpenEO backend, by default None
    """

    # Make a connection to the OpenEO backend
    connection = BACKEND_CONNECTIONS[backend_context.backend]()

    # Preparing the input cube for the inference
    inputs = scaleag_preprocessed_inputs(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        tile_size=tile_size,
    )

    JOB_OPTIONS = {
        "driver-memory": "4g",
        "executor-memory": "1g",
        "executor-memoryOverhead": "1g",
        "python-memory": "2g",
        "soft-errors": "true",
    }
    if job_options is not None:
        JOB_OPTIONS.update(job_options)

    inputs.execute_batch(
        outputfile=output_path,
        out_format="NetCDF",
        title="ScaleAg [collect_inputs] job",
        description="Job that collects inputs for ScaleAG inference",
        job_options=job_options,
    )
