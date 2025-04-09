from typing import Optional

from openeo import DataCube
from openeo_gfmap import TemporalContext
from openeo_gfmap.inference.model_inference import apply_model_inference
from openeo_gfmap.preprocessing.scaling import compress_uint8, compress_uint16

from scaleagdata_vito.openeo.parameters import (
    PostprocessParameters,
    ScaleAgParametersBinary,
    ScaleAgParametersMultiClass,
    ScaleAgParametersRegression,
    ScaleAgProductType,
)


def _regression_map(
    inputs: DataCube,
    temporal_extent: TemporalContext,
    parameters: ScaleAgParametersRegression,
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
    parameters: ScaleAgParametersRegression
        Parameters for the regression product inference pipeline
    postprocess_parameters: PostprocessParameters
        Parameters for the postprocessing
    Returns
    -------
    DataCube
        binary labels and probability
    """

    # Run feature computer

    regression = apply_model_inference(
        model_inference_class=parameters.classifier,
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
    regression = regression.reduce_dimension(dimension="t", reducer="mean")

    # Postprocess
    if postprocess_parameters.enable:
        if postprocess_parameters.save_intermediate:
            classes = classes.save_result(
                format="GTiff",
                options=dict(
                    filename_prefix=f"{ScaleAgProductType.REGRESSION.value}-raw_{temporal_extent.start_date}_{temporal_extent.end_date}"
                ),
            )
        regression = _postprocess(regression, postprocess_parameters)

    return regression


def _binary_map(
    inputs: DataCube,
    temporal_extent: TemporalContext,
    parameters: ScaleAgParametersBinary,
    postprocess_parameters: PostprocessParameters,
) -> DataCube:
    """Method to produce croptype map from preprocessed inputs, using
    a Presto feature extractor and a CatBoost classifier.

    Parameters
    ----------
    inputs : DataCube
        preprocessed input cube
    temporal_extent : TemporalContext
        temporal extent of the input cube
    parameters: ScaleAgParametersBinary
        Parameters for the binary product inference pipeline
    postprocess_parameters: PostprocessParameters
        Parameters for the postprocessing
        
    Returns
    -------
    DataCube
        croptype labels and probability
    """

    classes = apply_model_inference(
        model_inference_class=parameters.classifier,
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
                    filename_prefix=f"{ScaleAgProductType.BINARY.value}-raw_{temporal_extent.start_date}_{temporal_extent.end_date}"
                ),
            )
        classes = _postprocess(
            classes,
            postprocess_parameters,
        )

    # Cast to uint16
    classes = compress_uint8(classes)

    return classes

def _multiclass_map(
    inputs: DataCube,
    temporal_extent: TemporalContext,
    parameters: ScaleAgParametersMultiClass,
    postprocess_parameters: PostprocessParameters,
    lookup_table: Optional[dict] = None,
) -> DataCube:
    """Method to produce croptype map from preprocessed inputs, using
    a Presto feature extractor and a CatBoost classifier.

    Parameters
    ----------
    inputs : DataCube
        preprocessed input cube
    temporal_extent : TemporalContext
        temporal extent of the input cube
    parameters: ScaleAgParametersMultiClass
        Parameters for the multiclass product inference pipeline
    postprocess_parameters: PostprocessParameters
        Parameters for the postprocessing
    lookup_table: dict
        Mapping of class names to class labels, ordered by model output.

    Returns
    -------
    DataCube
        croptype labels and probability
    """

    classes = apply_model_inference(
        model_inference_class=parameters.classifier,
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
    
    parameters.update({"lookup_table": lookup_table})

    # Get rid of temporal dimension
    classes = classes.reduce_dimension(dimension="t", reducer="mean")

    # Postprocess
    if postprocess_parameters.enable:
        if postprocess_parameters.save_intermediate:
            classes = classes.save_result(
                format="GTiff",
                options=dict(
                    filename_prefix=f"{ScaleAgProductType.BINARY.value}-raw_{temporal_extent.start_date}_{temporal_extent.end_date}"
                ),
            )
        classes = _postprocess(
            classes,
            postprocess_parameters,
            lookup_table=lookup_table,
        )

    # Cast to uint16
    classes = compress_uint16(classes)

    return classes


def _postprocess(
    classes: DataCube,
    postprocess_parameters: "PostprocessParameters",
    lookup_table: Optional[dict],
) -> DataCube:
    """Method to postprocess the classes.

    Parameters
    ----------
    classes : DataCube
        classes to postprocess
    postprocess_parameters : PostprocessParameters
        parameter class for postprocessing
    lookup_table: dict
        Mapping of class names to class labels, ordered by model output.
    Returns
    -------
    DataCube
        postprocessed classes
    """

    # Run postprocessing on the raw classification output
    # Note that this uses the `apply_model_inference` method even though
    # it is not truly model inference
    parameters = postprocess_parameters.model_dump(exclude=["postprocessor"])
    parameters.update({"lookup_table": lookup_table})

    postprocessed_classes = apply_model_inference(
        model_inference_class=postprocess_parameters.postprocessor,
        cube=classes,
        parameters=parameters,
        size=[
            {"dimension": "x", "unit": "px", "value": 128},
            {"dimension": "y", "unit": "px", "value": 128},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 0},
            {"dimension": "y", "unit": "px", "value": 0},
        ],
    )

    return postprocessed_classes
