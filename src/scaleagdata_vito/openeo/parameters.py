from enum import Enum
from typing import Type

from openeo_gfmap.features.feature_extractor import PatchFeatureExtractor
from openeo_gfmap.inference.model_inference import ModelInference
from pydantic import BaseModel, Field, ValidationError, model_validator

from scaleagdata_vito.openeo.feature_extractor import PrestoFeatureExtractor
from scaleagdata_vito.openeo.inference import (
    ScaleAgBinaryClassifier,
    ScaleAgMulticlassClassifier,
    ScaleAgRegressor,
)
from scaleagdata_vito.openeo.postporcess import PostProcessor


class ScaleAgProductType(Enum):
    """Enum to define the different ScaleAg products."""

    MULTICLASS = "multiclass"
    REGRESSION = "regression"
    BINARY = "binary"


class ScaleAgParameters(BaseModel):
    """Parameters for the ScaleAg product inference pipeline. Types are enforced by Pydantic.

    Attributes
    ----------
    presto_url : str
        Public URL to the classifier model. The file should be an ONNX model.
    """
    
    presto_url: str


class ScaleAgParametersRegression(BaseModel):
    """Parameters for the ScaleAg product inference pipeline. Types are enforced by Pydantic.

    Attributes
    ----------
    presto_url : str
        Public URL to the Presto model. The file should be an ONNX model.
    feature_extractor : Type[PatchFeatureExtractor]
        Feature extractor to use for the inference. This class must be a
        subclass of GFMAP's `PatchFeatureExtractor` and returns float32
        features.
    classifier : Type[ModelInference]
        Classifier to use for the inference. This class must be a subclass of
        GFMAP's `ModelInference` and returns predictions/probabilities for
        regression task.
    """

    presto_url: str
    feature_extractor: Type[PatchFeatureExtractor] = Field(default=PrestoFeatureExtractor)
    classifier: Type[ModelInference] = Field(default=ScaleAgRegressor)
    
    @model_validator(mode="after")
    def check_udf_types(self):
        """Validates the FeatureExtractor and Classifier classes."""
        if not issubclass(self.feature_extractor, PatchFeatureExtractor):
            raise ValidationError(
                f"Feature extractor must be a subclass of PatchFeatureExtractor, got {self.feature_extractor}"
            )
        if not issubclass(self.classifier, ModelInference):
            raise ValidationError(
                f"Classifier must be a subclass of ModelInference, got {self.classifier}"
            )


class ScaleAgParametersBinary(BaseModel):
    """Parameters for the ScaleAg product inference pipeline. Types are enforced by Pydantic.

    Attributes
    ----------
    presto_url : str
        Public URL to the Presto model. The file should be an ONNX model.
    feature_extractor : Type[PatchFeatureExtractor]
        Feature extractor to use for the inference. This class must be a
        subclass of GFMAP's `PatchFeatureExtractor` and returns float32
        features.
    classifier : Type[ModelInference]
        Classifier to use for the inference. This class must be a subclass of
        GFMAP's `ModelInference` and returns predictions/probabilities for
        binary task.
    """
    presto_url: str
    feature_extractor: Type[PatchFeatureExtractor] = Field(default=PrestoFeatureExtractor)
    classifier: Type[ModelInference] = Field(default=ScaleAgBinaryClassifier)
    
    @model_validator(mode="after")
    def check_udf_types(self):
        """Validates the FeatureExtractor and Classifier classes."""
        if not issubclass(self.feature_extractor, PatchFeatureExtractor):
            raise ValidationError(
                f"Feature extractor must be a subclass of PatchFeatureExtractor, got {self.feature_extractor}"
            )
        if not issubclass(self.classifier, ModelInference):
            raise ValidationError(
                f"Classifier must be a subclass of ModelInference, got {self.classifier}"
            )
    
    
class ScaleAgParametersMultiClass(BaseModel):
    """Parameters for the ScaleAg product inference pipeline. Types are enforced by Pydantic.
    Attributes
    ----------
    presto_url : str
        Public URL to the Presto model. The file should be an ONNX model.
    feature_extractor : Type[PatchFeatureExtractor]
        Feature extractor to use for the inference. This class must be a
        subclass of GFMAP's `PatchFeatureExtractor` and returns float32
        features.
    classifier : Type[ModelInference]
        Classifier to use for the inference. This class must be a subclass of
        GFMAP's `ModelInference` and returns predictions/probabilities for
        multiclass task.
    
    """
    
    presto_url: str
    feature_extractor: Type[PatchFeatureExtractor] = Field(default=PrestoFeatureExtractor)
    classifier: Type[ModelInference] = Field(default=ScaleAgMulticlassClassifier)
    
    @model_validator(mode="after")
    def check_udf_types(self):
        """Validates the FeatureExtractor and Classifier classes."""
        if not issubclass(self.feature_extractor, PatchFeatureExtractor):
            raise ValidationError(
                f"Feature extractor must be a subclass of PatchFeatureExtractor, got {self.feature_extractor}"
            )
        if not issubclass(self.classifier, ModelInference):
            raise ValidationError(
                f"Classifier must be a subclass of ModelInference, got {self.classifier}"
            )


class PostprocessParameters(BaseModel):
    """Parameters for postprocessing. Types are enforced by Pydantic.

    Attributes
    ----------
    enable: bool (default=True)
        Whether to enable postprocessing.
    method: str (default="smooth_probabilities")
        The method to use for postprocessing. Must be one of ["smooth_probabilities", "majority_vote"]
    kernel_size: int (default=5)
        Used for majority vote postprocessing. Must be an odd number, larger than 1 and smaller than 25.
    save_intermediate: bool (default=False)
        Whether to save intermediate results (before applying the postprocessing).
        The intermediate results will be saved in the GeoTiff format.
    keep_class_probs: bool (default=False)
        If the per-class probabilities should be outputted in the final product.
    """

    enable: bool = Field(default=True)
    method: str = Field(default="smooth_probabilities")
    kernel_size: int = Field(default=5)
    save_intermediate: bool = Field(default=False)
    keep_class_probs: bool = Field(default=False)

    postprocessor: Type[ModelInference] = Field(default=PostProcessor)

    @model_validator(mode="after")
    def check_udf_types(self):
        """Validates the PostProcessor class."""
        if not issubclass(self.postprocessor, ModelInference):
            raise ValidationError(
                f"Postprocessor must be a subclass of PostProcessor, got {self.postprocessor}"
            )
        return self

    @model_validator(mode="after")
    def check_parameters(self):
        """Validates parameters."""
        if not self.enable and self.save_intermediate:
            raise ValueError(
                "Cannot save intermediate results if postprocessing is disabled."
            )

        if self.method not in ["smooth_probabilities", "majority_vote"]:
            raise ValueError(
                f"Method must be one of ['smooth_probabilities', 'majority_vote'], got {self.method}"
            )

        if self.method == "majority_vote":
            if self.kernel_size % 2 == 0:
                raise ValueError(
                    f"Kernel size for majority filtering should be an odd number, got {self.kernel_size}"
                )
            if self.kernel_size > 25:
                raise ValueError(
                    f"Kernel size for majority filtering should be an odd number smaller than 25, got {self.kernel_size}"
                )
            if self.kernel_size < 3:
                raise ValueError(
                    f"Kernel size for majority filtering should be an odd number larger than 1, got {self.kernel_size}"
                )

        return self
