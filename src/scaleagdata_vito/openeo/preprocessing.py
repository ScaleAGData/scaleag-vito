import pathlib
import warnings
from typing import Optional, Union

import geojson
import xarray as xr
from openeo import Connection, DataCube
from openeo_gfmap import BackendContext, FetchType, SpatialContext, TemporalContext
from openeo_gfmap.preprocessing.compositing import mean_compositing, median_compositing
from openeo_gfmap.preprocessing.sar import compress_backscatter_uint16
from worldcereal.openeo.preprocessing import (
    raw_datacube_DEM,
    raw_datacube_S1,
    raw_datacube_S2,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


def precomposited_datacube_METEO(
    connection: Connection,
    spatial_extent: SpatialContext,
    temporal_extent: TemporalContext,
) -> DataCube:
    """Extract the precipitation and temperature AGERA5 data from a
    pre-composited and pre-processed collection. The data is stored in the
    CloudFerro S3 stoage, allowing faster access and processing from the CDSE
    backend.

    Limitations:
        - Only two bands are available: precipitation-flux and temperature-mean.
        - This function does not support fetching points or polygons, but only
          tiles.
    """
    temporal_extent = [temporal_extent.start_date, temporal_extent.end_date]
    spatial_extent = dict(spatial_extent)

    # Monthly composited METEO data
    cube = connection.load_stac(
        "https://s3.waw3-1.cloudferro.com/swift/v1/agera_10d/stac/collection.json",  ####
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=["precipitation-flux", "temperature-mean"],
    )
    cube.result_node().update_arguments(featureflags={"tilesize": 1})
    cube = cube.rename_labels(
        dimension="bands", target=["AGERA5-PRECIP", "AGERA5-TMEAN"]
    )

    return cube


def scaleag_preprocessed_inputs_gfmap(
    connection: Connection,
    backend_context: BackendContext,
    spatial_extent: SpatialContext,
    temporal_extent: TemporalContext,
    period: str = "dekad",
    fetch_type: Optional[FetchType] = FetchType.POINT,
    disable_meteo: bool = False,
    tile_size: Optional[int] = None,
) -> DataCube:
    """
    Preprocesses data during OpenEO extraction to prepare inputs for Presto.
    Args:
        connection (Connection): The connection to the backend.
        backend_context (BackendContext): The backend context.
        spatial_extent (SpatialContext): The spatial extent.
        temporal_extent (TemporalContext): The temporal extent.
        period (str, optional): The period for compositing. Defaults to "dekad".
        fetch_type (Optional[FetchType], optional): The fetch type. Defaults to FetchType.POINT.
        disable_meteo (bool, optional): Flag to disable meteorological data. Defaults to False.
        tile_size (Optional[int], optional): The tile size. Defaults to None.
    Returns:
        DataCube: The preprocessed data cube.
    """
    # Extraction of S2 from GFMAP
    s2_data = raw_datacube_S2(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=[
            "S2-L2A-B02",
            "S2-L2A-B03",
            "S2-L2A-B04",
            "S2-L2A-B05",
            "S2-L2A-B06",
            "S2-L2A-B07",
            "S2-L2A-B08",
            "S2-L2A-B8A",
            "S2-L2A-B11",
            "S2-L2A-B12",
        ],
        fetch_type=fetch_type,
        filter_tile=False,
        distance_to_cloud_flag=False,
        additional_masks_flag=False,
        apply_mask_flag=True,
        tile_size=tile_size,
    )

    s2_data = median_compositing(s2_data, period=period)

    # Cast to uint16
    s2_data = s2_data.linear_scale_range(0, 65534, 0, 65534)

    # Extraction of the S1 data
    # Decides on the orbit direction from the maximum overlapping area of
    # available products.
    s1_data = raw_datacube_S1(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=[
            "S1-SIGMA0-VH",
            "S1-SIGMA0-VV",
        ],
        fetch_type=fetch_type,
        target_resolution=10.0,  # Compute the backscatter at 20m resolution, then upsample nearest neighbor when merging cubes
        orbit_direction=None,  # Make the query on the catalogue for the best orbit
        tile_size=tile_size,
    )

    s1_data = mean_compositing(s1_data, period=period)
    s1_data = compress_backscatter_uint16(backend_context, s1_data)

    dem_data = raw_datacube_DEM(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent,
        fetch_type=fetch_type,
    )

    dem_data = dem_data.linear_scale_range(0, 65534, 0, 65534)

    data = s2_data.merge_cubes(s1_data)
    data = data.merge_cubes(dem_data)

    if not disable_meteo:
        meteo_data = precomposited_datacube_METEO(
            connection=connection,
            spatial_extent=spatial_extent,
            temporal_extent=temporal_extent,
        )
        data = data.merge_cubes(meteo_data)
    return data


def run_openeo_extraction_job(gdf, output_path, job_params):
    """
    Runs an OpenEO extraction job using the provided parameters.
    Args:
        gdf (GeoDataFrame): The GeoDataFrame containing the spatial extent.
        output_path (str): The path to save the output file.
        job_params (dict): A dictionary containing the job parameters.
    Returns:
        None
    """

    geometry_latlon = geojson.loads(gdf.to_json())
    inputs = scaleag_preprocessed_inputs_gfmap(
        connection=job_params["connection"],
        backend_context=job_params["backend_context"],
        spatial_extent=geometry_latlon,
        temporal_extent=job_params["temporal_extent"],
        fetch_type=job_params["fetch_type"],
        disable_meteo=job_params["disable_meteo"],
    )
    cube = inputs.aggregate_spatial(geometries=geometry_latlon, reducer="mean")

    job = cube.create_job(
        outputfile=output_path,
        out_format=job_params["out_format"],
        title=job_params["title"],
        job_options={
            "driver-memory": "4g",
            "executor-memoryOverhead": "4g",
            "soft-error": True,
        },
    )
    job.start_and_wait()
    job.download_result(output_path)


def merge_datacubes(dataset, subset):
    if isinstance(dataset, Union[str, pathlib.PosixPath]):
        dataset = xr.load_dataset(dataset)
    subset = xr.load_dataset(subset)
    return xr.concat([dataset, subset], dim="feature")
