from typing import Optional

from openeo import Connection, DataCube
from openeo_gfmap import BackendContext, FetchType, SpatialContext, TemporalContext
from openeo_gfmap.preprocessing.compositing import mean_compositing, median_compositing
from openeo_gfmap.preprocessing.sar import compress_backscatter_uint16
from worldcereal.openeo.preprocessing import (
    raw_datacube_DEM,
    raw_datacube_S1,
    raw_datacube_S2,
)


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
        "https://s3.waw3-1.cloudferro.com/swift/v1/agera/stac/collection.json",  ####
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
