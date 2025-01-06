"""Extract S1, S2, METEO and DEM point data using OpenEO-GFMAP package."""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import geojson
import geopandas as gpd
import openeo
import pandas as pd
import pystac
from openeo_gfmap import Backend, BackendContext, FetchType, TemporalContext
from tqdm import tqdm

from scaleagdata_vito.openeo.extract import get_job_nb_polygons, pipeline_log
from scaleagdata_vito.openeo.preprocessing import scaleag_preprocessed_inputs


def generate_output_path_geometry_scaleag(
    root_folder: Path,
    geometry_index: int,
    row: pd.Series,
    asset_id: Optional[str] = None,
):
    """
    For geometry extractions, only one asset (a geoparquet file) is generated per job.
    Therefore geometry_index is always 0.
    It has to be included in the function signature to be compatible with the GFMapJobManager.
    """

    s2_tile_id = row.s2_tile
    utm_zone = str(s2_tile_id[0:2])

    # Create the subfolder to store the output
    subfolder = root_folder / utm_zone / s2_tile_id
    subfolder.mkdir(parents=True, exist_ok=True)

    return (
        subfolder
        / f"SCALEAG_{root_folder.name}_{row.start_date}_{row.end_date}_{s2_tile_id}_{row.id}{row.out_extension}"
    )


def create_job_dataframe_geometry_scaleag(
    backend: Backend,
    split_jobs: List[gpd.GeoDataFrame],
    start_date_user: str,
    end_date_user: str,
) -> pd.DataFrame:
    """Create a dataframe from the split jobs, containg all the necessary information to run the job."""
    rows = []

    # ensure start date is 1st day of month, end date is last day of month
    start_date = datetime.strptime(start_date_user, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_user, "%Y-%m-%d")

    start_date = start_date.replace(day=1)
    end_date = end_date.replace(day=1) + pd.offsets.MonthEnd(0)

    for job in tqdm(split_jobs):
        s2_tile = job.tile.iloc[0]
        variables = {
            "backend_name": backend.value,
            "out_prefix": "geometry-extraction",
            "out_extension": ".geoparquet",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "s2_tile": s2_tile,
            "geometry": job.to_json(),
        }

        rows.append(pd.Series(variables))

    return pd.DataFrame(rows)


def create_job_geometry_scaleag(
    row: pd.Series,
    connection: openeo.DataCube,
    provider,
    connection_provider,
    executor_memory: str = "5G",
    python_memory: str = "2G",
    max_executors: int = 22,
):
    """Creates an OpenEO BatchJob from the given row information."""

    # Load the temporal and spatial extent
    temporal_extent = TemporalContext(row.start_date, row.end_date)

    # Get the feature collection containing the geometry to the job
    geometry = geojson.loads(row.geometry)
    assert isinstance(geometry, geojson.FeatureCollection)
    assert len(geometry.features) > 0, "No geometries with the extract flag found"

    # Backend name and fetching type
    backend = Backend(row.backend_name)
    backend_context = BackendContext(backend)

    # Try to get s2 tile ID to filter the collection
    if "s2_tile" in row:
        pipeline_log.debug(f"Extracting data for S2 tile {row.s2_tile}")
        s2_tile = row.s2_tile
    else:
        s2_tile = None

    inputs = scaleag_preprocessed_inputs(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=geometry,
        temporal_extent=temporal_extent,
        fetch_type=FetchType.POINT,
        s2_tile=s2_tile,
    )

    # Finally, create a vector cube based on the Point geometries
    cube = inputs.aggregate_spatial(geometries=geometry, reducer="mean")

    # Increase the memory of the jobs depending on the number of polygons to extract
    number_points = get_job_nb_polygons(row)
    if pipeline_log is not None:
        pipeline_log.debug("Number of polygons to extract %s", number_points)

    job_options = {
        "driver-memory": "2G",
        "driver-memoryOverhead": "2G",
        "driver-cores": "1",
        "executor-memory": executor_memory,
        "python-memory": python_memory,
        "executor-cores": "1",
        "max-executors": max_executors,
        "soft-errors": "true",
    }

    return cube.create_job(
        out_format="Parquet",
        title=f"ScaleAgData_Geometry_Extraction_{row.s2_tile}",
        job_options=job_options,
    )


def post_job_action_geometry_scaleag(
    job_items: List[pystac.Item],
    row: pd.Series,
    parameters: Optional[dict] = None,
) -> list:
    for idx, item in enumerate(job_items):
        item_asset_path = Path(list(item.assets.values())[0].href)

        gdf = gpd.read_parquet(item_asset_path)

        # Convert the dates to datetime format
        gdf["date"] = pd.to_datetime(gdf["date"])
        gdf["lat"] = gdf.geometry.centroid.y
        gdf["lon"] = gdf.geometry.centroid.x

        # # For each sample, add start and end date to the dataframe
        # # is there a better way to do this, as this is already done in the job creation?
        sample_ids = gdf["sample_id"].unique()

        for sample_id in sample_ids:
            sample = gdf[gdf["sample_id"] == sample_id]
            start_date = pd.to_datetime(sample["date"].min()).replace(day=1)
            end_date = pd.to_datetime(sample["date"].max()).replace(
                day=1
            ) + pd.offsets.MonthEnd(0)
            start_date, end_date = start_date.strftime("%Y-%m-%d"), end_date.strftime(
                "%Y-%m-%d"
            )
            gdf.loc[gdf["sample_id"] == sample_id, "start_date"] = start_date
            gdf.loc[gdf["sample_id"] == sample_id, "end_date"] = end_date

        # Convert band dtype to uint16 (temporary fix)
        # TODO: remove this step when the issue is fixed on the OpenEO backend
        bands = [
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
            "S1-SIGMA0-VH",
            "S1-SIGMA0-VV",
            "elevation",
            "AGERA5-PRECIP",
            "AGERA5-TMEAN",
        ]
        gdf[bands] = gdf[bands].fillna(65535).astype("uint16")

        gdf.to_parquet(item_asset_path, index=False)

    return job_items
