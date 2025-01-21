"""Extract S1, S2, METEO and DEM point data using OpenEO-GFMAP package."""

import logging
import os
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional

import geojson
import geopandas as gpd
import openeo
import pandas as pd
import pystac
import requests
from openeo_gfmap import Backend, BackendContext, FetchType, TemporalContext
from tqdm import tqdm

from scaleagdata_vito.openeo.preprocessing import scaleag_preprocessed_inputs

# Logger used for the pipeline
pipeline_log = logging.getLogger("extraction_pipeline")

pipeline_log.setLevel(level=logging.INFO)

stream_handler = logging.StreamHandler()
pipeline_log.addHandler(stream_handler)

formatter = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s:  %(message)s")
stream_handler.setFormatter(formatter)


# Exclude the other loggers from other libraries
class ManagerLoggerFilter(logging.Filter):
    """Filter to only accept the OpenEO-GFMAP manager logs."""

    def filter(self, record):
        return record.name in [pipeline_log.name]


stream_handler.addFilter(ManagerLoggerFilter())


def upload_geoparquet_artifactory(
    gdf: gpd.GeoDataFrame, name: str, collection: str = ""
) -> str:
    """Upload the given GeoDataFrame to artifactory and return the URL of the
    uploaded file. Necessary as a workaround for Polygon sampling in OpenEO
    using custom CRS.
    """
    # Save the dataframe as geoparquet to upload it to artifactory
    temporary_file = NamedTemporaryFile()
    gdf.to_parquet(temporary_file.name)

    artifactory_username = os.getenv("ARTIFACTORY_USERNAME")
    artifactory_password = os.getenv("ARTIFACTORY_PASSWORD")

    if not artifactory_username or not artifactory_password:
        raise ValueError(
            "Artifactory credentials not found. Please set ARTIFACTORY_USERNAME and ARTIFACTORY_PASSWORD."
        )

    headers = {"Content-Type": "application/octet-stream"}

    upload_url = f"https://artifactory.vgt.vito.be/artifactory/auxdata-public/gfmap-temp/openeogfmap_dataframe_{collection}{name}.parquet"

    with open(temporary_file.name, "rb") as f:
        response = requests.put(
            upload_url,
            headers=headers,
            data=f,
            auth=(artifactory_username, artifactory_password),
            timeout=180,
        )

    response.raise_for_status()

    return upload_url


def get_job_nb_polygons(row: pd.Series) -> int:
    """Get the number of polygons in the geometry."""
    return len(
        list(
            filter(
                lambda feat: feat.properties.get("extract"),
                geojson.loads(row.geometry)["features"],
            )
        )
    )


def generate_output_path_sample_scaleag(
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


def create_job_dataframe_sample_scaleag(
    backend: Backend,
    split_jobs: List[gpd.GeoDataFrame],
    start_date_user: str,
    end_date_user: str,
) -> pd.DataFrame:
    """Create a dataframe from the split jobs, containg all the necessary information to run the job."""
    rows = []

    for job in tqdm(split_jobs):
        s2_tile = job.tile.iloc[0]

        job["lat"] = job.geometry.centroid.y
        job["lon"] = job.geometry.centroid.x

        if "date" in job.columns:
            job.rename(columns={"date": "original_date"}, inplace=True)
            job["original_date"] = pd.to_datetime(job["original_date"])

            min_time = job.original_date.min()
            max_time = job.original_date.max()

            # 9 months before and after the valid time
            start_date = min_time - pd.Timedelta(days=275)
            end_date = max_time + pd.Timedelta(days=275)
        else:
            start_date = datetime.strptime(start_date_user, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_user, "%Y-%m-%d")

        # ensure start date is 1st day of month, end date is last day of month
        start_date = start_date.replace(day=1)
        end_date = end_date.replace(day=1) + pd.offsets.MonthEnd(0)

        # Convert dates to string format
        start_date, end_date = start_date.strftime("%Y-%m-%d"), end_date.strftime(
            "%Y-%m-%d"
        )

        job["start_date"] = start_date
        job["end_date"] = end_date

        # # set again as string so that it is json serializable
        job["original_date"] = job.original_date.dt.strftime("%Y-%m-%d")

        variables = {
            "backend_name": backend.value,
            "out_prefix": "geometry-extraction",
            "out_extension": ".geoparquet",
            "start_date": start_date,
            "end_date": end_date,
            "s2_tile": s2_tile,
            "geometry": job.to_json(),
        }

        rows.append(pd.Series(variables))

    return pd.DataFrame(rows)


def create_job_sample_scaleag(
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


def post_job_action_sample_scaleag(
    job_items: List[pystac.Item],
    row: pd.Series,
    parameters: Optional[dict] = None,
) -> list:
    for idx, item in enumerate(job_items):
        item_asset_path = Path(list(item.assets.values())[0].href)

        gdf = gpd.read_parquet(item_asset_path)

        # Convert the dates to datetime format
        gdf["timestamp"] = pd.to_datetime(gdf["date"])
        gdf.drop(columns=["date"], inplace=True)

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
            "slope",
            "AGERA5-PRECIP",
            "AGERA5-TMEAN",
        ]
        gdf[bands] = gdf[bands].fillna(65535).astype("uint16")

        gdf.to_parquet(item_asset_path, index=False)

    return job_items


def generate_extraction_job_command(
    job_params, extraction_script_path="../scripts/extractions/extract.py"
):
    command = ["python", extraction_script_path]

    for key, value in job_params.items():
        if isinstance(value, bool):
            if value:  # Add flag if True
                command.append(f"--{key}")
        elif value is not None:  # Skip None values
            if key in ["output_folder", "input_df"]:
                command.extend([f"-{key}", str(value)])
            else:
                command.extend([f"--{key}", str(value)])
    return command
