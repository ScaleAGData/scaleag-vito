"""Common functions used by extraction scripts."""

import logging
import os
from tempfile import NamedTemporaryFile

import geojson
import geopandas as gpd
import pandas as pd
import requests

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
