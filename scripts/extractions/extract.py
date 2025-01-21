"""Main script to perform extractions. Each collection has it's specifities and
own functions, but the setup and main thread execution is done here."""

import argparse
import typing
from datetime import datetime
from enum import Enum
from functools import partial
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from openeo.rest import OpenEoApiError, OpenEoApiPlainError, OpenEoRestError
from openeo_gfmap import Backend
from openeo_gfmap.backend import cdse_connection
from openeo_gfmap.manager.job_manager import GFMAPJobManager
from openeo_gfmap.manager.job_splitters import split_job_s2grid

from scaleagdata_vito.openeo.extract import pipeline_log
from scaleagdata_vito.openeo.extract_sample_scaleag import (
    create_job_dataframe_sample_scaleag,
    create_job_sample_scaleag,
    generate_output_path_sample_scaleag,
    post_job_action_sample_scaleag,
)


class ExtractionCollection(Enum):
    """Collections that can be extracted in the extraction scripts."""

    SAMPLE_SCALEAG = "SAMPLE_SCALEAG"


def load_dataframe(df_path: Path) -> gpd.GeoDataFrame:
    """Load the input dataframe from the given path."""
    pipeline_log.info("Loading input dataframe from %s.", df_path)

    if df_path.name.endswith(".geoparquet"):
        return gpd.read_parquet(df_path)
    else:
        return gpd.read_file(df_path)


def prepare_job_dataframe(
    input_df: gpd.GeoDataFrame,
    collection: ExtractionCollection,
    max_locations: int,
    backend: Backend,
    start_date: str,
    end_date: str,
) -> gpd.GeoDataFrame:
    """Prepare the job dataframe to extract the data from the given input
    dataframe."""
    pipeline_log.info("Preparing the job dataframe.")
    pipeline_log.info("Performing splitting by s2 grid...")
    split_dfs = split_job_s2grid(input_df, max_points=max_locations)

    pipeline_log.info("Dataframes split to jobs, creating the job dataframe...")
    collection_switch: dict[ExtractionCollection, typing.Callable] = {
        ExtractionCollection.SAMPLE_SCALEAG: create_job_dataframe_sample_scaleag,
    }

    create_job_dataframe_fn = collection_switch.get(
        collection,
        lambda: (_ for _ in ()).throw(
            ValueError(f"Collection {collection} not supported.")
        ),
    )

    job_df = create_job_dataframe_fn(backend, split_dfs, start_date, end_date)
    pipeline_log.info("Job dataframe created with %s jobs.", len(job_df))

    return job_df


def setup_extraction_functions(
    collection: ExtractionCollection,
    memory: str,
    python_memory: str,
    max_executors: int,
) -> tuple[typing.Callable, typing.Callable, typing.Callable]:
    """Setup the datacube creation, path generation and post-job action
    functions for the given collection. Returns a tuple of three functions:
    1. The datacube creation function
    2. The output path generation function
    3. The post-job action function
    """

    datacube_creation = {
        ExtractionCollection.SAMPLE_SCALEAG: partial(
            create_job_sample_scaleag,
            executor_memory=memory,
            python_memory=python_memory,
            max_executors=max_executors,
        ),
    }

    datacube_fn = datacube_creation.get(
        collection,
        lambda: (_ for _ in ()).throw(
            ValueError(f"Collection {collection} not supported.")
        ),
    )

    path_fns = {
        ExtractionCollection.SAMPLE_SCALEAG: partial(
            generate_output_path_sample_scaleag
        ),
    }

    path_fn = path_fns.get(
        collection,
        lambda: (_ for _ in ()).throw(
            ValueError(f"Collection {collection} not supported.")
        ),
    )

    post_job_actions = {
        ExtractionCollection.SAMPLE_SCALEAG: partial(
            post_job_action_sample_scaleag,
        ),
    }

    post_job_fn = post_job_actions.get(
        collection,
        lambda: (_ for _ in ()).throw(
            ValueError(f"Collection {collection} not supported.")
        ),
    )

    return datacube_fn, path_fn, post_job_fn


def manager_main_loop(
    manager: GFMAPJobManager,
    collection: ExtractionCollection,
    job_df: gpd.GeoDataFrame,
    datacube_fn: typing.Callable,
    tracking_df_path: Path,
) -> None:
    """Main loop for the job manager, re-running it whenever an uncatched
    OpenEO exception occurs, and notifying the user through the Pushover API
    whenever the extraction start or an error occurs.
    """
    latest_exception_time = None
    exception_counter = 0

    while True:
        pipeline_log.info("Launching the jobs manager.")
        try:
            manager.run_jobs(job_df, datacube_fn, tracking_df_path)
            return
        except (
            OpenEoApiPlainError,
            OpenEoApiError,
            OpenEoRestError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.HTTPError,
        ) as e:
            pipeline_log.exception("An error occurred during the extraction.\n%s", e)
            if latest_exception_time is None:
                latest_exception_time = pd.Timestamp.now()
                exception_counter += 1
            # 30 minutes between each exception
            elif (datetime.now() - latest_exception_time).seconds < 1800:
                exception_counter += 1
            else:
                latest_exception_time = None
                exception_counter = 0

            if exception_counter >= 3:
                pipeline_log.error(
                    "Too many OpenEO exceptions occurred in a short amount of time, stopping the extraction..."
                )
                raise e
        except Exception as e:
            pipeline_log.exception(
                "An unexpected error occurred during the extraction.\n%s", e
            )
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from a collection")
    parser.add_argument(
        "-output_folder", type=Path, help="The folder where to store the extracted data"
    )
    parser.add_argument(
        "-input_df", type=Path, help="The input dataframe with the data to extract"
    )
    parser.add_argument(
        "--collection",
        type=ExtractionCollection,
        choices=list(ExtractionCollection),
        default=ExtractionCollection.SAMPLE_SCALEAG,
        help="The collection to extract",
    )
    parser.add_argument(
        "--start_date",
        type=str,
    )
    parser.add_argument(
        "--end_date",
        type=str,
    )
    parser.add_argument(
        "--max_locations",
        type=int,
        default=500,
        help="The maximum number of locations to extract per job",
    )
    parser.add_argument(
        "--memory",
        type=str,
        default="1800m",
        help="Memory to allocate for the executor.",
    )
    parser.add_argument(
        "--python_memory",
        type=str,
        default="1900m",
        help="Memory to allocate for the python processes as well as OrfeoToolbox in the executors.",
    )
    parser.add_argument(
        "--max_executors", type=int, default=22, help="Number of executors to run."
    )
    parser.add_argument(
        "--parallel_jobs",
        type=int,
        default=2,
        help="The maximum number of parrallel jobs to run at the same time.",
    )
    parser.add_argument(
        "--restart_failed",
        action="store_true",
        help="Restart the jobs that previously failed.",
    )
    parser.add_argument(
        "--unique_id_column",
        type=str,
        default="id",
        help="The column contianing the unique sample IDs",
    )
    args = parser.parse_args()

    # Fetches values and setups hardocded values
    collection = args.collection
    max_locations_per_job = args.max_locations
    backend = Backend.CDSE

    if not args.output_folder.is_dir():
        args.output_folder.mkdir(parents=True, exist_ok=True)
        # raise ValueError(f"Output folder {args.output_folder} does not exist.")

    tracking_df_path = Path(args.output_folder) / "job_tracking.csv"

    # Load the input dataframe and build the job dataframe
    input_df = load_dataframe(args.input_df)
    input_df["sample_id"] = input_df[args.unique_id_column]
    assert input_df["sample_id"].is_unique, "The unique ID column is not unique."

    job_df = None
    if not tracking_df_path.exists():
        job_df = prepare_job_dataframe(
            input_df,
            collection,
            max_locations_per_job,
            backend,
            args.start_date,
            args.end_date,
        )

    # Setup the extraction functions
    pipeline_log.info("Setting up the extraction functions.")
    datacube_fn, path_fn, post_job_fn = setup_extraction_functions(
        collection,
        args.memory,
        args.python_memory,
        args.max_executors,
    )

    # Initialize and setups the job manager
    pipeline_log.info("Initializing the job manager.")

    job_manager = GFMAPJobManager(
        output_dir=args.output_folder,
        output_path_generator=path_fn,
        post_job_action=post_job_fn,
        poll_sleep=60,
        n_threads=4,
        restart_failed=args.restart_failed,
        stac_enabled=False,
    )

    job_manager.add_backend(
        Backend.CDSE.value,
        cdse_connection,
        parallel_jobs=args.parallel_jobs,
    )

    manager_main_loop(job_manager, collection, job_df, datacube_fn, tracking_df_path)

    pipeline_log.info("Extraction completed successfully.")
