"""Main script to perform extractions. Each collection has it's specifities and
own functions, but the setup and main thread execution is done here."""

import argparse
from pathlib import Path

from openeo_gfmap import FetchType

from scaleagdata_vito.openeo.extract_sample_scaleag import ExtractionCollection, extract

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from a collection")
    parser.add_argument(
        "-output_folder", type=Path, help="The folder where to store the extracted data"
    )
    parser.add_argument(
        "--input_df", type=Path, help="The input dataframe with the data to extract"
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
        default=None,
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--spatial_extent",
        type=str,
        default=None,
        help="The spatial extent to extract in the inference routine. If not provided, the extent of the input dataframe will be used.",
    )
    parser.add_argument(
        "--composite_window",
        type=str,
        choices=["dekad", "month"],
        default="dekad",
        help="The temporal resolution of the data to extract. can be either month or dekad",
    )
    parser.add_argument(
        "--max_locations",
        type=int,
        default=50,
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
        default=10,
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
        default="",
        help="The column contianing the unique sample IDs",
    )
    parser.add_argument(
        "--routine",
        type=str,
        choices=["training", "inference"],
        default="training",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["NetCDF", "Parquet"],
        default="geoparquet",
        help="The format to store the extracted data.",
    )
    parser.add_argument(
        "--fetch_type",
        type=FetchType,
        choices=list(FetchType),
        default=FetchType.POINT,
        help="The type of data fetching to use.",
    )
    args = parser.parse_args()

    import json

    import pandas as pd

    input_filename = Path(
        "/data/users/Private/giorgia/git/GEOMaize/data/inference/inference_extent_50km_latlon.geojson"
    )
    args = pd.Series(
        dict(
            collection=ExtractionCollection.SAMPLE_SCALEAG,
            output_folder=Path(
                f"/data/users/Private/giorgia/git/GEOMaize/data/inference/month/ref_id={input_filename.stem}/"
            ),
            input_df=input_filename,
            start_date="2025-07-01",
            end_date="2025-11-30",
            unique_id_column="id",
            composite_window="month",
            max_locations=250,
            memory="1800m",
            executor_memory="3G",
            python_memory="3G",
            max_executors=22,
            parallel_jobs=10,
            soft_errors=0.1,
            restart_failed=False,
            output_format="netCDF",
            fetch_type=FetchType.TILE,
        )
    )
    output_json_path = args.output_folder / "extraction_args.json"
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        args_str = args.apply(str)
        json.dump(args_str.to_dict(), f, indent=4)
    extract(args)
