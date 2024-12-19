import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Optional, cast

import pandas as pd
import torch
from presto.utils import (
    DEFAULT_SEED,
    config_dir,
    data_dir,
    device,
    initialize_logging,
    seed_everything,
)

from scaleagdata_vito.presto.presto_eval import ScaleAGEval
from scaleagdata_vito.presto.presto_utils import load_pretrained_model

logger = logging.getLogger("__main__")

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name", type=str, default="")
argparser.add_argument("--path_to_config", type=str, default="")
argparser.add_argument(
    "--output_dir",
    type=str,
    default="",
    help="Parent directory to save output to, <output_dir>/wandb/ "
    "and <output_dir>/output/ will be written to. "
    "Leave empty to use the directory you are running this file from.",
)
argparser.add_argument(
    "--val_per_n_steps", type=int, default=-1, help="If -1, val every epoch"
)
argparser.add_argument("--seed", type=int, default=DEFAULT_SEED)
argparser.add_argument("--num_workers", type=int, default=4)
argparser.add_argument("--wandb", dest="wandb", action="store_true")
argparser.add_argument("--wandb_org", type=str)
argparser.add_argument(
    "--train_file",
    type=str,
    default="worldcereal_presto_cropland_nointerp_V1_TRAIN.parquet",
)
argparser.add_argument(
    "--val_file",
    type=str,
    default="worldcereal_presto_cropland_nointerp_V2_VAL.parquet",
)
argparser.add_argument("--dekadal", dest="dekadal", action="store_true")
argparser.add_argument("--ss_dekadal", dest="ss_dekadal", action="store_true")
argparser.add_argument("--finetuned", dest="finetuned", action="store_true")
argparser.add_argument("--data_dir", type=str, default="")
argparser.add_argument(
    "--path_to_weigths",
    type=str,
    default="",  # "/home/vito/millig/gio/models/presto_ss/monthly/output/2024_05_14_18_07_41_787101_szn0tnsi/models/28.pt",
)  # WC ss 30d Presto
argparser.add_argument("--target_name", type=str, default="median_yield")
argparser.add_argument("--task", type=str, default="regression")
argparser.add_argument("--num_outputs", type=int, default=1)
argparser.set_defaults(wandb=False)
argparser.set_defaults(dekadal=False)
argparser.set_defaults(ss_dekadal=False)
argparser.set_defaults(finetuned=False)
args = argparser.parse_args().__dict__

model_name = args["model_name"]
target_name = args["target_name"]
task = args["task"]
seed: int = args["seed"]
num_workers: int = args["num_workers"]
path_to_config = args["path_to_config"]
wandb_enabled: bool = args["wandb"]
wandb_org: str = args["wandb_org"]
finetuned: bool = args["finetuned"]
ss_dekadal: bool = args["ss_dekadal"]

seed_everything(seed)
output_parent_dir = (
    Path(args["output_dir"]) if args["output_dir"] else Path(__file__).parent
)
output_parent_dir.mkdir(parents=True, exist_ok=True)

path_to_weigths: Optional[Path] = None

if args["path_to_weigths"] is not None:
    path_to_weigths = Path(args["path_to_weigths"])

if args["data_dir"]:
    data_dir = Path(args["data_dir"])

run_id = None

if wandb_enabled:
    import wandb

    run = wandb.init(
        entity=wandb_org,
        project="scaleag",
        dir=output_parent_dir,
        name=model_name,
    )
    run_id = cast(wandb.sdk.wandb_run.Run, run).id

model_logging_dir = output_parent_dir / model_name  # timestamp_dirname(run_id)
model_logging_dir.mkdir(exist_ok=True, parents=True)
initialize_logging(model_logging_dir)
logger.info("Using output dir: %s" % model_logging_dir)

train_file: str = args["train_file"]
val_file: str = args["val_file"]
dekadal: bool = args["dekadal"]

logger.info("Setting up dataloaders")

train_df = pd.read_parquet(data_dir / train_file)
val_df = pd.read_parquet(data_dir / val_file)

logger.info("Setting up model")

model_kwargs = json.load(Path(config_dir / "default.json").open("r"))

model = load_pretrained_model(
    path_to_weigths=path_to_weigths,
    dekadal=dekadal,
    finetuned=finetuned,
    ss_dekadal=ss_dekadal,
    device=device,
    num_outputs=1,
)


if not task == "regression":
    model_modes = [
        "Random Forest Classifier",
        "Logistic Regression",
        "CatBoostClassifier",
    ]
else:
    model_modes = [
        "Random Forest Regressor",
        "Linear Regression",
        "CatBoostRegressor",
    ]

logger.info(f"Fine Tuning Presto on {task} task")
full_eval = ScaleAGEval(
    train_df,
    val_df,
    target_name,
    dekadal=dekadal,
    task=task,
)

results, finetuned_model, sklearn_models_trained = full_eval.finetuning_results(
    model, sklearn_model_modes=model_modes
)
logger.info(json.dumps(results, indent=2))

model_path = model_logging_dir / "models"
model_path.mkdir(exist_ok=True, parents=True)
finetuned_model_path = model_path / f"{model_name}_finetuned_model.pt"
torch.save(finetuned_model.state_dict(), finetuned_model_path)


# save ds models as well. assuming the orders of the models are the same as specified in model_modes
output_dir_ds_models = output_parent_dir / "ds_models"
output_dir_ds_models.mkdir(exist_ok=True, parents=True)

for i, ds_model in enumerate(sklearn_models_trained):
    ds_model_name = output_dir_ds_models / model_modes[i].replace(" ", "_")
    pickle.dump(model, open(f"{ds_model_name}.dsm", "wb"))
