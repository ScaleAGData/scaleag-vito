# import sys
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from loguru import logger
from prometheo import finetune
from prometheo.finetune import Hyperparams
from prometheo.models.presto import param_groups_lrd
from prometheo.models.presto.wrapper import PretrainedPrestoWrapper, load_presto_weights

# from prometheo.models import Presto
from prometheo.predictors import collate_fn
from scaleagdata_vito.presto.datasets import ScaleAgDataset
from scaleagdata_vito.presto.utils import (
    evaluate_finetuned_model,
    get_pretrained_model_url,
)
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader

DEFAULT_SEED = 42
use_balancing = True
extractions_name = "LPIS_Extractions_2024"
p = f"/projects/HEScaleAgData/timeseries_modelling/datasets/{extractions_name}/extractions/"
experiment_name = "presto_finetuning_LPIS_2024_refdata_multiclass_v2"
model_output_dir = (
    Path(f"/projects/HEScaleAgData/models/presto_finetuned/{extractions_name}")
    / experiment_name
)
model_output_dir.mkdir(parents=True, exist_ok=True)
# load extracted dataset
target_name = "yield_bin"
task_type = "multiclass"  # "regression" or "multiclass"
composite_window: Literal["dekad", "month"] = "dekad"
window_of_interest = ["2024-04-01", "2024-10-31"]

# df = load_dataset(
#     files_root_dir=p,
#     window_of_interest=window_of_interest,
#     no_data_value=65535,
#     composite_window=composite_window,
# )

# # prepare datasets for training
# RANDOM_SEED = 42  # for reproducibility
# df_train, df_val, df_test = train_test_val_split(
#     df=df, group_sample_by="parentname", sampling_frac=0.8, seed=RANDOM_SEED
# )

# # save dfs for reproducibility
# df_train.to_parquet(model_output_dir / "df_train.parquet", index=False)
# df_val.to_parquet(model_output_dir / "df_val.parquet", index=False)
# df_test.to_parquet(model_output_dir / "df_test.parquet", index=False)

woi_tag = (
    f"{window_of_interest[0].replace('-', '')}-{window_of_interest[1].replace('-', '')}"
)

df_train = pd.read_parquet(
    f"/projects/HEScaleAgData/reference_datasets/LPIS_Extractions_2024/df_train_{woi_tag}.parquet"
)
df_val = pd.read_parquet(
    f"/projects/HEScaleAgData/reference_datasets/LPIS_Extractions_2024/df_val_{woi_tag}.parquet"
)
df_test = pd.read_parquet(
    f"/projects/HEScaleAgData/reference_datasets/LPIS_Extractions_2024/df_test_{woi_tag}.parquet"
)

num_timesteps = df_train.available_timesteps.max()
nclasses = 1 if task_type == "regression" else df_train[target_name].nunique()
# compute and save target mean and std for unnormalizing predictions later
if task_type == "regression":
    regression = True
    target_mean = df_train[target_name].mean()
    target_std = df_train[target_name].std()

    with open(model_output_dir / "train_target_mean_and_std.txt", "w") as f:
        f.write(f"target_mean: {target_mean}\n")
        f.write(f"target_std: {target_std}\n")
    # initialize datasets
    train_ds = ScaleAgDataset(
        df_train,
        num_timesteps=num_timesteps,
        num_outputs=nclasses,
        task_type="regression",
        target_name=target_name,
        composite_window=composite_window,
        target_mean=target_mean,
        target_std=target_std,
    )
    val_ds = ScaleAgDataset(
        df_val,
        num_timesteps=num_timesteps,
        num_outputs=nclasses,
        task_type="regression",
        target_name=target_name,
        composite_window=composite_window,
        target_mean=target_mean,
        target_std=target_std,
    )

    test_ds = ScaleAgDataset(
        df_test,
        num_timesteps=num_timesteps,
        num_outputs=nclasses,
        task_type="regression",
        target_name=target_name,
        composite_window=composite_window,
        target_mean=target_mean,
        target_std=target_std,
    )
    # loss function for regression
    loss_fn = nn.MSELoss()

elif task_type == "multiclass":
    regression = False
    train_ds = ScaleAgDataset(
        df_train,
        num_timesteps=num_timesteps,
        num_outputs=nclasses,
        task_type="multiclass",
        target_name=target_name,
        composite_window=composite_window,
    )
    val_ds = ScaleAgDataset(
        df_val,
        num_timesteps=num_timesteps,
        num_outputs=nclasses,
        task_type="multiclass",
        target_name=target_name,
        composite_window=composite_window,
    )

    test_ds = ScaleAgDataset(
        df_test,
        num_timesteps=num_timesteps,
        num_outputs=nclasses,
        task_type="multiclass",
        target_name=target_name,
        composite_window=composite_window,
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=65535)

# Run finetuning
pretrained_model_path = get_pretrained_model_url(composite_window=composite_window)
batch_size = 64
num_workers = 2
patience = 5
max_epochs = 100
unfreeze_epoch = 5
lr = 2e-5
freeze_layers = ["encoder"]


if pretrained_model_path is None:
    logger.info("No pretrained model path provided. Using randomly initialized model.")

# if composite_window == "dekad":
# try:
model = PretrainedPrestoWrapper(
    num_outputs=nclasses,
    regression=regression,
)
model = load_presto_weights(model, pretrained_model_path, strict=False)
# except Exception as e:
#     model = PretrainedPrestoWrapper(
#         num_outputs=num_outputs,
#         regression=regression,
#         pretrained_model_path=pretrained_model_path,
#     )

hyperparams = Hyperparams(
    max_epochs=max_epochs,
    batch_size=batch_size,
    patience=patience,
    num_workers=num_workers,
    lr=lr,
)
parameters = param_groups_lrd(model)
optimizer = AdamW(parameters, lr=hyperparams.lr)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5, verbose=True, min_lr=1e-6
)
generator = torch.Generator()
generator.manual_seed(DEFAULT_SEED)

train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True if not use_balancing else False,
    sampler=(
        train_ds.get_balanced_sampler(
            generator=generator,
            method="log",
            clip_range=None,
            normalize=True,
        )
        if use_balancing
        else None
    ),
    generator=generator if not use_balancing else None,
    num_workers=num_workers,
    collate_fn=collate_fn,
)
val_dl = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=collate_fn,
)

logger.info(f"Finetuning the model on {train_ds.task_type} task")
finetuned_model = finetune.run_finetuning(
    model=model,
    train_dl=train_dl,
    val_dl=val_dl,
    experiment_name=experiment_name,
    output_dir=model_output_dir,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    hyperparams=hyperparams,
    setup_logging=True,  # Already setup logging
    freeze_layers=freeze_layers,
    unfreeze_epoch=unfreeze_epoch,
)

evaluate_finetuned_model(
    finetuned_model, test_ds, num_workers=num_workers, batch_size=batch_size
)
