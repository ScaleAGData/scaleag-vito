# import sys
from pathlib import Path
from typing import Literal

from loguru import logger
from prometheo import finetune
from prometheo.finetune import Hyperparams
from prometheo.models.presto import param_groups_lrd
from prometheo.models.presto.wrapper import PretrainedPrestoWrapper, load_presto_weights

# from prometheo.models import Presto
from prometheo.predictors import collate_fn
from scaleagdata_vito.presto.datasets import ScaleAgDataset
from scaleagdata_vito.presto.presto_df import load_dataset
from scaleagdata_vito.presto.utils import (
    evaluate_finetuned_model,
    get_pretrained_model_url,
    train_test_val_split,
)
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader

extractions_name = "LPIS_Extractions_2024"
p = f"/projects/HEScaleAgData/timeseries_modelling/datasets/{extractions_name}/extractions/"
experiment_name = "presto_finetuning_LPIS_2024_dekad_median_yield"
model_output_dir = (
    Path(f"/projects/HEScaleAgData/models/presto_finetuned/{extractions_name}")
    / experiment_name
)
model_output_dir.mkdir(parents=True, exist_ok=True)
# load extracted dataset
target_name = "median_yield"
composite_window: Literal["dekad", "month"] = "dekad"
window_of_interest = ["2024-04-01", "2024-10-31"]
df = load_dataset(
    files_root_dir=p,
    window_of_interest=window_of_interest,
    no_data_value=65535,
    composite_window=composite_window,
)

# prepare datasets for training
RANDOM_SEED = 42  # for reproducibility
df_train, df_val, df_test = train_test_val_split(
    df=df, group_sample_by="parentname", sampling_frac=0.8, seed=RANDOM_SEED
)

# save dfs for reproducibility
df_train.to_parquet(model_output_dir / "df_train.parquet", index=False)
df_val.to_parquet(model_output_dir / "df_val.parquet", index=False)
df_test.to_parquet(model_output_dir / "df_test.parquet", index=False)

# compute and save target mean and std for unnormalizing predictions later
target_mean = df_train[target_name].mean()
target_std = df_train[target_name].std()

with open(model_output_dir / "train_target_mean_and_std.txt", "w") as f:
    f.write(f"target_mean: {target_mean}\n")
    f.write(f"target_std: {target_std}\n")

# initialize datasets
num_timesteps = df_train.available_timesteps.max()

train_ds = ScaleAgDataset(
    df_train,
    num_timesteps=num_timesteps,
    task_type="regression",
    target_name=target_name,
    composite_window=composite_window,
    target_mean=target_mean,
    target_std=target_std,
)
val_ds = ScaleAgDataset(
    df_val,
    num_timesteps=num_timesteps,
    task_type="regression",
    target_name=target_name,
    composite_window=composite_window,
    target_mean=target_mean,
    target_std=target_std,
)

test_ds = ScaleAgDataset(
    df_test,
    num_timesteps=num_timesteps,
    task_type="regression",
    target_name=target_name,
    composite_window=composite_window,
    target_mean=target_mean,
    target_std=target_std,
)

# Run finetuning
pretrained_model_path = get_pretrained_model_url(composite_window=composite_window)
batch_size = 32
num_workers = 2
patience = 10
max_epochs = 100
unfreeze_epoch = 5
lr = 1e-4
freeze_layers = ["encoder"]

regression = True
num_outputs = 1
loss_fn = nn.MSELoss()

if pretrained_model_path is None:
    logger.info("No pretrained model path provided. Using randomly initialized model.")

# if composite_window == "dekad":
# try:
model = PretrainedPrestoWrapper(
    num_outputs=num_outputs,
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
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
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
