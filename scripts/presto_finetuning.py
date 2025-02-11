import random

# import sys
from pathlib import Path

from loguru import logger

# sys.path.append("/home/vito/millig/gio/prometheo/")
from prometheo import finetune
from prometheo.datasets.scaleag import ScaleAgDataset  # fix installation
from prometheo.finetune import Hyperparams
from prometheo.models.presto.wrapper import PretrainedPrestoWrapper, load_pretrained
from torch import nn

from scaleagdata_vito.presto.presto_df import load_dataset
from scaleagdata_vito.presto.utils import evaluate_finetuned_model

# load extracted dataset
window_of_interest = ["2022-04-01", "2022-10-31"]
df = load_dataset(
    files_root_dir="/projects/TAP/HEScaleAgData/data/AVR_subfields/extractions_31012025/",
    window_of_interest=window_of_interest,
    use_valid_time=False,
    required_min_timesteps=36,
    buffer_window=8,
    no_data_value=65535,
    composite_window="dekad",
)


#### prepare datasets for training
sampling_frac = 0.8
random.seed(42)
parentname = df.parentname.unique()
parentname_train = random.sample(list(parentname), int(len(parentname) * sampling_frac))
df_sample = df.copy()
df_train = df_sample[df_sample.parentname.isin(parentname_train)]
df_val = df_sample[~df_sample.parentname.isin(parentname_train)]

logger.info(f"Train size: {len(df_train)}")
logger.info(f"Val size: {len(df_val)}")


# initialize datasets
num_timesteps = df.available_timesteps.max()

train_ds = ScaleAgDataset(
    df_train,
    num_timesteps=num_timesteps,
    task_type="regression",
    target_name="median_yield",
    compositing_window="dekad",
    upper_bound=120000,
    lower_bound=10000,
)
val_ds = ScaleAgDataset(
    df_val,
    num_timesteps=num_timesteps,
    task_type="regression",
    target_name="median_yield",
    compositing_window="dekad",
    upper_bound=120000,
    lower_bound=10000,
)

#### load pretrained model
pretrained_model_path = (
    "/home/vito/millig/gio/models/presto_ss/dekadal/output/presto-ss-wc_10D.pt"
)
# pretrained_model_path = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/scaleagdata/models/presto-ss-wc_10D.pt"
model = PretrainedPrestoWrapper(
    num_outputs=1,
    regression=True,
)
model = load_pretrained(model, pretrained_model_path, strict=False)


# evaluate_finetuned_model(model, val_ds, num_workers=2, batch_size=32)
hyperparams = Hyperparams(max_epochs=50, batch_size=256, patience=1, num_workers=2)
output_dir = Path("/home/vito/millig/gio/presto_exp/prometheo_exp")

# set loss depending on the task type
if train_ds.task_type == "regression":
    loss_fn = nn.MSELoss()
elif train_ds.task_type == "binary":
    loss_fn = nn.BCEWithLogitsLoss()
else:
    loss_fn = nn.CrossEntropyLoss()

finetuned_model = finetune.run_finetuning(
    model,
    train_ds,
    val_ds,
    experiment_name="presto-ss-wc-10D-ft-dek",
    output_dir=output_dir,
    loss_fn=loss_fn,
    hyperparams=hyperparams,
)

finetuned_model = PretrainedPrestoWrapper(num_outputs=1, regression=True)
finetuned_model = load_pretrained(
    finetuned_model,
    "/home/vito/millig/gio/presto_exp/prometheo_exp/presto-ss-wc-10D-ft-dek.pt",
)

evaluate_finetuned_model(finetuned_model, val_ds, num_workers=2, batch_size=32)
