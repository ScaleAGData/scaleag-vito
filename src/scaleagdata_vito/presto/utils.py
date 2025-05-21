import os
import random
from pathlib import Path
from typing import Literal, Union

import catboost as cb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from loguru import logger
from prometheo import finetune
from prometheo.datasets.scaleag import ScaleAgDataset
from prometheo.finetune import Hyperparams
from prometheo.models.presto import param_groups_lrd
from prometheo.models.presto.wrapper import (
    PretrainedPrestoWrapper,
    dataset_to_model,
    load_presto_weights,
)
from prometheo.predictors import collate_fn, to_torchtensor
from prometheo.utils import device
from sklearn.metrics import (
    classification_report,
    explained_variance_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader

dir = (
    Path(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent / "resources"
)


def predict_with_head(
    dl: DataLoader,
    finetuned_model: PretrainedPrestoWrapper,
):
    all_preds, all_targets = [], []
    finetuned_model.eval()
    for batch in dl:
        with torch.no_grad():
            preds = finetuned_model(batch)
            # binary classification
            if dl.dataset.task_type == "binary":
                preds = nn.functional.sigmoid(preds)
            # multiclass classification
            elif dl.dataset.task_type == "multiclass":
                preds = nn.functional.softmax(preds, dim=-1)
            # Flatten predictions and targets
            preds = preds.cpu().numpy().flatten()
            targets = batch.label.float().numpy().flatten()

            all_preds.append(preds)
            all_targets.append(targets)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return all_preds, all_targets


def get_encodings(
    dl: DataLoader,
    finetuned_model: PretrainedPrestoWrapper,
    eval_pooling: Literal["global", "time", None] = "global",
):
    all_encodings = []
    all_targets = []
    finetuned_model.eval()
    for batch in dl:
        with torch.no_grad():
            s1_s2_era5_srtm, mask, dynamic_world = dataset_to_model(batch)
            encodings = finetuned_model.encoder(
                x=to_torchtensor(s1_s2_era5_srtm, device=device).float(),
                dynamic_world=to_torchtensor(dynamic_world, device=device).long(),
                latlons=to_torchtensor(batch.latlon, device=device).float(),
                mask=to_torchtensor(mask, device=device).long(),
                # presto wants 0 indexed months, not 1 indexed months
                month=to_torchtensor(batch.timestamps[:, :, 1] - 1, device=device),
                eval_pooling=eval_pooling,
            )
            all_encodings.append(encodings.numpy())
            all_targets.append(batch.label.float().numpy().flatten())
    all_encodings = np.concatenate(all_encodings)
    all_targets = np.concatenate(all_targets)
    return all_encodings, all_targets


def evaluate_finetuned_model(
    finetuned_model: PretrainedPrestoWrapper,
    test_ds: ScaleAgDataset,
    num_workers: int = 2,
    batch_size: int = 32,
):
    # Construct the dataloader

    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    logger.info(f"Evaluating the finetuned model on {test_ds.task_type} task")
    # predict with Presto head and evaluate depending on the task
    preds, targets = predict_with_head(test_dl, finetuned_model)
    if test_ds.task_type == "binary":
        preds = preds > 0.5
        metrics = classification_report(targets, preds, output_dict=True)
    elif test_ds.task_type == "multiclass":
        preds = [test_ds.index_to_class[int(t)] for t in preds]
        targets = [test_ds.index_to_class[int(t)] for t in targets]
        metrics = classification_report(targets, preds, output_dict=True)
    else:
        targets = test_ds.revert_to_original_units(targets)
        preds = test_ds.revert_to_original_units(preds)
        metrics = {
            "RMSE": float(np.sqrt(mean_squared_error(targets, preds))),
            "R2_score": float(r2_score(targets, preds)),
            "explained_var_score": float(explained_variance_score(targets, preds)),
            "MAPE": float(mean_absolute_percentage_error(targets, preds)),
        }

    return metrics


def load_finetuned_model(
    model_path: Union[Path, str],
    task_type: Literal["regression", "binary", "multiclass"] = "regression",
    num_outputs: int = 1,
):
    if task_type == "regression":
        model = PretrainedPrestoWrapper(
            num_outputs=1,
            regression=True,
        )

    elif task_type == "binary":
        model = PretrainedPrestoWrapper(
            num_outputs=1,
            regression=False,
        )

    else:
        if num_outputs == 1:
            raise ValueError(
                "num_outputs is 1, but task_type is multiclass.Please provide correct num_outputs"
            )
        model = PretrainedPrestoWrapper(
            num_outputs=num_outputs,
            regression=False,
        )
    return load_presto_weights(model, f"{model_path}.pt", strict=False)


def finetune_on_task(
    train_ds: ScaleAgDataset,
    val_ds: ScaleAgDataset,
    output_dir: Union[Path, str],
    experiment_name: str,
    pretrained_model_path: Union[Path, str, None] = None,
    max_epochs: int = 50,
    batch_size: int = 100,
    patience: int = 3,
    num_workers: int = 2,
):

    composite_window = train_ds.composite_window

    if train_ds.task_type == "regression":
        regression = True
        num_outputs = 1
        loss_fn = nn.MSELoss()
    elif train_ds.task_type == "binary":
        regression = False
        num_outputs = 1
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        regression = False
        num_outputs = train_ds.num_outputs
        loss_fn = nn.CrossEntropyLoss()

    if pretrained_model_path is None:
        logger.info(
            "No pretrained model path provided. Using randomly initialized model."
        )

    if composite_window == "dekad":
        model = PretrainedPrestoWrapper(
            num_outputs=num_outputs,
            regression=regression,
        )
        model = load_presto_weights(model, pretrained_model_path, strict=False)
    else:
        model = PretrainedPrestoWrapper(
            num_outputs=num_outputs,
            regression=regression,
            pretrained_model_path=pretrained_model_path,
        )

    hyperparams = Hyperparams(
        max_epochs=max_epochs,
        batch_size=batch_size,
        patience=patience,
        num_workers=num_workers,
    )
    parameters = param_groups_lrd(model)
    optimizer = AdamW(parameters, lr=hyperparams.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    logger.info(f"Finetuning the model on {train_ds.task_type} task")
    finetuned_model = finetune.run_finetuning(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        experiment_name=experiment_name,
        output_dir=output_dir,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        hyperparams=hyperparams,
        setup_logging=False,  # Already setup logging
    )
    return finetuned_model


def evaluate_downstream_model(
    finetuned_model: PretrainedPrestoWrapper,
    downstream_model: Union[cb.CatBoostRegressor, cb.CatBoostClassifier],
    test_ds: ScaleAgDataset,
    num_workers: int = 2,
    batch_size: int = 32,
    eval_pooling: Literal["global", "time", None] = "global",
):

    # Construct the dataloader
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    logger.info(f"Evaluating the finetuned model on {test_ds.task_type} task")
    # get encodings from Presto and predict and evaluate with downstream model
    encodings, targets = get_encodings(test_dl, finetuned_model, eval_pooling)
    preds = downstream_model.predict(encodings)
    if test_ds.task_type in ["binary", "multiclass"]:
        metrics = classification_report(targets, preds, output_dict=True)
    else:
        targets = test_ds.revert_to_original_units(targets)
        preds = test_ds.revert_to_original_units(preds)
        metrics = {
            "RMSE": float(np.sqrt(mean_squared_error(targets, preds))),
            "R2_score": float(r2_score(targets, preds)),
            "explained_var_score": float(explained_variance_score(targets, preds)),
            "MAPE": float(mean_absolute_percentage_error(targets, preds)),
        }

    return metrics


def train_test_val_split(
    df,
    group_sample_by=None,
    uniform_sample_by=None,
    sampling_frac=0.8,
    nmin_per_class=5,
):
    """
    Splits the data into train, val and test sets.
    The split is done based on the unique parentname values.
    """
    random.seed(3)
    if group_sample_by is not None:
        parentnames = df[group_sample_by].unique()
        parentname_train = random.sample(
            list(parentnames), int(len(parentnames) * sampling_frac)
        )
        df_sample = df.copy()
        df_train = df_sample[df_sample[group_sample_by].isin(parentname_train)]

        # split in val and test
        df_val_test = df_sample[~df_sample[group_sample_by].isin(parentname_train)]
        parentname_val_test = df_val_test[group_sample_by].unique()
        parentname_val = random.sample(
            list(parentname_val_test), int(len(parentname_val_test) * 0.5)
        )
        df_val = df_val_test[df_val_test[group_sample_by].isin(parentname_val)]
        df_test = df_val_test[~df_val_test[group_sample_by].isin(parentname_val)]

    elif uniform_sample_by is not None:
        group_counts = df[uniform_sample_by].value_counts()
        valid_groups = group_counts[group_counts >= nmin_per_class].index
        if len(valid_groups) != len(group_counts):
            logger.warning(
                f"Some groups have less than {nmin_per_class} samples. They will be excluded from the split."
            )
        else:
            logger.info(
                f"All groups have at least {nmin_per_class} samples. Proceeding with the split."
            )
        df_sample = df[df[uniform_sample_by].isin(valid_groups)].reset_index(drop=True)
        df_train = df_sample.groupby(uniform_sample_by).sample(
            frac=sampling_frac, random_state=3
        )
        df_val_test = df_sample[~df_sample.index.isin(df_train.index)]
        df_val = df_val_test.groupby(uniform_sample_by).sample(frac=0.5, random_state=3)
        df_test = df_val_test[~df_val_test.index.isin(df_val.index)]
    else:
        raise ValueError(
            "Either group_sample_by or uniform_sample_by must be provided to split the data."
        )

    logger.info(f"Training set size: {len(df_train)}")
    logger.info(f"Validation set size: {len(df_val)}")
    logger.info(f"Test set size: {len(df_test)}")

    return df_train, df_val, df_test


def plot_distribution(df, target_name, upper_bound=None, lower_bound=None):
    plt.figure(figsize=(8, 4))
    sns.histplot(df[target_name], bins=30, kde=True)
    if upper_bound is not None:
        plt.axvline(x=upper_bound, color="r", linestyle="--", label="Upper Bound")
    if lower_bound is not None:
        plt.axvline(x=lower_bound, color="g", linestyle="--", label="Lower Bound")
        plt.legend()
    plt.title(target_name)
    plt.show()


def get_pretrained_model_url(composite_window: Literal["dekad", "month"]):
    if composite_window == "dekad":
        try:
            return "https://artifactory.vgt.vito.be/artifactory/auxdata-public/scaleagdata/models/presto-ss-wc_10D.pt"
        except:
            return dir / "presto-ss-wc_10D.pt"
    else:
        try:
            return "https://artifactory.vgt.vito.be/artifactory/auxdata-public/scaleagdata/models/presto-ss-wc_30D.pt"
        except:
            return dir / "presto-ss-wc_30D.pt"


def get_resources_dir():
    return dir
