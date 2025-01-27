from typing import Literal, Union

import catboost as cb
import numpy as np
import torch
from loguru import logger
from prometheo.datasets.scaleag import ScaleAgDataset
from prometheo.models.presto.wrapper import PretrainedPrestoWrapper, dataset_to_model
from prometheo.predictors import to_torchtensor
from prometheo.utils import device
from sklearn.metrics import (
    classification_report,
    explained_variance_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from torch import nn
from torch.utils.data import DataLoader


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
