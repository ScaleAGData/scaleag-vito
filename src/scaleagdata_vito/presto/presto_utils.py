import io
from typing import Literal, cast

import numpy as np
import requests
import torch
from loguru import logger
from presto.presto import Presto, get_sinusoid_encoding_table
from presto.utils import device
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
from torch import nn

default_model_kwargs = {
    "encoder_embedding_size": 128,
    "channel_embed_ratio": 0.25,
    "month_embed_ratio": 0.25,
    "encoder_depth": 2,
    "mlp_ratio": 4,
    "encoder_num_heads": 8,
    "decoder_embedding_size": 128,
    "decoder_depth": 2,
    "decoder_num_heads": 8,
}


def load_pretrained_model_from_url(
    model_url, finetuned=False, ss_dekadal=False, strict=False
):
    if finetuned:
        # initialize architecture without loading pretrained model
        model = Presto.construct(**default_model_kwargs)
        # extend model architecture to dekadal
        model = reinitialize_pos_embedding(model, max_sequence_length=72)
        # if we try to load a PrestoFT model, the architecture will be encoder + head
        # so we run the command to construct the same FT model architecture to be able
        # to correctly load weights
        model = model.construct_finetuning_model(num_outputs=1)
        logger.info(" Initialize Presto dekadal architecture with dekadal PrestoFT...")
        response = requests.get(model_url)
        best_model = torch.load(io.BytesIO(response.content), map_location=device)
        model.load_state_dict(best_model, strict=strict)
    else:
        # load pretrained default Presto
        model = Presto.construct(**default_model_kwargs)
        if model_url != "":
            if ss_dekadal:
                logger.info(
                    " Initialize Presto dekadal architecture with 10d ss trained WorldCereal Presto weights..."
                )
                # if model was self-supervised trained as decadal, first reinitialize positional
                # embeddings then load weights
                model = reinitialize_pos_embedding(model, max_sequence_length=72)
                response = requests.get(model_url)
                best_model = torch.load(
                    io.BytesIO(response.content), map_location=device
                )
                model.load_state_dict(best_model, strict=strict)
            else:
                logger.info(
                    " Initialize Presto dekadal architecture with 30d ss trained WorldCereal Presto weights..."
                )
                # if the model was self-supervised trained as monthly, first load weights then
                # reinitialize positional embeddings
                response = requests.get(model_url)
                best_model = torch.load(
                    io.BytesIO(response.content), map_location=device
                )
                model = reinitialize_pos_embedding(model, max_sequence_length=72)
        else:
            logger.info(
                " Initialize Presto dekadal architecture with pretrained Presto weights..."
            )
            model = reinitialize_pos_embedding(model, max_sequence_length=72)
    model.to(device)
    return model


def reinitialize_pos_embedding(
    model, max_sequence_length: int
):  # PrestoFineTuningModel
    # reinitialize encoder pos embed to stretch max length of time series
    model.encoder.pos_embed = nn.Parameter(
        torch.zeros(1, max_sequence_length, model.encoder.pos_embed.shape[-1]),
        requires_grad=False,
    )
    pos_embed = get_sinusoid_encoding_table(
        model.encoder.pos_embed.shape[1], model.encoder.pos_embed.shape[-1]
    )
    model.encoder.pos_embed.data.copy_(pos_embed)

    # reinitialize decoder pos embed to stretch max length of time series
    model.decoder.pos_embed = nn.Parameter(
        torch.zeros(1, max_sequence_length, model.decoder.pos_embed.shape[-1]),
        requires_grad=False,
    )
    pos_embed = get_sinusoid_encoding_table(
        model.decoder.pos_embed.shape[1], model.decoder.pos_embed.shape[-1]
    )
    model.decoder.pos_embed.data.copy_(pos_embed)
    return model


def get_encodings(dl, pretrained_presto):
    pretrained_presto.eval()
    batch_encodings, batch_targets = [], []
    for x, y, dw, latlons, month, variable_mask in dl:
        batch_targets.append(y)
        x_f, dw_f, latlons_f, month_f, variable_mask_f = [
            t.to(device) for t in (x, dw, latlons, month, variable_mask)
        ]
        with torch.no_grad():
            cast(Presto, pretrained_presto).eval()
            encodings = (
                cast(Presto, pretrained_presto)
                .encoder(
                    x_f,
                    dynamic_world=dw_f.long(),
                    mask=variable_mask_f,
                    latlons=latlons_f,
                    month=month_f,
                )
                .cpu()
                .numpy()
            )
            batch_encodings.append(encodings)
    batch_encodings = np.concatenate(batch_encodings)
    batch_targets = np.concatenate(batch_targets)
    if len(batch_targets.shape) == 2 and batch_targets.shape[1] == 1:
        batch_targets = batch_targets.ravel()
    return batch_encodings, batch_targets


def predict_with_head(dl, finetuned_model):
    test_preds, targets = [], []
    for x, y, dw, latlons, month, variable_mask in dl:
        targets.append(y)
        x_f, dw_f, latlons_f, month_f, variable_mask_f = [
            t.to(device) for t in (x, dw, latlons, month, variable_mask)
        ]
        finetuned_model.eval()
        with torch.no_grad():
            preds = (
                finetuned_model(
                    x_f,
                    dynamic_world=dw_f.long(),
                    mask=variable_mask_f,
                    latlons=latlons_f,
                    month=month_f,
                )
                .squeeze(dim=1)
                .cpu()
                .numpy()
            )
            test_preds.append(preds)
    test_preds_np = np.concatenate(test_preds)
    target_np = np.concatenate(targets)
    return test_preds_np, target_np


def revert_to_original_units(y_norm, upper_bound, lower_bound):
    return y_norm * (upper_bound - lower_bound) + lower_bound


def evaluate(
    pretrained_model,
    ds_model,
    dl_val,
    up_val,
    low_val,
    task: Literal["regression", "binary", "multiclass"],
):
    """
    Evaluate the performance of a deep learning model on a validation dataset.

    Parameters:
    - pretrained_model: The pretrained model used for generating the encodings.
    - ds_model: The ML model trained for the downstream task to evaluate.
    - dl_val: The validation dataloader.
    - task: The type of task, which can be 'regression', 'binary', or 'multiclass'.

    Returns:
    - metrics: The performance metrics on validation dataset.

    """
    encodings, targets = get_encodings(dl_val, pretrained_model)
    targets = revert_to_original_units(targets, up_val, low_val)
    preds = ds_model.predict(encodings)
    preds = revert_to_original_units(preds, up_val, low_val)
    if task == "regression":
        metrics = {
            "RMSE": float(np.sqrt(mean_squared_error(targets, preds))),
            "R2_score": float(r2_score(targets, preds)),
            "explained_var_score": float(explained_variance_score(targets, preds)),
        }
    elif task == "binary":
        pass
    elif task == "multiclass":
        pass
    return metrics, preds, targets
