import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from presto.presto import (
    FinetuningHead,
    Presto,
    PrestoFineTuningModel,
    get_sinusoid_encoding_table,
    param_groups_lrd,
)
from presto.utils import DEFAULT_SEED, device
from scaleagdata.datasets import ScaleAG10DDataset, ScaleAGDataset
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    explained_variance_score,
    f1_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger("__main__")

SklearnStyleModel = Union[BaseEstimator, CatBoostClassifier]


@dataclass
class Hyperparams:
    lr: float = 2e-5
    max_epochs: int = 70
    batch_size: int = 64
    patience: int = 10
    num_workers: int = 4


class ScaleAGYieldEval:
    name = "ScaleAG_AVR_Yield"
    threshold = 0.05
    num_outputs = 1

    def __init__(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        target_name: str,
        seed: int = DEFAULT_SEED,
        dekadal: bool = True,
        task: Literal["regression", "binary", "multiclass"] = "regression",
    ):
        self.seed = seed

        # SAR cannot equal 0.0 since we take the log of it
        self.train_df = self.prep_dataframe(train_data, dekadal=dekadal)
        self.val_df = self.prep_dataframe(val_data, dekadal=dekadal)
        self.test_df = self.val_df.copy()

        self.target_name = target_name
        self.dekadal = dekadal
        self.task = task
        if task != "regression":
            self.num_outputs = len(self.train_df[target_name].unique())

        self.ds_class = ScaleAG10DDataset if dekadal else ScaleAGDataset

    @staticmethod
    def prep_dataframe(
        df: pd.DataFrame,
        dekadal: bool = False,
    ):
        # SAR cannot equal 0.0 since we take the log of it
        cols = [
            f"SAR-{s}-ts{t}-20m"
            for s in ["VV", "VH"]
            for t in range(36 if dekadal else 12)
        ]
        df = df.drop_duplicates(subset=["lat", "lon", "end_date"])
        df = df[~pd.isna(df).any(axis=1)]
        df = df[~(df.loc[:, cols] == 0.0).any(axis=1)]
        return df

    def _construct_finetuning_model(
        self, pretrained_model: Union[Presto, PrestoFineTuningModel]
    ) -> PrestoFineTuningModel:
        if isinstance(pretrained_model, PrestoFineTuningModel):
            # if we are passing a model that had already been fine tuned then we need
            # then we need to adjust a bit the initialization as the construct_finetuning_model
            # expects a Presto architecture but here we already have a PrestoFineTuningModel
            model = cast(Callable, self.construct_from_finetuned)(
                encoder=pretrained_model.encoder
            )
        else:
            model = cast(Callable, pretrained_model.construct_finetuning_model)(
                num_outputs=self.num_outputs
            )

        # reinitialize positional embedding if decadal
        if self.dekadal:
            max_sequence_length = 72  # can this be 36?
            old_pos_embed_device = model.encoder.pos_embed.device
            model.encoder.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    max_sequence_length,
                    model.encoder.pos_embed.shape[-1],
                    device=old_pos_embed_device,
                ),
                requires_grad=False,
            )
            pos_embed = get_sinusoid_encoding_table(
                model.encoder.pos_embed.shape[1], model.encoder.pos_embed.shape[-1]
            )
            model.encoder.pos_embed.data.copy_(
                pos_embed.to(device=old_pos_embed_device)
            )
        return model

    def construct_from_finetuned(
        self,
        encoder,
    ) -> PrestoFineTuningModel:
        head = FinetuningHead(
            num_outputs=self.num_outputs,
            hidden_size=encoder.embedding_size,
        )
        model = PrestoFineTuningModel(encoder, head).to(encoder.pos_embed.device)
        model.train()
        return model

    def setup_model_dict(self, models):
        model_dict = {}
        for model in models:
            if model == "Logistic Regression":
                model_dict[model] = LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=self.seed
                )
            elif model == "Random Forest Classifier":
                model_dict[model] = RandomForestClassifier(
                    class_weight="balanced", random_state=self.seed
                )
            elif model == "CatBoostClassifier":
                model_dict[model] = CatBoostClassifier(
                    iterations=8000,
                    depth=8,
                    learning_rate=0.05,
                    early_stopping_rounds=20,
                    l2_leaf_reg=3,
                    random_state=self.seed,
                    auto_class_weights="Balanced",
                )
            elif model == "Linear Regression":
                model_dict[model] = LinearRegression()

            elif model == "Random Forest Regressor":
                model_dict[model] = RandomForestRegressor(random_state=self.seed)

            elif model == "CatBoostRegressor":
                model_dict[model] = CatBoostRegressor(random_state=self.seed)
        return model_dict

    @torch.no_grad()
    def finetune_sklearn_model(
        self,
        dl: DataLoader,
        val_dl: DataLoader,
        pretrained_model: PrestoFineTuningModel,
        models: List[str] = ["Logistic Regression", "Random Forest Classifier"],
    ) -> Union[Sequence[BaseEstimator], Dict]:

        for model_mode in models:
            assert model_mode in [
                "Linear Regression",
                "Logistic Regression",
                "Random Forest Classifier",
                "CatBoostClassifier",
                "Random Forest Regressor",
                "CatBoostRegressor",
            ]
        pretrained_model.eval()

        def dataloader_to_encodings_and_targets(
            dl: DataLoader,
        ) -> Tuple[np.ndarray, np.ndarray]:
            encoding_list, target_list = [], []
            for x, y, dw, latlons, month, variable_mask in dl:
                x_f, dw_f, latlons_f, month_f, variable_mask_f = [
                    t.to(device) for t in (x, dw, latlons, month, variable_mask)
                ]
                target_list.append(y)
                with torch.no_grad():
                    encodings = (
                        pretrained_model.encoder(
                            x_f,
                            dynamic_world=dw_f.long(),
                            mask=variable_mask_f,
                            latlons=latlons_f,
                            month=month_f,
                        )
                        .cpu()
                        .numpy()
                    )
                    encoding_list.append(encodings)
            encodings_np = np.concatenate(encoding_list)
            targets = np.concatenate(target_list)
            if len(targets.shape) == 2 and targets.shape[1] == 1:
                targets = targets.ravel()
            return encodings_np, targets

        train_encodings, train_targets = dataloader_to_encodings_and_targets(dl)
        val_encodings, val_targets = dataloader_to_encodings_and_targets(val_dl)

        fit_models = []
        model_dict = self.setup_model_dict(models)
        for model in tqdm(models, desc="Fitting sklearn models"):
            if model == "CatBoostClassifier":
                fit_models.append(
                    clone(model_dict[model]).fit(
                        train_encodings,
                        train_targets,
                        eval_set=Pool(val_encodings, val_targets),
                    )
                )
            else:
                fit_models.append(
                    clone(model_dict[model]).fit(train_encodings, train_targets)
                )
        return fit_models

    # @staticmethod
    def _inference_for_dl(
        self,
        dl,
        finetuned_model: Union[PrestoFineTuningModel, SklearnStyleModel],
        pretrained_model: Optional[PrestoFineTuningModel] = None,
    ) -> Tuple:
        test_preds, targets = [], []

        for x, y, dw, latlons, month, variable_mask in dl:
            targets.append(y)
            x_f, dw_f, latlons_f, month_f, variable_mask_f = [
                t.to(device) for t in (x, dw, latlons, month, variable_mask)
            ]
            if isinstance(finetuned_model, PrestoFineTuningModel):
                finetuned_model.eval()
                preds = finetuned_model(
                    x_f,
                    dynamic_world=dw_f.long(),
                    mask=variable_mask_f,
                    latlons=latlons_f,
                    month=month_f,
                ).squeeze(dim=1)
                # binary classification
                if self.task == "binary":
                    preds = torch.sigmoid(preds).cpu().numpy()
                # multiclass classification
                elif self.task == "multiclass":
                    preds = nn.functional.softmax(preds).cpu().numpy()
                    preds = np.argmax(preds, axis=-1)
                # regression ????
                else:
                    preds = preds.cpu().numpy()

            else:
                cast(Presto, pretrained_model).eval()
                encodings = (
                    cast(Presto, pretrained_model)
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
                if self.task == "binary":
                    preds = finetuned_model.predict_proba(encodings)[:, 1]
                else:
                    preds = finetuned_model.predict(encodings)
            test_preds.append(preds)

        test_preds_np = np.concatenate(test_preds)
        target_np = np.concatenate(targets)
        return test_preds_np, target_np

    @torch.no_grad()
    def evaluate(
        self,
        finetuned_model: Union[PrestoFineTuningModel, BaseEstimator],
        pretrained_model: Optional[PrestoFineTuningModel] = None,
    ) -> Dict:
        test_ds = self.ds_class(self.test_df, self.target_name, self.task)

        dl = DataLoader(
            test_ds,
            batch_size=2048,  # 4096, #8192,
            shuffle=False,  # keep as False!
            num_workers=Hyperparams.num_workers,
        )
        assert isinstance(dl.sampler, torch.utils.data.SequentialSampler)

        test_preds_np, target_np = self._inference_for_dl(
            dl, finetuned_model, pretrained_model
        )
        # print(f"target {target_np}")
        # print(f"test preds {test_preds_np}")
        if self.task == "binary":
            test_preds_np = test_preds_np >= self.threshold
        elif self.task == "multiclass":
            # map indices to classes
            test_preds_np = [test_ds.index_to_class[int(t)] for t in test_preds_np]
            target_np = [test_ds.index_to_class[int(t)] for t in target_np]
        # targets are normalized during training to avoid gradient explosion.
        # revert to original units

        elif self.task == "regression":
            target_np = test_ds.revert_to_original_units(target_np)
            test_preds_np = test_ds.revert_to_original_units(test_preds_np)

        prefix = f"{self.name}_{finetuned_model.__class__.__name__}"

        if self.task == "binary":
            return {
                f"{prefix}_f1": float(f1_score(target_np, test_preds_np)),
                f"{prefix}_recall": float(recall_score(target_np, test_preds_np)),
                f"{prefix}_precision": float(precision_score(target_np, test_preds_np)),
            }
        elif self.task == "regression":
            return {
                f"{prefix}_RMSE": float(
                    np.sqrt(mean_squared_error(target_np, test_preds_np))
                ),
                f"{prefix}_R2_score": float(r2_score(target_np, test_preds_np)),
                f"{prefix}_explained_var_score": float(
                    explained_variance_score(target_np, test_preds_np)
                ),
                f"{prefix}_MAPE": float(
                    mean_absolute_percentage_error(target_np, test_preds_np)
                ),
            }
        else:
            labs = list(test_ds.index_to_class.values())
            return {
                f"{prefix}_f1": float(
                    f1_score(target_np, test_preds_np, labels=labs, average="macro")
                ),
                f"{prefix}_recall": float(
                    recall_score(target_np, test_preds_np, labels=labs, average="macro")
                ),
                f"{prefix}_precision": float(
                    precision_score(
                        target_np, test_preds_np, labels=labs, average="macro"
                    )
                ),
            }

    def finetune(self, pretrained_model) -> PrestoFineTuningModel:
        hyperparams = Hyperparams()
        model = self._construct_finetuning_model(pretrained_model)

        parameters = param_groups_lrd(model)
        optimizer = AdamW(parameters, lr=hyperparams.lr)

        train_ds = self.ds_class(self.train_df, self.target_name, self.task)

        val_ds = self.ds_class(self.val_df, self.target_name, self.task)

        if self.task == "regression":
            loss_fn = nn.MSELoss()
        elif self.task == "binary":
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.CrossEntropyLoss()

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        train_dl = DataLoader(
            train_ds,
            batch_size=hyperparams.batch_size,
            shuffle=True,
            num_workers=hyperparams.num_workers,
            generator=generator,
        )

        val_dl = DataLoader(
            val_ds,
            batch_size=hyperparams.batch_size,
            shuffle=False,
            num_workers=hyperparams.num_workers,
        )

        train_loss = []
        val_loss = []
        best_loss = None
        best_model_dict = None
        epochs_since_improvement = 0

        run = None
        try:
            import wandb

            run = wandb.run
        except ImportError:
            pass

        for _ in (pbar := tqdm(range(hyperparams.max_epochs), desc="Finetuning")):
            model.train()
            epoch_train_loss = 0.0
            for x, y, dw, latlons, month, variable_mask in tqdm(
                train_dl, desc="Training", leave=False
            ):
                x, y, dw, latlons, month, variable_mask = [
                    t.to(device) for t in (x, y, dw, latlons, month, variable_mask)
                ]
                optimizer.zero_grad()
                preds = model(
                    x,
                    dynamic_world=dw.long(),
                    mask=variable_mask,
                    latlons=latlons,
                    month=month,
                )
                if self.task == "multiclass":
                    loss = loss_fn(
                        preds.squeeze(-1), y.long()
                    )  # if ce y must be the index of the class
                else:
                    loss = loss_fn(preds.squeeze(-1), y.float())
                epoch_train_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss.append(epoch_train_loss / len(train_dl))

            model.eval()
            all_preds, all_y = [], []
            for x, y, dw, latlons, month, variable_mask in val_dl:
                x, y, dw, latlons, month, variable_mask = [
                    t.to(device) for t in (x, y, dw, latlons, month, variable_mask)
                ]
                with torch.no_grad():
                    preds = model(
                        x,
                        dynamic_world=dw.long(),
                        mask=variable_mask,
                        latlons=latlons,
                        month=month,
                    )

                    if self.task == "multiclass":
                        all_y.append(y.long())
                    else:
                        all_y.append(y.float())
                    all_preds.append(preds.squeeze(-1))

            val_loss.append(loss_fn(torch.cat(all_preds), torch.cat(all_y)))
            # val_loss.append(nn.functional.huber_loss(torch.cat(all_preds), torch.cat(all_y)))
            pbar.set_description(
                f"Train metric: {train_loss[-1]}, Val metric: {val_loss[-1]}"
            )

            if run is not None:
                wandb.log(
                    {
                        f"{self.name}_finetuning_val_loss": val_loss[-1],
                        f"{self.name}_finetuning_train_loss": train_loss[-1],
                    }
                )

            if best_loss is None:
                best_loss = val_loss[-1]
                best_model_dict = deepcopy(model.state_dict())
            else:
                if val_loss[-1] < best_loss:
                    best_loss = val_loss[-1]
                    best_model_dict = deepcopy(model.state_dict())
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                    if epochs_since_improvement >= hyperparams.patience:
                        logger.info("Early stopping!")
                        break
        assert best_model_dict is not None
        model.load_state_dict(best_model_dict)

        model.eval()
        return model

    def finetuning_results_sklearn(
        self, sklearn_model_modes: List[str], finetuned_model: PrestoFineTuningModel
    ) -> Dict:

        results_dict = {}
        if len(sklearn_model_modes) > 0:

            train_ds = self.ds_class(self.train_df, self.target_name, self.task)

            val_ds = self.ds_class(self.val_df, self.target_name, self.task)

            dl = DataLoader(
                train_ds,
                batch_size=2048,
                shuffle=False,
                num_workers=4,
            )

            val_dl = DataLoader(
                val_ds,
                batch_size=2048,
                shuffle=False,
                num_workers=4,
            )
            sklearn_models = self.finetune_sklearn_model(
                dl,
                val_dl,
                finetuned_model,
                models=sklearn_model_modes,
            )
            for sklearn_model in sklearn_models:
                logger.info(f"Evaluating {sklearn_model}...")
                results_dict.update(self.evaluate(sklearn_model, finetuned_model))

        return results_dict

    def finetuning_results(
        self,
        pretrained_model,
        sklearn_model_modes: List[str],
    ) -> Tuple[Dict, Optional[PrestoFineTuningModel]]:
        for model_mode in sklearn_model_modes:
            assert model_mode in [
                "Linear Regression",
                "Logistic Regression",
                "Random Forest Classifier",
                "CatBoostClassifier",
                "Random Forest Regressor",
                "CatBoostRegressor",
            ]
        results_dict = {}
        # we want to always finetune the model, since the sklearn models
        # will use the finetuned model as a base
        finetuned_model = self.finetune(pretrained_model)
        results_dict.update(self.evaluate(finetuned_model, None))
        results_dict.update(
            self.finetuning_results_sklearn(sklearn_model_modes, finetuned_model)
        )
        return results_dict, finetuned_model
