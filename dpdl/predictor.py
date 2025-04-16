import torch
import logging

import torch.nn.functional as F
import pandas as pd
import os
from torch.nn.parallel import DistributedDataParallel as DDP

from .datamodules import DataModule, DataModuleFactory
from .models import ModelBase
from .models.model_factory import ModelFactory
from .callbacks.callback_factory import CallbackHandler, CallbackFactory
from .utils import seed_everything

log = logging.getLogger(__name__)


class Predictor:

    def __init__(
        self,
        model: ModelBase,
        datamodule: DataModule,
        dataset_split: str,
        callback_handler: CallbackHandler,
    ):
        """
        init Predictor

        args:
            model (ModelBase): the model for prediction
            datamodule: provide the class for data preprocessing and dataloader
            dataset_split (str): for a specific split of dataset, e.g., 'train', 'valid', 'test'
            callback_handler (CallbackHandler): callback handler for prediction
            seed (int): random seed for reproducibility
        """
        self.model = model
        self.datamodule = datamodule
        self.dataset_split = dataset_split
        self.callback_handler = callback_handler

    def load_model(self, fpath: str):
        """
        load model from file

        args:
            fpath (optional, str): the path to the model file
        """
        raise NotImplementedError(
            "load_model function is not implemented in the Predictor class, will be done later"
        )

    def predict(self, configuration):
        """
        Perform prediction using the model on the specified dataset split.
        All ranks compute prediction in parallel using DDP, and rank 0 gathers 
        the results and saves them as a CSV file.

        Args:
            configuration: A Configuration object that provides dataset_split and log path.
        """

        model = self.model.cuda()
        model = DDP(model)
        model.eval()

        dataset_split = configuration.dataset_split
        dataloader = self.datamodule.get_dataloader(dataset_split)

        local_preds = []
        local_probs = []
        local_labels = []

        with torch.no_grad():
            for i, (X, y) in enumerate(dataloader):
                X = X.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)

                logits = model(X)
                probs = F.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1)
                conf = probs.values

                local_preds.append(pred)
                local_probs.append(conf)
                local_labels.append(y)

        local_preds = torch.cat(local_preds)
        local_probs = torch.cat(local_probs)
        local_labels = torch.cat(local_labels)

        world_size = torch.distributed.get_world_size()

        def alloc_like(x):
            return [torch.zeros_like(x) for _ in range(world_size)]

        gathered_preds = alloc_like(local_preds)
        gathered_probs = alloc_like(local_probs)
        gathered_labels = alloc_like(local_labels)

        torch.distributed.all_gather(gathered_preds, local_preds)
        torch.distributed.all_gather(gathered_probs, local_probs)
        torch.distributed.all_gather(gathered_labels, local_labels)

        if torch.distributed.get_rank() == 0:
            all_preds = torch.cat([t.cpu() for t in gathered_preds]).numpy()
            all_probs = torch.cat([t.cpu() for t in gathered_probs]).numpy()
            all_labels = torch.cat([t.cpu() for t in gathered_labels]).numpy()

            df = (
                pd.DataFrame({
                    "label": all_labels,
                    "prediction": all_preds,
                    "confidence": all_probs,
                })
                .sort_values("index")
                .reset_index(drop=True)
            )

            # Use the configuration to determine save path
            save_dir = os.path.join(configuration.log_dir, configuration.experiment_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"predictions_{dataset_split}.csv")

            df.to_csv(save_path, index=False)
            log.info(f"Saved predictions to: {save_path}")


class PredictorFactory:
    @staticmethod
    def get_predictor(config_manager) -> Predictor:
        """
        get predictor from configuration

        args:
            config_manager: configuration manager for managing configurations

        returns:
            Predictor: the predictor for prediction
        """
        configuration = config_manager.configuration
        hyperparams = config_manager.hyperparams

        datamodule = DataModuleFactory.get_datamodule(configuration, hyperparams)
        num_classes = datamodule.get_num_classes()

        model, transforms = ModelFactory.get_model(
            configuration, hyperparams, num_classes
        )
        datamodule.initialize(transforms)

        callback_handler = CallbackHandler(
            CallbackFactory.get_callbacks(configuration, hyperparams)
        )

        predictor = Predictor(
            model=model,
            datamodule=datamodule,
            dataset_split=configuration.dataset_split,
            callback_handler=callback_handler,
        )

        return predictor
