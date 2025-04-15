import torch
import logging
from .datamodules import DataModuleFactory
from .models.model_factory import ModelFactory
from .callbacks.callback_factory import CallbackHandler, CallbackFactory
from .utils import seed_everything

log = logging.getLogger(__name__)


class Predictor:
    def __init__(
        self,
        model: torch.nn.Module,
        datamodule,
        dataset_split: str,
        callback_handler: CallbackHandler,
        seed: int = 0,
    ):
        """
        init Predictor

        args:
            model (torch.nn.Module): the model for prediction
            datamodule: provide the class for data preprocessing and dataloader
            dataset_split (str): for a specific split of dataset, e.g., 'train', 'valid', 'test'
            callback_handler (CallbackHandler): callback handler for prediction
            seed (int): random seed for reproducibility
        """
        self.model = model
        self.datamodule = datamodule
        self.dataset_split = dataset_split
        self.callback_handler = callback_handler
        self.seed = seed

    def load_model(self, fpath: str):
        """
        load model from file

        args:
            fpath (optional, str): the path to the model file
        """
        raise NotImplementedError(
            "load_model function is not implemented in the Predictor class, will be done later"
        )
        try:
            self.model.load_model(fpath)
            log.info(f"load model from {fpath}")
        except AttributeError:
            state_dict = torch.load(fpath, map_location="cuda")
            self.model.load_state_dict(state_dict)
            log.info(f"load model from {fpath}")
        self.model.eval()

    def predict(self):
        """
        predict the dataset

        returns:
            torch.Tensor: the prediction result
        """
        self.model.eval()
        dataloader = self.datamodule.get_dataloader(self.dataset_split)
        predictions = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                self.callback_handler.call(
                    'on_predict_batch_start', self, batch_idx, batch
                )

                X = batch[0] if isinstance(batch, (list, tuple)) else batch
                X = X.cuda(non_blocking=True)
                logits = self.model(X)

                preds = torch.argmax(logits, dim=1)
                predictions.append(preds.cpu())

                self.callback_handler.call(
                    'on_predict_batch_end', self, batch_idx, batch, preds.cpu()
                )

        return torch.cat(predictions)


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
            seed=configuration.seed,
        )

        return predictor
