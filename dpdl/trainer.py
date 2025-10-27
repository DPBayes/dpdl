
from distutils import core
import os
import logging
import math
from collections.abc import Mapping
import opacus
import torch
import torchmetrics

from opacus.utils.batch_memory_manager import BatchMemoryManager

from .models.model_factory import ModelFactory
from .callbacks.callback_factory import CallbackHandler, CallbackFactory
from .configurationmanager import ConfigurationManager, Configuration, Hyperparameters
from .datamodules import DataModule, DataModuleFactory
from .optimizers import OptimizerFactory
from .loss_factory import LossFactory
from .metrics_factory import MetricsFactory
from .utils import seed_everything, shift_and_flatten

log = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,

        # essentials
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        datamodule: DataModule,

        # generic params
        epochs: int = 10,
        total_steps: int = None,
        validation_frequency: int = 1,
        seed: int = 0,
        physical_batch_size: int = 40,
        callback_handler: CallbackHandler = None,
        llm: bool = None,
        peft: str = None,
        task: str = None
    ):

        self.model = model
        self.optimizer = optimizer
        self.datamodule = datamodule
        self.epochs = epochs
        self.total_steps = total_steps
        self.validation_frequency = validation_frequency
        self.seed = seed
        self.physical_batch_size = physical_batch_size
        self.llm = llm
        self.peft = peft
        self.task = task

        if not callback_handler:
            self.callback_handler = CallbackHandler()
        else:
            self.callback_handler = callback_handler

        if self.epochs and self.total_steps:
            raise ValueError('You should provide either "epochs" or "total_steps", not both.')

        self.setup()
        #self.model = self.model.cuda()
    
    def setup(self):
        self.model = self.model.cuda()
        print('setup model before parallel', self.model)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        print('setup model after parallel', self.model)

    def fit(self):
        self.callback_handler.call('on_train_start', self)

        if self.total_steps:
            self._fit_total_steps()
        else:
            self._fit_epochs()

        self.callback_handler.call('on_train_end', self)

    def _fit_epochs(self):
        self.device = torch.cuda.current_device()
        for epoch in range(self.epochs):
            self.fit_one_epoch(epoch)

            if self.validation_frequency and epoch % self.validation_frequency == 0:
                if torch.distributed.get_rank() == 0:
                    self.validate(epoch)

                # other ranks will wait for validation
                torch.distributed.barrier()

    def _fit_total_steps(self):
        step = 0
        virtual_epoch = 0
        steps_per_epoch = self._calculate_steps_per_epoch()

        # start the first virtual epoch
        self._handle_virtual_epoch_start(virtual_epoch)

        while step < self.total_steps:
            for batch_idx, batch in enumerate(self.datamodule.get_dataloader('train')):
                if step >= self.total_steps:
                    break

                self.callback_handler.call('on_train_batch_start', self, batch_idx, batch)
                logical_batch_loss = self.fit_one_batch(batch_idx, batch)
                self.callback_handler.call('on_train_batch_end', self, batch_idx, batch, logical_batch_loss)

                step += 1

                if step % steps_per_epoch == 0:
                    self._handle_virtual_epoch_end(virtual_epoch)
                    virtual_epoch += 1

                    if self.validation_frequency and virtual_epoch % self.validation_frequency == 0:
                        if torch.distributed.get_rank() == 0:
                            self.validate(virtual_epoch)

                        # other ranks will wait for validation
                        torch.distributed.barrier()

                    # are we finished?
                    if step >= self.total_steps:
                        break

                    # start the next virtual epoch
                    self._handle_virtual_epoch_start(virtual_epoch)

        last_step_in_epoch = step % steps_per_epoch
        if last_step_in_epoch != 0:
            self._handle_virtual_epoch_end(virtual_epoch)

        assert step == self.total_steps, f'Mismatch in total steps count: Expected {self.total_steps} total steps, but stepped {step} times!'

    def _handle_virtual_epoch_start(self, epoch):
        self.model.train()
        self.callback_handler.call('on_train_epoch_start', self, epoch)

    def _handle_virtual_epoch_end(self, epoch):
        # compute the epoch metrics
        metrics = self._unwrap_model().train_metrics.compute()
        self._unwrap_model().train_metrics.reset()

        self.callback_handler.call('on_train_epoch_end', self, epoch, metrics)

    def fit_one_epoch(self, epoch):
        self.model.train()
        
        self.callback_handler.call('on_train_epoch_start', self, epoch)

        for batch_idx, batch in enumerate(self.datamodule.get_dataloader('train')):

            print('for batch idx',batch_idx, 'we have batch:\n',batch)
            
            self.callback_handler.call('on_train_batch_start', self, batch_idx, batch)
            if self.task in ['CausalLM','InstructLM']:
                logical_batch_loss = self.fit_one_batch_causal(batch_idx, batch)
            else:
                logical_batch_loss = self.fit_one_batch(batch_idx, batch)
            self.callback_handler.call('on_train_batch_end', self, batch_idx, batch, logical_batch_loss)
            
            print('---------------------------------- end batch ------------------------')

        # compute the epoch metrics
        metrics = self._unwrap_model().train_metrics.compute()
        self._unwrap_model().train_metrics.reset()

        if self.task == 'InstructLM':
            self.sample()

        self.callback_handler.call('on_train_epoch_end', self, epoch, metrics)


    def fit_one_batch_causal(self, batch_idx, batch):
        X, y = batch
        X = X.to(device= self.device, non_blocking=True)
        y = y.to(device= self.device, non_blocking=True)

        is_mapping = isinstance(X, Mapping)  # covers dict and HF BatchEncoding
        if is_mapping:
            for k, v in X.items():
                if isinstance(v, torch.Tensor):
                    X[k] = v.to(device=self.device, non_blocking=True)
        else:
            X = X.to(device=self.device, non_blocking=True)
        y = y.to(device=self.device, non_blocking=True)

        # gradient accumulation. split the batch to sub batches that fit in the GPU memory.
        # then process the sub batches one at a time and call backward.
        # when all the sub batches have been processed we can finally step the optimizer.
        if is_mapping:
            # split each tensor in the dict
            X_split = {k: v.split(self.physical_batch_size, dim=0) for k, v in X.items()}
            y_split = y.split(self.physical_batch_size, dim=0)
        else:
            X_split = X.split(self.physical_batch_size, dim=0)
            y_split = y.split(self.physical_batch_size, dim=0)
        
        N = len(y_split)

        # zero the grads as usually before doing anything
        self.optimizer.zero_grad()

        logical_batch_loss = 0
        # process the sub batches one at a time
        print('Number of physical batches', N)

        for i in range(N):
            if is_mapping:
                X_splitted = {k: X_split[k][i] for k in X_split}
            else:
                X_splitted = X_split[i]
            
            y_splitted = y_split[i]
            physical_batch = (X_splitted, y_splitted)

            self.callback_handler.call('on_train_physical_batch_start', self, i, physical_batch)

            logits = self.model(X_splitted)

            preds, y_splitted_flatten = shift_and_flatten(logits, y_splitted)
            preds_flat = torch.argmax(preds, dim=-1)

            #Loss needs the logits flatten
            loss = self._unwrap_model().criterion(preds, y_splitted_flatten) / N  # NB: normalize loss
            print('one batch loss',loss)
            loss.backward()
            logical_batch_loss += loss.item()
            print('logical batch loss',logical_batch_loss)

            #Perplexity needs the normal logits [batch_size, seq_len, vocab_size]
            self._unwrap_model().train_metrics['Perplexity'].update(logits, y_splitted)

            self._unwrap_model().train_metrics['MulticlassAccuracy'].update(preds_flat, y_splitted_flatten)
            # notify the callbacks of a physical batch end
            self.callback_handler.call('on_train_physical_batch_end', self, i, physical_batch, loss.item())
        
        # after accumulating the gradients for all the sub batches we can finally update weights.
        self.optimizer.step()

        return logical_batch_loss

    def fit_one_batch(self, batch_idx, batch):
        X, y = batch
        X = X.to(device= self.device, non_blocking=True)
        y = y.to(device= self.device, non_blocking=True)

        is_mapping = isinstance(X, Mapping)  # covers dict and HF BatchEncoding
        if is_mapping:
            for k, v in X.items():
                if isinstance(v, torch.Tensor):
                    X[k] = v.to(device=self.device, non_blocking=True)
        else:
            X = X.to(device=self.device, non_blocking=True)
        y = y.to(device=self.device, non_blocking=True)

        # gradient accumulation. split the batch to sub batches that fit in the GPU memory.
        # then process the sub batches one at a time and call backward.
        # when all the sub batches have been processed we can finally step the optimizer.
        if is_mapping:
            # split each tensor in the dict
            X_split = {k: v.split(self.physical_batch_size, dim=0) for k, v in X.items()}
            y_split = y.split(self.physical_batch_size, dim=0)
        else:
            X_split = X.split(self.physical_batch_size, dim=0)
            y_split = y.split(self.physical_batch_size, dim=0)
        
        N = len(y_split)

        # check the splits
        print("[DEBUG] check the splits")
        for k, v in X_split.items():
            print(f"length of {k}:", len(v))
            print(f"shape of {k}:", v[0].shape)
        print("length of y_split:", len(y_split))

        # zero the grads as usually before doing anything
        self.optimizer.zero_grad()

        logical_batch_loss = 0
        # process the sub batches one at a time
        print('Number of physical batches', N)

        for i in range(N):
            if is_mapping:
                X_splitted = {k: X_split[k][i] for k in X_split}
            else:
                X_splitted = X_split[i]
            
            y_splitted = y_split[i]
            physical_batch = (X_splitted, y_splitted)

            self.callback_handler.call('on_train_physical_batch_start', self, i, physical_batch)

            logits = self.model(X_splitted)
            print("logits: ", logits)
            print('logits shape',logits.shape)
            preds = torch.argmax(logits, dim=1)

            loss = self._unwrap_model().criterion(logits, y_splitted) / N  # NB: normalize loss
            print('one batch loss',loss)
            loss.backward()
            logical_batch_loss += loss.item()
            print('logical batch loss',logical_batch_loss)
            
            # update the metrics if there are any

            #preds = torch.argmax(logits, dim=1)

            self._unwrap_model().train_metrics.update(preds, y_splitted)
            #self.unwrap_llm_model().train_metrics.update(preds, y_split[i])

            # notify the callbacks of a physical batch end
            self.callback_handler.call('on_train_physical_batch_end', self, i, physical_batch, loss.item())
        
        # after accumulating the gradients for all the sub batches we can finally update weights.
        self.optimizer.step()

        return logical_batch_loss

    def unwrap_llm_model(self, m, target="base"):
        """
        Unwraps through DDP/DataParallel (.module) and wrappers (.model).

        target:
        - "base"    -> ModelBase        (outermost wrapper)
        - "hf_llm"  -> HF_llm           (HF wrapper)
        - "hf_core" -> HF core model    (e.g., BertForSequenceClassification)
        """
        # remove DDP/DataParallel
        while hasattr(m, "module"):
            m = m.module

        # ModelBase
        if target == "base":
            return m  

        # ModelBase -> HF_llm
        if hasattr(m, "model"):
            m = m.model
        if target == "hf_llm":
            return m  

        # HF_llm -> HF core
        if hasattr(m, "model"):
            m = m.model
        return m  
    
    
    def validate(self, epoch=None, enable_callbacks=True):
        return self._evaluate('validation', epoch, enable_callbacks)

    def test(self):
        return self._evaluate('test')

    def get_dataloader(self, name):
        return self.datamodule.get_dataloader(name)

    def get_datamodule(self):
        return self.datamodule

    def _evaluate(self, mode, epoch=None, enable_callbacks=True):
        if enable_callbacks:
            self.callback_handler.call(f'on_{mode}_epoch_start', self, epoch)

        self.model.eval()
        torch.set_grad_enabled(False)

        # record the loss separately, as we need to return it when
        # performing hyperparameter optimization
        evaluation_loss = 0

        if mode == 'validation':
            dataloader_name = 'valid'
            metrics_evaluator = self._unwrap_model().valid_metrics
        elif mode == 'test':
            dataloader_name = 'test'
            metrics_evaluator = self._unwrap_model().test_metrics
        else:
            raise ValueError(f'Unknown evaluation mode: "{mode}"')

        dataloader = self.datamodule.get_dataloader(dataloader_name)

        metrics_evaluator.reset()

        for batch_idx, batch in enumerate(dataloader):
            loss = self._evaluate_one_batch(mode, batch_idx, batch, enable_callbacks, metrics_evaluator)
            evaluation_loss += loss

        evaluation_loss /= len(dataloader)

        metrics = metrics_evaluator.compute()

        torch.set_grad_enabled(True)
        self.model.train()

        if enable_callbacks:
            self.callback_handler.call(f'on_{mode}_epoch_end', self, epoch, metrics)

        return evaluation_loss, metrics

    def _evaluate_one_batch(self, mode, batch_idx, batch, enable_callbacks, metrics_evaluator):
        if enable_callbacks:
            self.callback_handler.call(f'on_{mode}_batch_start', self, batch_idx, batch)

        X, y = batch
        X = X.to(device = self.device, non_blocking=True)
        y = y.to(device = self.device, non_blocking=True)
        
        if isinstance(X, Mapping):
            for k, v in X.items():
                X[k] = v.to(device=self.device, non_blocking=True)
        else:
            X = X.to(device=self.device, non_blocking=True)
        y = y.to(device=self.device, non_blocking=True)

        logits = self.model(X)
        if self.task in ['CausalLM','InstructLM']:
            preds, y_flatten = shift_and_flatten(logits, y)
            loss = self._unwrap_model().criterion(preds, y_flatten)
        else:
            loss = self._unwrap_model().criterion(logits, y)
        loss = loss.item()

        if self.task in ['CausalLM','InstructLM']:

            preds_flat = torch.argmax(preds, dim=1)

            metrics_evaluator['Perplexity'].update(logits, y)

            metrics_evaluator['MulticlassAccuracy'].update(preds_flat, y_flatten)

        else:
            preds = torch.argmax(logits, dim=1)
            metrics_evaluator.update(preds, y)

        if enable_callbacks:
            self.callback_handler.call(f'on_{mode}_batch_end', self, batch_idx, batch, loss)

        return loss

    def _unwrap_model(self):
        # the model is wrapped inside torch distributed,
        # here we just return the vanilla model
        #return self.model.module
        
        m = self.model
        # remove DDP/DataParallel
        while hasattr(m, "module"):
            m = m.module

        return m  #ModelBase

    def _calculate_steps_per_epoch(self):
        N = len(self.datamodule.get_dataloader('train').dataset)
        B = self.datamodule.batch_size
        return math.ceil(N / B)

    def save_model(self, fpath):

        if self.peft is not None and self.peft == 'lora':
            # Extract the directory from the path
            directory = os.path.dirname(fpath)

            # Create the directory if it doesn't exist
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            self._unwrap_model().save_pretrained(fpath)

        else:
            self._unwrap_model().save_model(fpath)

    def sample(self):

        self.model.eval()

        with torch.no_grad():

            for batch_idx, batch in enumerate(self.datamodule.get_dataloader('sample')):
                
                print('sample',batch)

                X = batch
                X = X.to(device= self.device, non_blocking=True)

                is_mapping = isinstance(X, Mapping)  # covers dict and HF BatchEncoding
                if is_mapping:
                    for k, v in X.items():
                        if isinstance(v, torch.Tensor):
                            X[k] = v.to(device=self.device, non_blocking=True)
                else:
                    X = X.to(device=self.device, non_blocking=True)

                # gradient accumulation. split the batch to sub batches that fit in the GPU memory.
                # then process the sub batches one at a time and call backward.
                # when all the sub batches have been processed we can finally step the optimizer.
                if is_mapping:
                    # split each tensor in the dict
                    X_split = {k: v.split(self.physical_batch_size, dim=0) for k, v in X.items()}
                else:
                    X_split = X.split(self.physical_batch_size, dim=0)
                  
                N = len(X_split)

                # process the sub batches one at a time
                print('Number of physical batches', N)

                for i in range(N):
                    if is_mapping:
                        X_splitted = {k: X_split[k][i] for k in X_split}
                    else:
                        X_splitted = X_split[i]
                    print(X_splitted)
                    generated_ids = self._unwrap_model().generate(X_splitted, max_new_tokens=128, temperature=0.7, do_sample=True)
                    print('sampled text decoded',self.datamodule.decode(generated_ids[0]))


class DifferentiallyPrivateTrainer(Trainer):
    def __init__(
        self,
        *,
        # privacy params
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        clipping_mode: str = 'flat',
        accountant: str = 'prv',
        poisson_sampling: bool = True,
        normalize_clipping: bool = False,
        secure_mode: bool = False,
        target_epsilon: float = None,
        target_delta: float = None,
        noise_batch_ratio: float = None,
        seed: int = 0,
        **kwargs,
    ):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.clipping_mode = clipping_mode
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.noise_batch_ratio = noise_batch_ratio
        self.seed = seed
        self.poisson_sampling = poisson_sampling
        self.normalize_clipping = normalize_clipping

        # setup opacus privacy engine
        privacy_engine_args = {
            'accountant': accountant,
            'secure_mode': secure_mode,
        }

        self.privacy_engine = opacus.PrivacyEngine(**privacy_engine_args)

        super().__init__(seed=seed, **kwargs)

    def _has_target_privacy_params(self):
        if self.target_epsilon == -1:
            return False

        if not self.target_epsilon:
            return False

        if self.target_epsilon and not self.target_delta:
            raise ValueError('Parameter "target_epsilon" and "target_delta" not given.')

        if self.noise_batch_ratio and not self.target_delta:
            raise ValueError('Parameter "target_epsilon" and "target_delta" not given.')

        if all([self.target_epsilon, self.noise_batch_ratio]):
            raise ValueError('Parameters "target_epsilon" and "noise_batch_ratio" are exclusive.')

        if all([self.target_epsilon, self.noise_multiplier]):
            raise ValueError('Parameters "target_epsilon" and "noise_multiplier" are exlusive.')

        if all([self.noise_batch_ratio, self.noise_multiplier]):
            raise ValueError('Parameters "noise_batch_ratio" and "noise_multiplier" are exclusive.')

        if self.target_epsilon and not self.target_delta:
            raise ValueError('Parameter "target_epsilon" present, but "target_delta" is missing.')

        return True

    def setup(self):
        noise_generator = torch.Generator(device=torch.cuda.current_device())
        if self.seed:
            noise_generator.manual_seed(self.seed)

        self.model = self.model.cuda()

        # let's be distributed by default and wrap the model for Opacus DDP.
        # DifferentiallyPrivateDistributedDataParallel is actually a no-op in Opacus, but
        # let's wrap anyway in case of future api changes. https://opacus.ai/tutorials/ddp_tutorial
        model = opacus.distributed.DifferentiallyPrivateDistributedDataParallel(self.model)

        print('The model after opacus DDP', model)

        optimizer = self.optimizer
        train_dataloader = self.datamodule.get_dataloader('train')

        # setup differential privacy for the model, optimize, and dataloader
        if self._has_target_privacy_params():
            dp_model, dp_optimizer, dp_dataloader = self.privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                max_grad_norm=self.max_grad_norm,
                clipping=self.clipping_mode,
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                epochs=self.epochs,
                noise_generator=noise_generator,
                poisson_sampling=self.poisson_sampling,
                normalize_clipping=self.normalize_clipping,
                total_steps=self.total_steps,
            )
        else:
            if self.target_epsilon == -1:
                self.noise_multiplier = 0

            if self.noise_batch_ratio:
                self.noise_multiplier = self.noise_batch_ratio * self.datamodule.batch_size

            dp_model, dp_optimizer, dp_dataloader = self.privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                clipping=self.clipping_mode,
                noise_generator=noise_generator,
                poisson_sampling=self.poisson_sampling,
                normalize_clipping=self.normalize_clipping,
                total_steps=self.total_steps,
            )

        # now we can start using the DP'ifyed stuff
        self.model = dp_model
        print("DP model: ", self.model)
        self.datamodule.set_dataloader('train', dp_dataloader)
        self.optimizer = dp_optimizer

    def get_epsilon(self):
        return self.privacy_engine.get_epsilon(self.target_delta)

    def _unwrap_model(self):
        # the model is wrapped inside Opacus, and Opacus distributed.
        # let's unwrap the vanilla model and return it
        return self.model._module.module

    def _fit_total_steps(self):
        # here we'll keep track of our approximate epochs
        virtual_epoch = 0

        # number of total steps taken
        step = 0

        # number of logical batches in an approximate epoch
        n_logical_batches = 0

        # track the logical batch loss here
        logical_batch_loss = 0

        # track the number of physical batches in a logical batch
        n_physical_batch_in_logical = 0

        # flag to indicate the beginning of a new logical batch
        logical_batch_begin = True

        # flag to indicate that a logical batch has been completed (set via the optimizer check)
        logical_batch_completed = False

        # to calculate the start/end of an epoch, we need the number
        # of steps in an epoch.
        steps_per_epoch = self._calculate_steps_per_epoch()

        # At the very start, call on_train_batch_start for the first logical batch.
        if logical_batch_begin:
            self.callback_handler.call('on_train_batch_start', self, n_logical_batches, None)
            logical_batch_begin = False

        # if 'total_steps' is set then Opacus will do the stepping for us, or
        # more precisely: the dataloader will have exactly 'total_steps' batches.
        # Here, we will spend approximately an epoch worth of those.
        with BatchMemoryManager(
            data_loader=self.datamodule.get_dataloader('train'),
            max_physical_batch_size=self.physical_batch_size,
            optimizer=self.optimizer,
        ) as virtual_dataloader:
            for batch_idx, batch in enumerate(virtual_dataloader):
                # first batch, we can start first epoch
                if batch_idx == 0:
                    self._handle_virtual_epoch_start(virtual_epoch)

                # now, let's check if we are going to reach the end of logical batch.
                # the optimizer will not skip next gradient update if we are not at
                # the end of the logical batch. there's currently pretty much no other
                # way to do it than this, because we don't know the size of the logical
                # batch that was sampled.
                if not self.optimizer._check_skip_next_step(False):
                    step += 1
                    logical_batch_completed = True
                else:
                    logical_batch_completed = False

                # notify the callbacks of a physical batch start
                self.callback_handler.call('on_train_physical_batch_start', self, batch_idx, batch)

                # let's fit this physical batch


                if self.task in ['CausalLM','InstructLM']:
                    batch_loss = self.fit_one_batch_causal(batch_idx, batch)
                else:
                    batch_loss = self.fit_one_batch(batch_idx, batch)
                #batch_loss = self.fit_one_batch(batch_idx, batch)

                # notify the callbacks of a physical batch end
                self.callback_handler.call('on_train_physical_batch_end', self, batch_idx, batch, batch_loss)

                # accumulate loss and count the number of physical batches in a logical batch
                logical_batch_loss += batch_loss
                n_physical_batch_in_logical += 1

                # if the logical batch is complete, notify batch end and reset counters
                if logical_batch_completed:
                    self.callback_handler.call(
                        'on_train_batch_end',
                        self,
                        n_logical_batches,
                        None,
                        logical_batch_loss / n_physical_batch_in_logical,  # mean of physical batch losses
                    )
                    n_logical_batches += 1
                    logical_batch_loss = 0
                    n_physical_batch_in_logical = 0

                    # the next iteration starts a new logical batch
                    logical_batch_begin = True

                # At the beginning of a new logical batch, call on_train_batch_start.
                if logical_batch_begin:
                    self.callback_handler.call('on_train_batch_start', self, n_logical_batches, None)
                    logical_batch_begin = False

                # and next we check for epoch end
                if (logical_batch_completed and step % steps_per_epoch == 0) or step == self.total_steps:
                    self._handle_virtual_epoch_end(virtual_epoch)

                    if self.validation_frequency and virtual_epoch % self.validation_frequency == 0:
                        # validate only on rank 0. no need to do distributed here,
                        # the computation is not heavy because we don't need gradients.
                        if torch.distributed.get_rank() == 0:
                            self.validate(virtual_epoch)

                        # other ranks will wait for validation
                        torch.distributed.barrier()

                    if step < self.total_steps:
                        virtual_epoch += 1
                        self._handle_virtual_epoch_start(virtual_epoch)
                        # Start a new logical batch for the new epoch.
                        self.callback_handler.call('on_train_batch_start', self, n_logical_batches, None)
                        logical_batch_begin = False

                # Reset the logical batch completion flag for the next iteration.
                logical_batch_completed = False

        if step != self.total_steps:
            log.warn(f'Was going to step for {self.total_steps}, but stepped only {step} steps.')

    def fit_one_batch(self, batch_idx, batch):

        self.optimizer.zero_grad()

        X, y = batch
        
        is_mapping = isinstance(X, Mapping)  # covers dict and HF BatchEncoding
        if is_mapping:
            for k, v in X.items():
                if isinstance(v, torch.Tensor):
                    X[k] = v.to(device=self.device, non_blocking=True)
        else:
            X = X.to(device=self.device, non_blocking=True)
        y = y.to(device=self.device, non_blocking=True)
        
        logits = self.model(X)

        print("logits: ", logits)
        
        loss = self._unwrap_model().criterion(logits, y)
        loss.backward()
        print('one batch loss',loss)
        
        self.optimizer.step()

        loss = loss.item()

        # update metrics if there are any
        preds = torch.argmax(logits, dim=1)
        self._unwrap_model().train_metrics.update(preds, y)

        return loss
    
    def fit_one_batch_causal(self, batch_idx, batch):
        self.optimizer.zero_grad()

        X, y = batch
        
        is_mapping = isinstance(X, Mapping)  # covers dict and HF BatchEncoding
        if is_mapping:
            for k, v in X.items():
                if isinstance(v, torch.Tensor):
                    X[k] = v.to(device=self.device, non_blocking=True)
        else:
            X = X.to(device=self.device, non_blocking=True)
        y = y.to(device=self.device, non_blocking=True)

        logits = self.model(X)

        preds, y_flatten = shift_and_flatten(logits, y)
        preds_flat = torch.argmax(preds, dim=-1)

        #Loss needs the logits flatten
        loss = self._unwrap_model().criterion(preds, y_flatten)
        loss.backward()

        self.optimizer.step()

        loss = loss.item()

        self._unwrap_model().train_metrics['Perplexity'].update(logits, y)

        self._unwrap_model().train_metrics['MulticlassAccuracy'].update(preds_flat, y_flatten)

        return loss

    def fit_one_epoch(self, epoch):
        self.model.train()
        self.callback_handler.call('on_train_epoch_start', self, epoch)

        logical_idx = 0
        logical_loss = 0.0
        phys_in_logical = 0
        in_new_logical = True

        with BatchMemoryManager(
            data_loader=self.datamodule.get_dataloader('train'),
            max_physical_batch_size=self.physical_batch_size,
            optimizer=self.optimizer,
        ) as virtual_dataloader:

            for phys_idx, batch in enumerate(virtual_dataloader):

                # if we're starting a new logical batch, signal start
                if in_new_logical:
                    self.callback_handler.call(
                        'on_train_batch_start', self, logical_idx, None
                    )
                    in_new_logical = False

                # physical‐batch callbacks
                self.callback_handler.call(
                    'on_train_physical_batch_start', self, phys_idx, batch
                )

                print('for batch idx',phys_idx, 'we have batch:\n',batch)

                if self.task in ['CausalLM','InstructLM']:
                    loss = self.fit_one_batch_causal(phys_idx, batch)
                else:
                    loss = self.fit_one_batch(phys_idx, batch)

                #loss = self.fit_one_batch(phys_idx, batch)

                self.callback_handler.call(
                    'on_train_physical_batch_end', self, phys_idx, batch, loss
                )

                # accumulate
                logical_loss += loss
                phys_in_logical += 1

                # check for logical‐batch boundary
                if not self.optimizer._check_skip_next_step(False):
                    avg = logical_loss / phys_in_logical
                    self.callback_handler.call(
                        'on_train_batch_end',
                        self,
                        logical_idx,
                        None,
                        avg,
                    )
                    logical_idx += 1
                    logical_loss = 0.0
                    phys_in_logical = 0
                    in_new_logical = True

        # wrap up epoch
        metrics = self._unwrap_model().train_metrics.compute()
        self._unwrap_model().train_metrics.reset()
        self.callback_handler.call('on_train_epoch_end', self, epoch, metrics)

    def save_model(self, fpath):
        self.model.module.save_model(fpath)


class TrainerFactory:
    @staticmethod
    def get_trainer(config_manager: ConfigurationManager) -> Trainer:

        if seed := config_manager.configuration.privacy:
            seed_everything(seed)

        # are we differentially private?
        if config_manager.configuration.privacy:
            return TrainerFactory._get_differentially_private_trainer(config_manager.configuration, config_manager.hyperparams)
        
        if config_manager.configuration.checkpoint_step_interval is not None:
            config_manager.configuration.checkpoints_dir = os.path.join(config_manager.configuration.log_dir,config_manager.configuration.experiment_name, 'checkpoints')

        return TrainerFactory._get_basic_trainer(config_manager.configuration, config_manager.hyperparams)

    @staticmethod
    def _get_basic_trainer(configuration: Configuration, hyperparams: Hyperparameters) -> Trainer:

        #
        # First create DataModule, it can figure out the number of classes
        
        datamodule = DataModuleFactory.get_datamodule(configuration, hyperparams)
        num_classes = datamodule.get_num_classes()
        # setup data, model, and optimizer
        loss_fn = LossFactory.get_loss(configuration)
        if num_classes is None:
            model, transforms = ModelFactory.get_model(configuration, hyperparams, num_classes, loss_fn, None)        
            num_classes = model.config.vocab_size
            metrics = MetricsFactory.get_metrics(configuration, num_classes)
            model.set_metrics(metrics)
        else:
            metrics = MetricsFactory.get_metrics(configuration, num_classes)
            model, transforms = ModelFactory.get_model(configuration, hyperparams, num_classes, loss_fn, metrics)        

        optimizer = OptimizerFactory.get_optimizer(configuration, hyperparams, model)

        # Initialize the datamodule with the transformations
        datamodule.initialize(transforms)

        # should we cache outputs from the feature extractor?
        if configuration.cache_features:
            # compute cache on rank 0 only
            if torch.distributed.get_rank() == 0:
                datamodule.cache_features(model)
                torch.distributed.barrier()
            else:
                torch.distributed.barrier()
                datamodule.cache_features(model)

        callback_handler = CallbackHandler(
            CallbackFactory.get_callbacks(configuration, hyperparams)
        )

        epochs, total_steps = TrainerFactory._get_epochs_and_steps(configuration, hyperparams, datamodule)

        # instantiate a trainer without dp
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            datamodule=datamodule,
            callback_handler=callback_handler,
            physical_batch_size=configuration.physical_batch_size,
            epochs=epochs,
            total_steps=total_steps,
            seed=configuration.seed,
            validation_frequency=configuration.validation_frequency,
            llm=configuration.llm,
            peft=configuration.peft,
            task=configuration.task
        )

        return trainer

    @staticmethod
    def _get_differentially_private_trainer(configuration: Configuration, hyperparams: Hyperparameters) -> Trainer:
        # Target delta calculation: A common heuristic is to use 1/N', with N'
        # being the size of the dataset rounded up to the nearest power of 10.
        # To avoid too large values of delta, let's pick a somewhat sensible
        # minimum of 1e-5.
        def _round_up_to_nearest_power_of_10(n):
            return 10 ** math.ceil(math.log10(n))

        def _calculate_target_delta(N):
            N_prime = _round_up_to_nearest_power_of_10(N)
            return min(1e-5, 1 / N_prime)

        def _get_target_privacy_params(hyperparams):
            N = len(datamodule.get_dataloader('train').dataset)
            target_delta = _calculate_target_delta(N)

            if torch.distributed.get_rank() == 0:
                log.info(f'Dataset size is {N}, setting target delta to: {target_delta}.')

            # are we given a target epsilon?
            if hyperparams.target_epsilon is not None:
                target_epsilon = hyperparams.target_epsilon
            else:
                target_epsilon = None

            return target_delta, target_epsilon

        # First initialize the DataModule, it will know about the number of classes
        datamodule = DataModuleFactory.get_datamodule(configuration, hyperparams)
        num_classes = datamodule.get_num_classes()

        # Now, setup data, model, and optimizer
        loss_fn = LossFactory.get_loss(configuration)

        if num_classes is None:
            model, transforms = ModelFactory.get_model(configuration, hyperparams, num_classes, loss_fn, None)        
            num_classes = model.config.vocab_size
            metrics = MetricsFactory.get_metrics(configuration, num_classes)
            model.set_metrics(metrics)
        else:
            metrics = MetricsFactory.get_metrics(configuration, num_classes)
            model, transforms = ModelFactory.get_model(configuration, hyperparams, num_classes, loss_fn, metrics)

        if not opacus.validators.ModuleValidator.is_valid(model):
            print('a module is not valid')
            model = opacus.validators.ModuleValidator.fix(model)
        optimizer = OptimizerFactory.get_optimizer(configuration, hyperparams, model)

        # The datamodule needs to be aware of the transformations, now we can initialize it
        datamodule.initialize(transforms)

        # Are we caching the outputs of the feature extractor
        if configuration.cache_features:
            # compute cache on rank 0 only
            if torch.distributed.get_rank() == 0:
                datamodule.cache_features(model)
                torch.distributed.barrier()
            else:
                torch.distributed.barrier()
                datamodule.cache_features(model)

        callback_handler = CallbackHandler(
            CallbackFactory.get_callbacks(configuration, hyperparams)
        )

        target_delta, target_epsilon = _get_target_privacy_params(hyperparams)
        epochs, total_steps = TrainerFactory._get_epochs_and_steps(configuration, hyperparams, datamodule)

        # instantiate a differentialy private trained
        trainer = DifferentiallyPrivateTrainer(
            model=model,
            optimizer=optimizer,
            datamodule=datamodule,
            # hypers
            epochs=epochs,
            total_steps=total_steps,
            noise_multiplier=hyperparams.noise_multiplier,
            max_grad_norm=hyperparams.max_grad_norm,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            noise_batch_ratio=hyperparams.noise_batch_ratio,
            poisson_sampling=configuration.poisson_sampling,
            normalize_clipping=configuration.normalize_clipping,
            # config
            accountant=configuration.accountant,
            secure_mode=configuration.secure_mode,
            clipping_mode=configuration.clipping_mode,
            physical_batch_size=configuration.physical_batch_size,
            seed=configuration.seed,
            callback_handler=callback_handler,
            validation_frequency=configuration.validation_frequency,
            llm=configuration.llm,
            peft=configuration.peft,
            task=configuration.task
        )

        return trainer

    @staticmethod
    def _get_epochs_and_steps(
        configuration: Configuration,
        hyperparams: Hyperparameters,
        datamodule: DataModule,
    ):
        """
        Compute the number of training epochs and total optimizer steps.

        If `use_steps=True`, we convert epochs to total_steps using ceil(N / B),
        which matches the default logic in Opacus:
            - sample_rate = 1 / ceil(N / B)
            - steps = int(1 / sample_rate) = ceil(N / B)

        However, default Opacus might still make more steps than us, because we
        cap the total number of steps exactly at `total_steps` and Opacus default
        (`use_steps=False`) always makes a full pass on the dataloader when feeding
        batches through the BatchMemoryManager.

        Returns:
            (epochs, total_steps): One of the values will be None depending on mode.
        """

        # If we're using step-based training and the number of epochs is specified,
        # convert epochs to total steps using the default Opacus logic.
        if configuration.use_steps and hyperparams.epochs:
            dataloader = datamodule.get_dataloader('train')

            # Match Opacus: steps_per_epoch = ceil(N / B)
            N = len(dataloader.dataset)
            B = datamodule.batch_size
            steps_per_epoch = math.ceil(N / B)
            total_steps = steps_per_epoch * hyperparams.epochs
            epochs = None

        # If total steps are manually specified in config
        elif configuration.use_steps and hyperparams.total_steps:
            total_steps = hyperparams.total_steps
            epochs = None

        # Standard epoch-based training
        else:
            total_steps = None
            epochs = hyperparams.epochs

        return epochs, total_steps
