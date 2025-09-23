import logging
import os
from collections import OrderedDict
from typing import Optional

import torch
from torch.func import grad, vmap

from .callbacks.callback_factory import CallbackFactory, CallbackHandler
from .datamodules import DataModule, DataModuleFactory
from .experimentmanager import (
    save_gradient_diagnostics,
    save_predict_metrics,
    save_predictions,
)
from .models.model_base import ModelBase
from .models.model_factory import ModelFactory
from .trainer import Trainer, TrainerFactory
from .utils import tensor_to_python_type
from .configurationmanager import ConfigurationManager

log = logging.getLogger(__name__)


def _all_gather_object_list(local_list):
    world = torch.distributed.get_world_size()
    gathered = [None] * world
    torch.distributed.all_gather_object(gathered, list(local_list))

    merged = []
    for part in gathered:
        merged.extend(part)

    return merged


class Predictor:

    def __init__(
        self,
        trainer: Trainer,
        dataset_split: str,
        config_manager: ConfigurationManager,
        save_gradient_data: Optional[bool] = False,
    ):
        """
        init Predictor

        args:
            trainer (Trainer): initialized Trainer object
            dataset_split (str): for a specific split of dataset, e.g., 'train', 'valid', 'test'
            save_gradient_data (bool): boolean indicating if we should save also gradient data
        """
        self.trainer = trainer
        self.dataset_split = dataset_split
        self.config_manager = config_manager
        self.save_gradient_data = save_gradient_data

    def predict(self, configuration):
        """
        Perform prediction using the model on the specified dataset split.
        All ranks compute prediction in parallel using DDP, and rank 0 gathers
        the results and saves them as a CSV file.

        Args:
            configuration: A Configuration object that provides dataset_split and log path.
        """

        # disable possible dropout, etc
        model = self.trainer.model
        model.eval()

        dataloader = self.trainer.get_dataloader(self.dataset_split)

        local_preds = []
        local_probs = []
        local_labels = []

        if self.save_gradient_data:
            grad_records = []  # collect here (label, pred, norm, gradient)

        self.trainer._unwrap_model().train_metrics.reset()

        # NB: We are using torch.func.grad which builds it's own graph
        with torch.no_grad():
            for i, (X, y) in enumerate(dataloader):

                if torch.distributed.get_rank() == 0:
                    log.info(f' - Predicting on batch {i}')

                X = X.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)

                logits = model(X)
                probs = torch.nn.functional.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                self.trainer._unwrap_model().train_metrics.update(preds, y)

                local_preds.append(preds.cpu())
                local_probs.append(probs.cpu())
                local_labels.append(y.cpu())

                if self.save_gradient_data:
                    # Compute full model gradients (body + head)
                    full_grads = self._per_sample_full_grads(model, X, y)  # [B, full_g_dim]
                    
                    norms = full_grads.norm(dim=1).clamp_min(1e-12)
                    
                    # Store raw gradients and compute G_C and N_C for different clipping values
                    for i, (lbl, pred, norm) in enumerate(zip(y.tolist(), preds.tolist(), norms.tolist())):
                        grad_vector = full_grads[i].cpu().numpy()  # Raw gradient vector
                        record = {
                            'label': lbl, 
                            'pred': pred, 
                            'norm': norm,
                            'gradient': grad_vector.tolist()  # Store raw gradient as list
                        }
                        grad_records.append(record)

        metrics = self.trainer._unwrap_model().train_metrics.compute()

        torch.distributed.barrier()

        gathered_preds = _all_gather_object_list(local_preds)
        gathered_probs = _all_gather_object_list(local_probs)
        gathered_labels = _all_gather_object_list(local_labels)

        if torch.distributed.get_rank() == 0:
            all_preds = torch.cat([t.cpu() for t in gathered_preds])
            all_probs = torch.cat([t.cpu() for t in gathered_probs])
            all_labels = torch.cat([t.cpu() for t in gathered_labels])

            save_predictions(
                self.config_manager,
                labels=all_labels,
                preds=all_preds,
                probs=all_probs,
                split=self.dataset_split,
            )

            save_predict_metrics(self.config_manager, metrics)

        torch.distributed.barrier()

        if self.save_gradient_data:
            gathered_grad_recs = _all_gather_object_list(grad_records)
            if torch.distributed.get_rank() == 0:
                # Save raw gradient data
                save_gradient_diagnostics(
                    self.config_manager,
                    gathered_grad_recs,
                    split=self.dataset_split,
                    filename=f'raw_gradients_{self.dataset_split}.csv'
                )
                
                # Compute and save G_C and N_C for different clipping thresholds
                self._compute_and_save_clipping_analysis(gathered_grad_recs)

    def load_model(
        self,
        fpath: Optional[str],
    ) -> None:
        """
        Load weights into self.trainer.model.

        Args:
            fpath: Path to checkpoint file on disk.
            strict: Passed to load_state_dict.
            map_location: torch.load map_location (e.g. cpu/cuda).
        """
        model = self.trainer._unwrap_model()
        model.load_model(fpath)

        if torch.distributed.get_rank() == 0:
            log.info(f'Loaded weights from: {fpath}')


    def _per_sample_head_grads(self, head: torch.nn.Module, H: torch.Tensor, y: torch.Tensor):
        if isinstance(head, torch.nn.Linear):
            W = head.weight              # [C,F]
            b = head.bias                # [C] or None
            H2 = H                       # [B,F]
        elif isinstance(head, torch.nn.Conv2d):
            kh, kw = head.kernel_size
            if kh != 1 or kw != 1:
                raise TypeError(f'Only Conv2d 1x1 heads are supported; got {head.kernel_size}')

            # Flatten conv -> linear
            W = head.weight.view(head.out_channels, -1)  # [C,F]
            b = head.bias                                # [C] or None
            if H.ndim != 4:
                raise TypeError(f'Expected H with 4 dims for Conv2d; got {tuple(H.shape)}')

            H2 = H.view(H.size(0), -1)                   # [B,F], handles [B,F,1,1] too
        else:
            raise TypeError(f'Unsupported classifier type: {type(head)}')

        C, F = W.shape
        w_dim = C * F
        b_dim = C if b is not None else 0
        g_dim = w_dim + b_dim

        if b is not None:
            def loss_one(w, b_, h, yi):
                logits = h @ w.T + b_
                return torch.nn.functional.cross_entropy(logits.unsqueeze(0), yi.unsqueeze(0), reduction='sum')
            gfun = grad(loss_one, argnums=(0, 1))
            gw, gb = vmap(gfun, in_dims=(None, None, 0, 0))(W, b, H2, y)  # gw [B,C,F], gb [B,C]
            gw = gw.reshape(gw.size(0), w_dim)
            g = torch.cat([gw, gb], dim=1)
        else:
            def loss_one(w, h, yi):
                logits = h @ w.T
                return torch.nn.functional.cross_entropy(logits.unsqueeze(0), yi.unsqueeze(0), reduction='sum')
            gfun = grad(loss_one, argnums=0)
            gw = vmap(gfun, in_dims=(None, 0, 0))(W, H2, y)  # [B,C,F]
            g = gw.reshape(gw.size(0), w_dim)

        return g, g_dim, (b is not None)

    def _per_sample_full_grads(self, model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
        """
        Compute per-sample gradients for the full model (body + head) using torch.func.
        This is a simplified implementation that computes gradients by example.
        
        Args:
            model: The full model
            X: Input batch [B, ...]
            y: Target labels [B]
            
        Returns:
            torch.Tensor: Per-sample gradients [B, total_grad_dim]
        """
        batch_size = X.size(0)
        
        # Get all trainable parameters and their shapes
        param_list = []
        param_shapes = []
        total_params = 0
        
        for param in model.parameters():
            if param.requires_grad:
                param_list.append(param)
                param_shapes.append(param.shape)
                total_params += param.numel()
        
        # Function to compute loss for a single example
        def single_example_loss(params_tuple, x_single, y_single):
            # Temporarily assign parameters
            param_idx = 0
            for i, param in enumerate(model.parameters()):
                if param.requires_grad:
                    param.data = params_tuple[param_idx].data
                    param_idx += 1
            
            # Forward pass
            logits = model(x_single.unsqueeze(0))
            loss = torch.nn.functional.cross_entropy(logits, y_single.unsqueeze(0), reduction='sum')
            return loss
        
        # Create grad function
        grad_fn = grad(single_example_loss, argnums=0)
        
        # Prepare parameters tuple
        params_tuple = tuple(param_list)
        
        # Compute per-sample gradients using vmap
        try:
            per_sample_grad_tuples = vmap(grad_fn, in_dims=(None, 0, 0))(params_tuple, X, y)
        except Exception as e:
            log.warning(f"vmap approach failed: {e}. Falling back to loop-based computation.")
            # Fallback: compute gradients one by one
            per_sample_grads_list = []
            
            for i in range(batch_size):
                x_single = X[i]
                y_single = y[i]
                
                # Zero gradients
                model.zero_grad()
                
                # Forward and backward for single example
                logits = model(x_single.unsqueeze(0))
                loss = torch.nn.functional.cross_entropy(logits, y_single.unsqueeze(0), reduction='sum')
                loss.backward()
                
                # Collect gradients
                grad_vec = []
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_vec.append(param.grad.view(-1))
                
                if grad_vec:
                    grad_vec = torch.cat(grad_vec)
                else:
                    grad_vec = torch.zeros(total_params, device=X.device)
                
                per_sample_grads_list.append(grad_vec)
            
            # Stack all gradients
            flattened_grads = torch.stack(per_sample_grads_list, dim=0)
            return flattened_grads
        
        # Flatten and concatenate all per-sample gradients from vmap
        flattened_grads = torch.zeros(batch_size, total_params, device=X.device)
        
        start_idx = 0
        for i, param_grad in enumerate(per_sample_grad_tuples):
            param_size = param_grad.view(batch_size, -1).size(1)
            flattened_grads[:, start_idx:start_idx + param_size] = param_grad.view(batch_size, -1)
            start_idx += param_size
        
        return flattened_grads

    def _compute_and_save_clipping_analysis(self, grad_records):
        """
        Compute G_C and N_C for different clipping thresholds C and save the results.
        
        Args:
            grad_records: List of records with 'gradient' and 'norm' fields
        """
        import numpy as np
        import pandas as pd
        from .experimentmanager import safe_open
        import pathlib
        
        # Extract gradients and norms
        gradients = []
        norms = []
        
        for record in grad_records:
            grad_vec = np.array(record['gradient'])
            gradients.append(grad_vec)
            norms.append(record['norm'])
        
        gradients = np.array(gradients)  # [N, d] where N is number of samples, d is gradient dimension
        norms = np.array(norms)  # [N]
        
        # Define clipping thresholds to analyze
        # Use percentiles of the norm distribution and some fixed values
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        C_values = [np.percentile(norms, p) for p in percentiles]
        
        # Add some fixed values that might be commonly used
        fixed_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        C_values.extend(fixed_values)
        
        # Remove duplicates and sort
        C_values = sorted(list(set(C_values)))
        
        # Compute G_C and N_C for each clipping threshold
        results = []
        
        for C in C_values:
            # Find indices of clipped gradients: I_C = {i: ||g_i|| > C}
            clipped_indices = norms > C
            
            if not np.any(clipped_indices):
                # No gradients are clipped at this threshold
                G_C = np.zeros_like(gradients[0])
                N_C = np.zeros_like(gradients[0])
                num_clipped = 0
            else:
                # G_C = sum of gradients that are clipped
                clipped_gradients = gradients[clipped_indices]
                G_C = np.sum(clipped_gradients, axis=0)
                
                # N_C = sum of normalized gradients that are clipped
                clipped_norms = norms[clipped_indices]
                normalized_clipped_grads = clipped_gradients / clipped_norms[:, np.newaxis]
                N_C = np.sum(normalized_clipped_grads, axis=0)
                
                num_clipped = len(clipped_gradients)
            
            # Compute norms of G_C and N_C
            G_C_norm = np.linalg.norm(G_C)
            N_C_norm = np.linalg.norm(N_C)
            
            # Compute dot product N_C^T @ G_C
            dot_product = np.dot(N_C, G_C)
            
            # Store results
            result = {
                'clipping_threshold_C': C,
                'num_total_samples': len(gradients),
                'num_clipped_samples': num_clipped,
                'clipped_proportion': num_clipped / len(gradients),
                'G_C_norm': G_C_norm,
                'N_C_norm': N_C_norm,
                'N_C_dot_G_C': dot_product,
                'N_C_norm_squared': N_C_norm**2,
            }
            
            results.append(result)
        
        # Save the analysis results
        log_dir = self.config_manager.configuration.log_dir
        experiment_name = self.config_manager.configuration.experiment_name
        full_log_dir = pathlib.Path(f'{log_dir}/{experiment_name}')
        full_log_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(results)
        out_path = full_log_dir / f'clipping_analysis_{self.dataset_split}.csv'
        
        with safe_open(out_path, 'w') as fh:
            fh.write(df.to_csv(index=False))
        
        log.info(f'Clipping analysis saved to {out_path}')
        
        # Also save the raw G_C and N_C vectors for further analysis if needed
        detailed_results = []
        
        for i, C in enumerate(C_values):
            clipped_indices = norms > C
            
            if np.any(clipped_indices):
                clipped_gradients = gradients[clipped_indices]
                G_C = np.sum(clipped_gradients, axis=0)
                
                clipped_norms = norms[clipped_indices]
                normalized_clipped_grads = clipped_gradients / clipped_norms[:, np.newaxis]
                N_C = np.sum(normalized_clipped_grads, axis=0)
            else:
                G_C = np.zeros_like(gradients[0])
                N_C = np.zeros_like(gradients[0])
            
            detailed_result = {
                'clipping_threshold_C': C,
                'G_C': G_C.tolist(),
                'N_C': N_C.tolist(),
            }
            detailed_results.append(detailed_result)
        
        # Save detailed results (G_C and N_C vectors)
        detailed_out_path = full_log_dir / f'clipping_vectors_{self.dataset_split}.csv'
        detailed_df = pd.DataFrame(detailed_results)
        
        with safe_open(detailed_out_path, 'w') as fh:
            fh.write(detailed_df.to_csv(index=False))
        
        log.info(f'Detailed clipping vectors saved to {detailed_out_path}')

    def _compute_class_prototypes(self):
        """
        Forward-only + torch.func grads to build per-class prototype directions (unit vectors).
        """

        if torch.distributed.get_rank() == 0:
            log.info('Computing class prorotypes for angle calculation.')

        model = self.trainer.model
        model.eval()

        head = self.trainer._unwrap_model().get_classifier()

        if not isinstance(head, (torch.nn.Linear, torch.nn.Conv2d)):
            raise TypeError(f'Predictor expects Linear or Conv2d(1x1) head; got {type(head)}')

        if isinstance(head, torch.nn.Linear):
            C, F = head.weight.shape
        elif isinstance(head, torch.nn.Conv2d):
            # kernel expected to be 1x1; we’ll guard later
            C = head.out_channels
            F = head.in_channels * head.kernel_size[0] * head.kernel_size[1]
        else:
            raise TypeError(f'Unsupported classifier type: {type(head)}')

        C = head.weight.shape[0]
        b_dim = C if head.bias is not None else 0
        g_dim = C * F + b_dim

        device = head.weight.device

        K = self.trainer.datamodule.get_num_classes()
        assert C == K, f'Head out_features={C} != num_classes={K}'

        proto_sum = torch.zeros(K, g_dim, device=device)
        counts = torch.zeros(K, dtype=torch.long, device=device)

        # Hook to capture head input H
        last = {'H': None}

        def _hook(mod, inputs, output):
            last['H'] = inputs[0].detach()

        hnd = head.register_forward_hook(_hook)

        dataloader = self.trainer.get_dataloader(self.dataset_split)

        for i, (X, Y) in enumerate(dataloader):
            if torch.distributed.get_rank() == 0:
                log.info(f' - Processing batch {i}')

            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)

            # Forward only to populate last['H']
            with torch.no_grad():
                _ = model(X)

            H = last['H']  # [B,F], captured by hook
            if H is None:
                raise RuntimeError('Classifier hook did not capture activations (H is None).')

            # Compute per-sample head gradients
            g, _, _ = self._per_sample_head_grads(head, H, Y)  # [B,g_dim]

            # Accumulate class prototypes
            for k in range(K):
                m = Y == k
                if m.any():
                    proto_sum[k] += g[m].sum(0)
                    counts[k] += int(m.sum())

            del H, g
            torch.cuda.empty_cache()

        hnd.remove()

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(proto_sum)
            torch.distributed.all_reduce(counts)

        proto_mean = torch.where(
            counts.view(-1, 1) > 0,
            proto_sum / counts.clamp_min(1).view(-1, 1),
            torch.zeros_like(proto_sum),
        )
        proto_unit = proto_mean / proto_mean.norm(dim=1, keepdim=True).clamp_min(1e-12)
        has_bias = head.bias is not None

        return proto_unit, g_dim, has_bias


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

        trainer = TrainerFactory.get_trainer(config_manager)

        predictor = Predictor(
            trainer=trainer,
            dataset_split=configuration.dataset_split,
            config_manager=config_manager,
            save_gradient_data=configuration.prediction_save_gradient_data,
        )

        if fpath := getattr(configuration, 'model_weights_path', False):
            predictor.load_model(configuration.model_weights_path)

        return predictor
