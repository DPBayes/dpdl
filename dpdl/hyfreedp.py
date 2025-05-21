import logging
import math
from typing import Optional, Sequence
import opacus

import scipy.optimize as opt
import torch
from opacus.accountants.utils import get_noise_multiplier
from torch.optim import Optimizer

from .configurationmanager import Configuration, Hyperparameters

log = logging.getLogger(__name__)


def _epsilon_from_sigma(sigma, sample_rate, delta, repeats, accountant_str='prv'):
    accountant = opacus.accountants.create_accountant(accountant_str)

    # Compose X repetitions of a Poisson‑subsampled Gaussian
    for _ in range(repeats):
        accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)

    return accountant.get_epsilon(delta=delta)


def split_noise_multiplier(
    epsilon, delta, T, B, N, K, accountant_str='prv', gamma=1.01
):
    sample_rate = B / N

    # Algorithm 2: Line 4
    # Sigma that would achieve (ε,δ) for *gradients only*
    sigma_ref = get_noise_multiplier(
        target_epsilon=epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        steps=T,
        accountant=accountant_str,
    )

    if torch.distributed.get_rank() == 0:
        log.info(
            f'******** HyFreeDP - SPLIT_NOISE_MULTIPLIER - epsilon={epsilon}, delta={delta}, B={B}, N={N}, T={T}, K={K}, WE GOT SIGMA_REF={sigma_ref}'
        )

    # Line 6
    # Reserve bit more for gradients, so we can split to
    # use with gradients and the loss privatization
    sigma_g = gamma * sigma_ref

    # Line 8
    epsilon_ours = _epsilon_from_sigma(sigma_g, sample_rate, delta, T, accountant_str)

    # Line 9 until end
    target = epsilon - epsilon_ours
    if torch.distributed.get_rank() == 0:
        log.info(f'*********** GOT TARGET: {target}')

    if target <= 0:
        raise ValueError(
            'HyFreeDP - gamma too large – gradient noise already exceeds epsilon'
        )

    L = math.ceil(3 * T / K)  # Line 8: 3*T/K

    def objective(sigma_l):
        epsilon_loss = _epsilon_from_sigma(
            sigma_l, sample_rate, delta, L, accountant_str
        )  # Line 8: 3*T/K

        return epsilon_loss - target

    sigma_star = get_noise_multiplier(
        target_epsilon=target,
        target_delta=delta,
        sample_rate=sample_rate,
        steps=L,
        accountant=accountant_str,
    )

    # Search around sigma_star
    lo = 0.5 * sigma_star
    hi = 2.0 * sigma_star

    # initialize search
    while objective(lo) < 0:  # lo must be negative
        lo /= 2

    while objective(hi) > 0:  # hi must be positive
        hi *= 2

    if torch.distributed.get_rank() == 0:
        log.info(f'HyFreeDP - Starting search with lo={lo}, hi={hi}, target={target}')

    # bisect
    sigma_l = opt.brentq(objective, a=lo, b=hi)

    return float(sigma_g), float(sigma_l)


_LR_RANGE_SCALE = 2.0  # how far η may jump each adaptation
_LR_SMOOTH = 0.1  # EMA smoothing factor for η updates (0=no smoothing, 1=freeze)


class HyFreeDPOptimizer(Optimizer):
    """Hyperparameter‑free learning‑rate adapter (DP‑ready)."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        loss_fn,
        optimizer: Optimizer,
        init_eta: float = 1e-4,
        K: int = 5,
        # DP parameters
        accountant=None,
        sample_rate: float = None,
        sigma_l: float = None,
        batch_size: int = None,
        debug: bool = False,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.dp_optim = optimizer
        self.K = K
        self.debug = debug
        self.eta = init_eta
        self.step_no = 0
        self.batch_size_fixed = batch_size

        # DP bookkeeping
        if accountant is None or sample_rate is None or sigma_l is None:
            raise ValueError(
                'For HyFreeDP, accountant, sample_rate and sigma_l are required.'
            )

        self.accountant = accountant
        self.sample_rate = sample_rate
        self.sigma_l = sigma_l

        # initial R_l (bootstrapped on first probe)
        self.R_l = 1.0
        self._sum_prev_probe = None

        # propagate initial LR to wrapped optimiser
        for g in self.dp_optim.param_groups:
            g['lr'] = init_eta

        super().__init__(model.parameters(), {})

    def step(self, closure=None):
        """Wrapper around the underlying optimiser step with LR adaptation."""
        # fetch per-sample loss + batch
        per_sample_loss = getattr(self, 'per_sample_loss', None)
        X = getattr(self, 'batch_inputs', None)
        y = getattr(self, 'batch_targets', None)

        if per_sample_loss is None or X is None or y is None:
            raise RuntimeError(
                'HyFreeDP - missing per_sample_loss or batch inputs/targets before step().'
            )

        B = (
            per_sample_loss.shape[0]
            if self.batch_size_fixed is None
            else self.batch_size_fixed
        )

        # Update R_l from last probe sum (post-initial)
        if self.step_no > 0 and self._sum_prev_probe is not None:
            self.R_l = abs(self._sum_prev_probe).item()

        # Does this step trigger probing?
        need_probe = self.step_no % self.K == 0

        # Baseline loss
        if need_probe:
            L0_priv = self._privatize_loss(per_sample_loss, B)

            # Privatize for probing
            self._sum_prev_probe = L0_priv.clone()
        else:
            # No probe - This won't be released
            L0_priv = per_sample_loss.mean()

        baseline_val = L0_priv.item()

        # Snapshot parameters for for probing
        if need_probe:
            param_before = {
                p: p.data.clone() for p in self.model.parameters() if p.grad is not None
            }

        # Perform the DP-optimizer's step
        self.dp_optim.step(closure)

        # Probing
        if need_probe:
            # Compute DP update direction phi = (w_before - w_after)/eta
            phi = {}
            for p in param_before:
                phi[p] = (param_before[p] - p.data) / self.eta

            # Restore to pre-step weights for clean probes
            self._restore_params(param_before)

            # Disable hooks if they exist
            try:
                self.model.disable_hooks()
            except:
                pass

            # Probe helper function
            def probe(sign: float):
                # w <- w - (sign * eta) * phi
                self._shift_params(phi, -sign * self.eta)
                per = self.loss_fn(self.model(X), y, reduction='none')
                Lp = self._privatize_loss(per, B)

                # accumulate for next R_l
                self._sum_prev_probe += Lp

                # restore
                self._restore_params(param_before)
                return Lp.item()

            with torch.no_grad():
                L_plus = probe(+1.0)  # evaluate at -eta
                L_minus = probe(-1.0)  # evaluate at +eta

            # Re-enable hooks
            try:
                self.model.enable_hooks()
            except:
                pass

            # Fit quadratic ΔL(η) ≈ -b η + (a/2) η^2
            etas = torch.tensor([0.0, self.eta, -self.eta])
            losses = torch.tensor([baseline_val, L_plus, L_minus])
            y_fit = losses - baseline_val
            X_fit = torch.stack([etas**2 / 2, -etas], dim=1)

            # Solve the quadratic
            a, b = torch.linalg.lstsq(X_fit, y_fit[:, None]).solution.squeeze()

            # Compute new η = -b/(2a) with sign check
            if a > 0 and b < 0:
                eta_star = -b / (2.0 * a)
                lo, hi = self.eta / _LR_RANGE_SCALE, self.eta * _LR_RANGE_SCALE
                eta_star = float(torch.clamp(eta_star, lo, hi))
            else:
                eta_star = self.eta

            # Smoothing (not mentioned in the paper, but does not work without)
            self.eta = max(1e-8, _LR_SMOOTH * self.eta + (1 - _LR_SMOOTH) * eta_star)
            for g in self.dp_optim.param_groups:
                g['lr'] = self.eta

        # clear temporary attributes
        for attr in ('per_sample_loss', 'batch_inputs', 'batch_targets'):
            if hasattr(self, attr):
                delattr(self, attr)

        if self.debug:
            log.info(
                f'[HyFreeDP] step={self.step_no:4d} η={self.eta:.3e} loss={baseline_val:.4f}'
            )

        self.step_no += 1

    def _privatize_loss(self, per_sample_loss: torch.Tensor, B: int) -> torch.Tensor:
        coeff = torch.clamp(self.R_l / per_sample_loss.abs(), max=1.0)
        L_clip = (coeff * per_sample_loss).mean()
        noise = torch.randn_like(L_clip) * (self.sigma_l * self.R_l / B)
        tilde_L = L_clip + noise
        self.accountant.step(
            noise_multiplier=self.sigma_l, sample_rate=self.sample_rate
        )
        return tilde_L

    def _shift_params(self, grads: dict, alpha: float):
        with torch.no_grad():
            for p, g in grads.items():
                p.data.add_(g, alpha=alpha)

    def _restore_params(self, backup: dict):
        with torch.no_grad():
            for p, val in backup.items():
                p.data.copy_(val)
