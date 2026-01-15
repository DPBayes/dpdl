import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class GradSampleParamSet:
    params: List[torch.nn.Parameter]
    batch_size: int


def build_param_name_map(model: torch.nn.Module) -> Dict[int, str]:
    name_by_id: Dict[int, str] = {}
    for name, p in model.named_parameters():
        name_by_id[id(p)] = name
    return name_by_id


def _compile_optional_regex(pattern: Optional[str]) -> Optional[re.Pattern]:
    if not pattern:
        return None
    return re.compile(pattern)


def select_grad_sample_params(
    optimizer: torch.optim.Optimizer,
    *,
    name_by_param_id: Optional[Dict[int, str]] = None,
    include_param_name_regex: Optional[str] = None,
    exclude_param_name_regex: Optional[str] = None,
    only_requires_grad: bool = True,
) -> GradSampleParamSet:
    include = _compile_optional_regex(include_param_name_regex)
    exclude = _compile_optional_regex(exclude_param_name_regex)

    params: List[torch.nn.Parameter] = []
    batch_size: Optional[int] = None

    for group in optimizer.param_groups:
        for p in group["params"]:
            if only_requires_grad and not p.requires_grad:
                continue

            gs = getattr(p, "grad_sample", None)
            if gs is None or gs.numel() == 0:
                continue

            if name_by_param_id is not None:
                pname = name_by_param_id.get(id(p), "")
                if include is not None and not include.search(pname):
                    continue
                if exclude is not None and exclude.search(pname):
                    continue

            m_i = int(gs.size(0))
            if batch_size is None:
                batch_size = m_i
            elif batch_size != m_i:
                raise ValueError("Inconsistent grad_sample batch sizes across params")

            params.append(p)

    if batch_size is None:
        batch_size = 0

    return GradSampleParamSet(params=params, batch_size=batch_size)


def per_sample_grad_norms(
    params: Sequence[torch.nn.Parameter],
    *,
    batch_size: int,
    eps: float = 1e-12,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Returns a tensor of shape [batch_size] with L2 norms of per-sample gradients,
    aggregated over the provided parameters.
    """
    if batch_size <= 0 or not params:
        return torch.empty((0,), dtype=dtype, device=device)

    norms_sq = None
    for p in params:
        gs = getattr(p, "grad_sample", None)
        if gs is None or gs.numel() == 0:
            continue

        part = (gs.view(batch_size, -1).to(dtype=dtype) ** 2).sum(dim=1)
        norms_sq = part if norms_sq is None else (norms_sq + part)

    if norms_sq is None:
        return torch.empty((0,), dtype=dtype, device=device)

    if device is not None:
        norms_sq = norms_sq.to(device)

    return norms_sq.clamp_min(eps).sqrt()

