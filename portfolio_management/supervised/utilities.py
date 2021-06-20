import torch
import torch.nn as nn
from typing import Union
from typing import Optional


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x) -> torch.Tensor:
        return x.view(*self.shape)


class GlobalAveragePooling1D(nn.Module):
    def __init__(self, dim: Optional[Union[int, tuple, list]]):
        super(GlobalAveragePooling1D, self).__init__()
        self.dim = dim

    def forward(self, x) -> torch.Tensor:
        return torch.mean(x, dim=self.dim)


def get_module(name: str, kwargs: Optional[dict] = None):
    kwargs = kwargs if kwargs is not None else {}
    if name == "View":
        return View(**kwargs)
    elif name == "GlobalAveragePooling1D":
        return GlobalAveragePooling1D(**kwargs)
    else:
        return getattr(nn, name)(**kwargs)


def get_sequential(module_list) -> nn.Sequential:
    modules = []
    for module in module_list:
        modules.append(get_module(**module))
    return nn.Sequential(*modules)
