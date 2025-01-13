from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
import torch


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]

    @classmethod
    def from_list(cls, gaussians):
        return cls(
            torch.cat([g.means for g in gaussians], dim=1),
            torch.cat([g.covariances for g in gaussians], dim=1),
            torch.cat([g.harmonics for g in gaussians], dim=1),
            torch.cat([g.opacities for g in gaussians], dim=1),
        )

    def requires_grad_(self, requires_grad=True):
        for v in self.__dict__.values():
            v.requires_grad_(requires_grad)
        return self
