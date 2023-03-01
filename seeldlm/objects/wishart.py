from typing import Optional
from numpy.typing import NDArray
from numpy.linalg import inv as inverse_matrix

from .utils import symmetrise


class InvWishartModel:
    def __init__(self, scale: "NDArray", shape: "int"):

        self.scale = scale
        self.shape = shape
        self.inv_scale: "Optional[NDArray]" = None

    def invert_scale(self) -> "NDArray":

        if self.inv_scale is None:
            self.inv_scale = inverse_matrix(symmetrise(self.scale))

        return self.inv_scale

    def derive_scale_em_estimate(self, true_wishart: "InvWishartModel") -> "NDArray":
        return (self.shape / true_wishart.shape) * true_wishart.scale
