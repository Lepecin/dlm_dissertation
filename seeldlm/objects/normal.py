from typing import Optional
from numpy.typing import NDArray
from numpy.linalg import inv as inverse_matrix

from .utils import symmetrise
from .wishart import InvWishartModel


class NormalModel:
    def __init__(self, mean: "NDArray", covariance: "NDArray"):
        self.mean = mean
        self.covariance = covariance
        self.inv_covariance: "Optional[NDArray]" = None

    def invert_covariance(self) -> "NDArray":

        if self.inv_covariance is None:
            self.inv_covariance = inverse_matrix(symmetrise(self.covariance))

        return self.inv_covariance

    def update_wishart(
        self, wishart: "InvWishartModel", observation: "NDArray"
    ) -> "InvWishartModel":

        error = self.mean - observation  # (P, N)
        inv_covariance = self.invert_covariance()  # (P, P)

        scale = wishart.scale + (error.T).dot(inv_covariance).dot(error)  # (N, N)
        shape = wishart.shape + self.mean.shape[0]  # int

        return InvWishartModel(scale, shape)

    def transform(self, matrix: "NDArray") -> "NormalModel":
        new_mean = matrix.dot(self.mean)
        new_covariance = matrix.dot(self.covariance).dot(matrix.T)
        return NormalModel(new_mean, new_covariance)

    def derive_row_covariance_em_estimate(
        self, true_normal: "NormalModel", true_wishart: "InvWishartModel"
    ) -> "NDArray":

        inv_scale = true_wishart.invert_scale()
        constant = true_wishart.shape / true_wishart.scale.shape[0]

        em_estimate = true_normal.covariance + constant * (
            true_normal.mean - self.mean
        ).dot(inv_scale).dot((true_normal.mean - self.mean).T)

        return em_estimate
