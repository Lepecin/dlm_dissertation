from numpy.typing import NDArray

from .normal import NormalModel


class TransitionModel:
    def __init__(
        self,
        bias: "NDArray",
        weights: "NDArray",
        covariance: "NDArray",
    ):

        self.bias = bias
        self.weights = weights
        self.covariance = covariance

    def observe(self, observation: "NDArray") -> "NormalModel":

        B = self.bias
        A = self.weights
        V = self.covariance

        mean = B + A.dot(observation)

        return NormalModel(mean, V)
