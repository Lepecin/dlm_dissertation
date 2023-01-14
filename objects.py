import numpy
from numpy import ndarray
from dataclasses import dataclass


def symmetrise(array: "ndarray"):

    return (array + array.T) / 2


@dataclass
class NormalModel:

    mean: "ndarray"
    covariance: "ndarray"


@dataclass
class TransitionModel:

    bias: "ndarray"
    weights: "ndarray"
    covariance: "ndarray"

    def observe(self, observation: "ndarray"):

        # Extract models of transition
        B = self.transition.bias
        A = self.transition.weights
        V = self.transition.covariance

        # Create new mean from observation
        mean = B + A.dot(observation)

        return NormalModel(mean, V)


@dataclass
class JointModel:

    normal: "NormalModel"
    transition: "TransitionModel"

    def norm_mutator(
        self,
    ):
        M = self.normal.mean
        S = self.normal.covariance
        B = self.transition.bias
        A = self.transition.weights
        V = self.transition.covariance

        # Create parameters of new model
        mean: "ndarray" = B + A.dot(M)
        covariance: "ndarray" = V + A.dot(S).dot(A.T)

        return NormalModel(mean, covariance)

    def trans_mutator(
        self,
    ):
        normal = self.normal_transmutator()
        M = self.normal.mean
        S = self.normal.covariance
        A = self.transition.weights

        # Create parameters of new transition
        inv_covariance = numpy.linalg.inv(symmetrise(normal.covariance))
        weights: "ndarray" = S.dot(A.T).dot(inv_covariance)
        bias: "ndarray" = M - weights.dot(normal.mean)
        variation: "ndarray" = S - weights.dot(normal.covariance).dot(weights.T)
        transition = TransitionModel(bias, weights, variation)

        return JointModel(normal, transition)


@dataclass
class InvWishart:

    scale: "ndarray"
    shape: "int"
