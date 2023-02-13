import numpy
from numpy import ndarray
from dataclasses import dataclass, field
from typing import Optional


def symmetrise(array: "ndarray"):

    return (array + array.T) / 2


@dataclass
class NormalModel:

    mean: "ndarray"
    covariance: "ndarray"
    inv_covariance: "Optional[ndarray]" = field(default=None, init=False)

    def invert_covariance(self):

        if self.inv_covariance is None:
            self.inv_covariance = numpy.linalg.inv(symmetrise(self.covariance))

        return self.inv_covariance

    def update_wishart(self, wishart: "InvWishart", observation: "ndarray"):

        error = self.mean - observation  # (P, N)
        inv_covariance = self.invert_covariance()  # (P, P)
        scale = wishart.scale + (error.T).dot(inv_covariance).dot(error)  # (N, N)

        shape = wishart.shape + self.mean.shape[0]  # int

        return InvWishart(scale, shape)


@dataclass
class TransitionModel:

    bias: "ndarray"
    weights: "ndarray"
    covariance: "ndarray"

    def observe(self, observation: "ndarray"):

        # Extract models of transition
        B = self.bias
        A = self.weights
        V = self.covariance

        # Create new mean from observation
        mean = B + A.dot(observation)

        return NormalModel(mean, V)


@dataclass
class JointModel:

    normal: "NormalModel"
    transition: "TransitionModel"

    def give_normal(self):
        return self.normal

    def give_transition(self):
        return self.transition

    def mutate_normal(self):

        M = self.normal.mean
        S = self.normal.covariance
        B = self.transition.bias
        A = self.transition.weights
        V = self.transition.covariance

        # Create parameters of new model
        mean: "ndarray" = B + A.dot(M)
        covariance: "ndarray" = V + A.dot(S).dot(A.T)

        return NormalModel(mean, covariance)

    def mutate_joint_model(self):

        normal = self.mutate_normal()
        M = self.normal.mean
        S = self.normal.covariance
        A = self.transition.weights

        inv_covariance = normal.invert_covariance()
        weights: "ndarray" = S.dot(A.T).dot(inv_covariance)
        bias: "ndarray" = M - weights.dot(normal.mean)
        variation: "ndarray" = S - weights.dot(normal.covariance).dot(weights.T)

        transition = TransitionModel(bias, weights, variation)

        return JointModel(normal, transition)


@dataclass
class InvWishart:

    scale: "ndarray"
    shape: "int"
