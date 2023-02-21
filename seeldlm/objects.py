import numpy
from typing import Optional


def symmetrise(array: "numpy.ndarray") -> "numpy.ndarray":

    return (array + array.T) / 2


class NormalModel:
    def __init__(self, mean: "numpy.ndarray", covariance: "numpy.ndarray"):
        self.mean = mean
        self.covariance = covariance
        self.inv_covariance: "Optional[numpy.ndarray]" = None

    def invert_covariance(self):

        if self.inv_covariance is None:
            self.inv_covariance = numpy.linalg.inv(symmetrise(self.covariance))

        return self.inv_covariance

    def update_wishart(self, wishart: "InvWishartModel", observation: "numpy.ndarray"):

        error = self.mean - observation  # (P, N)
        inv_covariance = self.invert_covariance()  # (P, P)

        scale = wishart.scale + (error.T).dot(inv_covariance).dot(error)  # (N, N)
        shape = wishart.shape + self.mean.shape[0]  # int

        return InvWishartModel(scale, shape)

    def transform(self, matrix: "numpy.ndarray") -> "NormalModel":
        new_mean = matrix.dot(self.mean)
        new_covariance = matrix.dot(self.covariance).dot(matrix.T)
        return NormalModel(new_mean, new_covariance)


class TransitionModel:
    def __init__(
        self,
        bias: "numpy.ndarray",
        weights: "numpy.ndarray",
        covariance: "numpy.ndarray",
    ):

        self.bias = bias
        self.weights = weights
        self.covariance = covariance

    def observe(self, observation: "numpy.ndarray"):

        B = self.bias
        A = self.weights
        V = self.covariance

        mean = B + A.dot(observation)

        return NormalModel(mean, V)


class JointModel:
    def __init__(self, normal: "NormalModel", transition: "TransitionModel"):

        self.normal = normal
        self.transition = transition

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

        mean: "numpy.ndarray" = B + A.dot(M)
        covariance: "numpy.ndarray" = V + A.dot(S).dot(A.T)

        return NormalModel(mean, covariance)

    def mutate_joint_model(self):

        normal = self.mutate_normal()
        M = self.normal.mean
        S = self.normal.covariance
        A = self.transition.weights

        inv_covariance = normal.invert_covariance()
        weights: "numpy.ndarray" = S.dot(A.T).dot(inv_covariance)
        bias: "numpy.ndarray" = M - weights.dot(normal.mean)
        variation: "numpy.ndarray" = S - weights.dot(normal.covariance).dot(weights.T)

        transition = TransitionModel(bias, weights, variation)

        return JointModel(normal, transition)

    def generate_normal(self) -> "NormalModel":

        new_normal = self.mutate_normal()

        M = self.normal.mean
        S = self.normal.covariance
        A = self.transition.weights
        alt_covariance = S.dot(A.T)

        new_mean = numpy.block([[new_normal], [M]])
        new_covariance = numpy.block(
            [[new_normal.covariance, alt_covariance.T], [alt_covariance, S]]
        )

        return NormalModel(new_mean, new_covariance)


class InvWishartModel:
    def __init__(self, scale: "numpy.ndarray", shape: "int"):

        self.scale = scale
        self.shape = shape
