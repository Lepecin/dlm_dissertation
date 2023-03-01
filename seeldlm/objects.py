from typing import Optional
from numpy.typing import NDArray
from numpy.linalg import inv as inverse_matrix
from numpy import block as block_matrix


def symmetrise(array: "NDArray") -> "NDArray":

    return (array + array.T) / 2


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


class JointModel:
    def __init__(self, normal: "NormalModel", transition: "TransitionModel"):

        self.normal = normal
        self.transition = transition

    def give_normal(self) -> "NormalModel":
        return self.normal

    def give_transition(self) -> "TransitionModel":
        return self.transition

    def mutate_normal(self) -> "NormalModel":

        M = self.normal.mean
        S = self.normal.covariance
        B = self.transition.bias
        A = self.transition.weights
        V = self.transition.covariance

        mean: "NDArray" = B + A.dot(M)
        covariance: "NDArray" = V + A.dot(S).dot(A.T)

        return NormalModel(mean, covariance)

    def mutate_joint_model(self) -> "JointModel":

        normal = self.mutate_normal()
        M = self.normal.mean
        S = self.normal.covariance
        A = self.transition.weights

        inv_covariance = normal.invert_covariance()
        weights: "NDArray" = S.dot(A.T).dot(inv_covariance)
        bias: "NDArray" = M - weights.dot(normal.mean)
        variation: "NDArray" = S - weights.dot(normal.covariance).dot(weights.T)

        transition = TransitionModel(bias, weights, variation)

        return JointModel(normal, transition)

    def generate_normal(self) -> "NormalModel":

        new_normal = self.mutate_normal()

        M = self.normal.mean
        S = self.normal.covariance
        A = self.transition.weights
        alt_covariance = S.dot(A.T)

        new_mean = block_matrix([[new_normal.mean], [M]])
        new_covariance = block_matrix(
            [[new_normal.covariance, alt_covariance.T], [alt_covariance, S]]
        )

        return NormalModel(new_mean, new_covariance)


class InvWishartModel:
    def __init__(self, scale: "NDArray", shape: "int"):

        self.scale = scale
        self.shape = shape
