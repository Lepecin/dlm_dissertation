from numpy.typing import NDArray
from numpy import block as block_matrix
from typing import Tuple

from .normal import NormalModel
from .transition import TransitionModel


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

    def transition_transumer(
        self,
    ) -> "Tuple[NormalModel, TransitionModel]":

        joint_model = self.mutate_joint_model()
        model = joint_model.give_normal()
        transition = joint_model.give_transition()

        return model, transition
