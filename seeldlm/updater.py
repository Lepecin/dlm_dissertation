from typing import Tuple
from numpy.typing import NDArray

from .objects import NormalModel, TransitionModel, JointModel, InvWishartModel
from .memory import MemoryDLM


class UpdaterDLM(MemoryDLM):
    def __init__(self, observed_period: "int", predicted_period: "int"):
        super().__init__(observed_period, predicted_period)

    def evolve(self, time: "int", evolver: "TransitionModel"):

        filtered_state: "NormalModel" = self.filtered_states.get_from_time(time)

        joint_model = JointModel(filtered_state, evolver)
        evolved_state, smoother = joint_model.transition_transumer()

        self.smoothers.set_at_time(time + 1, smoother)
        self.evolved_states.set_at_time(time + 1, evolved_state)

    def observe_evolved(self, time: "int", observer: "TransitionModel"):

        evolved_state: "NormalModel" = self.evolved_states.get_from_time(time + 1)

        joint_model = JointModel(evolved_state, observer)
        evolved_space, filterer = joint_model.transition_transumer()

        self.filterers.set_at_time(time + 1, filterer)
        self.evolved_spaces.set_at_time(time + 1, evolved_space)

    def filter(self, time: "int", observation: "NDArray"):

        evolved_space: "NormalModel" = self.evolved_spaces.get_from_time(time + 1)
        filterer: "TransitionModel" = self.filterers.get_from_time(time + 1)
        error: "InvWishartModel" = self.wisharts.get_from_time(time)

        filtered_state = filterer.observe(observation)
        error = evolved_space.update_wishart(error, observation)

        self.filtered_states.set_at_time(time + 1, filtered_state)
        self.wisharts.set_at_time(time + 1, error)

    def observe_filtered(self, time: "int", observer: "TransitionModel"):

        filtered_state: "NormalModel" = self.filtered_states.get_from_time(time + 1)

        joint_model = JointModel(filtered_state, observer)
        filtered_space = joint_model.mutate_normal()

        self.filtered_spaces.set_at_time(time + 1, filtered_space)

    def smoothen(self, time: "int"):

        smoother: "TransitionModel" = self.smoothers.get_from_time(self.S - time)
        smoothed_state: "NormalModel" = self.smoothed_states.get_from_time(
            self.S - time
        )

        joint_model = JointModel(smoothed_state, smoother)
        smoothed_state = joint_model.mutate_normal()

        self.smoothed_states.set_at_time(self.S - time - 1, smoothed_state)

    def observe_smoothed(self, time: "int", observer: "TransitionModel"):

        smoothed_state: "NormalModel" = self.smoothed_states.get_from_time(
            self.S - time
        )

        joint_model = JointModel(smoothed_state, observer)
        smoothed_space = joint_model.mutate_normal()

        self.smoothed_spaces.set_at_time(self.S - time, smoothed_space)

    def predict(self, time: "int", evolver: "TransitionModel"):

        predicted_state: "NormalModel" = self.predicted_states.get_from_time(
            self.S + time
        )

        joint_model = JointModel(predicted_state, evolver)
        predicted_state = joint_model.mutate_normal()

        self.predicted_states.set_at_time(self.S + time + 1, predicted_state)

    def observe_predicted(self, time: "int", observer: "TransitionModel"):

        predicted_state: "NormalModel" = self.predicted_states.get_from_time(
            self.S + time + 1
        )

        joint_model = JointModel(predicted_state, observer)
        predicted_space = joint_model.mutate_normal()

        self.predicted_spaces.set_at_time(self.S + time + 1, predicted_space)
