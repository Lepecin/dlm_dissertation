from numpy import ndarray
from typing import List, Any
from objects import NormalModel, TransitionModel, JointModel, InvWishartModel


class ModelContainer:
    def __init__(self, start: "int", end: "int"):

        self.container: "List[Any]" = (start - end + 1) * [None]
        self.start = start

    def get_from_time(self, time: "int") -> "Any":
        return self.container[time - self.start]

    def set_at_time(self, object: "Any", time: "int"):
        self.container[time - self.start] = object


class MemoryDLM:
    def __init__(self, observed_period: "int", predicted_period: "int"):

        S = observed_period
        P = predicted_period

        self.S = S
        self.P = P

        # -- Space Models
        self.filtered_states = ModelContainer(0, S)
        self.evolved_states = ModelContainer(1, S)
        self.smoothed_states = ModelContainer(0, S)
        self.predicted_states = ModelContainer(S, S + P)

        # -- Space Models
        self.filtered_spaces = ModelContainer(1, S)
        self.evolved_spaces = ModelContainer(1, S)
        self.smoothed_spaces = ModelContainer(1, S)
        self.predicted_spaces = ModelContainer(S, S + P)

        # -- Transitions
        self.smoothers = ModelContainer(1, S)
        self.filterers = ModelContainer(1, S)

        # -- Error Matrix
        self.wisharts = ModelContainer(0, S)


class UpdaterDLM(MemoryDLM):
    def __init__(self, observed_period: "int", predicted_period: "int"):
        super().__init__(observed_period, predicted_period)

    def evolve(self, time: "int", evolver: "TransitionModel"):

        filtered_state = self.filtered_states.get_from_time(time)

        joint_model = JointModel(filtered_state, evolver)
        joint_model = joint_model.mutate_joint_model()
        smoother = joint_model.give_transition()
        evolved_state = joint_model.give_normal()

        self.smoothers.set_at_time(smoother, time + 1)
        self.evolved_states.set_at_time(evolved_state, time + 1)

    def observe_evolved(self, time: "int", observer: "TransitionModel"):

        evolved_state = self.evolved_states.get_from_time(time + 1)

        joint_model = JointModel(evolved_state, observer)
        joint_model = joint_model.mutate_joint_model()
        filterer = joint_model.give_transition()
        evolved_space = joint_model.give_normal()

        self.filterers.set_at_time(filterer, time + 1)
        self.evolved_spaces.set_at_time(evolved_space, time + 1)

    def filter(self, time: "int", observation: "ndarray"):

        evolved_space: "NormalModel" = self.evolved_spaces.get_from_time(
            time + 1
        )  # time + 1
        filterer: "TransitionModel" = self.filterers.get_from_time(time + 1)  # time + 1
        error: "InvWishartModel" = self.wisharts.get_from_time(time)  # time

        filtered_state = filterer.observe(observation)
        error = evolved_space.update_wishart(error, observation)

        self.filtered_states.set_at_time(filtered_state, time + 1)
        self.wisharts.set_at_time(error, time + 1)

    def observe_filtered(self, time: "int", observer: "TransitionModel"):

        filtered_state = self.filtered_states.get_from_time(time + 1)  # time + 1

        joint_model = JointModel(filtered_state, observer)
        filtered_space = joint_model.mutate_normal()

        self.filtered_spaces.set_at_time(filtered_space, time + 1)

    def smoothen(self, time: "int"):

        smoother = self.smoothers.get_from_time(self.S - time)  # s - time
        smoothed_state = self.smoothed_states.get_from_time(self.S - time)  # s - time

        joint_model = JointModel(smoothed_state, smoother)
        smoothed_state = joint_model.mutate_normal()

        self.smoothed_states.set_at_time(smoothed_state, self.S - time - 1)

    def observe_smoothed(self, time: "int", observer: "TransitionModel"):

        smoothed_state = self.smoothed_states.get_from_time(self.S - time)  # s - time

        joint_model = JointModel(smoothed_state, observer)
        smoothed_space = joint_model.mutate_normal()

        self.smoothed_spaces.set_at_time(smoothed_space, self.S - time)

    def predict(self, time: "int", evolver: "TransitionModel"):

        predicted_state = self.predicted_states.get_from_time(self.S + time)  # s + time

        joint_model = JointModel(predicted_state, evolver)
        predicted_state = joint_model.mutate_normal()

        self.predicted_states.set_at_time(predicted_state, self.S + time + 1)

    def observe_predicted(self, time: "int", observer: "TransitionModel"):

        predicted_state = self.predicted_states.get_from_time(
            self.S + time + 1
        )  # s + time + 1

        joint_model = JointModel(predicted_state, observer)
        predicted_space = joint_model.mutate_normal()

        self.predicted_spaces.set_at_time(self.S + time + 1)


class PrimeMemoryDLM:
    def __init__(
        self,
        observed_period: "int",
        predicted_period: "int",
        observations: "ndarray",
        primordial_model: "NormalModel",
        primordial_error: "InvWishartModel",
        evolvers: "ModelContainer",
        observers: "ModelContainer",
    ):

        self.S = observed_period
        self.P = predicted_period

        # -- Data
        self.observations = observations

        # -- Models
        self.primordial_model = primordial_model
        self.primordial_error = primordial_error

        # -- Transitions
        self.evolvers = evolvers
        self.observers = observers
