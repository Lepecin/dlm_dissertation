from objects import NormalModel, TransitionModel, JointModel, InvWishart
from typing import List
from dataclasses import dataclass, field
from numpy import ndarray


@dataclass(repr=False)
class MemoryDLM:

    # -- Space Models
    filtered_states: "List[NormalModel]" = field(default_factory=list)
    evolved_states: "List[NormalModel]" = field(default_factory=list)
    smoothed_states: "List[NormalModel]" = field(default_factory=list)
    predicted_states: "List[NormalModel]" = field(default_factory=list)

    # -- Space Models
    filtered_spaces: "List[NormalModel]" = field(default_factory=list)
    evolved_spaces: "List[NormalModel]" = field(default_factory=list)
    smoothed_spaces: "List[NormalModel]" = field(default_factory=list)
    predicted_spaces: "List[NormalModel]" = field(default_factory=list)

    # -- Transitions
    smoothers: "List[TransitionModel]" = field(default_factory=list)
    filterers: "List[TransitionModel]" = field(default_factory=list)

    # -- Error Matrix
    wisharts: "List[InvWishart]" = field(default_factory=list)

    def evolve(self, time: "int", evolver: "TransitionModel"):

        filtered_state = self.filtered_states[time]

        joint_model = JointModel(filtered_state, evolver)
        joint_model = joint_model.mutate_joint_model()
        smoother = joint_model.transition
        evolved_state = joint_model.normal

        self.smoothers.append(smoother)
        self.evolved_states.append(evolved_state)

    def observe_evolved(self, time: "int", observer: "TransitionModel"):

        evolved_state = self.evolved_states[time]

        joint_model = JointModel(evolved_state, observer)
        joint_model = joint_model.mutate_joint_model()
        filterer = joint_model.transition
        evolved_space = joint_model.normal

        self.filterers.append(filterer)
        self.evolved_spaces.append(evolved_space)

    def filter(self, time: "int", observation: "ndarray"):

        evolved_space = self.evolved_spaces[time]
        filterer: "TransitionModel" = self.filterers[time]
        error: "InvWishart" = self.wisharts[time]

        filtered_state = filterer.observe(observation)
        error = evolved_space.update_wishart(error, observation)

        self.filtered_states.append(filtered_state)
        self.wisharts.append(error)

    def observe_filtered(self, time: "int", observer: "TransitionModel"):

        filtered_state = self.filtered_states[time + 1]

        joint_model = JointModel(filtered_state, observer)
        filtered_space = joint_model.mutate_normal()

        self.filtered_spaces.append(filtered_space)

    def smoothen(self, time: "int"):

        smoother = self.smoothers[-1 - time]
        smoothed_state = self.smoothed_states[time]

        joint_model = JointModel(smoothed_state, smoother)
        smoothed_state = joint_model.mutate_normal()

        self.smoothed_states.append(smoothed_state)

    def observe_smoothed(self, time: "int", observer: "TransitionModel"):

        smoothed_state = self.smoothed_states[time]

        joint_model = JointModel(smoothed_state, observer)
        smoothed_space = joint_model.mutate_normal()

        self.smoothed_spaces.append(smoothed_space)

    def predict(self, time: "int", evolver: "TransitionModel"):

        predicted_state = self.predicted_states[time]

        joint_model = JointModel(predicted_state, evolver)
        predicted_state = joint_model.mutate_normal()

        self.predicted_states.append(predicted_state)

    def observe_predicted(self, time: "int", observer: "TransitionModel"):

        predicted_state = self.predicted_states[time + 1]

        joint_model = JointModel(predicted_state, observer)
        predicted_space = joint_model.mutate_normal()

        self.predicted_spaces.append(predicted_space)


@dataclass(repr=False)
class PrimeMemoryDLM:

    # -- Periods
    forward_period: "int"
    beyond_period: "int"

    # -- Data
    observations: "ndarray"

    # -- Models
    primordial_model: "NormalModel"
    primordial_error: "InvWishart"

    # -- Transitions
    evolvers: "List[TransitionModel]" = field(default_factory=list)
    observers: "List[TransitionModel]" = field(default_factory=list)
