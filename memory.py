from objects import NormalModel, TransitionDensity
from typing import List
from dataclasses import dataclass


@dataclass
class DLModelMemory:

    # Create state variable
    state: "str" = "Nihilated"

    # Filtered models
    fst: "List[NormalModel]" = list()
    fsp: "List[NormalModel]" = list()

    # Evolved models
    est: "List[NormalModel]" = list()
    esp: "List[NormalModel]" = list()

    # Observed models
    osp: "List[NormalModel]" = list()

    # Smoothed models
    sst: "List[NormalModel]" = list()
    ssp: "List[NormalModel]" = list()

    # Predicted models
    pst: "List[NormalModel]" = list()
    psp: "List[NormalModel]" = list()

    # Transitions
    smoothers: "List[TransitionDensity]" = list()
    evolvers: "List[TransitionDensity]" = list()
    predictors: "List[TransitionDensity]" = list()
    filters: "List[TransitionDensity]" = list()

    def check(self: "DLModelMemory", state: "str") -> "None":

        if not (self.state == state):
            raise BaseException(f"memory must be in state {state}")

    def restate(self: "DLModelMemory", state: "str") -> "None":

        self.state = state

    def checknstate(self: "DLModelMemory", state: "str", restate: "str") -> "None":

        self.check(state)
        self.restate(restate)
