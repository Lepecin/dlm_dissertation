from memory import MemoryDLM, PrimeMemoryDLM
from objects import JointModel


def forward(prime_memory: "PrimeMemoryDLM") -> "MemoryDLM":

    # Create memory
    memory = MemoryDLM()
    # Append the primordial state to filtered states
    memory.filtered_states.append(prime_memory.primordial_model)
    # For all times over the forward period
    for time in range(prime_memory.forward_period):
        memory = forward_cycle(memory, prime_memory, time)

    return memory


def forward_cycle(memory, prime_memory, time):

    filtered_state = memory.filtered_states[time]
    observation = prime_memory.observations[time]
    evolver = prime_memory.evolver
    observer = prime_memory.observer

    # -- Evolution
    joint_model = JointModel(filtered_state, evolver)
    joint_model = joint_model.trans_mutator()
    evolved_state = joint_model.normal
    smoother = joint_model.transition
    memory.smoothers.append(smoother)
    memory.evolved_states.append(evolved_state)

    # -- Observation Preperation
    observation = observation.reshape((-1, 1))

    # -- Evolved Observation
    joint_model = JointModel(evolved_state, observer)
    joint_model = joint_model.trans_mutator()
    evolved_space = joint_model.normal
    filterer = joint_model.transition
    memory.evolved_spaces.append(evolved_space)

    # -- Filtration
    filtered_state = filterer.observe(observation)
    memory.filtered_states.append(filtered_state)

    # -- Filtered Observation
    joint_model = JointModel(filtered_state, observer)
    filtered_space = joint_model.norm_mutator()
    memory.filtered_spaces.append(filtered_space)

    return memory


def backward(
    prime_memory: "PrimeMemoryDLM",
    memory: "MemoryDLM",
) -> "MemoryDLM":

    # Extract final filtered state
    filtered_state = memory.filtered_states[prime_memory.forward_period]
    memory.smoothed_states.append(filtered_state)

    # For all times over the forward period
    for time in range(prime_memory.forward_period):
        memory = backward_cycle(memory, prime_memory, time)

    def relister(l):
        return [l[-1 - i] for i in range(len(l))]

    memory.smoothed_states = relister(memory.smoothed_states)
    memory.smoothed_spaces = relister(memory.smoothed_spaces)

    return memory


def backward_cycle(memory, prime_memory, time):

    observer = prime_memory.observer

    # -- Smoothed Observation
    smoothed_state = memory.smoothed_states[time]
    joint_model = JointModel(smoothed_state, observer)
    smoothed_space = joint_model.norm_mutator()
    memory.smoothed_spaces.append(smoothed_space)

    # -- Smoothing
    smoother = memory.smoothers[-1 - time]
    joint_model = JointModel(smoothed_state, smoother)
    smoothed_state = joint_model.norm_mutator()
    memory.smoothed_states.append(smoothed_state)

    return memory


def beyond(
    prime_memory: "PrimeMemoryDLM",
    memory: "MemoryDLM",
) -> "MemoryDLM":

    # Extract final filtered state
    filtered_state = memory.filtered_states[prime_memory.forward_period]
    memory.predicted_states.append(filtered_state)
    filtered_space = memory.filtered_spaces[prime_memory.forward_period - 1]
    memory.predicted_spaces.append(filtered_space)

    for time in range(prime_memory.beyond_period):
        memory = beyond_cycle(memory, prime_memory, time)

    return memory


def beyond_cycle(memory, prime_memory, time):

    evolver = prime_memory.evolver
    observer = prime_memory.observer

    # -- Prediction
    predicted_state = memory.predicted_states[time]
    joint_model = JointModel(predicted_state, evolver)
    predicted_state = joint_model.norm_mutator()
    memory.predicted_states.append(predicted_state)

    # -- Predicted Observation
    joint_model = JointModel(predicted_state, observer)
    predicted_space = joint_model.norm_mutator()
    memory.predicted_spaces.append(predicted_space)

    return memory
