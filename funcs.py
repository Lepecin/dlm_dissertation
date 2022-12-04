from operators import transmutator, observator
from utils import relister
from memory import MemoryDLM, PrimeMemoryDLM


def new_forward(prime_memory: "PrimeMemoryDLM") -> "MemoryDLM":

    # Create memory
    memory = MemoryDLM()

    # Append the primordial state to filtered states
    memory.filtered_states.append(prime_memory.primordial_model)

    # For all times over the forward period
    for time in range(prime_memory.forward_period):

        # -- Evolution

        # Pick the filtered state model of the given time
        filtered_state = memory.filtered_states[time]

        # Create evolved state and smoother
        evolved_state, smoother = transmutator(
            model=filtered_state, transition=prime_memory.evolver
        )

        # Append the smoother to list of smoothers
        memory.smoothers.append(smoother)

        # Append the evolved states to list of evolved states
        memory.evolved_states.append(evolved_state)

        # -- Evolved Observation

        # Create evolved space and filterer
        evolved_space, filterer = transmutator(
            model=evolved_state, transition=prime_memory.observer
        )

        # Append evolved space to list of evolved spaces
        memory.evolved_spaces.append(evolved_space)

        # -- Filration

        # Extract current observation
        observation = prime_memory.observations[time]

        # Create filtered state
        filtered_state = observator(observation=observation, transition=filterer)

        # Append filtered state to list of filtered states
        memory.filtered_states.append(filtered_state)

        # -- Filtered Observation

        # Create filtered space
        filtered_space = transmutator(
            model=filtered_state, transition=prime_memory.observer, model_only=True
        )

        # Append filtered space to list of filtered spaces
        memory.filtered_spaces.append(filtered_space)

    return memory


def new_backward(
    prime_memory: "PrimeMemoryDLM",
    memory: "MemoryDLM",
) -> "MemoryDLM":

    # Extract final filtered state
    filtered_state = memory.filtered_states[prime_memory.forward_period]

    # Append last filtered state to list of smoothed states
    memory.smoothed_states.append(filtered_state)

    # For all times over the forward period
    for time in range(prime_memory.forward_period):

        # -- Smoothed Observation

        # Extract current smoothed state
        smoothed_state = memory.smoothed_states[time]

        # Create smoothed space
        smoothed_space = transmutator(
            model=smoothed_state, transition=prime_memory.observer, model_only=True
        )

        # Append smoothed space to list of smoothed spaces
        memory.smoothed_spaces.append(smoothed_space)

        # -- Smoothing

        # Extract current smoother
        smoother = memory.smoothers[-1 - time]

        # Create smoothed state
        smoothed_state = transmutator(
            model=smoothed_state, transition=smoother, model_only=True
        )

        # Append smoothed state to list of smoothed states
        memory.smoothed_states.append(smoothed_state)

    memory.smoothed_states = relister(memory.smoothed_states)

    memory.smoothed_spaces = relister(memory.smoothed_spaces)

    return memory


def new_beyond(
    prime_memory: "PrimeMemoryDLM",
    memory: "MemoryDLM",
):

    # Extract final filtered state
    filtered_state = memory.filtered_states[prime_memory.forward_period]

    memory.predicted_states.append(filtered_state)

    # Extract final filtered space
    filtered_space = memory.filtered_spaces[prime_memory.forward_period - 1]

    memory.predicted_spaces.append(filtered_space)

    for time in range(prime_memory.beyond_period):

        # -- Prediction

        predicted_state = memory.predicted_states[time]

        predicted_state = transmutator(
            model=predicted_state, transition=prime_memory.evolver, model_only=True
        )

        memory.predicted_states.append(predicted_state)

        # -- Predicted Observation

        predicted_space = transmutator(
            model=predicted_state, transition=prime_memory.observer, model_only=True
        )

        memory.predicted_spaces.append(predicted_space)

    return memory
