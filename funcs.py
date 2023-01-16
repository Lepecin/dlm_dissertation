from memory import MemoryDLM, PrimeMemoryDLM


def forward(prime_memory: "PrimeMemoryDLM", memory: "MemoryDLM") -> "MemoryDLM":

    forward_period = prime_memory.forward_period
    primordial_model = prime_memory.primordial_model
    evolvers = prime_memory.evolvers
    observers = prime_memory.observers
    observations = prime_memory.observations

    memory.filtered_states.append(primordial_model)

    for time in range(forward_period):

        observation = observations[time]
        observation = observation.reshape((-1, 1))
        evolver = evolvers[0]
        observer = observers[0]

        memory.evolve(time, evolver)
        memory.observe_evolved(time, observer)
        memory.filter(time, observation)
        memory.observe_filtered(time, observer)

    return memory


def backward(
    prime_memory: "PrimeMemoryDLM",
    memory: "MemoryDLM",
) -> "MemoryDLM":

    forward_period = prime_memory.forward_period
    observers = prime_memory.observers

    filtered_state = memory.filtered_states[forward_period]
    memory.smoothed_states.append(filtered_state)

    for time in range(forward_period):

        observer = observers[0]

        memory.observe_smoothed(time, observer)
        memory.smoothen(time)

    def relister(l):
        return [l[-1 - i] for i in range(len(l))]

    memory.smoothed_states = relister(memory.smoothed_states)
    memory.smoothed_spaces = relister(memory.smoothed_spaces)

    return memory


def beyond(
    prime_memory: "PrimeMemoryDLM",
    memory: "MemoryDLM",
) -> "MemoryDLM":

    forward_period = prime_memory.forward_period
    beyond_period = prime_memory.beyond_period
    evolvers = prime_memory.evolvers
    observers = prime_memory.observers

    filtered_state = memory.filtered_states[forward_period]
    memory.predicted_states.append(filtered_state)

    filtered_space = memory.filtered_spaces[forward_period - 1]
    memory.predicted_spaces.append(filtered_space)

    for time in range(beyond_period):

        evolver = evolvers[0]
        observer = observers[0]

        memory.predict(time, evolver)
        memory.observe_predicted(time, observer)

    return memory
