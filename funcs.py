from memory import UpdaterDLM, PrimeMemoryDLM


def forward(prime_memory: "PrimeMemoryDLM", memory: "UpdaterDLM") -> "UpdaterDLM":

    observed_period = prime_memory.S
    primordial_model = prime_memory.primordial_model
    primordial_error = prime_memory.primordial_error
    evolvers = prime_memory.evolvers
    observers = prime_memory.observers
    observations = prime_memory.observations

    memory.filtered_states.set_at_time(0, primordial_model)
    memory.wisharts.set_at_time(0, primordial_error)

    for time in range(observed_period):

        observation = observations[time]
        observation = observation.reshape((-1, 1))
        evolver = evolvers.get_from_time(0)
        observer = observers.get_from_time(0)

        memory.evolve(time, evolver)
        memory.observe_evolved(time, observer)
        memory.filter(time, observation)
        memory.observe_filtered(time, observer)

    return memory


def backward(
    prime_memory: "PrimeMemoryDLM",
    memory: "UpdaterDLM",
) -> "UpdaterDLM":

    observed_period = prime_memory.S
    observers = prime_memory.observers

    filtered_state = memory.filtered_states.get_from_time(observed_period)
    memory.smoothed_states.set_at_time(observed_period, filtered_state)

    for time in range(observed_period):

        observer = observers.get_from_time(0)

        memory.observe_smoothed(time, observer)
        memory.smoothen(time)

    return memory


def beyond(
    prime_memory: "PrimeMemoryDLM",
    memory: "UpdaterDLM",
) -> "UpdaterDLM":

    observed_period = prime_memory.S
    predicted_period = prime_memory.P
    evolvers = prime_memory.evolvers
    observers = prime_memory.observers

    filtered_state = memory.filtered_states.get_from_time(observed_period)
    memory.predicted_states.set_at_time(observed_period, filtered_state)

    filtered_space = memory.filtered_spaces.get_from_time(observed_period)
    memory.predicted_spaces.set_at_time(observed_period, filtered_space)

    for time in range(predicted_period):

        evolver = evolvers[0]
        observer = observers[0]

        memory.predict(time, evolver)
        memory.observe_predicted(time, observer)

    return memory
