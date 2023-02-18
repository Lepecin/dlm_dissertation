from .memory import PrimeMemoryDLM
from .updater import UpdaterDLM


class ModellerDLM:
    def __init__(self, prime_memory: "PrimeMemoryDLM"):
        self.prime_memory = prime_memory
        self.memory = UpdaterDLM(prime_memory.S, prime_memory.P)

    def forward(self):

        observed_period = self.prime_memory.S
        primordial_model = self.prime_memory.primordial_model
        primordial_error = self.prime_memory.primordial_error
        evolvers = self.prime_memory.evolvers
        observers = self.prime_memory.observers
        observations = self.prime_memory.observations

        self.memory.filtered_states.set_at_time(0, primordial_model)
        self.memory.wisharts.set_at_time(0, primordial_error)

        for time in range(observed_period):

            observation = observations[time]
            observation = observation.reshape((-1, 1))
            evolver = evolvers.get_from_time(0)
            observer = observers.get_from_time(0)

            self.memory.evolve(time, evolver)
            self.memory.observe_evolved(time, observer)
            self.memory.filter(time, observation)
            self.memory.observe_filtered(time, observer)

    def backward(self):

        observed_period = self.prime_memory.S
        observers = self.prime_memory.observers

        filtered_state = self.memory.filtered_states.get_from_time(observed_period)
        self.memory.smoothed_states.set_at_time(observed_period, filtered_state)

        for time in range(observed_period):

            observer = observers.get_from_time(0)

            self.memory.observe_smoothed(time, observer)
            self.memory.smoothen(time)

    def beyond(self):

        observed_period = self.prime_memory.S
        predicted_period = self.prime_memory.P
        evolvers = self.prime_memory.evolvers
        observers = self.prime_memory.observers

        filtered_state = self.memory.filtered_states.get_from_time(observed_period)
        self.memory.predicted_states.set_at_time(observed_period, filtered_state)

        filtered_space = self.memory.filtered_spaces.get_from_time(observed_period)
        self.memory.predicted_spaces.set_at_time(observed_period, filtered_space)

        for time in range(predicted_period):

            evolver = evolvers.get_from_time(0)
            observer = observers.get_from_time(0)

            self.memory.predict(time, evolver)
            self.memory.observe_predicted(time, observer)

    def get_memory(self):
        return self.memory

    def get_prime_memory(self):
        return self.prime_memory
