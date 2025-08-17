from typing import Callable, List

import numpy as np

class HMM:

    def __init__(
        self,
        sensor_model: Callable[[str, int], float],
        transition_model: Callable[[int, int], float],
        num_states: int,
    ):
        self.sensor_model = sensor_model
        self.transition_model = transition_model
        self.num_states = num_states
        self.current_timestep = 0
        self.probabilities = [1/self.num_states] * num_states
        """
        Inputs:
        - sensor_model: the sensor model of the HMM.
          This is a function that takes in an observation E
          (represented as a string 'A', 'B', ...) and a state S
          (reprensented as a natural number 0, 1, ...) and
          outputs the probability of observing E in state S.

        - transition_model: the transition model of the HMM.
          This is a function that takes in two states, s and s',
          and outputs the probability of transitioning from
          state s to state s'.

        - num_states: this is the number of hidden states in the HMM, an integer
        """

    def tell(self, observation: str):
        """
        Takes in an observation and records it.

        Input:
        - observation: The observation at the current timestep, a string

        Output:
        - None
        """
        update = [0] * self.num_states

        for state in range(self.num_states):
          update[state] = self.sensor_model(observation, state) * sum([self.transition_model(_, state) * self.probabilities[_] for _ in range(self.num_states)])

        total = sum(update)

        for i in range(self.num_states):
          self.probabilities[i] = update[i] / total
        
        self.current_timestep += 1

    def ask(self, time: int) -> List[float]:
        """
        Takes in a timestep that is greater than or equal to
        the current timestep and outputs a probability distribution
        (represented as a list) over states for that timestep.
        The index of the probability is the state it corresponds to.

        Input:
        - time: the timestep to get the observation distribution for, an integer

        Output:
        - a probability distribution over the hidden state for the given timestep, a list of numbers
        """
        if time == 0:
          return [1/self.num_states] * self.num_states

        if time == self.current_timestep:
          return self.probabilities

        prev_distribution = self.probabilities.copy()
        current_distribution = [0] * self.num_states
        difference = time - self.current_timestep

        while difference > 0:
          for state in range(self.num_states):
            current_distribution[state] = sum([self.transition_model(_, state) * prev_distribution[_] for _ in range(self.num_states)])
          prev_distribution = current_distribution.copy()
          difference -= 1

        return current_distribution
