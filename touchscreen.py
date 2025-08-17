import numpy as np

class touchscreenHMM:

    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.velocity = 0
        self.timestep = 0
        self.previous_state = np.zeros((self.height, self.width))

    def _sensor_model(self, observation: np.ndarray, state: np.ndarray) -> float:
        """
        This is the sensor model to get the probability of getting an observation from a state.

        Input:
        - observation: A 2D NumPy array filled with 0s, and a single 1 denoting a touch location.
        - state:       A 2D NumPy array filled with 0s, and a single 1 denoting a touch location.

        Output:
        - The probability of observing that observation from that given state, a number.
        """
        obs_coordinate = np.argwhere(observation == 1)
        state_coordinate = np.argwhere(state == 1)

        if len(obs_coordinate) == 0 or len(state_coordinate) == 0:
            return 0

        row1, col1 = obs_coordinate[0] 
        row2, col2 = state_coordinate[0]

        if row1 == row2 and col1 == col2:
            if row1 == self.height-1 or row1 == 0 or col1 == self.width-1 or col1 == 0:
                return 1/7
            else:
                return 1/9

        x_diff = abs(row1 - row2)
        y_diff = abs(col1 - col2)
        distance = np.sqrt((x_diff ** 2) + (y_diff ** 2))

        return 1/9 * (1 / distance)


    def _transition_model(self, old_state: np.ndarray, new_state: np.ndarray) -> float:
        """
        Transition model to go from the old state to the new state.

        Input:
        - old_state: A 2D NumPy array filled with 0s, and a single 1 denoting a touch location.
        - new_state: A 2D NumPy array filled with 0s, and a single 1 denoting a touch location.

        Output:
        - The probability of transitioning from the old state to the new state, a number.
        """
        old = np.argwhere(old_state == 1)
        new = np.argwhere(new_state == 1)

        if len(old) == 0 or len(new) == 0:
            return 0

        row1, col1 = old[0] 
        row2, col2 = new[0]

        x_diff = abs(row1 - row2)
        y_diff = abs(col1 - col2)
        
        if x_diff == 0 and y_diff==0 and (row1 == 0 or row1 == self.height-1 or col1 == 0 or col1 == self.width-1):
            return 1/7
        if x_diff<=1 and y_diff<=1:
            return 1/9
        else:
            return 0
        
        

    def filter_noisy_data(self, frame: np.ndarray) -> np.ndarray:
        """
        Passes in a noisy simualation. It returns the distribution where you think the 
        actual position of the finger is in the same format that it is passed in as.

        Input:
        - frame: A noisy frame to run your HMM on. This is a 2D NumPy array
                 filled with 0s, and a single 1 denoting a touch location.

        Output:
        - A 2D NumPy array with the probabilities of the actual finger location.
        """
        if self.timestep == 0:
            self.previous_state = frame.copy()

        current = np.zeros((self.height, self.width))

        total = 0.0
        # Iterate through all elements in the matrix
        for row in range(self.height):
            for column in range(self.width):
                state = np.zeros((self.height, self.width))
                state[row][column] = 1

                total_transition_prob = 0.0
                # Iterate through adjacent elements including itself
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        r, c = row + i, column + j
                        if 0 <= r < self.height and 0 <= c < self.width:
                            adjacent_state = np.zeros((self.height, self.width))
                            adjacent_state[r][c] = 1
                            transition_prob = self._transition_model(adjacent_state, state)
                            total_transition_prob += transition_prob * self.previous_state[r][c]

                current[row][column] = self._sensor_model(frame, state) * total_transition_prob
                total += current[row][column]

        if total != 0:
            current /= total

        self.previous_state = current.copy()
        self.timestep += 1

        return current



if __name__ == "__main__":
    from touchscreen_helpers.generate_data import create_simulations

    # Create an instance of the touchscreenHMM class
    hmm_filter = touchscreenHMM()

    # Sample noisy frame (a 2D NumPy array)
    noisy_frame = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    # Perform filtering on the noisy frame
    filtered_state = hmm_filter.filter_noisy_data(noisy_frame)

    # Print the filtered state (probabilities of the actual finger location)
    print("Filtered State:")
    print(filtered_state)
    pass
