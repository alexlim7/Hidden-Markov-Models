# Hidden Markov Models

This project implements Hidden Markov Models (HMMs) to infer hidden states from observable sequences. The goal is to develop a system that can predict hidden states over time based on observed data.

## Features

- **HMM Class**: Core class implementing the HMM functionalities.
  - `tell(observation)`: Updates the model with a new observation.
  - `filter()`: Computes the probability distribution over hidden states at the current timestep.
  - `predict()`: Computes the probability distribution over hidden states at future timesteps.
  - `__str__()`: Returns a string representation of the model's current state.

## How to Run

The main entry point is `touchscreen_runner.py` or `hmm_runner.py`. You can run simulations, visualize results, or evaluate the HMM model using command-line arguments. Example commands:

```bash
# Run a touchscreen simulation with default size and 100 frames
python touchscreen_runner.py

# Run a simulation with specific width, height, and number of frames
python touchscreen_runner.py --width 30 --height 30 --frames 200

# Visualize the simulation
python touchscreen_runner.py --visualize
```

## Project Structure

```text
.
├── touchscreen_helpers       # Helper functions for touchscreen simulation
├── hmm.py                    # Contains the HMM class and related functions
├── hmm_runner.py             # Runs the HMM with simulated touchscreen data
├── touchscreen.py            # Simulates touchscreen inputs
├── touchscreen_runner.py     # Runs the touchscreen simulation
├── unit_tests.py             # Unit tests for HMM functionalities
