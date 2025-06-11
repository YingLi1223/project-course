# Reinforcement Learning Based Frequency Control

This project provides a minimal skeleton for studying reinforcement learning control of converter-based renewable energy systems. The example focuses on a modified IEEE 33-bus distribution system with inverter-based PV generators. A lightweight `gym` environment is included to model frequency dynamics and allow integration of custom RL agents.

## Structure

- `envs/freq_control_env.py` – OpenAI Gym style environment implementation
- `agents/agent_interface.py` – base `Agent` class and a simple `RandomAgent` used for testing
- `models/` – placeholder directory for policy or value networks
- `train.py` – mock training loop invoking the environment and a random agent
- `data/` – sample CSV files containing load and irradiance time series
- `config.py` – configuration parameters such as inertia constant and time step

The environment implements a simple swing equation:

```math
\frac{df}{dt} = \frac{P_{gen} - P_{load} - D(f - f_0)}{2H}
```

where `f_0` is the nominal frequency, `H` the inertia constant, and `D` the damping coefficient. Actions correspond to generator power references that influence frequency.

Feel free to extend the agent and model implementations with more sophisticated algorithms or integrate a power flow solver such as `pypower` or `pandapower`.

