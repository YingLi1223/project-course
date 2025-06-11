"""Frequency control environment for reinforcement learning."""

import csv
from pathlib import Path
from typing import Tuple

import numpy as np
from gym import Env, spaces


class FrequencyControlEnv(Env):
    """OpenAI Gym-style environment for frequency control."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, config: dict, data_path: Path):
        super().__init__()
        self.config = config
        self.data_path = data_path

        self.time_step = config.get("time_step", 1.0)
        self.nominal_frequency = config.get("nominal_frequency", 50.0)
        self.H = config.get("H", 5.0)
        self.D = config.get("D", 1.0)
        self.num_gens = config.get("num_gens", 1)

        # Load data
        self.load_profile = self._load_csv(data_path / "load.csv")
        self.irradiance_profile = self._load_csv(data_path / "irradiance.csv")
        self.total_steps = len(self.load_profile)

        # Spaces
        # Observation: [frequency, load, irradiance]
        high = np.array([np.finfo(np.float32).max] * 3, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Action: power reference for each generator (normalized)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_gens,),
            dtype=np.float32,
        )

        self.state = None
        self.current_step = 0

    @staticmethod
    def _load_csv(path: Path) -> np.ndarray:
        data = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    data.append(float(row[0]))
        return np.array(data, dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Run one time step."""
        freq, load, irradiance = self.state

        # Simple power generation model from irradiance and action
        p_gen = irradiance + np.sum(action)

        # Swing equation
        df_dt = (p_gen - load - self.D * (freq - self.nominal_frequency)) / (2 * self.H)
        freq = freq + df_dt * self.time_step

        self.current_step += 1
        done = self.current_step >= self.total_steps

        if not done:
            load = self.load_profile[self.current_step]
            irradiance = self.irradiance_profile[self.current_step]
        next_state = np.array([freq, load, irradiance], dtype=np.float32)
        self.state = next_state

        # Reward: penalize frequency deviation and action magnitude
        freq_dev = freq - self.nominal_frequency
        reward = - (freq_dev ** 2 + 0.01 * np.sum(np.square(action)))

        return next_state, reward, done, {}

    def reset(self) -> np.ndarray:
        """Reset environment state."""
        self.current_step = 0
        load = self.load_profile[self.current_step] if len(self.load_profile) > 0 else 0.0
        irradiance = self.irradiance_profile[self.current_step] if len(self.irradiance_profile) > 0 else 0.0
        freq = self.nominal_frequency
        self.state = np.array([freq, load, irradiance], dtype=np.float32)
        return self.state

    def render(self, mode="human") -> None:
        """Render environment state."""
        freq, load, irradiance = self.state
        print(f"Step: {self.current_step} Freq: {freq:.2f}Hz Load: {load:.2f} Irr: {irradiance:.2f}")


