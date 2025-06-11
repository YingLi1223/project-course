"""Project configuration parameters."""

from pathlib import Path

CONFIG = {
    "time_step": 1.0,
    "nominal_frequency": 50.0,
    "H": 5.0,  # inertia constant
    "D": 1.0,  # damping
    "num_gens": 1,
    "learning_rate": 0.001,
    "episodes": 1,
    "steps_per_episode": 10,
}

DATA_PATH = Path(__file__).parent / "data"

