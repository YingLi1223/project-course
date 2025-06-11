"""Mock training loop for frequency control RL."""


from agents.agent_interface import RandomAgent
from config import CONFIG, DATA_PATH
from envs.freq_control_env import FrequencyControlEnv


def main() -> None:
    """Run a simple training loop with a random agent."""
    env = FrequencyControlEnv(CONFIG, DATA_PATH)
    agent = RandomAgent(env.action_space)

    for episode in range(CONFIG["episodes"]):
        obs = env.reset()
        total_reward = 0.0
        for _ in range(CONFIG["steps_per_episode"]):
            action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.learn()
            obs = next_obs
            total_reward += reward
            env.render()
            if done:
                break
        print(f"Episode {episode} Total reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()

