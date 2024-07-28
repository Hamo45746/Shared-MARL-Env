import os
import sys
from memray import Tracker
import psutil

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import Environment

def profile_environment():
    config_path = 'path/to/your/config.yaml'  # Update this path
    env = Environment(config_path)
    
    # Run a simulation
    env.reset()
    for _ in range(100):  # or however many steps you typically run
        action_dict = {agent_id: agent.get_next_action() for agent_id, agent in enumerate(env.agents)}
        env.step(action_dict)

    # Print memory usage
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024} MB")

if __name__ == "__main__":
    with Tracker("env_profile.bin", follow_fork=True):
        profile_environment()

print("Profiling complete. Check env_profile.bin for results.")