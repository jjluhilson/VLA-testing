import gymnasium as gym
import gymnasium_robotics
import time

print("Initializing MuJoCo Gymnasium Environment...")

# "human" render mode opens the physical MuJoCo viewer window
env = gym.make("FetchReach-v4", render_mode="human")
observation, info = env.reset()

print("Environment loaded successfully! Running steps...")

try:
    for step in range(500):
        # 1. Random action - to be replaced by VLA model
        action = env.action_space.sample()

        # 2. Step the simulation
        observation, reward, terminated, truncated, info = env.step(action)

        # Slow down the loop slightly so you can see the robot move
        time.sleep(0.02)

        if terminated or truncated:
            observation, info = env.reset()
            print(
                f"Environment reset at step {step + 1}. Reason was: {'terminated' if terminated else 'truncated'}"
            )

except KeyboardInterrupt:
    print("\nSimulation stopped by user.")

finally:
    env.close()
    print("Environment closed cleanly.")
