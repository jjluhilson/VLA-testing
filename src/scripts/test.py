import gymnasium as gym
import gymnasium_robotics
import cv2
import numpy as np
import time


# --- MOCK VLA MODEL ---
def vla_predict(image_tensor, text_command):
    """
    This simulates your VLA. In reality, you will pass the image and text
    through HuggingFace transformers (e.g., OpenVLA, RT-1) here.
    """
    # E.g., inputs = processor(text, image_tensor).to("cuda")
    #       outputs = model.generate(**inputs)

    # Returning a random action for the POC
    return np.random.uniform(-1.0, 1.0, size=(4,))


print("Initializing MuJoCo Environment for VLA...")

# Switch to rgb_array so Python can capture the raw pixel tensors
env = gym.make("FetchReach-v4", render_mode="rgb_array")
observation, info = env.reset()

# The natural language instruction for this task
instruction = "reach for the red sphere"

print("Environment loaded! Press 'q' in the image window to quit.")

try:
    for step in range(500):
        # --- 1. IMAGE EXTRACTION ---
        # Grab the standard third-person view as a Numpy Array (RGB format)
        workspace_img = env.render()

        # If you added a wrist camera to the XML, you would grab it like this:
        # wrist_img = env.unwrapped.mujoco_renderer.render(render_mode="rgb_array", camera_name="wrist_cam")

        # --- 2. VLA PRE-PROCESSING ---
        # Neural networks usually expect 224x224 or 256x256 image sizes.
        # We resize the raw image to match the VLA's required input resolution.
        vla_input = cv2.resize(workspace_img, (224, 224))

        # Normalize the image if your specific model requires it (0-1 range)
        vla_input_normalized = vla_input.astype(np.float32) / 255.0

        # --- 3. VLA INFERENCE ---
        # Pass the processed image and text to the model to get the joint movements
        action = vla_predict(vla_input_normalized, instruction)

        # --- 4. STEP SIMULATION ---
        observation, reward, terminated, truncated, info = env.step(action)

        # --- 5. VISUALIZATION (Debugging) ---
        # OpenCV expects BGR color format, so we flip the colors from Gym's RGB
        display_img = cv2.cvtColor(workspace_img, cv2.COLOR_RGB2BGR)

        # Show the feed on your desktop
        cv2.imshow("VLA Vision Feed (Third Person)", display_img)

        # cv2.waitKey handles both the display update and the time.sleep()
        # 20ms wait = ~50 FPS
        if cv2.waitKey(20) & 0xFF == ord("q"):
            print("Quit command received.")
            break

        # --- 6. ENVIRONMENT RESET ---
        if terminated or truncated:
            observation, info = env.reset()
            reason = "terminated (success)" if terminated else "truncated (timeout)"
            print(f"Environment reset at step {step + 1}. Reason: {reason}")

except KeyboardInterrupt:
    print("\nSimulation stopped by user.")

finally:
    env.close()
    cv2.destroyAllWindows()
    print("Environment closed cleanly.")
