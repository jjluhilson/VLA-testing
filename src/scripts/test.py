import gymnasium as gym
import gymnasium_robotics
import cv2
import numpy as np
import time
import mujoco


# --- MOCK VLA MODEL ---
def vla_predict(workspace_img, wrist_img, text_command):
    return np.random.uniform(-1.0, 1.0, size=(4,))


print("Initializing MuJoCo Environment for VLA...")

# Switch to rgb_array so Python can capture the raw pixel tensors
env = gym.make("FetchReach-v4", render_mode="rgb_array")
observation, info = env.reset()

# --- SETUP VIRTUAL WRIST CAMERA ---
# 1. Create a custom MuJoCo camera object
wrist_cam = mujoco.MjvCamera()
# 2. Set it to 'Tracking' mode so it follows a specific body part
wrist_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

model = env.unwrapped.model

# 3. Find the ID of the 'gripper_link' and tell the camera to track it
gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0:gripper_link")
wrist_cam.trackbodyid = gripper_id

# 4. Configure the camera's position relative to the gripper
wrist_cam.distance = 0.2  # Zoom in close to the fingers
wrist_cam.elevation = -45  # Angle it slightly downward
wrist_cam.azimuth = 90  # Look forward along the axis of the arm

# 5. Create a dedicated off-screen renderer for this wrist camera
wrist_renderer = mujoco.Renderer(model, height=224, width=224)

# The natural language instruction for this task
instruction = "reach for the red sphere"

print("Environment loaded! Press 'q' in the image window to quit.")

for i in range(model.ngeom):
    if "mocap" in mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[i]
    ):
        model.geom_rgba[i, 3] = 0

try:
    for step in range(500):
        # --- 1. EXTRACT WORKSPACE VIEW (Third Person) ---
        workspace_img_raw = env.render()
        workspace_img = cv2.resize(workspace_img_raw, (224, 224))

        # --- 2. EXTRACT WRIST VIEW (Eye-in-Hand) ---
        # Update the renderer's scene with current physics data and our tracking camera
        wrist_renderer.update_scene(env.unwrapped.data, camera=wrist_cam)
        scene = wrist_renderer.scene
        wrist_img = wrist_renderer.render()

        # --- 3. VLA PRE-PROCESSING ---
        ws_tensor = workspace_img.astype(np.float32) / 255.0
        wrist_tensor = wrist_img.astype(np.float32) / 255.0

        # --- 4. VLA INFERENCE ---
        action = vla_predict(ws_tensor, wrist_tensor, instruction)

        # --- 5. STEP SIMULATION ---
        observation, reward, terminated, truncated, info = env.step(action)

        # --- 6. VISUALIZATION (Debugging) ---
        # Stitch the two images horizontally to display them side-by-side
        combined_img = cv2.hconcat([workspace_img, wrist_img])
        display_img = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)

        # Draw labels on the UI
        cv2.putText(
            display_img,
            "Workspace View",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            display_img,
            "Wrist View",
            (234, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        cv2.imshow("VLA Multi-View Feed", display_img)

        if cv2.waitKey(20) & 0xFF == ord("q"):
            print("Quit command received.")
            break

        # --- 7. ENVIRONMENT RESET ---
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
