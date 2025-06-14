✅ 🔧 What this code does:
Fully ROS2 native using rclpy

Subscribes to your /robot/aligned_frame topic

Extracts only the wrist_back color image

Resizes & normalizes image using lerobot’s transform factory

Extracts TCP pose & gripper state from aligned message

Builds correct input dictionary for SmallVLA

Runs model inference live

Publishes resulting TCP target pose & gripper commands to:

/teleop (for robot arm)

/gripper/target (for gripper open/close)

✅ 🔧 What you need installed
rclpy (ROS2 Humble installed properly)

sereact_humanoid_msg ROS2 package (already provided to you)

lerobot installed via:

bash
Copy
Edit
pip install git+https://github.com/huggingface/lerobot.git
cv_bridge (usually installed via ROS2 Humble apt packages)

bash
Copy
Edit
sudo apt install ros-humble-cv-bridge-python
✅ 🔧 How to run it
Activate your ROS2 workspace.

Start ROS2 as usual (ensure your robot drivers are up and streaming aligned frames).

Run:

bash
Copy
Edit
python3 smallvla_inference_node.py
You should see live inference and action publishing in your terminal.

🔥 Good news:
This structure will work both for:

The pretrained model (lerobot/smallvla)

Your fine-tuned model once you replace it with your own checkpoint via:

python
Copy
Edit
SmallVLA.load_from_checkpoint("path/to/your/fine_tuned.ckpt")