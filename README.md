# bottender


## Mounting a bag to the docker container:

```bash

docker run -it --rm \
  -v /Users/pkm/Downloads/rosbag2_2025_06_12-18_29_00:/rosbags/my_bag \
  ros2-humble-mamba

docker build -t ros2-humble-mamba .
docker run -it ros2-humble-mamba

ros2 bag play rosbags/my_bag --rate 0.25 --loop


# Moung repository to docker container
docker run -it --rm \
  -v /Users/you/Projects/bottender:/workspace/bottender \
  -w /bottender \
  ros2-humble-mamba


# Mounting both the example bag and the repository
docker run -it --rm \
  -v /Users/pkm/Projects/bottender:/workspace/bottender -v /Users/pkm/Downloads/rosbag2_2025_06_12-18_29_00:/rosbags/my_bag \
  -w /bottender \
  ros2-humble-mamba


# Creating a new package with ROS2
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python gripper_loop --dependencies rclpy std_msgs sereact_humanoid_msg


ros2 pkg create --build-type ament_python ai_hackathon
colcon build
source install/setup.bash
ros2 run ai_hackathon my_node

# Run final docker container (from within bottender directory)
docker build -t bottender-dev .
docker run -it --rm bottender-dev

rm -rf build/ install/ log/
colcon build
# If the above does not work, try:
# colcon build --merge-install --cmake-clean-cache

```