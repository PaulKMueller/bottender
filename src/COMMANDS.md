# setup env
source /opt/ros/humble/setup.bash
# install packages
colcon build --symlink-install
# activate(?) packages
source install/setup.bash
# run node
ros2 run ai_hackathon my_node