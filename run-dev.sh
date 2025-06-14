#!/bin/bash

docker build -t ros2-humble-mamba .

docker run -it --rm \
  -v $(pwd):/workspace/bottender \
  -w /workspace/bottender \
  ros2-humble-mamba