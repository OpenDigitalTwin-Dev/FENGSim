sudo docker rm test

#!/bin/bash
IMAGE_NAME="ubuntu:24.04"
#IMAGE_NAME="ros2:latest"
WORKSPACE_DIR="$(pwd)"

sudo xhost +si:localuser:root

#xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge -

sudo chmod 777 /tmp/.docker.xauth

XAUTH=/tmp/.docker.xauth

sudo docker run --name test -it --privileged --network host \
    --shm-size=5G \
    -w /home/test \
    -v $HOME/FENGSim:/home/test/FENGSim \
    -e DISPLAY=$DISPLAY \
    -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    $IMAGE_NAME /bin/bash
