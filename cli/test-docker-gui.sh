sudo docker rm test

file1=$0
file2="${file1:1}"
path=$PWD$file2
echo "path: $path"
homepath="${path%FENGSim*}"
homepath+=FENGSim
echo "homepath: $homepath"
cd $homepath
cd cli

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
    -v $homepath:/home/test/FENGSim \
    -e DISPLAY=$DISPLAY \
    -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    $IMAGE_NAME /bin/bash
