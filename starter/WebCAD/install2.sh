#!/bin/sh

echo "xixixi :-)"

cd /var/www/html
ls
#cp sources.list /etc/apt

apt update
apt -y install cmake libreadline-dev dialog
apt -y install cmake-curses-gui
apt -y install g++
apt -y install build-essential libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev
apt -y install libfreetype-dev

apt -y install php
apt -y install libapache2-mod-php
apt -y install 7zip
cp dir.conf /etc/apache2/mods-enabled/

7z x oce-OCE-0.18.7z -o./
cd oce-OCE-0.18
mkdir build
cd build
cmake ..
make -j4
make install
cd ..
cd ..

mkdir build
cd build
cmake ..
make
cd ..
chmod -R 777 Models
.//build/oceSolver
chmod -R 777 Models

