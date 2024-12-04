#!/bin/sh

sudo docker stop webgl
sudo docker rm webgl
sudo docker rmi ubuntu/apache2

sudo rm -rf build
sudo rm -rf oce-OCE-0.18
