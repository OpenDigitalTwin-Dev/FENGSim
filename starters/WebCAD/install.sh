#!/bin/sh

#sudo docker run -p 80:80 -v ./:/usr/local/apache2/htdocs/ --name webgl -d httpd
sudo docker run -p 80:80 -v ./:/var/www/html --name webgl -d ubuntu/apache2
sudo docker exec -it webgl /bin/bash -c ".//var/www/html/install2.sh"

