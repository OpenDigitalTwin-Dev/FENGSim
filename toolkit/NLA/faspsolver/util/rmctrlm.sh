#!/bin/bash

## This shell command removes CTRL-M symbol from a Windows format file. 
## It will leave a copy of the old version as a .bak file. 

cp $1 $1.bak
sed -e's///g' $1.bak > $1


