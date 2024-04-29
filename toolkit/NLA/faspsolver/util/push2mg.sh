#!/bin/sh

# backup server info
SERVER="103.3.62.110"
USERID="root"

# optional parameters
OPT="--delete --progress --times --exclude *~"

# upload: Should be ran in doc directory with htdocs as a subdir
if [ ! -d htdocs/download ]; then
    mkdir htdocs/download
fi

# cp ../../download/.ht* htdocs/download
cp ../../download/*.zip htdocs/download
cp ../../download/*.pdf htdocs/download
rsync -avz $1 ${OPT} ./htdocs/* ${USERID}@${SERVER}:/www/wwwroot/103.3.62.110/fasp
