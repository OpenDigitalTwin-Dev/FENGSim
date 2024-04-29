#!/bin/sh
# 
# usage is fhead.sh ${CMAKE_CURRENT_SOURCE_DIR} where we assume that
# ${CMAKE_CURRENT_SOURCE_DIR} has subdirectories src and include and
# the function names from src/*.c are put into the include/fasp_functs.h
set +x
/bin/cat $1/src/*.c $1/extra/interface/*.c \
        | awk -v name="fasp_functs.h" -f mkheaders.awk \
        > $1/include/fasp_functs.h
set -x
