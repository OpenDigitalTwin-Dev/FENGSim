#!/bin/sh

for i in `ls -1 *.c` ; do 
    sed -e 's/	[	]*/    /g' $i > changed_$i ;  
    echo "$i converted to  changed_$i" ; 
done

exit(0)
