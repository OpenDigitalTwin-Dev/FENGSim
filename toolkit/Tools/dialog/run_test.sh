#!/bin/sh
# $Id: run_test.sh,v 1.3 2022/12/29 16:13:11 tom Exp $
##############################################################################
# Copyright (c) 2020,2022 Thomas E. Dickey                                   #
#                                                                            #
# Permission is hereby granted, free of charge, to any person obtaining a    #
# copy of this software and associated documentation files (the "Software"), #
# to deal in the Software without restriction, including without limitation  #
# the rights to use, copy, modify, merge, publish, distribute, distribute    #
# with modifications, sublicense, and/or sell copies of the Software, and to #
# permit persons to whom the Software is furnished to do so, subject to the  #
# following conditions:                                                      #
#                                                                            #
# The above copyright notice and this permission notice shall be included in #
# all copies or substantial portions of the Software.                        #
#                                                                            #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    #
# THE ABOVE COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        #
# DEALINGS IN THE SOFTWARE.                                                  #
#                                                                            #
# Except as contained in this notice, the name(s) of the above copyright     #
# holders shall not be used in advertising or otherwise to promote the sale, #
# use or other dealings in this Software without prior written               #
# authorization.                                                             #
##############################################################################

failed() {
	echo "? $*" >&2
	exit 1
}

: "${DIALOG=./dialog}"
CONFIG=samples
INPUTS=inputs.rc
OUTPUT=output.rc

[ $# != 0 ] && CONFIG="$1"

[ -f "$DIALOG" ] || failed "no such file: $DIALOG"
[ -d "$CONFIG" ] || failed "no such directory: $CONFIG"

create_rc="`$DIALOG --help | grep create-rc`"
if [ -z "$create_rc" ]
then
	echo "This version of dialog does not support --create-rc"
	exit
fi

for rcfile in "$CONFIG"/*.rc
do
	echo "** $rcfile"
	DIALOGRC="$rcfile" $DIALOG --create-rc $OUTPUT
	sed -e '/^#/d' "$OUTPUT" >"$INPUTS"
	mv -f $INPUTS $OUTPUT
	sed -e '/^#/d' "$rcfile" >"$INPUTS"
	diff -u $INPUTS $OUTPUT | \
		sed	-e "s,$INPUTS,$rcfile," \
			-e "s,$OUTPUT,$rcfile-test,"
done

rm -f $INPUTS $OUTPUT
