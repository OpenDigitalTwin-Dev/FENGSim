#!/bin/bash

./regression.ex > out/reg.new
sdiff out/reg.out out/reg.new | grep "|"
rm -f out/reg.new

