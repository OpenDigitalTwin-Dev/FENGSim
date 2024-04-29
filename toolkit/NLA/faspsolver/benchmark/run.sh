#!bin/sh

source ~/.bashrc

# help 
# ./benchmark.ex -help

# mtx
./benchmark.ex -f input.dat -startID 1 -endID 235 -mps 100000 -mat_type 0 -mat_dir mtx 2>&1 |tee -a prob1-235.log

# csr  
# ./benchmark.ex -f input-RHD.dat -startID 1 -endID 5 -mat_type 1 -mat_dir RHD -read_rhs 1 2>&1 |tee -a probRHD1-5.log

