######################################################
#
#	Makefile
#
#	libr3d.a
#
#	See Readme.md and r3d.h/r2d.h for usage
#
######################################################

######## User options #########

# Use single-precision computations (not recommended) 
#OPT += -DSINGLE_PRECISION

# Use an octree to efficiently search enclosed regions
# For most use cases, this gives negligible improvement 
#OPT += -DUSE_TREE

# Turn off clipping in voxelization routines
#OPT += -DNO_CLIPPING

# Turn off reduction in voxelization routines
#OPT += -DNO_REDUCTION

###############################

CC = gcc
CFLAGS = -Wall -I. -O3 
SRC = r3d.c r2d.c
DEPS = r3d.h r2d.h Makefile
OBJ = $(SRC:.c=.o)

all: libr3d.a

libr3d.a: $(OBJ)
	ar -rs $@ $^

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(OPT)

clean:
	rm -rf libr3d.a $(OBJ) 
