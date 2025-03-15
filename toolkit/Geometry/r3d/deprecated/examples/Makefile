######################################################
#
#	Makefile
#
#	for r3d examples
#
#	usage: `make`
#
######################################################

R3D_HOME = ..
SRC_VOX = voxelize.c 
SRC_CLIP = clip.c

CC = gcc
CFLAGS = -Wall -I. -O3 
INC += -I$(R3D_HOME)
LIB += -L$(R3D_HOME)
LDFLAGS += -lr3d -lm
OBJ_VOX = $(SRC_VOX:.c=.o)
OBJ_CLIP = $(SRC_CLIP:.c=.o)

all: r3d voxelize 

r3d: 
	make -C $(R3D_HOME)

voxelize: $(OBJ_VOX)
	$(CC) $(LIB) $(OBJ_VOX) -o $@ $(LDFLAGS) $(CFLAGS)

clip: $(OBJ_CLIP)
	$(CC) $(LIB) $(OBJ_CLIP) -o $@ $(LDFLAGS) $(CFLAGS)

%.o: %.c $(DEPS)
	$(CC) $(INC) -c -o $@ $< $(CFLAGS) $(OPT)

clean:
	rm -rf clip voxelize $(OBJ_VOX) $(OBJ_CLIP) 
