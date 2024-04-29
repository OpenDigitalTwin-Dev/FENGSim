#######################################################################
# Fast Auxiliary Space Preconditioners (FASP)
# 
########################################################################
# Makefile for building the tutorial executables. If the file
# ../Config.mk exists it will take the values of configuration
# variables from there. Otherwise, they have to be user supplied below.
########################################################################

fasp_prefix = not-defined-yet
fasp_library = not-defined-yet
CC = not-defined-yet
FC = not-defined-yet
CXX = not-defined-yet

# include the configuration written by CMake at config time if found
sinclude ../Config.mk

ifeq ($(fasp_prefix),not-defined-yet)
    fasp_prefix = ..
endif
ifeq ($(fasp_library),not-defined-yet)
    fasp_library = libfasp.a
endif
ifeq ($(CC),not-defined-yet) 
   CC=gcc 
endif
ifeq ($(FC),not-defined-yet)
   FC=gfortran
endif
ifeq ($(CXX),not-defined-yet)
   CXX=g++
endif

CFLAGS=-I$(fasp_prefix)/include 
CFLAGS+=-O3
FFLAGS=-I$(fasp_prefix)/include 
FFLAGS+=-O3 
LINKER = $(FC) # because of linking with Fortran files
LFLAGS = -L$(fasp_prefix)/lib -lfasp 
fasp_lib_file=$(fasp_prefix)/lib/$(fasp_library) 

examples = poisson-amg-c.ex poisson-its-c.ex poisson-pcg-c.ex \
           poisson-gmg-c.ex spe01-its-c.ex \
           poisson-amg-f.ex poisson-pcg-f.ex
examples_f = $(filter %-f.ex,$(examples))
examples_c = $(filter-out %-f.ex,$(examples))

.PHONY:	all clean

all:	$(examples_c) $(examples_f)

%-c.ex:	main/%.c $(fasp_lib_file)
	@$(CC) -c $(CFLAGS) -o main/$@.o $<
	@$(LINKER) -o $@ main/$@.o $(LFLAGS)
	@echo 'Building executable file $@'

%-f.ex:	main/%.f90 $(fasp_lib_file)
	@$(FC) -c $(FFLAGS) -o main/$@.o $< 
	@$(FC) -o $@ main/$@.o $(LFLAGS)
	@echo 'Building executable file $@'

$(fasp_lib_file):	
	$(error The FASP library $@ is not found)

clean:
	@-rm -f *.o main/*.o *~

distclean: clean
	@-rm -f poisson-amg-c.ex poisson-its-c.ex poisson-pcg-c.ex \
                poisson-gmg-c.ex spe01-its-c.ex \
                poisson-amg-f.ex poisson-pcg-f.ex 

