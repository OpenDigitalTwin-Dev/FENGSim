# file:    Makefile.inc
# purpose: make includes
# $Header: /public/M++/conf/Makefile.inc,v 1.18 2009-09-21 16:35:27 wieners Exp $

CC = mpicxx
F77 = mpif77

.SUFFIXES:
.SUFFIXES: .o .C .d .f

.PHONY: clean depend distclean explicit new lib

#LIB_DIR = -L../../lib
LIB_DIR = -L../../lib 

DEBUGOPTIONS = -g
OPIMIZEOPTIONS = -O3
#LINK_LIB = -llapack -lg2c -lilu_ml -lsuperlu
#LINK_LIB = -lsuperlu -llapack -lblas -lCGAL -lCGAL_Core -lgmp -l_CuraEngine -lclipper -lgtest -lArcus -lgomp
#LINK_LIB = -lsuperlu -llapack -lblas -lCGAL -lCGAL_Core -lgmp -lmpfr -lGmsh -ltet
LINK_LIB = -lsuperlu -llapack -lblas 


GENERAL_COMPILEOPTIONS = $(INCLUDE_DIR) -fPIC
GENERAL_LINKOPTIONS = $(LIB_DIR)

COMPILEOPTIONS = $(GENERAL_COMPILEOPTIONS) $(DEBUGOPTIONS) $(OPIMIZEOPTIONS)
LINKOPTIONS = $(GENERAL_LINKOPTIONS) -lm++ $(LINK_LIB)

.C.o:
	$(CC) -c $(COMPILEOPTIONS) -I../../src $<

#   mpicxx -c $(INCLUDE_DIR) -fPIC -g -03 -I../../src 

.C.d:
	($(SHELL) -ec '$(CC) -MM -c -I../../src $< \
			| sed '\''s/\($*\)\.o[ :]*/\1.o $@ : /g'\'' > $@; \
            [ -s $@ ] || rm -f $@')

.f.o:   $(F77) -c $(COMPILEOPTIONS) -I../../src $<
