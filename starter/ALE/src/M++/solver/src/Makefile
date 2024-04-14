# file:    Makefile
# purpose: make M++
# $Header: /public/M++/BEM/src/Makefile,v 1.3 2008-10-07 09:37:17 wieners Exp $

SOURCES = Main.C \
	  Poisson.C \
	  PoissonPrePost.C

include ../../conf/Makefile.inc

LIB = ../../lib/libm++.a 

EXECUTABLE = ../../PoissonRun

OBJECTS = $(SOURCES:.C=.o)

DEPENDENCIES = $(SOURCES:.C=.d)

COMPILEOPTIONS = -O3 -time -w $(INCLUDE_DIR) -fPIC
#COMPILEOPTIONS = -g -time -w $(INCLUDE_DIR) -fPIC

$(EXECUTABLE): $(DEPENDENCIES) Makefile $(OBJECTS) $(LIB)
	$(CC) -o $(EXECUTABLE) $(OBJECTS) $(LINKOPTIONS) 

default: $(DEPENDENCIES) $(CONFIG) Makefile $(OBJECTS)
	$(MAKE) $(EXECUTABLE)

include $(DEPENDENCIES)

.PHONY: clean depend distclean explicit new

clean:
	rm -f $(EXECUTABLE) $(OBJECTS)

cleanall: clean
	rm -f *.d 

distclean: cleanall
	rm -f *~

depend: $(DEPENDENCIES)
	cat $(DEPENDENCIES) > depend

new:
	rm -f $(EXECUTABLE)
	$(MAKE);
