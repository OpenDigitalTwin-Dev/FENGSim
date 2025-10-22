# file:    Makefile
# purpose: make M++
# $Header: /public/M++/src/Makefile,v 1.6 2008-12-05 14:20:52 maurer Exp $

SOURCES = Identify.C MatrixGraph.C Algebra.C Shape.C \
	  Point.C Parallel.C Time.C ctools.C IO.C Mesh.C \
          Cell.C Distribution.C Plot.C \
	  Quadrature.C DoF.C Sparse.C\
	  Interface.C Discretization.C \
	  LinearSolver.C DD.C Preconditioner.C TimeSeries.C \
	  ESolver.C Transfer.C \
	  Element.C FaceElement.C Newton.C Newmark.C Euler.C Small.C Schur.C 

include ../conf/Makefile.inc

LIB = ../lib/libm++.a

OBJECTS = $(SOURCES:.C=.o)

DEPENDENCIES = $(SOURCES:.C=.d)

default: $(DEPENDENCIES) $(CONFIG) Makefile $(OBJECTS)
	$(MAKE) lib

lib: $(OBJECTS)
	ar -ru $(LIB) $(OBJECTS)

include $(DEPENDENCIES)

clean:
	rm -f $(EXECUTABLE) $(OBJECTS) $(LIB)

cleanall: clean
	rm -f *.d *~

distclean: cleanall
	rm -f *~

depend: $(DEPENDENCIES)
	cat $(DEPENDENCIES) > depend

new:
	rm -f $(EXECUTABLE)
	$(MAKE);
