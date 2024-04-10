# file: Makefile
# author: Christian Wieners
# $Header: /public/M++/Makefile,v 1.20 2009-11-24 10:48:32 wieners Exp $

.PHONY: clean depend distclean explicit new lib

default:
	cd src; $(MAKE)

lib:
	cd src; $(MAKE)
#	cd ../FEniCS/dolfin-2017.1.0/build; $(MAKE)
#	cd /home/jiping/software/cgal-master/build; $(MAKE); sudo make install	

AdditiveManufacturing: lib
	cd AdditiveManufacturing/src; $(MAKE)

Cura: lib
	cd Cura/src; $(MAKE)

Poisson: lib
	cd Poisson/src; $(MAKE)

solver: lib
	cd solver/src; $(MAKE)

Heat: lib
	cd Heat/src; $(MAKE)

Elasticity: lib
	cd Elasticity/src; $(MAKE)

Machining: lib
	rm MachiningRun; rm MachiningRun/src/*.o; rm MachiningRun/src/*.d; cd Machining/src; $(MAKE)

ThermoElasticity: lib
	cd ThermoElasticity/src; $(MAKE)

ElastoPlasticity: default
	cd ElastoPlasticity/src; $(MAKE)

Materials: lib
	cd Materials/src; $(MAKE)

Solids: lib
	cd Solids/src; $(MAKE)

PoissonDolfin: lib
	rm Dolfin; rm PoissonDolfin/src/Main.o; cd PoissonDolfin/src; $(MAKE)

OCE: lib
	rm oce; rm OCE/src/Main.o; cd OCE/src; $(MAKE)

Plasticity: lib
	cd Plasticity/src; $(MAKE)

StaticPlasticity: lib
	cd StaticPlasticity/src; $(MAKE)

Stokes: lib
	cd Stokes/src; $(MAKE)

Porous: lib
	cd Porous/src; $(MAKE)

Cosserat: lib
	cd Cosserat/src; $(MAKE)

Fruitfly: lib
	cd Fruitfly/src; $(MAKE)

Meshes: lib
	cd Meshes/src; $(MAKE)

NumericalAnalysis: lib
	cd NumericalAnalysis/src; $(MAKE)

Edyn: lib
	cd Edyn/src; $(MAKE)

FDTD: lib
	cd FDTD/src; $(MAKE)

DG: lib
	cd DG/src; $(MAKE)

BEM1.0a: lib
	cd BEM1.0a/src; $(MAKE)

BEM1.0b: lib
	cd BEM1.0b/src; $(MAKE)

BEM1.0c: lib
	cd BEM1.0c/src; $(MAKE)

BEM1.0d: lib
	cd BEM1.0d/src; $(MAKE)

BEM1.0e: lib
	cd BEM1.0e/src; $(MAKE)

BEM2.0a: lib
	cd BEM2.0a/src; $(MAKE)

Laplace: lib
	cd Laplace/src; $(MAKE)

EMTH: lib
	cd EMTH/src; $(MAKE)

Maxwell: lib
	cd Maxwell/src; $(MAKE)

MaxwellComplex: lib
	cd MaxwellComplex/src; $(MAKE)

TransientEddyCurrent: lib
	cd TransientEddyCurrent/src; $(MAKE)

MagnetoStatics: lib
	cd MagnetoStatics/src; $(MAKE)

ElectroMagnetismTH: lib
	cd ElectroMagnetismTH/src; $(MAKE)

ConvectionDiffusion: lib
	cd ConvectionDiffusion/src; $(MAKE)

ChemicalProcessModeling: lib
	cd ChemicalProcessModeling/src; $(MAKE)

Heat: lib
	cd Heat/src; $(MAKE)

TimeConvectionDiffusion: lib
	cd TimeConvectionDiffusion/src; $(MAKE)

FokkerPlanck: lib
	cd FokkerPlanck/src; $(MAKE)

Reactors: lib
	cd Reactors/src; $(MAKE)

Pandas: lib
	cd Pandas/src; $(MAKE)

pandaslib: Makefile
	cd Pandas/src; $(MAKE) pandaslib; cd ..;

pandaslibdebug: Makefile
	cd Pandas/src; $(MAKE) pandaslibdebug; cd ..;

cleanpandas: Makefile
	cd Pandas/src; $(MAKE) cleanpandas; cd ..;

ErrorBound: lib
	cd ErrorBound/src; $(MAKE)

Scattering: lib
	cd Scattering/src; $(MAKE)

MultiSpecRed6: lib
	cd MultiSpecRed6/src; $(MAKE)

Wave: lib
	cd Wave/src; $(MAKE)

Schurtest: lib
	cd Schurtest/src; $(MAKE)

double:
	cd src; cp CompilerDOUBLE.h Compiler.h

complex:
	cd src; cp CompilerCOMPLEX.h Compiler.h

clean:
	cd src; $(MAKE) clean
	cd ErrorBound/src; $(MAKE) clean
	cd Plasticity/src; $(MAKE) clean
	cd Stokes/src; $(MAKE) clean
	cd Cosserat/src; $(MAKE) clean
	cd Materials/src; $(MAKE) clean
	cd Fruitfly/src; $(MAKE) clean
	cd Meshes/src; $(MAKE) clean
	cd Porous/src; $(MAKE) clean
	cd Wave/src; $(MAKE) clean
	cd BEM1.0a/src; $(MAKE) clean
	cd BEM1.0b/src; $(MAKE) clean
	cd BEM1.0c/src; $(MAKE) clean	
	cd BEM1.0d/src; $(MAKE) clean	
	cd BEM1.0e/src; $(MAKE) clean	
	cd Laplace/src; $(MAKE) clean	
	cd Poisson/src; $(MAKE) clean		
	cd PoissonDolfin/src; $(MAKE) clean		
	cd EMTH/src; $(MAKE) clean		
	cd TransientEddyCurrent/src; $(MAKE) clean			
	cd Maxwell/src; $(MAKE) clean			
	cd MaxwellComplex/src; $(MAKE) clean			
	cd MagnetoStatics/src; $(MAKE) clean				
	cd ElectroMagnetismTH/src; $(MAKE) clean				
	cd ConvectionDiffusion/src; $(MAKE) clean	
	cd ConvectionDiffusionReaction/src; $(MAKE) clean	
	cd Heat/src; $(MAKE) clean
	cd TimeConvectionDiffusion/src; $(MAKE) clean	
	cd FokkerPlanck/src; $(MAKE) clean	
	cd Schurtest/src; $(MAKE) clean

cleanall:
	cd src; $(MAKE) cleanall
	cd ErrorBound/src; $(MAKE) cleanall
	cd Plasticity/src; $(MAKE) cleanall
	cd Stokes/src; $(MAKE) cleanall
	cd Cosserat/src; $(MAKE) cleanall
	cd Materials/src; $(MAKE) cleanall
	cd Fruitfly/src; $(MAKE) cleanall
	cd Meshes/src; $(MAKE) cleanall
	cd Porous/src; $(MAKE) cleanall
	cd Wave/src; $(MAKE) cleanall
	cd BEM1.0a/src; $(MAKE) cleanall
	cd BEM1.0b/src; $(MAKE) cleanall
	cd BEM1.0c/src; $(MAKE) cleanall
	cd BEM1.0d/src; $(MAKE) cleanall
	cd BEM1.0e/src; $(MAKE) cleanall
	cd Laplace/src; $(MAKE) cleanall
	cd Poisson/src; $(MAKE) cleanall		
	cd PoissonDolfin/src; $(MAKE) cleanall		
	cd EMTH/src; $(MAKE) cleanall		
	cd TransientEddyCurrent/src; $(MAKE) cleanall		
	cd Maxwell/src; $(MAKE) cleanall		
	cd MaxwellComplex/src; $(MAKE) cleanall		
	cd MagnetoStatics/src; $(MAKE) cleanall			
	cd ElectroMagnetismTH/src; $(MAKE) cleanall			
	cd ConvectionDiffusion/src; $(MAKE) cleanall
	cd ConvectionDiffusionReaction/src; $(MAKE) cleanall
	cd Heat/src; $(MAKE) cleanall	
	cd TimeConvectionDiffusion/src; $(MAKE) cleanall	
	cd FokkerPlanck/src; $(MAKE) cleanall	
	cd Schurtest/src; $(MAKE) cleanall

distclean:
	\rm -f *~ */*~ */*/*~
	cd src; $(MAKE) distclean
	cd ErrorBound/src; $(MAKE) distclean
	cd Plasticity/src; $(MAKE) distclean
	cd Stokes/src; $(MAKE) distclean
	cd Cosserat/src; $(MAKE) distclean
	cd Materials/src; $(MAKE) distclean
	cd Fruitfly/src; $(MAKE) distclean
	cd Meshes/src; $(MAKE) distclean
	cd Porous/src; $(MAKE) distclean
	cd Wave/src; $(MAKE) distclean
	cd BEM1.0a/src; $(MAKE) distclean
	cd BEM1.0b/src; $(MAKE) distclean
	cd BEM1.0c/src; $(MAKE) distclean
	cd BEM1.0d/src; $(MAKE) distclean
	cd BEM1.0e/src; $(MAKE) distclean
	cd Laplace/src; $(MAKE) distclean	
	cd Poisson/src; $(MAKE) distclean	
	cd PoissonDolfin/src; $(MAKE) distclean	
	cd EMTH/src; $(MAKE) distclean	
	cd TransientEddyCurrent/src; $(MAKE) distclean	
	cd Maxwell/src; $(MAKE) distclean	
	cd MaxwellComplex/src; $(MAKE) distclean	
	cd MagnetoStatics/src; $(MAKE) distclean	
	cd ElectroMagnetismTH/src; $(MAKE) distclean	
	cd ConvectionDiffusion/src; $(MAKE) distclean
	cd ConvectionDiffusionReaction/src; $(MAKE) distclean
	cd Heat/src; $(MAKE) distclean
	cd TimeConvectionDiffusion/src; $(MAKE) distclean
	cd FokkerPlanck/src; $(MAKE) distclean
	cd Schurtest/src; $(MAKE) distclean

dxlink: 
	cd tools/dx; $(MAKE)

dxnew: 
	rm -f dxlink;
	$(MAKE) dxlink;

cleandx:
	cd tools/dx; $(MAKE) clean; cd ../..;

new:
	rm -f M++
	cd src;	$(MAKE) new; cd ..;

x_dx:
	\rm -f data/dx/*

x_gp:
	\rm -f data/gp/*

x_tp:
	\rm -f data/tp/*

x_vtk:
	\rm -f data/vtk/*

xdata: 
	$(MAKE) -s -i x_dx; $(MAKE) -s -i x_gp; $(MAKE) -s -i x_tp; $(MAKE) -s -i x_vtk;

xlog:
	\rm -f log/log*

ylog:
	\rm -f log/*

xxx:
	$(MAKE) -s xlog; $(MAKE) -s -i xdata;

yyy:
	$(MAKE) -s ylog; $(MAKE) -s -i xdata;
