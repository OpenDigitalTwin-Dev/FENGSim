# AM

## Installation and Complications

To install FENGSim on Ubuntu 24.04:

```
	sudo apt install git
	git clone https://github.com/OpenDigitalTwin-Dev/FENGSim.git
	cd FENGSim/cli
	./install.sh or ./uninstall.sh
```

To install AM in the FENGSim/starter directory on Ubuntu 24.04:

```
	cd FENGSim/starter
	git clone https://github.com/OpenDigitalTwin-Dev/AM.git
	./install
	mkdir build
	cd build
	cmake ..
	make -j4
```

## References

In the AM/docs/refs/solid_mechanics directory, there are eight books focused on solid mechanics. Lunzhi Xu's book serves as a great starting point. The work by ROBERT J. ASARO and VLADO A. LUBARDA provides a broad introduction. Meanwhile, the book by EA de Souza Neto, D Peric, and DRJ Owen covers finite element methods in solid mechanics. 

In the AM/docs/refs/solid_mechanics directory, you can find several papers authored by C. Wieners that discuss the M++ finite element software as well as the multigrid approach for elasticity and elastoplasticity.
