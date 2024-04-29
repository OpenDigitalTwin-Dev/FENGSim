# Fast Auxiliary Space Preconditioning (FASP) Solver Library: README

## Introduction
The FASP package is designed for developing and testing new efficient solvers 
and preconditioners for discrete partial differential equations (PDEs) or 
systems of PDEs. The main components of the package are standard Krylov methods, 
algebraic multigrid methods, and incomplete factorization methods. Based on 
these standard techniques, we build efficient solvers, based on the framework 
of Auxiliary Space Preconditioning, for several complicated applications. 
Current examples include the fluid dynamics, underground water simulation, 
the black oil model in reservoir simulation, and so on. 

## Install
To compile, you need a C99 compiler (and a F90 compiler if you need Fortran 
examples). By default, we use GNU gcc/gfortan, respectively.

Configuring and building the FASP library and test suite requires CMake 2.8 or
higher <http://www.cmake.org/>.

The command to configure is:

``` bash
    > mkdir Build; cd Build; cmake .. 
```

After successfully configing the environment, just run:

``` bash
    > make  # to compile the FASP lib only; do not install
```

To install the FASP library and executables, run:

``` bash
    > make install  # to compile and install the FASP lib
```

Note: The default prefix is the FASP source directory.

## Compatibility
This package has been tested with on the following platforms: 

- Linux: gcc/gfortran, icc/ifort
- Windows XP, 7, 10: icc/ifort
- Mac with Intel CPU: gcc/gfortran, icc/ifort, clang
- Mac with Apple Silicon: gcc/gfortran, clang

## License
This software is free software distributed under the Lesser General Public 
License or LGPL, version 3.0 or any later versions. This software distributed 
in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even 
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License 
along with FASP. If not, see <http://www.gnu.org/licenses/>.
