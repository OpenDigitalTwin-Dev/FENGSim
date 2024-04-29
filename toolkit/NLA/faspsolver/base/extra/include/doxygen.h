/*! \file doxygen.h
 *  \brief Main page for Doygen documentation
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2010--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Less Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */
 
/** \mainpage Introduction
 *
 * Over the last few decades, researchers have expended significant effort on
 * developing efficient iterative methods for solving discretized partial differential
 * equations (PDEs). Though these efforts have yielded many mathematically optimal
 * solvers such as the multigrid method, the unfortunate reality is that multigrid
 * methods have not been much used in practical applications. This marked gap between
 * theory and practice is mainly due to the fragility of traditional multigrid (MG)
 * methodology and the complexity of its implementation. We aim to develop techniques
 * and the corresponding software that will narrow this gap, specifically by developing
 * mathematically optimal solvers that are robust and easy to use in practice.
 *
 * We believe that there is no one-size-for-all solution method for discrete linear
 * systemsfrom different applications. And, efficient iterative solvers can be
 * constructed by taking the properties of PDEs and discretizations into account. In
 * this project, we plan to construct a pool of discrete problems arising from partial
 * differential equations (PDEs) or PDE systems and efficient linear solvers for these
 * problems. We mainly utilize the methodology of Auxiliary Space Preconditioning (ASP)
 * to construct efficient linear solvers. Due to this reason, this software package
 * is called Fast Auxiliary Space Preconditioning or FASP for short.
 * 
 * The levels of abstraction are designed as follows:
 *
 * - Level 0 (Aux*.c): Auxiliary functions (timing, memory, threading, ...)
 *
 * - Level 1 (Bla*.c): Basic linear algebra subroutines (SpMV, RAP, ILU, SWZ, ...)
 *
 * - Level 2 (Itr*.c): Iterative methods and smoothers (Jacobi, GS, SOR, Poly, ...)
 *
 * - Level 3 (Kry*.c): Krylov iterative methods (CG, BiCGstab, MinRes, GMRES, ...)
 *
 * - Level 4 (Pre*.c): Preconditioners (GMG, AMG, FAMG, ...)
 *
 * - Level 5 (Sol*.c): User interface for FASP solvers (Solvers, wrappers, ...)
 *
 * - Level x (Xtr*.c): Interface to external packages (Mumps, Umfpack, ...)
 *
 * FASP contains the kernel part and several applications (ranging from fluid dynamics  
 * to reservoir simulation). The kernel part is open-source and licensed under GNU 
 * Lesser General Public License or LGPL version 3.0 or later. Some of the applications 
 * contain contributions from and owned partially by other parties.
 *
 * > For the moment, FASP is under alpha testing. If you wish to obtain a current version
 * > of FASP or you have any questions, feel free to contact us at faspdev@gmail.com.
 * 
 * This software distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
 * See the GNU Lesser General Public License for more details.
 *
 */
 
/**
 * \page download How to obtain FASP
 *
 * The most updated version of FASP can be downloaded from
 *
 * > http://www.multigrid.org/fasp/download/faspsolver.zip
 *
 * We use Git as our main version control tool. Git is easy to use and it is available
 * at all OS platforms. For people who is interested in the developer version, you can 
 * obtain the FASP package with Git: 
 * 
 * > $ git clone git@github.com:FaspDevTeam/faspsolver.git
 * 
 * will give you the developer version of the FASP package. 
 *
 */

/**
 * \page build Building and Installation
 *
 * This is a simple instruction on building and testing. For more details, please refer to 
 * the README files and the short
 * <a href="http://www.multigrid.org/fasp/download/userguide.pdf">User's Guide</a> in 
 * "faspsolver/doc/".
 *
 * To compile, you need a Fortran and a C compiler.  First, you can type in the "faspsolver/"
 * root directory:
 *
 * > $ mkdir Build; cd Build; cmake ..
 * 
 * which will config the environment automatically. And, then, you can need to type:
 *
 * > $ make install
 *
 * which will make the FASP shared static library and install to PREFIX/. By default, FASP 
 * libraries and executables will be installed in the FASP home directory "faspsolver/".
 *
 * There is a simple GUI tool for building and installing FASP included in the package. You 
 * need Tcl/Tk support in your computer. You may call this GUI by run in the root directory:
 *
 * > $ wish fasp_install.tcl
 *
 * If you need to see the detailed usage of "make" or need any help, please type:
 *
 * > $ make help
 *
 * After installation, tutorial examples can be found in "tutorial/".
 *
 */ 


/**
 * \page developers Developers
 *
 * Project leader:
 *
 * - Xu, Jinchao (Penn State University, USA)
 *
 * Project coordinator:
 *
 * - Zhang, Chensong (Chinese Academy of Sciences, China)
 *
 * Current active developers (in alphabetic order):
 *
 * - Feng, Chunsheng (Xiangtan University, China)
 *
 * - Zhang, Chensong (Chinese Academy of Sciences, China)
 *
 * With contributions from (in alphabetic order):
 *
 * - Brannick, James (Penn State University, USA)
 *
 * - Chen, Long (University of California, Irvine, USA)
 *
 * - Hu, Xiaozhe (Tufts University, USA)
 *
 * - Huang, Feiteng (Sichuan University, China)
 *
 * - Huang, Xuehai (Shanghai Jiaotong University, China)
 * 
 * - Li, Zheng (Xiangtan University, China)
 *
 * - Qiao, Changhe (Penn State University, USA)
 *
 * - Shu, Shi (Xiangtan University, China)
 *
 * - Sun, Pengtao (University of Nevada, Las Vegas, USA)
 *
 * - Yang, Kai (Penn State University, USA)
 *
 * - Yue, Xiaoqiang (Xiangtan University, China)
 *
 * - Wang, Lu (LLNL, USA)
 *
 * - Wang, Ziteng (University of Alabama, USA)
 *
 * - Zhang, Shiquan (Sichuan University, China)
 *
 * - Zhang, Shuo (Chinese Academy of Sciences, China)
 *
 * - Zhang, Hongxuan (Penn State Univeristy, USA)
 *
 * - Zhang, Weifeng (Kunming University of Science and Technology, China)
 *
 * - Zhou, Zhiyang (Xiangtan University, China)
 *
 */
   
/**
 * \page doxygen_comment Doxygen
 *
 * We use Doxygen as our automatically documentation generator which will make our 
 * future maintainance minimized. You can obtain the software (Windows, Linux and 
 * OS X) as well as its manual on the official website 
 * 
 * http://www.doxygen.org
 *
 * For an ordinary user, Doxygen is completely trivial to use. We only need to use 
 * some special marker in the usual comment as we put in c-files. 
 *
 */
 
/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
 
