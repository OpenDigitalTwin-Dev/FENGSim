!> \file  testf.f90
!>
!> \brief Test function for F90 interfaces
!>
!>---------------------------------------------------------------------------------
!>  Copyright (C) 2009--Present by the FASP team. All rights reserved.
!>  Released under the terms of the GNU Lesser General Public License 3.0 or later.
!>---------------------------------------------------------------------------------

program test

  implicit none

  double precision, dimension(:), allocatable :: u,b
  double precision, dimension(:), allocatable :: a
  integer,          dimension(:), allocatable :: ia,ja

  double precision :: tol
  integer          :: iufile, n, nnz, i, prt_lvl, maxit, stype

  ! Step 0: user defined variables
  stype = 2  
  prt_lvl = 3
  maxit = 100
  tol = 1.0d-8
  iufile = 1

  ! Step 1: read A and b 

  !===> Read data A from file

  open(unit=iufile,file='../data/csrmat_FE.dat')

  read(iufile,*) n

  allocate(ia(1:n+1))   
  read(iufile,*) (ia(i),i=1,n+1)   
  nnz=ia(n+1)-ia(1)

  allocate(ja(1:nnz),a(1:nnz))
  read(iufile,*) (ja(i),i=1,nnz)
  read(iufile,*) (a(i),i=1,nnz)   

  close(iufile)

  !===> Read data b from file

  open(unit=iufile,file='../data/rhs_FE.dat')

  read(iufile,*) n
  allocate(b(1:n))
  read(iufile,*) (b(i),i=1,n)
  close(iufile)

  !===> Shift the index to start from 0 (for C routines)
  forall (i=1:n+1) ia(i)=ia(i)-1
  forall (i=1:nnz) ja(i)=ja(i)-1

  ! Step 2: Solve the system 

  !===> Initial guess
  allocate(u(1:n))
  u=0.0d0 

  if (stype == 1) then 
     call fasp_fwrapper_dcsr_amg(n,nnz,ia,ja,a,b,u,tol,maxit,prt_lvl);
  else if (stype == 2) then 
     call fasp_fwrapper_dcsr_krylov_amg(n,nnz,ia,ja,a,b,u,tol,maxit,prt_lvl);
  else
     write(*,*) '### ERROR: Unknown solver type!'
  end if

  ! Step 3: Clean up memory
  deallocate(ia,ja,a)
  deallocate(b,u)

end program test
