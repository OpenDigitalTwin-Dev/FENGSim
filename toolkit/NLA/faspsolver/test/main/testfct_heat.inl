/*! \file  testfct_heat.inl
 *
 *  \brief Test functions for the Heat equation
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */
 
#include <math.h>

#define pi 3.1415926535897932
#define DIRICHLET 1
#define INTERIROI 0

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn double f (double *p)
 *
 * \brief Load f of 2D heat transfer's equation: du/dt - u_xx - u_yy = f
 *
 * \param *p   the x-y-t-axis value of the point
 * \return    function value
 *
 * \note Right hand side when the true solution u is sin(pi*x)*sin(pi*y)*t
 *
 * \author Feiteng Huang
 * \date   03/30/2012
 */
double f (double *p)
{
    return 2*pi*pi*sin(pi*p[0])*sin(pi*p[1])*p[2] + sin(pi*p[0])*sin(pi*p[1]);
}

/**
 * \fn double u (double *p)
 *
 * \brief Exact solution u
 *
 * \param *p   the x-y-t-axis value of the point
 * \return    function value 
 *
 * \author Feiteng Huang
 * \date   03/30/2012
 */
double u (double *p)
{
    return sin(pi*p[0])*sin(pi*p[1])*p[2];
}

/**
 * \fn double u_x (double *p)
 *
 * \brief X-directional partial derivative of u
 *
 * \param *p   the x-y-t-axis value of the point
 * \return    x-directional partial derivative of true solution u 
 *
 * \author Feiteng Huang
 * \date   03/30/2012
 */
double u_x (double *p)
{
    return pi*cos(pi*p[0])*sin(pi*p[1])*p[2];
}

/**
 * \fn double u_y(double *p)
 *
 * \brief Y-directional partial derivative of u
 *
 * \param *p   the x-y-t-axis value of the point
 * \return    y-directional partial derivative of true solution u
 *
 * \author Feiteng Huang
 * \date   03/30/2012
 */
double u_y (double *p)
{
    return pi*sin(pi*p[0])*cos(pi*p[1])*p[2];
}

/**
 * \fn int bd_flag (double *p)
 *
 * \brief boundary flag for node
 *
 * \param *p  the x-y-t-axis value of the point
 * \return    function value 
 *
 * \author Feiteng Huang
 * \date   04/05/2012
 */
int bd_flag (double *p)
{
    // set all boundary to Dirichlet boundary
    if (p[0] < 1e-15 || 1-p[0] < 1e-15 || p[1] < 1e-15 || 1-p[1] < 1e-15) 
        return DIRICHLET;
    else
        return INTERIORI;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
