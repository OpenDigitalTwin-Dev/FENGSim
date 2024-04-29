/*! \file  testfct_poisson.inl
 *
 *  \brief Test functions for the Poisson's equation
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */
 
#include <math.h>

#define DIRICHLET 1
#define INTERIROI 0

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn double f (double *p)
 *
 * \brief Load f of 2D Poisson's equation: - u_xx - u_yy = f
 *
 * \param *p  the x-y-axis value of the point
 * \return    function value
 *
 * \note Right hand side when the true solution u is -cos(x)*sin(y)
 *
 * \author Xuehai Huang
 * \date   03/29/2009
 */
double f (double *p)
{
    return -2.0*cos(p[0])*sin(p[1]);
}

/**
 * \fn double u (double *p)
 *
 * \brief Exact solution u
 *
 * \param *p  the x-y-axis value of the point
 * \return    function value 
 *
 * \author Xuehai Huang
 * \date   03/29/2009
 */
double u (double *p)
{
    return -cos(p[0])*sin(p[1]);
}

/**
 * \fn double u_x (double *p)
 *
 * \brief X-directional partial derivative of u
 *
 * \param *p  the x-y-axis value of the point
 * \return    x-directional partial derivative of true solution u 
 *
 * \author Xuehai Huang
 * \date   03/29/2009
 */
double u_x (double *p)
{
    return sin(p[0])*sin(p[1]);
}

/**
 * \fn double u_y(double *p)
 *
 * \brief Y-directional partial derivative of u
 *
 * \param *p  the x-y-axis value of the point
 * \return    y-directional partial derivative of true solution u 
 *
 * \author Xuehai Huang
 * \date   03/29/2009
 */
double u_y (double *p)
{
    return -cos(p[0])*cos(p[1]);
}

/**
 * \fn int bd_flag (double *p)
 *
 * \brief boundary flag for node
 *
 * \param *p  the x-y-axis value of the point
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
