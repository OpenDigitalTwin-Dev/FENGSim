/*! \file  FemBasis.inl
 *
 *  \brief Finite Element Basis Functions
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

/** 
* \fn static double basisP1 (int index, double lambda[2])
*
* \brief Basis functions of P1 finite element
*
* \param index         Index of the basis function
* \param lambda        Reference coordinates
*
* \return phi
*
* \author Feiteng Huang
* \date   04/01/2012
*/
static double basisP1 (int index, 
                       double lambda[2])
{
    double phi;

    if (index == 2)
        phi = 1 - lambda[0] - lambda[1];
    else
        phi = lambda[index];

    return phi;
}

/** 
* \fn static void gradBasisP1 (double nodes[3][2], double s, int index, double phi[2])
*
* \brief Gradient of basis functions of P1 finite element
*
* \param nodes         Vertices of the triangle
* \param s             Area of the triangle
* \param index         Index of the basis function
* \param phi           Gradient of basis function (OUT)
*
* \author Xuehai Huang
* \date   03/29/2009
* 
* Modified by Feiteng Huang 04/01/2012, also change the name of function
*/
static void gradBasisP1 (double nodes[3][2], 
                         double s, 
                         int index, 
                         double phi[2])
{
    const int node1 = (index+1)%3, node2 = (index+2)%3;
    phi[0]=(nodes[node1][1]-nodes[node2][1])/(2.0*s);
    phi[1]=(nodes[node2][0]-nodes[node1][0])/(2.0*s);
}

/**
* \fn static double areaT (double x1,double x2,double x3,
*                          double y1,double y2,double y3)
*
* \brief Get area for triangle p1(x1,y1),p2(x2,y2),p3(x3,y3)
*
* \param x1    x-axis value of the point p1
* \param x2    x-axis value of the point p2
* \param x3    x-axis value of the point p3
* \param y1    y-axis value of the point p1
* \param y2    y-axis value of the point p2
* \param y3    y-axis value of the point p3
*
* \return      Area of the triangle
*
* \author Xuehai Huang
* \date   03/29/2009
*/
static double areaT (double x1,
                     double x2,
                     double x3,
                     double y1,
                     double y2,
                     double y3)
{
    return ((x2-x1)*(y3-y1)-(y2-y1)*(x3-x1))/2;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
