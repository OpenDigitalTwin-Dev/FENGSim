/*! \file  AuxConvert.c
 *
 *  \brief Utilities for encoding format conversion
 *
 *  \note  This file contains Level-0 (Aux) functions.
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn unsigned long fasp_aux_change_endian4 (const unsigned long x)
 *
 * \brief Swap order for different endian systems
 *
 * \param x   An unsigned long integer
 *
 * \return    Unsigend long ineger after swapping
 *
 * \author Chensong Zhang
 * \date   11/16/2009
 */
unsigned long fasp_aux_change_endian4 (const unsigned long x)
{
    unsigned char *ptr = (unsigned char *)&x;
    return (ptr[0] << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3];
}

/**
 * \fn double fasp_aux_change_endian8 (const double x)
 *
 * \brief Swap order for different endian systems
 *
 * \param x   A unsigned long integer
 *
 * \return    Unsigend long ineger after swapping
 *
 * \author Chensong Zhang
 * \date   11/16/2009
 */
double fasp_aux_change_endian8 (const double x)
{
    double dbl;
    unsigned char *bytes, *buffer;
    
    buffer=(unsigned char *)&dbl;
    bytes=(unsigned char *)&x;
    
    buffer[0]=bytes[7];
    buffer[1]=bytes[6];
    buffer[2]=bytes[5];
    buffer[3]=bytes[4];
    buffer[4]=bytes[3];
    buffer[5]=bytes[2];
    buffer[6]=bytes[1];
    buffer[7]=bytes[0];
    return dbl;
}

/**
 * \fn double fasp_aux_bbyteToldouble (const unsigned char bytes[])
 *
 * \brief Swap order of double-precision float for different endian systems
 *
 * \param bytes  A unsigned char
 *
 * \return       Unsigend long ineger after swapping
 *
 * \author Chensong Zhang
 * \date   11/16/2009
 */
double fasp_aux_bbyteToldouble (const unsigned char bytes[])
{
    double dbl;
    unsigned char *buffer;
    buffer=(unsigned char *)&dbl;
    buffer[0]=bytes[7];
    buffer[1]=bytes[6];
    buffer[2]=bytes[5];
    buffer[3]=bytes[4];
    buffer[4]=bytes[3];
    buffer[5]=bytes[2];
    buffer[6]=bytes[1];
    buffer[7]=bytes[0];
    return dbl;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
