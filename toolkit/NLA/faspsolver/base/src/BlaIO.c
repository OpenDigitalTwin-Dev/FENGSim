/*! \file  BlaIO.c
 *
 *  \brief Matrix/vector input/output subroutines
 *
 *  \note  Read, write or print a matrix or a vector in various formats
 *
 *  \note  This file contains Level-1 (Bla) functions. It requires:
 *         AuxArray.c, AuxConvert.c, AuxMemory.c, AuxMessage.c, AuxVector.c,
 *         BlaFormat.c, BlaSparseBSR.c, BlaSparseCOO.c, BlaSparseCSR.c,
 *         and BlaSpmvCSR.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include "fasp.h"
#include "fasp_functs.h"
#include "hb_io.h"

// Flags which indicates lengths of INT and REAL numbers
int ilength; /**< Length of INT in byte */
int dlength; /**< Length of REAL in byte */

/*---------------------------------*/
/*--  Declare Private Functions  --*/
/*---------------------------------*/

#include "BlaIOUtil.inl"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_dcsrvec_read1 (const char *filename, dCSRmat *A, dvector *b)
 *
 * \brief Read A and b from a SINGLE disk file
 *
 * \param filename  File name
 * \param A         Pointer to the CSR matrix
 * \param b         Pointer to the dvector
 *
 * \note
 *      This routine reads a dCSRmat matrix and a dvector vector from a single
 *      disk file. The difference between this and fasp_dcoovec_read is that this
 *      routine support non-square matrices.
 *
 * \note File format:
 *   - nrow ncol         % number of rows and number of columns
 *   - ia(j), j=0:nrow   % row index
 *   - ja(j), j=0:nnz-1  % column index
 *   - a(j), j=0:nnz-1   % entry value
 *   - n                 % number of entries
 *   - b(j), j=0:n-1     % entry value
 *
 * \author Xuehai Huang
 * \date   03/29/2009
 *
 * Modified by Chensong Zhang on 03/14/2012
 */
void fasp_dcsrvec_read1(const char* filename, dCSRmat* A, dvector* b)
{
    int  i, m, n, idata;
    REAL ddata;

    // Open input disk file
    FILE* fp = fopen(filename, "r");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: reading file %s ...\n", __FUNCTION__, filename);

    skip_comments(fp); // skip the comments in the beginning --zcs 06/30/2020

    // Read CSR matrix
    if (fscanf(fp, "%d %d", &m, &n) > 0) {
        A->row = m;
        A->col = n;
    } else {
        fasp_chkerr(ERROR_WRONG_FILE, filename);
    }

    A->IA = (INT*)fasp_mem_calloc(m + 1, sizeof(INT));
    for (i = 0; i <= m; ++i) {
        if (fscanf(fp, "%d", &idata) > 0)
            A->IA[i] = idata;
        else {
            fasp_chkerr(ERROR_WRONG_FILE, filename);
        }
    }

    INT nnz = A->IA[m] - A->IA[0];

    A->nnz = nnz;
    A->JA  = (INT*)fasp_mem_calloc(nnz, sizeof(INT));
    A->val = (REAL*)fasp_mem_calloc(nnz, sizeof(REAL));

    for (i = 0; i < nnz; ++i) {
        if (fscanf(fp, "%d", &idata) > 0)
            A->JA[i] = idata;
        else {
            fasp_chkerr(ERROR_WRONG_FILE, filename);
        }
    }

    for (i = 0; i < nnz; ++i) {
        if (fscanf(fp, "%lf", &ddata) > 0)
            A->val[i] = ddata;
        else {
            fasp_chkerr(ERROR_WRONG_FILE, filename);
        }
    }

    // Read RHS vector
    if (fscanf(fp, "%d", &m) > 0) b->row = m;

    b->val = (REAL*)fasp_mem_calloc(m, sizeof(REAL));

    for (i = 0; i < m; ++i) {
        if (fscanf(fp, "%lf", &ddata) > 0)
            b->val[i] = ddata;
        else {
            fasp_chkerr(ERROR_WRONG_FILE, filename);
        }
    }

    fclose(fp);
}

/**
 * \fn void fasp_dcsrvec_read2 (const char *filemat, const char *filerhs,
 *                              dCSRmat *A, dvector *b)
 *
 * \brief Read A and b from two separate disk files
 *
 * \param filemat  File name for matrix
 * \param filerhs  File name for right-hand side
 * \param A        Pointer to the dCSR matrix
 * \param b        Pointer to the dvector
 *
 * \note  This routine reads a dCSRmat matrix and a dvector vector from a disk file.
 *
 * \note
 * CSR matrix file format:
 *   - nrow              % number of columns (rows)
 *   - ia(j), j=0:nrow   % row index
 *   - ja(j), j=0:nnz-1  % column index
 *   - a(j), j=0:nnz-1   % entry value
 *
 * \note
 * RHS file format:
 *   - n                 % number of entries
 *   - b(j), j=0:nrow-1  % entry value
 *
 * \note Indices start from 1, NOT 0!!!
 *
 * \author Zhiyang Zhou
 * \date   2010/08/06
 *
 * Modified by Chensong Zhang on 2012/01/05
 */
void fasp_dcsrvec_read2(const char* filemat, const char* filerhs, dCSRmat* A,
                        dvector* b)
{
    int i, n, tempi;

    /* read the matrix from file */
    FILE* fp = fopen(filemat, "r");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filemat);

    printf("%s: reading file %s ...\n", __FUNCTION__, filemat);

    skip_comments(fp); // skip the comments in the beginning --zcs 06/30/2020

    if (fscanf(fp, "%d\n", &n) > 0) {
        A->row = n;
        A->col = n;
        A->IA  = (INT*)fasp_mem_calloc(n + 1, sizeof(INT));
    } else {
        fasp_chkerr(ERROR_WRONG_FILE, filemat);
    }

    for (i = 0; i <= n; ++i) {
        if (fscanf(fp, "%d\n", &tempi) > 0)
            A->IA[i] = tempi - 1;
        else {
            fasp_chkerr(ERROR_WRONG_FILE, filemat);
        }
    }

    INT nz = A->IA[n];
    A->nnz = nz;
    A->JA  = (INT*)fasp_mem_calloc(nz, sizeof(INT));
    A->val = (REAL*)fasp_mem_calloc(nz, sizeof(REAL));

    for (i = 0; i < nz; ++i) {
        if (fscanf(fp, "%d\n", &tempi) > 0)
            A->JA[i] = tempi - 1;
        else {
            fasp_chkerr(ERROR_WRONG_FILE, filemat);
        }
    }

    for (i = 0; i < nz; ++i) {
        if (fscanf(fp, "%le\n", &(A->val[i])) <= 0) {
            fasp_chkerr(ERROR_WRONG_FILE, filemat);
        }
    }

    fclose(fp);

    /* Read the rhs from file */
    b->row = n;
    b->val = (REAL*)fasp_mem_calloc(n, sizeof(REAL));

    fp = fopen(filerhs, "r");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filerhs);

    printf("%s: reading file %s ...\n", __FUNCTION__, filerhs);

    if (fscanf(fp, "%d\n", &n) < 0) fasp_chkerr(ERROR_WRONG_FILE, filerhs);

    if (n != b->row) {
        printf("### WARNING: rhs size = %d, matrix size = %d!\n", n, b->row);
        fasp_chkerr(ERROR_MAT_SIZE, filemat);
    }

    for (i = 0; i < n; ++i) {
        if (fscanf(fp, "%le\n", &(b->val[i])) <= 0) {
            fasp_chkerr(ERROR_WRONG_FILE, filerhs);
        }
    }

    fclose(fp);
}

/**
 * \fn void fasp_dcsr_read (const char *filename, dCSRmat *A)
 *
 * \brief Read A from matrix disk file in IJ format
 *
 * \param filename  Char for matrix file name
 * \param A         Pointer to the CSR matrix
 *
 * \author Ziteng Wang
 * \date   12/25/2012
 */
void fasp_dcsr_read(const char* filename, dCSRmat* A)
{
    int  i, m, idata;
    REAL ddata;

    // Open input disk file
    FILE* fp = fopen(filename, "r");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: reading file %s ...\n", __FUNCTION__, filename);

    skip_comments(fp); // skip the comments in the beginning --zcs 06/30/2020

    // Read CSR matrix
    if (fscanf(fp, "%d", &m) > 0)
        A->row = A->col = m;
    else {
        fasp_chkerr(ERROR_WRONG_FILE, filename);
    }

    A->IA = (INT*)fasp_mem_calloc(m + 1, sizeof(INT));
    for (i = 0; i <= m; ++i) {
        if (fscanf(fp, "%d", &idata) > 0)
            A->IA[i] = idata;
        else {
            fasp_chkerr(ERROR_WRONG_FILE, filename);
        }
    }

    // If IA starts from 1, shift by -1
    if (A->IA[0] == 1)
        for (i = 0; i <= m; ++i) A->IA[i]--;

    INT nnz = A->IA[m] - A->IA[0];

    A->nnz = nnz;
    A->JA  = (INT*)fasp_mem_calloc(nnz, sizeof(INT));
    A->val = (REAL*)fasp_mem_calloc(nnz, sizeof(REAL));

    for (i = 0; i < nnz; ++i) {
        if (fscanf(fp, "%d", &idata) > 0)
            A->JA[i] = idata;
        else {
            fasp_chkerr(ERROR_WRONG_FILE, filename);
        }
    }

    // If JA starts from 1, shift by -1
    if (A->JA[0] == 1)
        for (i = 0; i < nnz; ++i) A->JA[i]--;

    for (i = 0; i < nnz; ++i) {
        if (fscanf(fp, "%lf", &ddata) > 0)
            A->val[i] = ddata;
        else {
            fasp_chkerr(ERROR_WRONG_FILE, filename);
        }
    }

    fclose(fp);
}

/**
 * \fn void fasp_dcoo_read (const char *filename, dCSRmat *A)
 *
 * \brief Read A from matrix disk file in IJ format -- indices starting from 0
 *
 * \param filename  File name for matrix
 * \param A         Pointer to the CSR matrix
 *
 * \note File format:
 *   - nrow ncol nnz     % number of rows, number of columns, and nnz
 *   - i  j  a_ij        % i, j a_ij in each line
 *
 * \note After reading, it converts the matrix to dCSRmat format.
 *
 * \author Xuehai Huang, Chensong Zhang
 * \date   03/29/2009
 */
void fasp_dcoo_read(const char* filename, dCSRmat* A)
{
    int  i, j, k, m, n, nnz;
    REAL value;

    FILE* fp = fopen(filename, "r");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: reading file %s ...\n", __FUNCTION__, filename);

    skip_comments(fp); // skip the comments in the beginning --zcs 06/30/2020

    if (fscanf(fp, "%d %d %d", &m, &n, &nnz) <= 0) {
        fasp_chkerr(ERROR_WRONG_FILE, filename);
    }

    dCOOmat Atmp = fasp_dcoo_create(m, n, nnz);

    for (k = 0; k < nnz; k++) {
        if (fscanf(fp, "%d %d %le", &i, &j, &value) != EOF) {
            Atmp.rowind[k] = i;
            Atmp.colind[k] = j;
            Atmp.val[k]    = value;
        } else {
            fasp_chkerr(ERROR_WRONG_FILE, filename);
        }
    }

    fclose(fp);

    fasp_format_dcoo_dcsr(&Atmp, A);
    fasp_dcoo_free(&Atmp);
}

/**
 * \fn void fasp_dcoo_read1 (const char *filename, dCSRmat *A)
 *
 * \brief Read A from matrix disk file in IJ format -- indices starting from 1
 *
 * \param filename  File name for matrix
 * \param A         Pointer to the CSR matrix
 *
 * \note File format:
 *   - nrow ncol nnz     % number of rows, number of columns, and nnz
 *   - i  j  a_ij        % i, j a_ij in each line
 *
 * \author Xiaozhe Hu, Chensong Zhang
 * \date   03/24/2013
 *
 * Modified by Chensong Zhang on 01/12/2019: Convert COO to CSR
 */
void fasp_dcoo_read1(const char* filename, dCSRmat* A)
{
    int  i, j, k, m, n, nnz;
    REAL value;

    FILE* fp = fopen(filename, "r");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: reading file %s ...\n", __FUNCTION__, filename);

    skip_comments(fp); // skip the comments in the beginning --zcs 06/30/2020

    if (fscanf(fp, "%d %d %d", &m, &n, &nnz) <= 0) {
        fasp_chkerr(ERROR_WRONG_FILE, filename);
    }

    dCOOmat Atmp = fasp_dcoo_create(m, n, nnz);

    for (k = 0; k < nnz; k++) {
        if (fscanf(fp, "%d %d %le", &i, &j, &value) != EOF) {
            Atmp.rowind[k] = i - 1;
            Atmp.colind[k] = j - 1;
            Atmp.val[k]    = value;
        } else {
            fasp_chkerr(ERROR_WRONG_FILE, filename);
        }
    }

    fclose(fp);

    fasp_format_dcoo_dcsr(&Atmp, A);
    fasp_dcoo_free(&Atmp);
}

/**
 * \fn void fasp_dcoovec_bin_read (const char *fni, const char *fnj, const char *fna,
 *                                 const char *fnb, dCSRmat *A, dvector *b)
 *
 * \brief Read A from matrix disk files in IJ format (three binary files)
 *
 * \param fni       File name for matrix i-index
 * \param fnj       File name for matrix j-index
 * \param fna       File name for matrix values
 * \param fnb       File name for vector values
 * \param A         Pointer to the CSR matrix
 * \param b         Pointer to the vector
 *
 * \note After reading, it converts the matrix to dCSRmat format.
 *
 * \author Chensong Zhang
 * \date   08/27/2022
 */
void fasp_dcoovec_bin_read(const char* fni, const char* fnj, const char* fna,
                           const char* fnb, dCSRmat* A, dvector* b)
{
    size_t n, type, nnz, i;
    FILE*  fp;

    fp = fopen(fnb, "rb");
    if (fp == NULL) {
        fasp_chkerr(ERROR_WRONG_FILE, fnb);
    }
    printf("%s: reading file %s ...\n", __FUNCTION__, fnb);
    fread(&n, sizeof(size_t), 1, fp);
    b->row = n;
    b->val = (double*)fasp_mem_calloc(n, sizeof(double));
    fread(b->val, sizeof(double), n, fp);
    fclose(fp);

    fp = fopen(fni, "rb");
    if (fp == NULL) {
        fasp_chkerr(ERROR_WRONG_FILE, fni);
    }
    printf("%s: reading file %s ...\n", __FUNCTION__, fni);
    fread(&type, sizeof(size_t), 1, fp);
    fread(&nnz, sizeof(size_t), 1, fp);
    dCOOmat Atmp = fasp_dcoo_create(n, n, nnz);
    Atmp.rowind  = (int*)fasp_mem_calloc(nnz, sizeof(int));
    fread(Atmp.rowind, sizeof(int), nnz, fp);
    for (i = 0; i < nnz; i++) Atmp.rowind[i] = Atmp.rowind[i] - 1;
    fclose(fp);

    fp = fopen(fnj, "rb");
    if (fp == NULL) {
        fasp_chkerr(ERROR_WRONG_FILE, fnj);
    }
    printf("%s: reading file %s ...\n", __FUNCTION__, fnj);
    fread(&type, sizeof(size_t), 1, fp);
    fread(&nnz, sizeof(size_t), 1, fp);
    Atmp.colind = (int*)fasp_mem_calloc(nnz, sizeof(int));
    fread(Atmp.colind, sizeof(int), nnz, fp);
    for (i = 0; i < nnz; i++) Atmp.colind[i] = Atmp.colind[i] - 1;
    fclose(fp);

    fp = fopen(fna, "rb");
    if (fp == NULL) {
        fasp_chkerr(ERROR_WRONG_FILE, fna);
    }
    printf("%s: reading file %s ...\n", __FUNCTION__, fna);
    fread(&type, sizeof(size_t), 1, fp);
    fread(&nnz, sizeof(size_t), 1, fp);
    Atmp.val = (double*)fasp_mem_calloc(nnz, sizeof(double));
    fread(Atmp.val, sizeof(double), nnz, fp);
    fclose(fp);

    fasp_format_dcoo_dcsr(&Atmp, A);
    fasp_dcoo_free(&Atmp);
}

/**
 * \fn void fasp_dcoo_shift_read (const char *filename, dCSRmat *A)
 *
 * \brief Read A from matrix disk file in IJ format -- indices starting from 0
 *
 * \param filename  File name for matrix
 * \param A         Pointer to the CSR matrix
 *
 * \note File format:
 *   - nrow ncol nnz     % number of rows, number of columns, and nnz
 *   - i  j  a_ij        % i, j a_ij in each line
 *
 * \note i and j suppose to start with index 1!!!
 *
 * \note After read in, it shifts the index to C fashion and converts the matrix
 *       to dCSRmat format.
 *
 * \author Xiaozhe Hu
 * \date   04/01/2014
 */
void fasp_dcoo_shift_read(const char* filename, dCSRmat* A)
{
    int  i, j, k, m, n, nnz;
    REAL value;

    FILE* fp = fopen(filename, "r");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: reading file %s ...\n", __FUNCTION__, filename);

    skip_comments(fp); // skip the comments in the beginning --zcs 06/30/2020

    if (fscanf(fp, "%d %d %d", &m, &n, &nnz) <= 0) {
        fasp_chkerr(ERROR_WRONG_FILE, filename);
    }

    dCOOmat Atmp = fasp_dcoo_create(m, n, nnz);

    for (k = 0; k < nnz; k++) {
        if (fscanf(fp, "%d %d %le", &i, &j, &value) != EOF) {
            Atmp.rowind[k] = i - 1;
            Atmp.colind[k] = j - 1;
            Atmp.val[k]    = value;
        } else {
            fasp_chkerr(ERROR_WRONG_FILE, filename);
        }
    }

    fclose(fp);

    fasp_format_dcoo_dcsr(&Atmp, A);
    fasp_dcoo_free(&Atmp);
}

/**
 * \fn void fasp_dmtx_read (const char *filename, dCSRmat *A)
 *
 * \brief Read A from matrix disk file in MatrixMarket general format
 *
 * \param filename  File name for matrix
 * \param A         Pointer to the CSR matrix
 *
 * \note File format:
 *   This routine reads a MatrixMarket general matrix from a mtx file.
 *   And it converts the matrix to dCSRmat format. For details of mtx format,
 *   please refer to http://math.nist.gov/MatrixMarket/.
 *
 * \note Indices start from 1, NOT 0!!!
 *
 * \author Chensong Zhang
 * \date   09/05/2011
 */
void fasp_dmtx_read(const char* filename, dCSRmat* A)
{
    int  i, j, m, n, nnz;
    INT  innz; // index of nonzeros
    REAL value;

    FILE* fp = fopen(filename, "r");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: reading file %s ...\n", __FUNCTION__, filename);

    skip_comments(fp); // skip the comments in the beginning --zcs 06/30/2020

    if (fscanf(fp, "%d %d %d", &m, &n, &nnz) <= 0) {
        fasp_chkerr(ERROR_WRONG_FILE, filename);
    }

    dCOOmat Atmp = fasp_dcoo_create(m, n, nnz);

    innz = 0;

    while (innz < nnz) {
        if (fscanf(fp, "%d %d %le", &i, &j, &value) != EOF) {
            Atmp.rowind[innz] = i - 1;
            Atmp.colind[innz] = j - 1;
            Atmp.val[innz]    = value;
            innz              = innz + 1;
        } else {
            fasp_chkerr(ERROR_WRONG_FILE, filename);
        }
    }

    fclose(fp);

    fasp_format_dcoo_dcsr(&Atmp, A);
    fasp_dcoo_free(&Atmp);
}

/**
 * \fn void fasp_dmtxsym_read (const char *filename, dCSRmat *A)
 *
 * \brief Read A from matrix disk file in MatrixMarket sym format
 *
 * \param filename  File name for matrix
 * \param A         Pointer to the CSR matrix
 *
 * \note File format:
 *   This routine reads a MatrixMarket symmetric matrix from a mtx file.
 *   And it converts the matrix to dCSRmat format. For details of mtx format,
 *   please refer to http://math.nist.gov/MatrixMarket/.
 *
 * \note Indices start from 1, NOT 0!!!
 *
 * \author Chensong Zhang
 * \date   09/02/2011
 */
void fasp_dmtxsym_read(const char* filename, dCSRmat* A)
{
    int  i, j, m, n, nnz;
    int  innz; // index of nonzeros
    REAL value;

    FILE* fp = fopen(filename, "r");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: reading file %s ...\n", __FUNCTION__, filename);

    skip_comments(fp); // skip the comments in the beginning --zcs 06/30/2020

    if (fscanf(fp, "%d %d %d", &m, &n, &nnz) <= 0) {
        fasp_chkerr(ERROR_WRONG_FILE, filename);
    }

    nnz          = 2 * (nnz - m) + m; // adjust for sym problem
    dCOOmat Atmp = fasp_dcoo_create(m, n, nnz);

    innz = 0;

    while (innz < nnz) {
        if (fscanf(fp, "%d %d %le", &i, &j, &value) != EOF) {

            if (i == j) {
                Atmp.rowind[innz] = i - 1;
                Atmp.colind[innz] = j - 1;
                Atmp.val[innz]    = value;
                innz              = innz + 1;
            } else {
                Atmp.rowind[innz]     = i - 1;
                Atmp.rowind[innz + 1] = j - 1;
                Atmp.colind[innz]     = j - 1;
                Atmp.colind[innz + 1] = i - 1;
                Atmp.val[innz]        = value;
                Atmp.val[innz + 1]    = value;
                innz                  = innz + 2;
            }
        } else {
            fasp_chkerr(ERROR_WRONG_FILE, filename);
        }
    }

    fclose(fp);

    fasp_format_dcoo_dcsr(&Atmp, A);
    fasp_dcoo_free(&Atmp);
}

/**
 * \fn void fasp_dstr_read (const char *filename, dSTRmat *A)
 *
 * \brief Read A from a disk file in dSTRmat format
 *
 * \param filename  File name for the matrix
 * \param A         Pointer to the dSTRmat
 *
 * \note
 *      This routine reads a dSTRmat matrix from a disk file. After done, it converts
 *      the matrix to dCSRmat format.
 *
 * \note File format:
 *   - nx, ny, nz
 *   - nc: number of components
 *   - nband: number of bands
 *   - n: size of diagonal, you must have diagonal
 *   - diag(j), j=0:n-1
 *   - offset, length: offset and length of off-diag1
 *   - offdiag(j), j=0:length-1
 *
 * \author Xuehai Huang
 * \date   03/29/2009
 */
void fasp_dstr_read(const char* filename, dSTRmat* A)
{
    int  nx, ny, nz, nxy, ngrid, nband, nc, offset;
    int  i, k, n;
    REAL value;

    FILE* fp = fopen(filename, "r");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: reading file %s ...\n", __FUNCTION__, filename);

    skip_comments(fp); // skip the comments in the beginning --zcs 06/30/2020

    // read dimension of the problem
    if (fscanf(fp, "%d %d %d", &nx, &ny, &nz) > 0) {
        A->nx = nx;
        A->ny = ny;
        A->nz = nz;
    } else {
        fasp_chkerr(ERROR_WRONG_FILE, filename);
    }

    nxy      = nx * ny;
    ngrid    = nxy * nz;
    A->nxy   = nxy;
    A->ngrid = ngrid;

    // read number of components
    if (fscanf(fp, "%d", &nc) > 0)
        A->nc = nc;
    else {
        fasp_chkerr(ERROR_WRONG_FILE, filename);
    }

    // read number of bands
    if (fscanf(fp, "%d", &nband) > 0)
        A->nband = nband;
    else {
        fasp_chkerr(ERROR_WRONG_FILE, filename);
    }

    A->offsets = (INT*)fasp_mem_calloc(nband, sizeof(INT));

    // read diagonal
    if (fscanf(fp, "%d", &n) > 0) {
        A->diag = (REAL*)fasp_mem_calloc(n, sizeof(REAL));
    } else {
        fasp_chkerr(ERROR_WRONG_FILE, filename);
    }

    for (i = 0; i < n; ++i) {
        if (fscanf(fp, "%le", &value) > 0)
            A->diag[i] = value;
        else {
            fasp_chkerr(ERROR_WRONG_FILE, filename);
        }
    }

    // read offdiags
    k          = nband;
    A->offdiag = (REAL**)fasp_mem_calloc(nband, sizeof(REAL*));
    while (k--) {
        // read number band k
        if (fscanf(fp, "%d %d", &offset, &n) > 0) {
            A->offsets[nband - k - 1] = offset;
        } else {
            fasp_chkerr(ERROR_WRONG_FILE, filename);
        }

        A->offdiag[nband - k - 1] = (REAL*)fasp_mem_calloc(n, sizeof(REAL));
        for (i = 0; i < n; ++i) {
            if (fscanf(fp, "%le", &value) > 0) {
                A->offdiag[nband - k - 1][i] = value;
            } else {
                fasp_chkerr(ERROR_WRONG_FILE, filename);
            }
        }
    }

    fclose(fp);
}

/**
 * \fn void fasp_dbsr_read (const char *filename, dBSRmat *A)
 *
 * \brief Read A from a disk file in dBSRmat format
 *
 * \param filename   File name for matrix A
 * \param A          Pointer to the dBSRmat A
 *
 * \note
 *   This routine reads a dBSRmat matrix from a disk file in the following format:
 *
 * \note File format:
 *   - ROW, COL, NNZ
 *   - nb: size of each block
 *   - storage_manner: storage manner of each block
 *   - ROW+1: length of IA
 *   - IA(i), i=0:ROW
 *   - NNZ: length of JA
 *   - JA(i), i=0:NNZ-1
 *   - NNZ*nb*nb: length of val
 *   - val(i), i=0:NNZ*nb*nb-1
 *
 * \author Xiaozhe Hu
 * \date   10/29/2010
 */
void fasp_dbsr_read(const char* filename, dBSRmat* A)
{
    int    ROW, COL, NNZ, nb, storage_manner;
    int    i, n;
    int    index;
    REAL   value;
    size_t status;

    FILE* fp = fopen(filename, "r");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: reading file %s ...\n", __FUNCTION__, filename);

    skip_comments(fp); // skip the comments in the beginning --zcs 06/30/2020

    status = fscanf(fp, "%d %d %d", &ROW, &COL, &NNZ); // dimensions of the problem
    fasp_chkerr(status, filename);
    A->ROW = ROW;
    A->COL = COL;
    A->NNZ = NNZ;

    status = fscanf(fp, "%d", &nb); // read the size of each block
    fasp_chkerr(status, filename);
    A->nb = nb;

    status = fscanf(fp, "%d", &storage_manner); // read the storage_manner
    fasp_chkerr(status, filename);
    A->storage_manner = storage_manner;

    // allocate memory space
    fasp_dbsr_alloc(ROW, COL, NNZ, nb, storage_manner, A);

    // read IA
    status = fscanf(fp, "%d", &n);
    fasp_chkerr(status, filename);
    for (i = 0; i < n; ++i) {
        status = fscanf(fp, "%d", &index);
        fasp_chkerr(status, filename);
        A->IA[i] = index;
    }

    // read JA
    status = fscanf(fp, "%d", &n);
    fasp_chkerr(status, filename);
    for (i = 0; i < n; ++i) {
        status = fscanf(fp, "%d", &index);
        fasp_chkerr(status, filename);
        A->JA[i] = index;
    }

    // read val
    status = fscanf(fp, "%d", &n);
    fasp_chkerr(status, filename);
    for (i = 0; i < n; ++i) {
        status = fscanf(fp, "%le", &value);
        fasp_chkerr(status, filename);
        A->val[i] = value;
    }

    fclose(fp);
}

/**
 * \fn void fasp_dvecind_read (const char *filename, dvector *b)
 *
 * \brief Read b from matrix disk file
 *
 * \param filename  File name for vector b
 * \param b         Pointer to the dvector b (output)
 *
 * \note File Format:
 *     - nrow
 *     - ind_j, val_j, j=0:nrow-1
 *
 * \note Because the index is given, order is not important!
 *
 * \author Chensong Zhang
 * \date   03/29/2009
 */
void fasp_dvecind_read(const char* filename, dvector* b)
{
    int    i, n, index;
    REAL   value;
    size_t status;

    printf("%s: reading file %s ...\n", __FUNCTION__, filename);

    FILE* fp = fopen(filename, "r");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    skip_comments(fp); // skip the comments in the beginning --zcs 06/30/2020

    status = fscanf(fp, "%d", &n);
    fasp_dvec_alloc(n, b);

    for (i = 0; i < n; ++i) {

        status = fscanf(fp, "%d %le", &index, &value);

        if (value > BIGREAL || index >= n) {
            fasp_dvec_free(b);
            fclose(fp);

            printf("### ERROR: Wrong index = %d or value = %lf\n", index, value);
            fasp_chkerr(ERROR_INPUT_PAR, __FUNCTION__);
        }

        b->val[index] = value;
    }

    fclose(fp);
    fasp_chkerr(status, filename);
}

/**
 * \fn void fasp_dvec_read (const char *filename, dvector *b)
 *
 * \brief Read b from a disk file in array format
 *
 * \param filename  File name for vector b
 * \param b         Pointer to the dvector b (output)
 *
 * \note File Format:
 *   - nrow
 *   - val_j, j=0:nrow-1
 *
 * \author Chensong Zhang
 * \date   03/29/2009
 */
void fasp_dvec_read(const char* filename, dvector* b)
{
    int    i, n;
    REAL   value;
    size_t status;

    FILE* fp = fopen(filename, "r");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: reading file %s ...\n", __FUNCTION__, filename);

    skip_comments(fp); // skip the comments in the beginning --zcs 06/30/2020

    status = fscanf(fp, "%d", &n);

    fasp_dvec_alloc(n, b);

    for (i = 0; i < n; ++i) {

        status    = fscanf(fp, "%le", &value);
        b->val[i] = value;

        if (value > BIGREAL) {
            fasp_dvec_free(b);
            fclose(fp);

            printf("### ERROR: Wrong value = %lf!\n", value);
            fasp_chkerr(ERROR_INPUT_PAR, __FUNCTION__);
        }
    }

    fclose(fp);
    fasp_chkerr(status, filename);
}

/**
 * \fn void fasp_ivecind_read (const char *filename, ivector *b)
 *
 * \brief Read b from matrix disk file
 *
 * \param filename  File name for vector b
 * \param b         Pointer to the dvector b (output)
 *
 * \note File Format:
 *   - nrow
 *   - ind_j, val_j ... j=0:nrow-1
 *
 * \author Chensong Zhang
 * \date   03/29/2009
 */
void fasp_ivecind_read(const char* filename, ivector* b)
{
    int    i, n, index, value;
    size_t status;

    FILE* fp = fopen(filename, "r");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: reading file %s ...\n", __FUNCTION__, filename);

    skip_comments(fp); // skip the comments in the beginning --zcs 06/30/2020

    status = fscanf(fp, "%d", &n);
    fasp_ivec_alloc(n, b);

    for (i = 0; i < n; ++i) {
        status        = fscanf(fp, "%d %d", &index, &value);
        b->val[index] = value;
    }

    fclose(fp);
    fasp_chkerr(status, filename);
}

/**
 * \fn void fasp_ivec_read (const char *filename, ivector *b)
 *
 * \brief Read b from a disk file in array format
 *
 * \param filename  File name for vector b
 * \param b         Pointer to the dvector b (output)
 *
 * \note File Format:
 *   - nrow
 *   - val_j, j=0:nrow-1
 *
 * \author Xuehai Huang
 * \date   03/29/2009
 */
void fasp_ivec_read(const char* filename, ivector* b)
{
    int    i, n, value;
    size_t status;

    FILE* fp = fopen(filename, "r");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: reading file %s ...\n", __FUNCTION__, filename);

    skip_comments(fp); // skip the comments in the beginning --zcs 06/30/2020

    status = fscanf(fp, "%d", &n);
    fasp_ivec_alloc(n, b);

    for (i = 0; i < n; ++i) {
        status    = fscanf(fp, "%d", &value);
        b->val[i] = value;
    }

    fclose(fp);
    fasp_chkerr(status, filename);
}

/**
 * \fn void fasp_dcsrvec_write1 (const char *filename, dCSRmat *A, dvector *b)
 *
 * \brief Write A and b to a SINGLE disk file
 *
 * \param filename  File name
 * \param A         Pointer to the CSR matrix
 * \param b         Pointer to the dvector
 *
 * \note
 *      This routine writes a dCSRmat matrix and a dvector vector to a single disk file.
 *
 * \note File format:
 *   - nrow ncol         % number of rows and number of columns
 *   - ia(j), j=0:nrow   % row index
 *   - ja(j), j=0:nnz-1  % column index
 *   - a(j), j=0:nnz-1   % entry value
 *   - n                 % number of entries
 *   - b(j), j=0:n-1     % entry value
 *
 * \author Feiteng Huang
 * \date   05/19/2012
 *
 * Modified by Chensong on 12/26/2012
 */
void fasp_dcsrvec_write1(const char* filename, dCSRmat* A, dvector* b)
{
    INT m = A->row, n = A->col, nnz = A->nnz;
    INT i;

    FILE* fp = fopen(filename, "w");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    /* write the matrix to file */
    printf("%s: reading file %s ...\n", __FUNCTION__, filename);

    fprintf(fp, "%d %d\n", m, n);
    for (i = 0; i < m + 1; ++i) {
        fprintf(fp, "%d\n", A->IA[i]);
    }
    for (i = 0; i < nnz; ++i) {
        fprintf(fp, "%d\n", A->JA[i]);
    }
    for (i = 0; i < nnz; ++i) {
        fprintf(fp, "%le\n", A->val[i]);
    }

    m = b->row;

    /* write the rhs to file */
    printf("%s: writing to file %s ...\n", __FUNCTION__, filename);

    fprintf(fp, "%d\n", m);

    for (i = 0; i < m; ++i) fprintf(fp, "%le\n", b->val[i]);

    fclose(fp);
}

/**
 * \fn void fasp_dcsrvec_write2 (const char *filemat, const char *filerhs,
 *                               dCSRmat *A, dvector *b)
 *
 * \brief Write A and b to two separate disk files
 *
 * \param filemat  File name for matrix
 * \param filerhs  File name for right-hand side
 * \param A        Pointer to the dCSR matrix
 * \param b        Pointer to the dvector
 *
 * \note
 *      This routine writes a dCSRmat matrix and a dvector vector to two disk files.
 *
 * \note
 * CSR matrix file format:
 *   - nrow              % number of columns (rows)
 *   - ia(j), j=0:nrow   % row index
 *   - ja(j), j=0:nnz-1  % column index
 *   - a(j),  j=0:nnz-1  % entry value
 *
 * \note
 * RHS file format:
 *   - n                 % number of entries
 *   - b(j), j=0:nrow-1  % entry value
 *
 * \note Indices start from 1, NOT 0!!!
 *
 * \author Feiteng Huang
 * \date   05/19/2012
 */
void fasp_dcsrvec_write2(const char* filemat, const char* filerhs, dCSRmat* A,
                         dvector* b)
{
    INT m = A->row, nnz = A->nnz;
    INT i;

    FILE* fp = fopen(filemat, "w");

    /* write the matrix to file */
    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filemat);

    printf("%s: writing to file %s ...\n", __FUNCTION__, filemat);

    fprintf(fp, "%d\n", m);
    for (i = 0; i < m + 1; ++i) {
        fprintf(fp, "%d\n", A->IA[i] + 1);
    }
    for (i = 0; i < nnz; ++i) {
        fprintf(fp, "%d\n", A->JA[i] + 1);
    }
    for (i = 0; i < nnz; ++i) {
        fprintf(fp, "%le\n", A->val[i]);
    }

    fclose(fp);

    m = b->row;

    fp = fopen(filerhs, "w");

    /* write the rhs to file */
    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filerhs);

    printf("%s: writing to file %s ...\n", __FUNCTION__, filerhs);

    fprintf(fp, "%d\n", m);

    for (i = 0; i < m; ++i) fprintf(fp, "%le\n", b->val[i]);

    fclose(fp);
}

/**
 * \fn void fasp_dcoo_write (const char *filename, dCSRmat *A)
 *
 * \brief Write a matrix to disk file in IJ format (coordinate format)
 *
 * \param A         pointer to the dCSRmat matrix
 * \param filename  char for vector file name
 *
 * \note
 *      The routine writes the specified REAL vector in COO format.
 *      Refer to the reading subroutine \ref fasp_dcoo_read.
 *
 * \note File format:
 *   - The first line of the file gives the number of rows, the
 *   number of columns, and the number of nonzeros.
 *   - Then gives nonzero values in i j a(i,j) format.
 *
 * \author Chensong Zhang
 * \date   03/29/2009
 */
void fasp_dcoo_write(const char* filename, dCSRmat* A)
{
    const INT m = A->row, n = A->col;
    INT       i, j;

    FILE* fp = fopen(filename, "w");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: writing to file %s ...\n", __FUNCTION__, filename);

    fprintf(fp, "%d  %d  %d\n", m, n, A->nnz);
    for (i = 0; i < m; ++i) {
        for (j = A->IA[i]; j < A->IA[i + 1]; j++)
            fprintf(fp, "%d  %d  %0.15e\n", i, A->JA[j], A->val[j]);
    }

    fclose(fp);
}

/**
 * \fn void fasp_dstr_write (const char *filename, dSTRmat *A)
 *
 * \brief Write a dSTRmat to a disk file
 *
 * \param filename  File name for A
 * \param A         Pointer to the dSTRmat matrix A
 *
 * \note  The routine writes the specified REAL vector in STR format.
 *        Refer to the reading subroutine \ref fasp_dstr_read.
 *
 * \author Shiquan Zhang
 * \date   03/29/2010
 */
void fasp_dstr_write(const char* filename, dSTRmat* A)
{
    const INT nx = A->nx, ny = A->ny, nz = A->nz;
    const INT ngrid = A->ngrid, nband = A->nband, nc = A->nc;

    INT* offsets = A->offsets;

    INT i, k, n;

    FILE* fp = fopen(filename, "w");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: writing to file %s ...\n", __FUNCTION__, filename);

    fprintf(fp, "%d  %d  %d\n", nx, ny, nz); // write dimension of the problem

    fprintf(fp, "%d\n", nc); // read number of components

    fprintf(fp, "%d\n", nband); // write number of bands

    // write diagonal
    n = ngrid * nc * nc;    // number of nonzeros in each band
    fprintf(fp, "%d\n", n); // number of diagonal entries
    for (i = 0; i < n; ++i) fprintf(fp, "%le\n", A->diag[i]);

    // write offdiags
    k = nband;
    while (k--) {
        INT offset = offsets[nband - k - 1];
        n          = (ngrid - ABS(offset)) * nc * nc; // number of nonzeros in each band
        fprintf(fp, "%d  %d\n", offset, n);           // read number band k
        for (i = 0; i < n; ++i) {
            fprintf(fp, "%le\n", A->offdiag[nband - k - 1][i]);
        }
    }

    fclose(fp);
}

/**
 * \fn void fasp_dbsr_print (const char *filename, dBSRmat *A)
 *
 * \brief Print a dBSRmat to a disk file in a readable format
 *
 * \param filename  File name for A
 * \param A         Pointer to the dBSRmat matrix A
 *
 * \author Chensong Zhang
 * \date   01/07/2021
 */
void fasp_dbsr_print(const char* filename, dBSRmat* A)
{
    const INT ROW = A->ROW;
    const INT nb  = A->nb;
    const INT nb2 = nb * nb;

    INT*  ia  = A->IA;
    INT*  ja  = A->JA;
    REAL* val = A->val;

    INT i, j, k, ind;

    FILE* fp = fopen(filename, "w");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: printing to file %s ...\n", __FUNCTION__, filename);

    for (i = 0; i < ROW; i++) {
        for (k = ia[i]; k < ia[i + 1]; k++) {
            j = ja[k];
            fprintf(fp, "A[%d,%d]=\n", i, j);
            for (ind = 0; ind < nb2; ind++) {
                fprintf(fp, "%+.15E  ", val[k * nb2 + ind]);
            }
            fprintf(fp, "\n");
        }
    }
}

/**
 * \fn void fasp_dbsr_write (const char *filename, dBSRmat *A)
 *
 * \brief Write a dBSRmat to a disk file
 *
 * \param filename  File name for A
 * \param A         Pointer to the dBSRmat matrix A
 *
 * \note  The routine writes the specified REAL vector in BSR format.
 *        Refer to the reading subroutine \ref fasp_dbsr_read.
 *
 * \author Shiquan Zhang
 * \date   10/29/2010
 */
void fasp_dbsr_write(const char* filename, dBSRmat* A)
{
    const INT ROW = A->ROW, COL = A->COL, NNZ = A->NNZ;
    const INT nb = A->nb, storage_manner = A->storage_manner;

    INT*  ia  = A->IA;
    INT*  ja  = A->JA;
    REAL* val = A->val;

    INT i, n;

    FILE* fp = fopen(filename, "w");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: writing to file %s ...\n", __FUNCTION__, filename);

    fprintf(fp, "%d  %d  %d\n", ROW, COL, NNZ); // write dimension of the block matrix

    fprintf(fp, "%d\n", nb); // write the size of each block

    fprintf(fp, "%d\n", storage_manner); // write storage manner of each block

    // write A->IA
    n = ROW + 1;            // length of A->IA
    fprintf(fp, "%d\n", n); // length of A->IA
    for (i = 0; i < n; ++i) fprintf(fp, "%d\n", ia[i]);

    // write A->JA
    n = NNZ;                // length of A->JA
    fprintf(fp, "%d\n", n); // length of A->JA
    for (i = 0; i < n; ++i) fprintf(fp, "%d\n", ja[i]);

    // write A->val
    n = NNZ * nb * nb;      // length of A->val
    fprintf(fp, "%d\n", n); // length of A->val
    for (i = 0; i < n; ++i) fprintf(fp, "%le\n", val[i]);

    fclose(fp);
}

/**
 * \fn void fasp_dvec_write (const char *filename, dvector *vec)
 *
 * \brief Write a dvector to disk file
 *
 * \param vec       Pointer to the dvector
 * \param filename  File name
 *
 * \author Xuehai Huang
 * \date   03/29/2009
 */
void fasp_dvec_write(const char* filename, dvector* vec)
{
    INT m = vec->row, i;

    FILE* fp = fopen(filename, "w");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: writing to file %s ...\n", __FUNCTION__, filename);

    fprintf(fp, "%d\n", m);

    for (i = 0; i < m; ++i) fprintf(fp, "%0.15e\n", vec->val[i]);

    fclose(fp);
}

/**
 * \fn void fasp_dvecind_write (const char *filename, dvector *vec)
 *
 * \brief Write a dvector to disk file in coordinate format
 *
 * \param vec       Pointer to the dvector
 * \param filename  File name
 *
 * \note The routine writes the specified REAL vector in IJ format.
 *   - The first line of the file is the length of the vector;
 *   - After that, each line gives index and value of the entries.
 *
 * \author Xuehai Huang
 * \date   03/29/2009
 */
void fasp_dvecind_write(const char* filename, dvector* vec)
{
    INT m = vec->row, i;

    FILE* fp = fopen(filename, "w");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: writing to file %s ...\n", __FUNCTION__, filename);

    fprintf(fp, "%d\n", m);

    for (i = 0; i < m; ++i) fprintf(fp, "%d %le\n", i, vec->val[i]);

    fclose(fp);
}

/**
 * \fn void fasp_ivec_write (const char *filename, ivector *vec)
 *
 * \brief Write a ivector to disk file in coordinate format
 *
 * \param vec       Pointer to the dvector
 * \param filename  File name
 *
 * \note The routine writes the specified INT vector in IJ format.
 *   - The first line of the file is the length of the vector;
 *   - After that, each line gives index and value of the entries.
 *
 * \author Xuehai Huang
 * \date   03/29/2009
 */
void fasp_ivec_write(const char* filename, ivector* vec)
{
    INT m = vec->row, i;

    FILE* fp = fopen(filename, "w");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: writing to file %s ...\n", __FUNCTION__, filename);

    // write number of nonzeros
    fprintf(fp, "%d\n", m);

    // write index and value each line
    for (i = 0; i < m; ++i) fprintf(fp, "%d %d\n", i, vec->val[i] + 1);

    fclose(fp);
}

/**
 * \fn void fasp_dvec_print (const INT n, dvector *u)
 *
 * \brief Print first n entries of a vector of REAL type
 *
 * \param n   An interger (if n=0, then print all entries)
 * \param u   Pointer to a dvector
 *
 * \author Chensong Zhang
 * \date   03/29/2009
 */
void fasp_dvec_print(const INT n, dvector* u)
{
    INT i;
    INT NumPrint = n;

    if (n <= 0) NumPrint = u->row; // print all

    for (i = 0; i < NumPrint; ++i) printf("vec_%d = %15.10E\n", i, u->val[i]);
}

/**
 * \fn void fasp_ivec_print (const INT n, ivector *u)
 *
 * \brief Print first n entries of a vector of INT type
 *
 * \param n   An interger (if n=0, then print all entries)
 * \param u   Pointer to an ivector
 *
 * \author Chensong Zhang
 * \date   03/29/2009
 */
void fasp_ivec_print(const INT n, ivector* u)
{
    INT i;
    INT NumPrint = n;

    if (n <= 0) NumPrint = u->row; // print all

    for (i = 0; i < NumPrint; ++i) printf("vec_%d = %d\n", i, u->val[i]);
}

/**
 * \fn void fasp_dcsr_print (const dCSRmat *A)
 *
 * \brief Print out a dCSRmat matrix in coordinate format
 *
 * \param A   Pointer to the dCSRmat matrix A
 *
 * \author Xuehai Huang
 * \date   03/29/2009
 */
void fasp_dcsr_print(const dCSRmat* A)
{
    const INT m = A->row, n = A->col;
    INT       i, j;

    printf("nrow = %d, ncol = %d, nnz = %d\n", m, n, A->nnz);
    for (i = 0; i < m; ++i) {
        for (j = A->IA[i]; j < A->IA[i + 1]; j++)
            printf("A_(%d,%d) = %+.15E\n", i, A->JA[j], A->val[j]);
    }
}

/**
 * \fn void fasp_dcoo_print (const dCOOmat *A)
 *
 * \brief Print out a dCOOmat matrix in coordinate format
 *
 * \param A   Pointer to the dCOOmat matrix A
 *
 * \author Ziteng Wang
 * \date   12/24/2012
 */
void fasp_dcoo_print(const dCOOmat* A)
{
    INT k;

    printf("nrow = %d, ncol = %d, nnz = %d\n", A->row, A->col, A->nnz);
    for (k = 0; k < A->nnz; k++) {
        printf("A_(%d,%d) = %+.15E\n", A->rowind[k], A->colind[k], A->val[k]);
    }
}

/**
 * \fn void fasp_dbsr_write_coo (const char *filename, const dBSRmat *A)
 *
 * \brief Print out a dBSRmat matrix in coordinate format for matlab spy
 *
 * \param filename   Name of file to write to
 * \param A          Pointer to the dBSRmat matrix A
 *
 * \author Chunsheng Feng
 * \date   11/14/2013
 *
 * Modified by Chensong Zhang on 06/14/2014: Fix index problem.
 */
void fasp_dbsr_write_coo(const char* filename, const dBSRmat* A)
{

    INT i, j, k, l;
    INT nb, nb2;
    nb  = A->nb;
    nb2 = nb * nb;

    FILE* fp = fopen(filename, "w");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

#if DEBUG_MODE > PRINT_MIN
    printf("### DEBUG: nrow = %d, ncol = %d, nnz = %d, nb = %d\n", A->ROW, A->COL,
           A->NNZ, A->nb);
    printf("### DEBUG: storage_manner = %d\n", A->storage_manner);
#endif

    printf("%s: writing to file %s ...\n", __FUNCTION__, filename);

    // write dimension of the block matrix
    fprintf(fp, "%% dimension of the block matrix and nonzeros %d  %d  %d\n", A->ROW,
            A->COL, A->NNZ);
    // write the size of each block
    fprintf(fp, "%% the size of each block %d\n", A->nb);
    // write storage manner of each block
    fprintf(fp, "%% storage manner of each block %d\n", A->storage_manner);

    for (i = 0; i < A->ROW; i++) {
        for (j = A->IA[i]; j < A->IA[i + 1]; j++) {
            for (k = 0; k < A->nb; k++) {
                for (l = 0; l < A->nb; l++) {
                    fprintf(fp, "%d %d %+.15E\n", i * nb + k + 1, A->JA[j] * nb + l + 1,
                            A->val[j * nb2 + k * nb + l]);
                }
            }
        }
    }

    fclose(fp);
}

/**
 * \fn void fasp_dcsr_write_coo (const char *filename, const dCSRmat *A)
 *
 * \brief Print out a dCSRmat matrix in coordinate format for matlab spy
 *
 * \param filename   Name of file to write to
 * \param A          Pointer to the dCSRmat matrix A
 *
 * \author Chunsheng Feng
 * \date   11/14/2013
 *
 * \note Output indices start from 1 instead of 0!
 */
void fasp_dcsr_write_coo(const char* filename, const dCSRmat* A)
{

    INT i, j;

#if DEBUG_MODE > PRINT_MIN
    printf("nrow = %d, ncol = %d, nnz = %d\n", A->row, A->col, A->nnz);
#endif

    FILE* fp = fopen(filename, "w");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: writing to file %s ...\n", __FUNCTION__, filename);

    // write dimension of the matrix
    fprintf(fp, "%% dimension of the matrix and nonzeros %d  %d  %d\n", A->row, A->col,
            A->nnz);

    for (i = 0; i < A->row; i++) {
        for (j = A->IA[i]; j < A->IA[i + 1]; j++) {
            fprintf(fp, "%d %d %+.15E\n", i + 1, A->JA[j] + 1, A->val[j]);
        }
    }

    fclose(fp);
}

/**
 * \fn void fasp_dcsr_write_mtx (const char *filename, const dCSRmat *A)
 *
 * \brief Print out a dCSRmat matrix in coordinate format for MatrixMarket
 *
 * \param filename   Name of file to write to
 * \param A          Pointer to the dCSRmat matrix A
 *
 * \author Chensong Zhang
 * \date   08/28/2022
 *
 * \note Output indices start from 1 instead of 0!
 */
void fasp_dcsr_write_mtx(const char* filename, const dCSRmat* A)
{
    INT i, j;

#if DEBUG_MODE > PRINT_MIN
    printf("nrow = %d, ncol = %d, nnz = %d\n", A->row, A->col, A->nnz);
#endif

    FILE* fp = fopen(filename, "w");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: writing to file %s ...\n", __FUNCTION__, filename);

    // write dimension of the matrix
    fprintf(fp, "%% MatrixMarket matrix coordinate general\n");
    fprintf(fp, "%d  %d  %d\n", A->row, A->col, A->nnz);

    for (i = 0; i < A->row; i++) {
        for (j = A->IA[i]; j < A->IA[i + 1]; j++) {
            fprintf(fp, "%d %d %+.15E\n", i + 1, A->JA[j] + 1, A->val[j]);
        }
    }

    fclose(fp);
}

/**
 * \fn void fasp_dstr_print (const dSTRmat *A)
 *
 * \brief Print out a dSTRmat matrix in coordinate format
 *
 * \param A		Pointer to the dSTRmat matrix A
 *
 * \author Ziteng Wang
 * \date   12/24/2012
 */
void fasp_dstr_print(const dSTRmat* A)
{
    // TODO: To be added later! --Chensong
}

/**
 * \fn fasp_matrix_read (const char *filename, void *A)
 *
 * \brief Read matrix from different kinds of formats from both ASCII and binary files
 *
 * \param filename   File name of matrix file
 * \param A          Pointer to the matrix
 *
 * \note Flags for matrix file format:
 *   - fileflag			 % fileflag = 1: binary, fileflag = 0000: ASCII
 *	 - formatflag		 % a 3-digit number for internal use, see below
 *   - matrix			 % different types of matrix
 *
 * \note Meaning of formatflag:
 *   - matrixflag        % first digit of formatflag
 *		 + matrixflag = 1: CSR format
 *		 + matrixflag = 2: BSR format
 *		 + matrixflag = 3: STR format
 *		 + matrixflag = 4: COO format
 *		 + matrixflag = 5: MTX format
 *		 + matrixflag = 6: MTX symmetrical format
 *	 - ilength			 % third digit of formatflag, length of INT
 *	 - dlength			 % fourth digit of formatflag, length of REAL
 *
 * \author Ziteng Wang
 * \date   12/24/2012
 *
 * Modified by Chensong Zhang on 05/01/2013
 */
void fasp_matrix_read(const char* filename, void* A)
{

    int    index, flag;
    SHORT  EndianFlag;
    size_t status;

    FILE* fp = fopen(filename, "rb");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: reading file %s ...\n", __FUNCTION__, filename);

    status = fread(&index, sizeof(INT), 1, fp);
    fasp_chkerr(status, filename);

    // matrix stored in ASCII format
    if (index == 808464432) {

        fclose(fp);
        fp = fopen(filename, "r"); // reopen file of reading file in ASCII

        status = fscanf(fp, "%d\n", &flag); // jump over the first line
        fasp_chkerr(status, __FUNCTION__);

        status = fscanf(fp, "%d\n", &flag); // reading the format information
        fasp_chkerr(status, __FUNCTION__);

        flag = (INT)flag / 100;

        switch (flag) {
            case 0:
                fasp_dcsr_read_s(fp, (dCSRmat*)A);
                break;
            case 1:
                fasp_dcoo_read_s(fp, (dCSRmat*)A);
                break;
            case 2:
                fasp_dbsr_read_s(fp, (dBSRmat*)A);
                break;
            case 3:
                fasp_dstr_read_s(fp, (dSTRmat*)A);
                break;
            case 4:
                fasp_dcoo_read_s(fp, (dCSRmat*)A);
                break;
            case 5:
                fasp_dmtx_read_s(fp, (dCSRmat*)A);
                break;
            case 6:
                fasp_dmtxsym_read_s(fp, (dCSRmat*)A);
                break;
            default:
                printf("### ERROR: Unknown flag %d in %s!\n", flag, filename);
                fasp_chkerr(ERROR_WRONG_FILE, __FUNCTION__);
        }

        fclose(fp);
        return;
    }

    // matrix stored in binary format

    // test Endian consistence of machine and file
    EndianFlag = index;

    status = fread(&index, sizeof(INT), 1, fp);
    fasp_chkerr(status, filename);

    index   = endian_convert_int(index, sizeof(INT), EndianFlag);
    flag    = (INT)index / 100;
    ilength = (INT)(index - flag * 100) / 10;
    dlength = index % 10;

    switch (flag) {
        case 1:
            fasp_dcsr_read_b(fp, (dCSRmat*)A, EndianFlag);
            break;
        case 2:
            fasp_dbsr_read_b(fp, (dBSRmat*)A, EndianFlag);
            break;
        case 3:
            fasp_dstr_read_b(fp, (dSTRmat*)A, EndianFlag);
            break;
        case 4:
            fasp_dcoo_read_b(fp, (dCSRmat*)A, EndianFlag);
            break;
        case 5:
            fasp_dmtx_read_b(fp, (dCSRmat*)A, EndianFlag);
            break;
        case 6:
            fasp_dmtxsym_read_b(fp, (dCSRmat*)A, EndianFlag);
            break;
        default:
            printf("### ERROR: Unknown flag %d in %s!\n", flag, filename);
            fasp_chkerr(ERROR_WRONG_FILE, __FUNCTION__);
    }

    fclose(fp);
}

/**
 * \fn void fasp_matrix_read_bin (const char *filename, void *A)
 *
 * \brief Read matrix in binary format
 *
 * \param filename   File name of matrix file
 * \param A          Pointer to the matrix
 *
 * \author Xiaozhe Hu
 * \date   04/14/2013
 *
 * Modified by Chensong Zhang on 05/01/2013: Use it to read binary files!!!
 */
void fasp_matrix_read_bin(const char* filename, void* A)
{
    int    index, flag;
    SHORT  EndianFlag = 1;
    size_t status;

    FILE* fp = fopen(filename, "rb");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: reading file %s ...\n", __FUNCTION__, filename);

    status = fread(&index, sizeof(INT), 1, fp);
    fasp_chkerr(status, filename);

    index = endian_convert_int(index, sizeof(INT), EndianFlag);

    flag    = (INT)index / 100;
    ilength = (int)(index - flag * 100) / 10;
    dlength = index % 10;

    switch (flag) {
        case 1:
            fasp_dcoo_read_b(fp, (dCSRmat*)A, EndianFlag);
            break;
        case 2:
            fasp_dbsr_read_b(fp, (dBSRmat*)A, EndianFlag);
            break;
        case 3:
            fasp_dstr_read_b(fp, (dSTRmat*)A, EndianFlag);
            break;
        case 4:
            fasp_dcsr_read_b(fp, (dCSRmat*)A, EndianFlag);
            break;
        case 5:
            fasp_dmtx_read_b(fp, (dCSRmat*)A, EndianFlag);
            break;
        case 6:
            fasp_dmtxsym_read_b(fp, (dCSRmat*)A, EndianFlag);
            break;
        default:
            printf("### ERROR: Unknown flag %d in %s!\n", flag, filename);
            fasp_chkerr(ERROR_WRONG_FILE, __FUNCTION__);
    }

    fclose(fp);
}

/**
 * \fn fasp_matrix_write (const char *filename, void *A, const INT flag)
 *
 * \brief write matrix from different kinds of formats from both ASCII and binary files
 *
 * \param filename   File name of matrix file
 * \param A          Pointer to the matrix
 * \param flag       Type of file and matrix, a 3-digit number
 *
 * \note Meaning of flag:
 *   - fileflag			 % fileflag = 1: binary, fileflag = 0: ASCII
 *	 - matrixflag
 *		+ matrixflag = 1: CSR format
 *		+ matrixflag = 2: BSR format
 *		+ matrixflag = 3: STR format
 *
 * \note Matrix file format:
 *   - fileflag			 % fileflag = 1: binary, fileflag = 0000: ASCII
 *	 - formatflag		 % a 3-digit number
 *   - matrixflag		 % different kinds of matrix judged by formatflag
 *
 * \author Ziteng Wang
 * \date   12/24/2012
 */
void fasp_matrix_write(const char* filename, void* A, const INT flag)
{
    INT   fileflag, matrixflag;
    FILE* fp;

    matrixflag = flag % 100;
    fileflag   = (INT)flag / 100;

    // write matrix in ASCII file
    if (!fileflag) {

        fp = fopen(filename, "w");

        if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

        printf("%s: writing to file %s ...\n", __FUNCTION__, filename);

        fprintf(fp, "%d%d%d%d\n", fileflag, fileflag, fileflag, fileflag);

        fprintf(fp, "%d%d%d\n", matrixflag, (int)sizeof(INT), (int)sizeof(REAL));

        switch (matrixflag) {
            case 1:
                fasp_dcsr_write_s(fp, (dCSRmat*)A);
                break;
            case 2:
                fasp_dbsr_write_s(fp, (dBSRmat*)A);
                break;
            case 3:
                fasp_dstr_write_s(fp, (dSTRmat*)A);
                break;
            default:
                printf("### WARNING: Unknown matrix flag %d\n", matrixflag);
        }
        fclose(fp);
        return;
    }

    // write matrix in binary file
    fp = fopen(filename, "wb");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filename);

    printf("%s: writing to file %s ...\n", __FUNCTION__, filename);

    INT putflag = fileflag * 100 + sizeof(INT) * 10 + sizeof(REAL);
    fwrite(&putflag, sizeof(INT), 1, fp);

    switch (matrixflag) {
        case 1:
            fasp_dcsr_write_b(fp, (dCSRmat*)A);
            break;
        case 2:
            fasp_dbsr_write_b(fp, (dBSRmat*)A);
            break;
        case 3:
            fasp_dstr_write_b(fp, (dSTRmat*)A);
            break;
        default:
            printf("### WARNING: Unknown matrix flag %d\n", matrixflag);
    }

    fclose(fp);
}

/**
 * \fn fasp_vector_read (const char *filerhs, void *b)
 *
 * \brief Read RHS vector from different kinds of formats in ASCII or binary files
 *
 * \param filerhs File name of vector file
 * \param b Pointer to the vector
 *
 * \note Matrix file format:
 *   - fileflag			 % fileflag = 1: binary, fileflag = 0000: ASCII
 *	 - formatflag		 % a 3-digit number
 *   - vector			 % different kinds of vector judged by formatflag
 *
 * \note Meaning of formatflag:
 *   - vectorflag        % first digit of formatflag
 *		 + vectorflag = 1: dvec format
 *		 + vectorflag = 2: ivec format
 *		 + vectorflag = 3: dvecind format
 *		 + vectorflag = 4: ivecind format
 *	 - ilength			 % second digit of formatflag, length of INT
 *	 - dlength			 % third digit of formatflag, length of REAL
 *
 * \author Ziteng Wang
 * \date   12/24/2012
 */
void fasp_vector_read(const char* filerhs, void* b)
{
    int    index, flag;
    SHORT  EndianFlag;
    size_t status;

    FILE* fp = fopen(filerhs, "rb");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filerhs);

    printf("%s: reading file %s ...\n", __FUNCTION__, filerhs);

    status = fread(&index, sizeof(INT), 1, fp);
    fasp_chkerr(status, filerhs);

    // vector stored in ASCII
    if (index == 808464432) {

        fclose(fp);
        fp = fopen(filerhs, "r");

        if (!fscanf(fp, "%d\n", &flag))
            printf("### ERROR: File format problem in %s!\n", __FUNCTION__);
        // TODO: Check why skip this flag ??? --Chensong

        if (!fscanf(fp, "%d\n", &flag))
            printf("### ERROR: File format problem in %s!\n", __FUNCTION__);
        flag = (int)flag / 100;

        switch (flag) {
            case 1:
                fasp_dvec_read_s(fp, (dvector*)b);
                break;
            case 2:
                fasp_ivec_read_s(fp, (ivector*)b);
                break;
            case 3:
                fasp_dvecind_read_s(fp, (dvector*)b);
                break;
            case 4:
                fasp_ivecind_read_s(fp, (ivector*)b);
                break;
        }
        fclose(fp);
        return;
    }

    // vector stored in binary
    EndianFlag = index;
    status     = fread(&index, sizeof(INT), 1, fp);
    fasp_chkerr(status, filerhs);

    index   = endian_convert_int(index, sizeof(INT), EndianFlag);
    flag    = (int)index / 100;
    ilength = (int)(index - 100 * flag) / 10;
    dlength = index % 10;

    switch (flag) {
        case 1:
            fasp_dvec_read_b(fp, (dvector*)b, EndianFlag);
            break;
        case 2:
            fasp_ivec_read_b(fp, (ivector*)b, EndianFlag);
            break;
        case 3:
            fasp_dvecind_read_b(fp, (dvector*)b, EndianFlag);
            break;
        case 4:
            fasp_ivecind_read_b(fp, (ivector*)b, EndianFlag);
            break;
        default:
            printf("### ERROR: Unknown flag %d in %s!\n", flag, filerhs);
            fasp_chkerr(ERROR_WRONG_FILE, __FUNCTION__);
    }

    fclose(fp);
}

/**
 * \fn fasp_vector_write (const char *filerhs, void *b, const INT flag)
 *
 * \brief write RHS vector from different kinds of formats in both ASCII and binary
 *files
 *
 * \param filerhs File name of vector file
 *
 * \param b Pointer to the vector
 *
 * \param flag Type of file and vector, a 2-digit number
 *
 * \note Meaning of the flags
 *   - fileflag			 % fileflag = 1: binary, fileflag = 0: ASCII
 *	 - vectorflag
 *		 + vectorflag = 1: dvec format
 *		 + vectorflag = 2: ivec format
 *		 + vectorflag = 3: dvecind format
 *		 + vectorflag = 4: ivecind format
 *
 * \note Matrix file format:
 *   - fileflag			 % fileflag = 1: binary, fileflag = 0000: ASCII
 *	 - formatflag		 % a 2-digit number
 *   - vectorflag		 % different kinds of vector judged by formatflag
 *
 * \author Ziteng Wang
 * \date   12/24/2012
 *
 * Modified by Chensong Zhang on 05/02/2013: fix a bug when writing in binary format
 */
void fasp_vector_write(const char* filerhs, void* b, const INT flag)
{

    INT   fileflag, vectorflag;
    FILE* fp;

    fileflag   = (int)flag / 10;
    vectorflag = (int)flag % 10;

    // write vector in ASCII
    if (!fileflag) {
        fp = fopen(filerhs, "w");

        if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filerhs);

        printf("%s: writing to file %s ...\n", __FUNCTION__, filerhs);

        fprintf(fp, "%d%d%d%d\n", fileflag, fileflag, fileflag, fileflag);

        fprintf(fp, "%d%d%d\n", vectorflag, (int)sizeof(INT), (int)sizeof(REAL));

        switch (vectorflag) {
            case 1:
                fasp_dvec_write_s(fp, (dvector*)b);
                break;
            case 2:
                fasp_ivec_write_s(fp, (ivector*)b);
                break;
            case 3:
                fasp_dvecind_write_s(fp, (dvector*)b);
                break;
            case 4:
                fasp_ivecind_write_s(fp, (ivector*)b);
                break;
            default:
                printf("### WARNING: Unknown vector flag %d\n", vectorflag);
        }

        fclose(fp);
        return;
    }

    // write vector in binary
    fp = fopen(filerhs, "wb");

    if (fp == NULL) fasp_chkerr(ERROR_OPEN_FILE, filerhs);

    printf("%s: writing to file %s ...\n", __FUNCTION__, filerhs);

    INT putflag = vectorflag * 100 + sizeof(INT) * 10 + sizeof(REAL);
    fwrite(&putflag, sizeof(INT), 1, fp);

    switch (vectorflag) {
        case 1:
            fasp_dvec_write_b(fp, (dvector*)b);
            break;
        case 2:
            fasp_ivec_write_b(fp, (ivector*)b);
            break;
        case 3:
            fasp_dvecind_write_b(fp, (dvector*)b);
            break;
        case 4:
            fasp_ivecind_write_b(fp, (ivector*)b);
            break;
        default:
            printf("### WARNING: Unknown vector flag %d\n", vectorflag);
    }

    fclose(fp);
}

/**
 * \fn fasp_hb_read (const char *input_file, dCSRmat *A, dvector *b)
 *
 * \brief Read matrix and right-hans side from a HB format file
 *
 * \param input_file   File name of vector file
 * \param A            Pointer to the matrix
 * \param b            Pointer to the vector
 *
 * \note Modified from the C code hb_io_prb.c by John Burkardt, which is NOT part
 *       of the FASP project!
 *
 * \author Xiaoehe Hu
 * \date   05/30/2014
 */
void fasp_hb_read(const char* input_file, dCSRmat* A, dvector* b)
{
    //-------------------------
    // Setup local variables
    //-------------------------
    // variables for FASP
    dCSRmat tempA;

    // variables for hb_io

    int*    colptr = NULL;
    double* exact  = NULL;
    double* guess  = NULL;
    int     i;
    int     indcrd;
    char*   indfmt = NULL;
    FILE*   input;
    int     j;
    char*   key    = NULL;
    char*   mxtype = NULL;
    int     ncol;
    int     neltvl;
    int     nnzero;
    int     nrhs;
    int     nrhsix;
    int     nrow;
    int     ptrcrd;
    char*   ptrfmt = NULL;
    int     rhscrd;
    char*   rhsfmt = NULL;
    int*    rhsind = NULL;
    int*    rhsptr = NULL;
    char*   rhstyp = NULL;
    double* rhsval = NULL;
    double* rhsvec = NULL;
    int*    rowind = NULL;
    char*   title  = NULL;
    int     totcrd;
    int     valcrd;
    char*   valfmt = NULL;
    double* values = NULL;

    printf("\n");
    printf("HB_FILE_READ reads all the data in an HB file.\n");

    printf("\n");
    printf("Reading the file '%s'\n", input_file);

    input = fopen(input_file, "rt");

    if (!input) {
        printf("### ERROR: Fail to open the file [%s]\n", input_file);
        fasp_chkerr(ERROR_OPEN_FILE, __FUNCTION__);
    }

    //-------------------------
    // Reading...
    //-------------------------
    hb_file_read(input, &title, &key, &totcrd, &ptrcrd, &indcrd, &valcrd, &rhscrd,
                 &mxtype, &nrow, &ncol, &nnzero, &neltvl, &ptrfmt, &indfmt, &valfmt,
                 &rhsfmt, &rhstyp, &nrhs, &nrhsix, &colptr, &rowind, &values, &rhsval,
                 &rhsptr, &rhsind, &rhsvec, &guess, &exact);

    //-------------------------
    // Printing if needed
    //-------------------------
#if DEBUG_MODE > PRINT_MIN
    /*
     Print out the header information.
     */
    hb_header_print(title, key, totcrd, ptrcrd, indcrd, valcrd, rhscrd, mxtype, nrow,
                    ncol, nnzero, neltvl, ptrfmt, indfmt, valfmt, rhsfmt, rhstyp, nrhs,
                    nrhsix);
    /*
     Print the structure information.
     */
    hb_structure_print(ncol, mxtype, nnzero, neltvl, colptr, rowind);

    /*
     Print the values.
     */
    hb_values_print(ncol, colptr, mxtype, nnzero, neltvl, values);

    if (0 < rhscrd) {
        /*
         Print a bit of the right hand sides.
         */
        if (rhstyp[0] == 'F') {
            r8mat_print_some(nrow, nrhs, rhsval, 1, 1, 5, 5, "  Part of RHS");
        } else if (rhstyp[0] == 'M' && mxtype[2] == 'A') {
            i4vec_print_part(nrhs + 1, rhsptr, 10, "  Part of RHSPTR");
            i4vec_print_part(nrhsix, rhsind, 10, "  Part of RHSIND");
            r8vec_print_part(nrhsix, rhsvec, 10, "  Part of RHSVEC");
        } else if (rhstyp[0] == 'M' && mxtype[2] == 'E') {
            r8mat_print_some(nnzero, nrhs, rhsval, 1, 1, 5, 5, "  Part of RHS");
        }
        /*
         Print a bit of the starting guesses.
         */
        if (rhstyp[1] == 'G') {
            r8mat_print_some(nrow, nrhs, guess, 1, 1, 5, 5, "  Part of GUESS");
        }
        /*
         Print a bit of the exact solutions.
         */
        if (rhstyp[2] == 'X') {
            r8mat_print_some(nrow, nrhs, exact, 1, 1, 5, 5, "  Part of EXACT");
        }
    }
#endif

    //-------------------------
    // Closing
    //-------------------------
    fclose(input);

    //-------------------------
    // Convert to FASP format
    //-------------------------

    // convert matrix
    if (ncol != nrow) {
        printf("### ERROR: The matrix is not square! [%s]\n", __FUNCTION__);
        goto FINISHED;
    }

    tempA = fasp_dcsr_create(nrow, ncol, nnzero);

    for (i = 0; i <= ncol; i++) tempA.IA[i] = colptr[i] - 1;
    for (i = 0; i < nnzero; i++) tempA.JA[i] = rowind[i] - 1;
    fasp_darray_cp(nnzero, values, tempA.val);

    // if the matrix is symmeric
    if (mxtype[1] == 'S') {

        // A = A'+ A
        dCSRmat tempA_tran;
        fasp_dcsr_trans(&tempA, &tempA_tran);
        fasp_blas_dcsr_add(&tempA, 1.0, &tempA_tran, 1.0, A);
        fasp_dcsr_free(&tempA);
        fasp_dcsr_free(&tempA_tran);

        // modify diagonal entries
        for (i = 0; i < A->row; i++) {

            for (j = A->IA[i]; j < A->IA[i + 1]; j++) {

                if (A->JA[j] == i) {
                    A->val[j] = A->val[j] / 2;
                    break;
                }
            }
        }
    }
    // if the matrix is not symmetric
    else {
        fasp_dcsr_trans(&tempA, A);
        fasp_dcsr_free(&tempA);
    }

    // convert right hand side

    if (nrhs == 0) {

        printf("### ERROR: No right hand side! [%s]\n", __FUNCTION__);
        goto FINISHED;
    } else if (nrhs > 1) {

        printf("### ERROR: More than one right hand side! [%s]\n", __FUNCTION__);
        goto FINISHED;
    } else {

        fasp_dvec_alloc(nrow, b);
        fasp_darray_cp(nrow, rhsval, b->val);
    }

    //-------------------------
    // Cleanning
    //-------------------------
FINISHED:
    if (colptr) free(colptr);
    if (exact) free(exact);
    if (guess) free(guess);
    if (rhsind) free(rhsind);
    if (rhsptr) free(rhsptr);
    if (rhsval) free(rhsval);
    if (rhsvec) free(rhsvec);
    if (rowind) free(rowind);
    if (values) free(values);

    return;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
