/*! \file  BlaIOUtil.inl
 *
 *  \brief Read, write, and print subroutines.
 *
 *  \note  This file contains Level-1 (Bla) functions, which are used in:
 *         BlaIO.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include "fasp.h"
#include "fasp_functs.h"

/**
 * \fn static inline void skip_comments (FILE * fp)
 *
 * \brief Skip the comments in the beginning of data file
 *
 * \param FILE        File handler
 *
 * \author Chensong Zhang
 * \date   2020-06-30
 */
static inline void skip_comments (FILE * fp)
{
    while ( 1 ) {
        char  buffer[500];
        int   loc = ftell(fp); // record the current position
        int   val = fscanf(fp,"%s",buffer); // read in a string
        if ( val!=1 || val==EOF ) {
            printf("### ERROR: Could not get any data!\n");
            exit(ERROR_WRONG_FILE);
        }
        if ( buffer[0]=='%' || buffer[0]=='!' || buffer[0]=='/' ) {
            if (fscanf(fp, "%*[^\n]")) {/* skip rest of line and do nothing */ };
            continue;
        }
        else {
            fseek(fp, loc, SEEK_SET); // back to the beginning of this line
            break;
        }
    }
}

/**
 * \fn static inline INT endian_convert_int (const INT inum, const INT length,
 *                                           const SHORT EndianFlag)
 *
 * \brief Swap order of an INT number
 *
 * \param inum        An INT value
 * \param length     Length of INT: 2 for short, 4 for int, 8 for long
 * \param EndianFlag  If EndianFlag = 1, it returns inum itself
 *                    If EndianFlag = 2, it returns the swapped inum
 *
 * \return Value of inum or swapped inum
 *
 * \author Ziteng Wang
 * \date   2012-12-24
 */
static inline INT endian_convert_int (const INT  inum,
                                      const INT  length,
                                      const INT  EndianFlag)
{
    INT iretVal = 0, i;
    char *intToConvert = ( char * ) & inum;
    char *returnInt = ( char * ) & iretVal;
    
    if ( EndianFlag == 1 ) return inum;
    else {
        for (i = 0; i < length; i++) {
            returnInt[i] = intToConvert[length-i-1];
        }
        return iretVal;
    }
}

/**
 * \fn static inline REAL endian_convert_real (const REAL rnum, const INT length,
 *                                             const SHORT EndianFlag)
 *
 * \brief Swap order of a REAL number
 *
 * \param rnum        An REAL value
 * \param length      Length of INT: 2 for short, 4 for int, 8 for long
 * \param EndianFlag  If EndianFlag = 1, it returns rnum itself
 *                    If EndianFlag = 2, it returns the swapped rnum
 *
 * \return Value of rnum or swapped rnum
 *
 * \author Ziteng Wang
 * \date   2012-12-24
 */
static inline REAL endian_convert_real (const REAL  rnum,
                                        const INT   length,
                                        const INT   EndianFlag)
{
    REAL dretVal = 0.0;
    char *realToConvert = (char *) & rnum;
    char *returnReal    = (char *) & dretVal;
    INT  i;
    
    if (EndianFlag==1) return rnum;
    else {
        for (i = 0; i < length; i++) {
            returnReal[i] = realToConvert[length-i-1];
        }
        return dretVal;
    }
}

static inline void fasp_dcsr_read_s (FILE        *fp,
                                     dCSRmat     *A)
{
    int   status;
    INT   i,m,nnz,idata;
    REAL  ddata;
    
    // Read CSR matrix
    status = fscanf(fp, "%d", &m);
    A->row=m;
    
    A->IA = (INT *)fasp_mem_calloc(m+1, sizeof(INT));
    for ( i = 0; i <= m; ++i ) {
        status = fscanf(fp, "%d", &idata);
        A->IA[i] = idata;
    }
    
    nnz = A->IA[m]-A->IA[0]; A->nnz=nnz;
    
    A->JA  = (INT *)fasp_mem_calloc(nnz, sizeof(INT));
    A->val = (REAL*)fasp_mem_calloc(nnz, sizeof(REAL));
    
    for ( i = 0; i < nnz; ++i ) {
        status = fscanf(fp, "%d", &idata);
        A->JA[i] = idata;
    }
    
    for ( i = 0; i < nnz; ++i ) {
        status = fscanf(fp, "%lf", &ddata);
        A->val[i]= ddata;
    }

    fasp_chkerr(status, __FUNCTION__);
}

static inline void fasp_dcsr_read_b (FILE        *fp,
                                     dCSRmat     *A,
                                     const SHORT  EndianFlag)
{
    size_t   status;
    INT      i,m,nnz,idata;
    REAL     ddata;
    
    // Read CSR matrix
    status = fread(&idata, ilength, 1, fp);
    A->row = endian_convert_int(idata, ilength, EndianFlag);
    m = A->row;
    
    A->IA = (INT *)fasp_mem_calloc(m+1, sizeof(INT));
    for ( i = 0; i <= m; ++i ) {
        status = fread(&idata, ilength, 1, fp);
        A->IA[i] = endian_convert_int(idata, ilength, EndianFlag);
    }
    
    nnz=A->IA[m]-A->IA[0]; A->nnz=nnz;
    
    A->JA=(INT *)fasp_mem_calloc(nnz, sizeof(INT));
    A->val=(REAL*)fasp_mem_calloc(nnz, sizeof(REAL));
    
    for ( i = 0; i < nnz; ++i ) {
        status = fread(&idata, ilength, 1, fp);
        A->JA[i] = endian_convert_int(idata, ilength, EndianFlag);
    }
    
    for ( i = 0; i < nnz; ++i ) {
        status = fread(&ddata, dlength, 1, fp);
        A->val[i] = endian_convert_real(ddata, dlength, EndianFlag);
    }

    fasp_chkerr(status, __FUNCTION__);
}

static inline void fasp_dcoo_read_s (FILE        *fp,
                                     dCSRmat     *A)
{
    INT   i,j,k,m,n,nnz;
    REAL  value;
    int   status;
    
    status = fscanf(fp,"%d %d %d",&m,&n,&nnz);
    
    dCOOmat Atmp = fasp_dcoo_create(m,n,nnz);
    
    for ( k = 0; k < nnz; k++ ) {
        status = fscanf(fp, "%d %d %le", &i, &j, &value);
        if ( status != EOF ) {
            Atmp.rowind[k] = i;
            Atmp.colind[k] = j;
            Atmp.val[k]    = value;
        }
        else {
            fasp_chkerr(ERROR_WRONG_FILE, __FUNCTION__);
        }
    }

    fasp_format_dcoo_dcsr(&Atmp,A);
    fasp_dcoo_free(&Atmp);

    fasp_chkerr(status, __FUNCTION__);
}

static inline void fasp_dcoo_read_b (FILE        *fp,
                                     dCSRmat     *A,
                                     const SHORT  EndianFlag)
{
    INT     k,m,n,nnz,index;
    REAL    value;
    size_t  status;
    
    status = fread(&m, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    m = endian_convert_int(m, ilength, EndianFlag);

    status = fread(&n, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    n = endian_convert_int(n, ilength, EndianFlag);

    status = fread(&nnz, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    nnz = endian_convert_int(nnz, ilength, EndianFlag);
    
    dCOOmat Atmp=fasp_dcoo_create(m,n,nnz);
    
    for (k = 0; k < nnz; k++) {
        if ( fread(&index, ilength, 1, fp) !=EOF ) {
            Atmp.rowind[k] = endian_convert_int(index, ilength, EndianFlag);

            status = fread(&index, ilength, 1, fp);
            fasp_chkerr(status, __FUNCTION__);
            Atmp.colind[k] = endian_convert_int(index, ilength, EndianFlag);

            status = fread(&value, sizeof(REAL), 1, fp);
            fasp_chkerr(status, __FUNCTION__);
            Atmp.val[k] = endian_convert_real(value, sizeof(REAL), EndianFlag);
        }
        else {
            fasp_chkerr(ERROR_WRONG_FILE, __FUNCTION__);
        }
    }
    
    fasp_format_dcoo_dcsr(&Atmp, A);
    fasp_dcoo_free(&Atmp);
}

static inline void fasp_dbsr_read_s (FILE        *fp,
                                     dBSRmat     *A)
{
    INT   ROW, COL, NNZ, nb, storage_manner;
    INT   i, n, index;
    REAL  value;
    int   status;
    
    status = fscanf(fp, "%d %d %d", &ROW,&COL,&NNZ); // read dimension of the problem
    fasp_chkerr(status, __FUNCTION__);
    A->ROW = ROW; A->COL = COL; A->NNZ = NNZ;
    
    status = fscanf(fp, "%d", &nb); // read the size
    fasp_chkerr(status, __FUNCTION__);
    A->nb = nb;
    
    status = fscanf(fp, "%d", &storage_manner); // read the storage_manner
    fasp_chkerr(status, __FUNCTION__);
    A->storage_manner = storage_manner;
    
    // allocate memory space
    fasp_dbsr_alloc(ROW, COL, NNZ, nb, storage_manner, A);
    
    // read IA
    status = fscanf(fp, "%d", &n);
    fasp_chkerr(status, __FUNCTION__);

    for ( i = 0; i < n; ++i ) {
        status = fscanf(fp, "%d", &index);
        fasp_chkerr(status, __FUNCTION__);
        A->IA[i] = index;
    }
    
    // read JA
    status = fscanf(fp, "%d", &n);
    fasp_chkerr(status, __FUNCTION__);

    for ( i = 0; i < n; ++i ){
        status = fscanf(fp, "%d", &index);
        fasp_chkerr(status, __FUNCTION__);
        A->JA[i] = index;
    }
    
    // read val
    status = fscanf(fp, "%d", &n);
    fasp_chkerr(status, __FUNCTION__);

    for ( i = 0; i < n; ++i ) {
        status = fscanf(fp, "%le", &value);
        fasp_chkerr(status, __FUNCTION__);
        A->val[i] = value;
    }
    
}

static inline void fasp_dbsr_read_b (FILE        *fp,
                                     dBSRmat     *A,
                                     const SHORT  EndianFlag)
{
    INT     ROW, COL, NNZ, nb, storage_manner;
    INT     i, n, index;
    REAL    value;
    size_t  status;
    
    // read dimension of the problem
    status = fread(&ROW, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    A->ROW = endian_convert_int(ROW, ilength, EndianFlag);

    status = fread(&COL, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    A->COL = endian_convert_int(COL, ilength, EndianFlag);

    status = fread(&NNZ, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    A->NNZ = endian_convert_int(NNZ, ilength, EndianFlag);
    
    status = fread(&nb, ilength, 1, fp); // read the size
    fasp_chkerr(status, __FUNCTION__);
    A->nb = endian_convert_int(nb, ilength, EndianFlag);
    
    status = fread(&storage_manner, 1, ilength, fp); // read the storage manner
    fasp_chkerr(status, __FUNCTION__);
    A->storage_manner = endian_convert_int(storage_manner, ilength, EndianFlag);
    
    // allocate memory space
    fasp_dbsr_alloc(ROW, COL, NNZ, nb, storage_manner, A);
    
    // read IA
    status = fread(&n, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    for ( i = 0; i < n; i++ ) {
        status = fread(&index, 1, ilength, fp);
        fasp_chkerr(status, __FUNCTION__);
        A->IA[i] = endian_convert_int(index, ilength, EndianFlag);
    }
    
    // read JA
    status = fread(&n, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    for ( i = 0; i < n; i++ ) {
        status = fread(&index, ilength, 1, fp);
        fasp_chkerr(status, __FUNCTION__);
        A->JA[i] = endian_convert_int(index, ilength, EndianFlag);
    }
    
    // read val
    status = fread(&n, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    for ( i = 0; i < n; i++ ) {
        status = fread(&value, sizeof(REAL), 1, fp);
        fasp_chkerr(status, __FUNCTION__);
        A->val[i] = endian_convert_real(value, sizeof(REAL), EndianFlag);
    }

}

static inline void fasp_dstr_read_s (FILE        *fp,
                                     dSTRmat     *A)
{
    INT   nx, ny, nz, nxy, ngrid, nband, nc, offset;
    INT   i, k, n;
    REAL  value;
    int   status;
    
    status = fscanf(fp,"%d %d %d",&nx,&ny,&nz); // read dimension of the problem
    fasp_chkerr(status, __FUNCTION__);
    A->nx = nx; A->ny = ny; A->nz = nz;
    
    nxy = nx*ny; ngrid = nxy*nz;
    A->nxy = nxy; A->ngrid = ngrid;
    
    status = fscanf(fp,"%d",&nc); // read number of components
    fasp_chkerr(status, __FUNCTION__);
    A->nc = nc;
    
    status = fscanf(fp,"%d",&nband); // read number of bands
    fasp_chkerr(status, __FUNCTION__);
    A->nband = nband;
    
    A->offsets=(INT*)fasp_mem_calloc(nband, ilength);
    
    // read diagonal
    status = fscanf(fp, "%d", &n);
    fasp_chkerr(status, __FUNCTION__);
    A->diag=(REAL *)fasp_mem_calloc(n, sizeof(REAL));
    for ( i = 0; i < n; ++i ) {
        status = fscanf(fp, "%le", &value);
        fasp_chkerr(status, __FUNCTION__);
        A->diag[i] = value;
    }
    
    // read offdiags
    k = nband;
    A->offdiag=(REAL **)fasp_mem_calloc(nband, sizeof(REAL *));
    while ( k-- ) {
        status = fscanf(fp,"%d %d",&offset,&n); // read number band k
        fasp_chkerr(status, __FUNCTION__);
        A->offsets[nband-k-1]=offset;
        
        A->offdiag[nband-k-1]=(REAL *)fasp_mem_calloc(n, sizeof(REAL));
        for ( i = 0; i < n; ++i ) {
            status = fscanf(fp, "%le", &value);
            fasp_chkerr(status, __FUNCTION__);
            A->offdiag[nband-k-1][i] = value;
        }
    }
    
}

static inline void fasp_dstr_read_b (FILE        *fp,
                                     dSTRmat     *A,
                                     const SHORT  EndianFlag)
{
    INT     nx, ny, nz, nxy, ngrid, nband, nc, offset;
    INT     i, k, n;
    REAL    value;
    size_t  status;
    
    // read dimension of the problem
    status = fread(&nx, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    A->nx = endian_convert_int(nx, ilength, EndianFlag);

    status = fread(&ny, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    A->ny = endian_convert_int(ny, ilength, EndianFlag);

    status = fread(&nz, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    A->nz = endian_convert_int(nz, ilength, EndianFlag);
    
    nxy = nx*ny; ngrid = nxy*nz;
    A->nxy = nxy; A->ngrid = ngrid;
    
    // read number of components
    status = fread(&nc, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    A->nc = nc;
    
    // read number of bands
    status = fread(&nband, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    A->nband = nband;
    
    A->offsets=(INT*)fasp_mem_calloc(nband, ilength);
    
    // read diagonal
    status = fread(&n, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    n = endian_convert_int(n, ilength, EndianFlag);
    A->diag=(REAL *)fasp_mem_calloc(n, sizeof(REAL));

    for ( i = 0; i < n; i++ ) {
        status = fread(&value, sizeof(REAL), 1, fp);
        fasp_chkerr(status, __FUNCTION__);
        A->diag[i]=endian_convert_real(value, sizeof(REAL), EndianFlag);
    }
    
    // read offdiags
    k = nband;
    A->offdiag=(REAL **)fasp_mem_calloc(nband, sizeof(REAL *));

    while ( k-- ) {
        status = fread(&offset, ilength, 1, fp);
        fasp_chkerr(status, __FUNCTION__);
        A->offsets[nband-k-1]=endian_convert_int(offset, ilength, EndianFlag);;
        
        status = fread(&n, ilength, 1, fp);
        fasp_chkerr(status, __FUNCTION__);
        n = endian_convert_int(n, ilength, EndianFlag);
        A->offdiag[nband-k-1]=(REAL *)fasp_mem_calloc(n, sizeof(REAL));
        for ( i = 0; i < n; i++ ) {
            status = fread(&value, sizeof(REAL), 1, fp);
            fasp_chkerr(status, __FUNCTION__);
            A->offdiag[nband-k-1][i]=endian_convert_real(value, sizeof(REAL), EndianFlag);
        }
    }

}

static inline void fasp_dmtx_read_s (FILE        *fp,
                                     dCSRmat     *A)
{
    INT   i,j,m,n,nnz;
    INT   innz; // index of nonzeros
    REAL  value;
    int   status;
    
    status = fscanf(fp,"%d %d %d",&m,&n,&nnz);
    
    dCOOmat Atmp=fasp_dcoo_create(m,n,nnz);
    
    innz = 0;
    
    while (innz < nnz) {
        status = fscanf(fp, "%d %d %le", &i, &j, &value);
        if ( status != EOF ) {
            Atmp.rowind[innz]=i-1;
            Atmp.colind[innz]=j-1;
            Atmp.val[innz] = value;
            innz = innz + 1;
        }
        else {
            fasp_chkerr(ERROR_WRONG_FILE, __FUNCTION__);
        }
    }
    
    fasp_format_dcoo_dcsr(&Atmp,A);
    fasp_dcoo_free(&Atmp);

    fasp_chkerr(status, __FUNCTION__);
}

static inline void fasp_dmtx_read_b (FILE        *fp,
                                     dCSRmat     *A,
                                     const SHORT  EndianFlag)
{
    INT     m,n,k,nnz;
    INT     index;
    REAL    value;
    size_t  status;
    
    status = fread(&m, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    m = endian_convert_int(m, ilength, EndianFlag);

    status = fread(&n, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    n = endian_convert_int(n, ilength, EndianFlag);

    status = fread(&nnz, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    nnz = endian_convert_int(nnz, ilength, EndianFlag);
    
    dCOOmat Atmp=fasp_dcoo_create(m,n,nnz);
    
    for (k = 0; k < nnz; k++) {
        if ( fread(&index, ilength, 1, fp) !=EOF ) {
            Atmp.rowind[k] = endian_convert_int(index, ilength, EndianFlag)-1;

            status = fread(&index, ilength, 1, fp);
            fasp_chkerr(status, __FUNCTION__);
            Atmp.colind[k] = endian_convert_int(index, ilength, EndianFlag)-1;

            status = fread(&value, sizeof(REAL), 1, fp);
            fasp_chkerr(status, __FUNCTION__);
            Atmp.val[k] = endian_convert_real(value, sizeof(REAL), EndianFlag);
        }
        else {
            fasp_chkerr(ERROR_WRONG_FILE, __FUNCTION__);
        }
    }
    
    fasp_format_dcoo_dcsr(&Atmp, A);
    fasp_dcoo_free(&Atmp);

}

static inline void fasp_dmtxsym_read_s (FILE        *fp,
                                        dCSRmat     *A)
{
    INT   i,j,m,n,nnz;
    INT   innz; // index of nonzeros
    REAL  value;
    int   status;
    
    status = fscanf(fp,"%d %d %d",&m,&n,&nnz);
    
    nnz = 2*(nnz-m) + m; // adjust for sym problem
    
    dCOOmat Atmp=fasp_dcoo_create(m,n,nnz);
    
    innz = 0;
    
    while (innz < nnz) {
        status = fscanf(fp, "%d %d %le", &i, &j, &value);
        if ( status != EOF ) {
            if (i==j) {
                Atmp.rowind[innz]=i-1;
                Atmp.colind[innz]=j-1;
                Atmp.val[innz] = value;
                innz = innz + 1;
            }
            else {
                Atmp.rowind[innz]=i-1; Atmp.rowind[innz+1]=j-1;
                Atmp.colind[innz]=j-1; Atmp.colind[innz+1]=i-1;
                Atmp.val[innz] = value; Atmp.val[innz+1] = value;
                innz = innz + 2;
            }
        }
        else {
            fasp_chkerr(ERROR_WRONG_FILE, __FUNCTION__);
        }
    }
    
    fasp_format_dcoo_dcsr(&Atmp,A);
    fasp_dcoo_free(&Atmp);

    fasp_chkerr(status, __FUNCTION__);
}

static inline void fasp_dmtxsym_read_b (FILE        *fp,
                                        dCSRmat     *A,
                                        const SHORT  EndianFlag)
{
    INT     m,n,nnz;
    INT     innz;
    INT     index[2];
    REAL    value;
    size_t  status;
    
    status = fread(&m, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    m = endian_convert_int(m, ilength, EndianFlag);

    status = fread(&n, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    n = endian_convert_int(n, ilength, EndianFlag);

    status = fread(&nnz, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    nnz = endian_convert_int(nnz, ilength, EndianFlag);
    nnz = 2*(nnz-m) + m; // adjust for sym problem
    
    dCOOmat Atmp=fasp_dcoo_create(m,n,nnz);
    
    innz = 0;
    
    while (innz < nnz) {
        if ( fread(index, ilength, 2, fp) !=EOF ) {
            
            if (index[0]==index[1]) {
                INT indextemp = index[0];
                Atmp.rowind[innz] = endian_convert_int(indextemp, ilength, EndianFlag)-1;
                indextemp = index[1];
                Atmp.colind[innz] = endian_convert_int(indextemp, ilength, EndianFlag)-1;

                status = fread(&value, sizeof(REAL), 1, fp);
                fasp_chkerr(status, __FUNCTION__);
                Atmp.val[innz] = endian_convert_real(value, sizeof(REAL), EndianFlag);
                innz = innz + 1;
            }
            else {
                INT indextemp = index[0];
                Atmp.rowind[innz] = endian_convert_int(indextemp, ilength, EndianFlag)-1;
                Atmp.rowind[innz+1] = Atmp.rowind[innz];
                indextemp = index[1];
                Atmp.colind[innz] = endian_convert_int(indextemp, ilength, EndianFlag)-1;
                Atmp.colind[innz+1] = Atmp.colind[innz];

                status = fread(&value, sizeof(REAL), 1, fp);
                fasp_chkerr(status, __FUNCTION__);
                Atmp.val[innz] = endian_convert_real(value, sizeof(REAL), EndianFlag);
                Atmp.val[innz+1] = Atmp.val[innz];
                innz = innz + 2;
            }
            
        }
        else {
            fasp_chkerr(ERROR_WRONG_FILE, __FUNCTION__);
        }
    }
    
    fasp_format_dcoo_dcsr(&Atmp,A);
    fasp_dcoo_free(&Atmp);
}

static inline void fasp_dcsr_write_s (FILE        *fp,
                                      dCSRmat     *A)
{
    const INT m=A->row, n=A->col;
    INT i;
    
    fprintf(fp,"%d  %d  %d\n",m,n,A->nnz);
    
    for ( i = 0; i < m+1; ++i ) fprintf(fp,"%d\n", A->IA[i]);
    
    for ( i = 0; i < A->nnz; ++i ) fprintf(fp,"%d\n", A->JA[i]);
    
    for ( i = 0; i < A->nnz; ++i ) fprintf(fp,"%le\n", A->val[i]);
}

static inline void fasp_dcsr_write_b (FILE        *fp,
                                      dCSRmat     *A)
{
    const INT m=A->row, n=A->col;
    INT i, j, nnz, index;
    REAL value;
    
    nnz = A->nnz;
    fwrite(&m, sizeof(INT), 1, fp);
    fwrite(&n, sizeof(INT), 1, fp);
    fwrite(&nnz, sizeof(INT), 1, fp);
    for ( i = 0; i < m; i++ ) {
        for (j = A->IA[i]; j < A->IA[i+1]; j++) {
            fwrite(&i, sizeof(INT), 1, fp);
            index = A->JA[j];
            value = A->val[j];
            fwrite(&index, sizeof(INT), 1, fp);
            fwrite(&value, sizeof(REAL), 1, fp);
        }
    }
    
    fclose(fp);
}

static inline void fasp_dbsr_write_s (FILE        *fp,
                                      dBSRmat     *A)
{
    const INT ROW = A->ROW, COL = A->COL, NNZ = A->NNZ;
    const INT nb = A->nb, storage_manner = A->storage_manner;
    
    INT  *ia  = A->IA;
    INT  *ja  = A->JA;
    REAL *val = A->val;
    
    INT i, n;
    
    fprintf(fp,"%d  %d  %d\n",ROW,COL,NNZ); // write dimension of the block matrix
    
    fprintf(fp,"%d\n",nb); // write the size
    
    fprintf(fp,"%d\n",storage_manner); // write storage manner
    
    // write A->IA
    n = ROW+1; // length of A->IA
    fprintf(fp,"%d\n",n); // length of A->IA
    for ( i = 0; i < n; ++i ) fprintf(fp, "%d\n", ia[i]);
    
    // write A->JA
    n = NNZ; // length of A->JA
    fprintf(fp, "%d\n", n); // length of A->JA
    for ( i = 0; i < n; ++i ) fprintf(fp, "%d\n", ja[i]);
    
    // write A->val
    n = NNZ*nb*nb; // length of A->val
    fprintf(fp, "%d\n", n); // length of A->val
    for ( i = 0; i < n; ++i ) fprintf(fp, "%le\n", val[i]);
}

static inline void fasp_dbsr_write_b (FILE        *fp,
                                      dBSRmat     *A)
{
    const INT ROW = A->ROW, COL = A->COL, NNZ = A->NNZ;
    const INT nb = A->nb, storage_manner = A->storage_manner;
    
    INT  *ia  = A->IA;
    INT  *ja  = A->JA;
    REAL *val = A->val;
    
    INT i, n, index;
    REAL value;
    
    // write dimension of the block matrix
    fwrite(&ROW, sizeof(INT), 1, fp);
    fwrite(&COL, sizeof(INT), 1, fp);
    fwrite(&NNZ, sizeof(INT), 1, fp);
    
    fwrite(&nb, sizeof(INT), 1, fp); // write the size
    
    fwrite(&storage_manner, sizeof(INT), 1, fp);
    
    // write A.IA
    n = ROW+1;
    fwrite(&n, sizeof(INT), 1, fp);
    for ( i = 0; i < n; i++ ) {
        index = ia[i];
        fwrite(&index, sizeof(INT), 1, fp);
    }
    
    // write A.JA
    n = NNZ;
    fwrite(&n, sizeof(INT), 1, fp);
    for ( i = 0; i < n; i++ ) {
        index = ja[i];
        fwrite(&index, sizeof(INT), 1, fp);
    }
    
    // write A.val
    n = NNZ*nb*nb;
    fwrite(&n,sizeof(INT), 1, fp);
    for ( i = 0; i < n; i++ ) {
        value = val[i];
        fwrite(&value, sizeof(REAL), 1, fp);
    }
}

static inline void fasp_dstr_write_s (FILE        *fp,
                                      dSTRmat     *A)
{
    const INT nx=A->nx, ny=A->ny, nz=A->nz;
    const INT ngrid=A->ngrid, nband=A->nband, nc=A->nc;
    
    INT *offsets=A->offsets;
    
    INT i, k, n;
    
    fprintf(fp,"%d  %d  %d\n",nx,ny,nz); // write dimension of the problem
    
    fprintf(fp,"%d\n",nc); // read number of components
    
    fprintf(fp,"%d\n",nband); // write number of bands
    
    // write diagonal
    n=ngrid*nc*nc; // number of nonzeros in each band
    fprintf(fp,"%d\n",n); // number of diagonal entries
    for ( i = 0; i < n; ++i ) fprintf(fp, "%le\n", A->diag[i]);
    
    // write offdiags
    k = nband;
    while ( k-- ) {
        INT offset=offsets[nband-k-1];
        n=(ngrid-ABS(offset))*nc*nc; // number of nonzeros in each band
        fprintf(fp,"%d  %d\n",offset,n); // read number band k
        for ( i = 0; i < n; ++i ) {
            fprintf(fp, "%le\n", A->offdiag[nband-k-1][i]);
        }
    }
    
}

static inline void fasp_dstr_write_b (FILE        *fp,
                                      dSTRmat     *A)
{
    const INT nx=A->nx, ny=A->ny, nz=A->nz;
    const INT ngrid=A->ngrid, nband=A->nband, nc=A->nc;
    
    INT *offsets=A->offsets;
    
    INT i, k, n;
    REAL value;
    
    // write dimension of the problem
    fwrite(&nx, sizeof(INT), 1, fp);
    fwrite(&ny, sizeof(INT), 1, fp);
    fwrite(&nz, sizeof(INT), 1, fp);
    
    fwrite(&nc, sizeof(INT), 1, fp);  // read number of components
    
    fwrite(&nband, sizeof(INT), 1, fp); // write number of bands
    
    // write diagonal
    n=ngrid*nc*nc; // number of nonzeros in each band
    fwrite(&n, sizeof(INT), 1, fp); // number of diagonal entries
    for ( i = 0; i < n; i++ ) {
        value = A->diag[i];
        fwrite(&value, sizeof(REAL), 1, fp);
    }
    
    // write offdiags
    k = nband;
    while ( k-- ) {
        INT offset=offsets[nband-k-1];
        n=(ngrid-ABS(offset))*nc*nc; // number of nonzeros in each band
        fwrite(&offset, sizeof(INT), 1, fp);
        fwrite(&n, sizeof(INT), 1, fp);
        for ( i = 0; i < n; i++ ) {
            value = A->offdiag[nband-k-1][i];
            fwrite(&value, sizeof(REAL), 1, fp);
        }
    }
    
}

static inline void fasp_dvec_read_s (FILE        *fp,
                                     dvector     *b)
{
    
    INT   i, n;
    REAL  value;
    INT   status;
    
    status = fscanf(fp,"%d",&n);
    fasp_dvec_alloc(n,b);
    
    for ( i = 0; i < n; ++i ) {
        status = fscanf(fp, "%le", &value);
        b->val[i] = value;
    }

    fasp_chkerr(status, __FUNCTION__);
}

static inline void fasp_dvec_read_b (FILE        *fp,
                                     dvector     *b,
                                     const SHORT  EndianFlag)
{
    INT     i, n;
    REAL    value;
    size_t  status;
    
    status = fread(&n, ilength, 1, fp);
    n = endian_convert_int(n, ilength, EndianFlag);
    fasp_dvec_alloc(n,b);
    
    for ( i = 0; i < n; i++ ) {
        status = fread(&value, dlength, 1, fp);
        b->val[i]=endian_convert_real(value, dlength, EndianFlag);
    }

    fasp_chkerr(status, __FUNCTION__);
}

static inline void fasp_ivec_read_s (FILE        *fp,
                                     ivector     *b)
{
    INT   i, n, value;
    INT   status;
    
    status = fscanf(fp,"%d",&n);
    fasp_ivec_alloc(n,b);
    
    for ( i = 0; i < n; ++i ) {
        status = fscanf(fp, "%d", &value);
        b->val[i] = value;
    }

    fasp_chkerr(status, __FUNCTION__);
}

static inline void fasp_ivec_read_b (FILE        *fp,
                                     ivector     *b,
                                     const SHORT  EndianFlag)
{
    INT     i, n, value;
    size_t  status;
    
    status = fread(&n, ilength, 1, fp);
    n = endian_convert_int(n, ilength, EndianFlag);
    fasp_ivec_alloc(n,b);
    
    for ( i = 0; i < n; i++ ) {
        status = fread(&value, dlength, 1, fp);
        b->val[i]=endian_convert_real(value, dlength, EndianFlag);
    }
    
    fclose(fp);
    fasp_chkerr(status, __FUNCTION__);
}

static inline void fasp_dvecind_read_s (FILE        *fp,
                                        dvector     *b)
{
    INT   i, n, index;
    REAL  value;
    INT   status;
    
    status = fscanf(fp,"%d",&n);
    fasp_dvec_alloc(n,b);
    
    for ( i = 0; i < n; ++i ) {
        status = fscanf(fp, "%d %le", &index, &value);
        b->val[index] = value;
    }

    fasp_chkerr(status, __FUNCTION__);
}

static inline void fasp_dvecind_read_b (FILE        *fp,
                                        dvector     *b,
                                        const SHORT  EndianFlag)
{
    INT     i, n, index;
    REAL    value;
    size_t  status;
    
    status = fread(&n, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);

    n = endian_convert_int(n, ilength, EndianFlag);
    fasp_dvec_alloc(n,b);
    
    for ( i = 0; i < n; i++ ) {
        status = fread(&index, ilength, 1, fp);
        fasp_chkerr(status, __FUNCTION__);

        status = fread(&value, dlength, 1, fp);
        fasp_chkerr(status, __FUNCTION__);

        index = endian_convert_int(index, ilength, EndianFlag);
        value = endian_convert_real(value, ilength, EndianFlag);
        b->val[index] = value;
    }
}

static inline void fasp_ivecind_read_s (FILE        *fp,
                                        ivector     *b)
{
    INT   i, n, index, value;
    int   status;
    
    status = fscanf(fp,"%d",&n);
    fasp_chkerr(status, __FUNCTION__);

    fasp_ivec_alloc(n,b);
    
    for ( i = 0; i < n; ++i ) {
        status = fscanf(fp, "%d %d", &index, &value);
        fasp_chkerr(status, __FUNCTION__);
        b->val[index] = value;
    }
}

static inline void fasp_ivecind_read_b (FILE        *fp,
                                        ivector     *b,
                                        const SHORT  EndianFlag)
{
    INT     i, n, index, value;
    size_t  status;
    
    status = fread(&n, ilength, 1, fp);
    fasp_chkerr(status, __FUNCTION__);
    n = endian_convert_int(n, ilength, EndianFlag);
    fasp_ivec_alloc(n,b);
    
    for ( i = 0; i < n; i++ ) {
        status = fread(&index, ilength, 1, fp);
        fasp_chkerr(status, __FUNCTION__);

        status = fread(&value, dlength, 1, fp);
        fasp_chkerr(status, __FUNCTION__);

        index = endian_convert_int(index, ilength, EndianFlag);
        value = endian_convert_real(value, dlength, EndianFlag);
        b->val[index] = value;
    }
}

static inline void fasp_dvec_write_s (FILE        *fp,
                                      dvector     *vec)
{
    INT m = vec->row, i;
    
    fprintf(fp,"%d\n",m);
    
    for ( i = 0; i < m; ++i ) fprintf(fp,"%le\n",vec->val[i]);
}

static inline void fasp_dvec_write_b (FILE        *fp,
                                      dvector     *vec)
{
    INT m = vec->row, i;
    REAL value;
    
    fwrite(&m, sizeof(INT), 1, fp);
    
    for ( i = 0; i < m; i++ ) {
        value = vec->val[i];
        fwrite(&value, sizeof(REAL), 1, fp);
    }
}

static inline void fasp_ivec_write_s (FILE        *fp,
                                      ivector     *vec)
{
    INT m = vec->row, i;
    
    fprintf(fp,"%d\n",m);
    
    for ( i = 0; i < m; ++i ) fprintf(fp,"%d %d\n",i,vec->val[i]);
}

static inline void fasp_ivec_write_b (FILE        *fp,
                                      ivector     *vec)
{
    INT m = vec->row, i, value;
    
    fwrite(&m, sizeof(INT), 1, fp);
    
    for ( i = 0; i < m; i++ ) {
        value = vec->val[i];
        fwrite(&i, sizeof(INT), 1, fp);
        fwrite(&value, sizeof(INT), 1, fp);
    }
}

static inline void fasp_dvecind_write_s (FILE        *fp,
                                         dvector     *vec)
{
    INT m = vec->row, i;
    
    fprintf(fp,"%d\n",m);
    
    for ( i = 0; i < m; ++i ) fprintf(fp,"%d %le\n",i,vec->val[i]);
}

static inline void fasp_dvecind_write_b (FILE        *fp,
                                         dvector     *vec)
{
    INT m = vec->row, i;
    REAL value;
    
    fwrite(&m, sizeof(INT), 1, fp);
    
    for ( i = 0; i < m; i++ ) {
        value = vec->val[i];
        fwrite(&i, sizeof(INT), 1, fp);
        fwrite(&value, sizeof(REAL), 1, fp);
    }
}

static inline void fasp_ivecind_write_b (FILE        *fp,
                                         ivector     *vec)
{
    INT m = vec->row, i;
    INT value;
    
    fwrite(&m, sizeof(INT), 1, fp);
    
    for ( i = 0; i < m; i++ ) {
        value = vec->val[i];
        fwrite(&i, sizeof(INT), 1, fp);
        fwrite(&value, sizeof(INT), 1, fp);
    }
}

static inline void fasp_ivecind_write_s (FILE        *fp,
                                         ivector     *vec)
{
    INT m = vec->row, i;
    
    fprintf(fp,"%d\n",m);
    
    for ( i = 0; i < m; ++i ) fprintf(fp, "%d %d\n", i, vec->val[i]);
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/

