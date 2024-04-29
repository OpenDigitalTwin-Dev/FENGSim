/*! \file  AuxGraphics.c
 *
 *  \brief Graphical output for CSR matrix
 *
 *  \note  This file contains Level-0 (Aux) functions. It requires:
 *         AuxMemory.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <math.h>

#include "fasp.h"
#include "fasp_grid.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--  Declare Private Functions  --*/
/*---------------------------------*/

static void put_byte (FILE *fp, const int c);
static void put_word (FILE *fp, const int w);
static void put_dword (FILE *fp, const int d);
static int write_bmp16 (const char *fname, int m, int n, const char map[]);

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_dcsr_subplot (const dCSRmat *A, const char *filename, int size)
 *
 * \brief Write sparse matrix pattern in BMP file format
 *
 * \param A         Pointer to the dCSRmat matrix
 * \param filename  File name
 * \param size      size*size is the picture size for the picture
 *
 * \author Chensong Zhang
 * \date   03/29/2009
 *
 * \note The routine fasp_dcsr_subplot writes pattern of the specified dCSRmat
 *       matrix in uncompressed BMP file format (Windows bitmap) to a binary
 *       file whose name is specified by the character string filename.
 *
 * Each pixel corresponds to one matrix element. The pixel colors have
 * the following meaning:
 *
 *  White    structurally zero element
 *  Blue     positive element
 *  Red      negative element
 *  Brown    nearly zero element
 */
void fasp_dcsr_subplot (const dCSRmat  *A,
                        const char     *filename,
                        int             size)
{
    INT m = A->row, n = A->col, minmn = MIN(m,n);
    int i, j, k;
	char *map; 

    if ( size>minmn ) size = minmn;
	map = (char *)fasp_mem_calloc(size * size, sizeof(char));
    
    printf("Writing matrix pattern to `%s'...\n",filename);
    
    memset((void *)map, 0x0F, size * size);
    
    for (i = 0; i < size; ++i) {
        for (j = A->IA[i]; j < A->IA[i+1]; ++j) {
            if (A->JA[j]<size) {
                k = size*i + A->JA[j];
                if (map[k] != 0x0F)
                    map[k] = 0x0F;
                else if (A->val[j] > 1e-20)
                    map[k] = 0x09; /* bright blue */
                else if (A->val[j] < -1e-20)
                    map[k] = 0x0C; /* bright red */
                else
                    map[k] = 0x06; /* brown */
            } // end if
        } // end for j
    } // end for i
    
    write_bmp16(filename, size, size, map);
    
    fasp_mem_free(map); map = NULL;
}

/**
 * \fn void fasp_dcsr_plot (const dCSRmat *A, const char *fname)
 *
 * \brief Write dCSR sparse matrix pattern in BMP file format
 *
 * \param A       Pointer to the dBSRmat matrix
 * \param fname   File name to plot to
 *
 * \author Chunsheng Feng
 * \date   11/16/2013
 *
 * \note The routine fasp_dcsr_plot writes pattern of the specified dCSRmat
 *       matrix in uncompressed BMP file format (Windows bitmap) to a binary
 *       file whose name is specified by the character string filename.
 *
 * Each pixel corresponds to one matrix element. The pixel colors have
 * the following meaning:
 *
 *  White    structurally zero element
 *  Black    zero element
 *  Blue     positive element
 *  Red      negative element
 *  Brown    nearly zero element
 */
void fasp_dcsr_plot (const dCSRmat  *A,
                     const char     *fname)
{
    FILE *fp;
    INT offset, bmsize, i, j, b;
    INT n = A->col, m = A->row;
    INT size;
    
    INT col;
    REAL val;
    char *map;
    
    size = ( (n+7)/8 )*8;
    
    map = (char *)fasp_mem_calloc(size, sizeof(char));
    
    memset(map, 0x0F, size);
    
    if (!(1 <= m && m <= 32767))
        printf("### ERROR: Invalid num of rows %d! [%s]\n", m, __FUNCTION__);
    
    if (!(1 <= n && n <= 32767))
        printf("### ERROR: Invalid num of cols %d! [%s]\n", n, __FUNCTION__);
    
    fp = fopen(fname, "wb");
    if (fp == NULL) {
        printf("### ERROR: Unable to create `%s'! [%s]\n", fname, __FUNCTION__);
        goto FINISH;
    }
    
    offset = 14 + 40 + 16 * 4;
    bmsize = (4 * n + 31) / 32;
    /* struct BMPFILEHEADER (14 bytes) */
    /* UINT bfType */          put_byte(fp, 'B'); put_byte(fp, 'M');
    /* DWORD bfSize */         put_dword(fp, offset + bmsize * 4);
    /* UINT bfReserved1 */     put_word(fp, 0);
    /* UNIT bfReserved2 */     put_word(fp, 0);
    /* DWORD bfOffBits */      put_dword(fp, offset);
    /* struct BMPINFOHEADER (40 bytes) */
    /* DWORD biSize */         put_dword(fp, 40);
    /* LONG biWidth */         put_dword(fp, n);
    /* LONG biHeight */        put_dword(fp, m);
    /* WORD biPlanes */        put_word(fp, 1);
    /* WORD biBitCount */      put_word(fp, 4);
    /* DWORD biCompression */  put_dword(fp, 0 /* BI_RGB */);
    /* DWORD biSizeImage */    put_dword(fp, 0);
    /* LONG biXPelsPerMeter */ put_dword(fp, 2953 /* 75 dpi */);
    /* LONG biYPelsPerMeter */ put_dword(fp, 2953 /* 75 dpi */);
    /* DWORD biClrUsed */      put_dword(fp, 0);
    /* DWORD biClrImportant */ put_dword(fp, 0);
    /* struct RGBQUAD (16 * 4 = 64 bytes) */
    /* CGA-compatible colors: */
    /* 0x00 = black */         put_dword(fp, 0x000000);
    /* 0x01 = blue */          put_dword(fp, 0x000080);
    /* 0x02 = green */         put_dword(fp, 0x008000);
    /* 0x03 = cyan */          put_dword(fp, 0x008080);
    /* 0x04 = red */           put_dword(fp, 0x800000);
    /* 0x05 = magenta */       put_dword(fp, 0x800080);
    /* 0x06 = brown */         put_dword(fp, 0x808000);
    /* 0x07 = light gray */    put_dword(fp, 0xC0C0C0);
    /* 0x08 = dark gray */     put_dword(fp, 0x808080);
    /* 0x09 = bright blue */   put_dword(fp, 0x0000FF);
    /* 0x0A = bright green */  put_dword(fp, 0x00FF00);
    /* 0x0B = bright cyan */   put_dword(fp, 0x00FFFF);
    /* 0x0C = bright red */    put_dword(fp, 0xFF0000);
    /* 0x0D = bright magenta */put_dword(fp, 0xFF00FF);
    /* 0x0E = yellow */        put_dword(fp, 0xFFFF00);
    /* 0x0F = white */         put_dword(fp, 0xFFFFFF);
    /* pixel data bits */
    b = 0;
    
    ////----------------------------------------------------------------------------------------
    //	for(i=((m+7)/8)*8 - 1; i>=m; i--){
    //		memset(map, 0x0F, size);
    //        for (j = 0; j < size; ++j) {
    //            b <<= 4;
    //            b |= (j < n ? map[j] & 15 : 0);
    //            if (j & 1) put_byte(fp, b);
    //        }
    //	}
    ////----------------------------------------------------------------------------------------
    
    for ( i = A->row-1; i >=0; i-- ) {
        memset(map, 0x0F, size);
        
        for ( j = A->IA[i]; j < A->IA[i+1]; j++ ) {
            col =  A->JA[j];
            val =  A->val[j];
            if (map[col] != 0x0F)
                map[col] = 0x0F;
            else if ( val > 1e-20)
                map[col] = 0x09; /* bright blue */
            else if ( val < -1e-20)
                map[col] = 0x0C; /* bright red */
            else if (val == 0)
                map[col] = 0x00; /* bright red */
            else
                map[col] = 0x06; /* brown */
        } // for j
        
        for (j = 0; j < size; ++j) {
            b <<= 4;
            b |= (j < n ? map[j] & 15 : 0);
            if (j & 1) put_byte(fp, b);
        }
    }
    
    fflush(fp);
    if (ferror(fp)) {
        printf("### ERROR: Write error on `%s'! [%s]\n", fname, __FUNCTION__);
    }

FINISH: if (fp != NULL) fclose(fp);
    
    fasp_mem_free(map); map = NULL;
}

/**
 * \fn void fasp_dbsr_subplot (const dBSRmat *A, const char *filename, int size)
 *
 * \brief Write sparse matrix pattern in BMP file format
 *
 * \param A         Pointer to the dBSRmat matrix
 * \param filename  File name
 * \param size      size*size is the picture size for the picture
 *
 * \author Chunsheng Feng
 * \date   11/16/2013
 *
 * \note The routine fasp_dbsr_subplot writes pattern of the specified dBSRmat
 *       matrix in uncompressed BMP file format (Windows bitmap) to a binary
 *       file whose name is specified by the character string filename.
 *
 * Each pixel corresponds to one matrix element. The pixel colors have
 * the following meaning:
 *
 *  White    structurally zero element
 *  Black    zero element
 *  Blue     positive element
 *  Red      negative element
 *  Brown    nearly zero element
 */
void fasp_dbsr_subplot (const dBSRmat  *A,
                        const char     *filename,
                        int             size)
{
	INT m = A->ROW;
    INT n = A->COL;
	INT nb = A->nb;
	INT nb2 = nb*nb;
    INT offset;
    INT row, col, i, j, k, l, minmn=nb*MIN(m,n);
    REAL val;
    char *map;

    if (size>minmn) size=minmn;
    
    printf("Writing matrix pattern to `%s'...\n",filename);
    
    map = (char *)fasp_mem_calloc(size * size, sizeof(char));
    
    memset((void *)map, 0x0F, size * size);
    
    for ( i = 0; i < size/nb; i++ ) {
        
        for ( j = A->IA[i]; j < A->IA[i+1]; j++ ) {
            for ( k = 0; k < A->nb; k++ ) {
                for ( l = 0; l < A->nb; l++ ) {
                    
                    row = i*nb + k;
                    col =  A->JA[j]*nb + l;
                    val = A->val[ A->JA[j]*nb2 + k*nb + l];
                    
                    if (col<size) {
                        
                        offset = size*row + col;
                        
                        if (map[offset] != 0x0F)
                            map[offset] = 0x0F;
                        else if ( val > 1e-20)
                            map[offset] = 0x09; /* bright blue */
                        else if ( val < -1e-20)
                            map[offset] = 0x0C; /* bright red */
                        else if (val == 0)
                            map[offset] = 0x00; /* bright red */
                        else
                            map[offset] = 0x06; /* brown */
                    } // end if
                }
            }
        }
    }
    
    write_bmp16(filename, size, size, map);
    
    fasp_mem_free(map); map = NULL;
}

/**
 * \fn void fasp_dbsr_plot (const dBSRmat *A, const char *fname)
 *
 * \brief Write dBSR sparse matrix pattern in BMP file format
 *
 * \param A       Pointer to the dBSRmat matrix
 * \param fname   File name
 *
 * \author Chunsheng Feng
 * \date   11/16/2013
 *
 * \note The routine fasp_dbsr_plot writes pattern of the specified dBSRmat
 *       matrix in uncompressed BMP file format (Windows bitmap) to a binary
 *       file whose name is specified by the character string filename.
 *
 * Each pixel corresponds to one matrix element. The pixel colors have
 * the following meaning:
 *
 *  White    structurally zero element
 *  Black    zero element
 *  Blue     positive element
 *  Red      negative element
 *  Brown    nearly zero element
 */
void fasp_dbsr_plot (const dBSRmat  *A,
                     const char     *fname)
{
    FILE *fp;
    INT offset, bmsize, i, j, b;
    INT size;
    INT nb = A->nb;
    INT nb2 = nb*nb;
    INT n = A->COL*A->nb, m = A->ROW*A->nb;
	INT col,k,l;
    REAL val;
    char *map;
    
    size = ( (n+7)/8 )*8;
    
    map = (char *)fasp_mem_calloc(size, sizeof(char));
    
    memset((void *)map, 0x0F, size);
    
    if (!(1 <= m && m <= 32767))
        printf("### ERROR: Invalid num of rows %d! [%s]\n", m, __FUNCTION__);

    if (!(1 <= n && n <= 32767))
        printf("### ERROR: Invalid num of cols %d! [%s]\n", n, __FUNCTION__);

    fp = fopen(fname, "wb");
    if (fp == NULL) {
        printf("### ERROR: Unable to create `%s'! [%s]\n", fname, __FUNCTION__);
        goto FINISH;
    }
    
    offset = 14 + 40 + 16 * 4;
    bmsize = (4 * n + 31) / 32;
    /* struct BMPFILEHEADER (14 bytes) */
    /* UINT bfType */          put_byte(fp, 'B'); put_byte(fp, 'M');
    /* DWORD bfSize */         put_dword(fp, offset + bmsize * 4);
    /* UINT bfReserved1 */     put_word(fp, 0);
    /* UNIT bfReserved2 */     put_word(fp, 0);
    /* DWORD bfOffBits */      put_dword(fp, offset);
    /* struct BMPINFOHEADER (40 bytes) */
    /* DWORD biSize */         put_dword(fp, 40);
    /* LONG biWidth */         put_dword(fp, n);
    /* LONG biHeight */        put_dword(fp, m);
    /* WORD biPlanes */        put_word(fp, 1);
    /* WORD biBitCount */      put_word(fp, 4);
    /* DWORD biCompression */  put_dword(fp, 0 /* BI_RGB */);
    /* DWORD biSizeImage */    put_dword(fp, 0);
    /* LONG biXPelsPerMeter */ put_dword(fp, 2953 /* 75 dpi */);
    /* LONG biYPelsPerMeter */ put_dword(fp, 2953 /* 75 dpi */);
    /* DWORD biClrUsed */      put_dword(fp, 0);
    /* DWORD biClrImportant */ put_dword(fp, 0);
    /* struct RGBQUAD (16 * 4 = 64 bytes) */
    /* CGA-compatible colors: */
    /* 0x00 = black */         put_dword(fp, 0x000000);
    /* 0x01 = blue */          put_dword(fp, 0x000080);
    /* 0x02 = green */         put_dword(fp, 0x008000);
    /* 0x03 = cyan */          put_dword(fp, 0x008080);
    /* 0x04 = red */           put_dword(fp, 0x800000);
    /* 0x05 = magenta */       put_dword(fp, 0x800080);
    /* 0x06 = brown */         put_dword(fp, 0x808000);
    /* 0x07 = light gray */    put_dword(fp, 0xC0C0C0);
    /* 0x08 = dark gray */     put_dword(fp, 0x808080);
    /* 0x09 = bright blue */   put_dword(fp, 0x0000FF);
    /* 0x0A = bright green */  put_dword(fp, 0x00FF00);
    /* 0x0B = bright cyan */   put_dword(fp, 0x00FFFF);
    /* 0x0C = bright red */    put_dword(fp, 0xFF0000);
    /* 0x0D = bright magenta */put_dword(fp, 0xFF00FF);
    /* 0x0E = yellow */        put_dword(fp, 0xFFFF00);
    /* 0x0F = white */         put_dword(fp, 0xFFFFFF);
    /* pixel data bits */
    b = 0;
    
    ////----------------------------------------------------------------------------------------
    //	for(i=size-1; i>=m; i--){
    //		memset(map, 0x0F, size);
    //        for (j = 0; j < size; ++j) {
    //            b <<= 4;
    //            b |= (j < n ? map[j] & 15 : 0);
    //            if (j & 1) put_byte(fp, b);
    //        }
    //	}
    ////----------------------------------------------------------------------------------------
    
    for ( i = A->ROW-1; i >=0; i-- ) {
        
        for ( k = A->nb-1; k >=0; k-- ) {
            
            memset(map, 0x0F, size);
            
            for ( j = A->IA[i]; j < A->IA[i+1]; j++ ) {
                for ( l = 0; l < A->nb; l++ ) {
                    
                    col =  A->JA[j]*nb + l;
                    val = A->val[ A->JA[j]*nb2 + k*nb + l];
                    
                    if (map[col] != 0x0F)
                        map[col] = 0x0F;
                    else if ( val > 1e-20)
                        map[col] = 0x09; /* bright blue */
                    else if ( val < -1e-20)
                        map[col] = 0x0C; /* bright red */
                    else if (val == 0)
                        map[col] = 0x00; /* bright red */
                    else
                        map[col] = 0x06; /* brown */
                } // for l
            } // for j
            
            
            for (j = 0; j < size; ++j) {
                b <<= 4;
                b |= (j < n ? map[j] & 15 : 0);
                if (j & 1) put_byte(fp, b);
            }
            
        }
    }
    
    fflush(fp);
    if (ferror(fp)) {
        printf("### ERROR: Write error on `%s'! [%s]\n", fname, __FUNCTION__);
    }
    
FINISH: if (fp != NULL) fclose(fp);
    
    fasp_mem_free(map); map = NULL;
}

/*!
 * \fn void fasp_grid2d_plot (pgrid2d pg, int level)
 *
 * \brief Output grid to a EPS file
 *
 * \param pg     Pointer to grid in 2d
 * \param level  Number of levels
 *
 * \author Chensong Zhang
 * \date   03/29/2009
 */
void fasp_grid2d_plot (pgrid2d  pg,
                       int      level)
{
    FILE *datei;
    char buf[120];
    INT i;
    REAL xmid,ymid,xc,yc;
    
    sprintf(buf,"Grid_ref_level%d.eps",level);
    datei = fopen(buf,"w");
    if(datei==NULL) {
        printf("Opening file %s fails!\n", buf);
        return;
    }
    
    fprintf(datei, "%%!PS-Adobe-2.0-2.0 EPSF-2.0\n");
    fprintf(datei, "%%%%BoundingBox: 0 0 550 550\n");
    fprintf(datei, "25 dup translate\n");
    fprintf(datei, "%f setlinewidth\n",0.2);
    fprintf(datei, "/Helvetica findfont %f scalefont setfont\n",64.0*pow(0.5,level));
    fprintf(datei, "/b{0 setgray} def\n");
    fprintf(datei, "/r{1.0 0.6 0.6  setrgbcolor} def\n");
    fprintf(datei, "/u{0.1 0.7 0.1  setrgbcolor} def\n");
    fprintf(datei, "/d{0.1 0.1 1.0  setrgbcolor} def\n");
    fprintf(datei, "/cs{closepath stroke} def\n");
    fprintf(datei, "/m{moveto} def\n");
    fprintf(datei, "/l{lineto} def\n");
    
    fprintf(datei,"b\n");
    for (i=0; i<pg->triangles; ++i) {
        xc = (pg->p[pg->t[i][0]][0]+pg->p[pg->t[i][1]][0]+pg->p[pg->t[i][2]][0])*150.0;
        yc = (pg->p[pg->t[i][0]][1]+pg->p[pg->t[i][1]][1]+pg->p[pg->t[i][2]][1])*150.0;
        
        xmid = pg->p[pg->t[i][0]][0]*450.0;
        ymid = pg->p[pg->t[i][0]][1]*450.0;
        fprintf(datei,"%.1f %.1f m ",0.9*xmid+0.1*xc,0.9*ymid+0.1*yc);
        xmid = pg->p[pg->t[i][1]][0]*450.0;
        ymid = pg->p[pg->t[i][1]][1]*450.0;
        fprintf(datei,"%.1f %.1f l ",0.9*xmid+0.1*xc,0.9*ymid+0.1*yc);
        xmid = pg->p[pg->t[i][2]][0]*450.0;
        ymid = pg->p[pg->t[i][2]][1]*450.0;
        fprintf(datei,"%.1f %.1f l ",0.9*xmid+0.1*xc,0.9*ymid+0.1*yc);
        fprintf(datei,"cs\n");
    }
    fprintf(datei,"r\n");
    for(i=0; i<pg->vertices; ++i) {
        xmid = pg->p[i][0]*450.0;
        ymid = pg->p[i][1]*450.0;
        fprintf(datei,"%.1f %.1f m ",xmid,ymid);
        fprintf(datei,"(%d) show\n ",i);
    }
    fprintf(datei,"u\n");
    for(i=0; i<pg->edges; ++i) {
        xmid = 0.5*(pg->p[pg->e[i][0]][0]+pg->p[pg->e[i][1]][0])*450.0;
        ymid = 0.5*(pg->p[pg->e[i][0]][1]+pg->p[pg->e[i][1]][1])*450.0;
        fprintf(datei,"%.1f %.1f m ",xmid,ymid);
        fprintf(datei,"(%d) show\n ",i);
        
        xmid = pg->p[pg->e[i][0]][0]*450.0;
        ymid = pg->p[pg->e[i][0]][1]*450.0;
        fprintf(datei,"%.1f %.1f m ",xmid,ymid);
        xmid = pg->p[pg->e[i][1]][0]*450.0;
        ymid = pg->p[pg->e[i][1]][1]*450.0;
        fprintf(datei,"%.1f %.1f l ",xmid,ymid);
        fprintf(datei,"cs\n");
    }
    fprintf(datei,"d\n");
    for(i=0; i<pg->triangles; ++i) {
        xmid = (pg->p[pg->t[i][0]][0]+pg->p[pg->t[i][1]][0]+pg->p[pg->t[i][2]][0])*150.0;
        ymid = (pg->p[pg->t[i][0]][1]+pg->p[pg->t[i][1]][1]+pg->p[pg->t[i][2]][1])*150.0;
        fprintf(datei,"%.1f %.1f m ",xmid,ymid);
        fprintf(datei,"(%d) show\n ",i);
    }
    fprintf(datei, "showpage\n");
    fclose(datei);
}

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

/*!
 * \fn static void put_byte (FILE *fp, const int c)
 *
 * \brief Write to byte to file
 *
 * \param fp pointer to file
 * \param c  byte to write
 *
 */
static void put_byte (FILE       *fp,
                      const int   c)
{
    fputc(c, fp);
    return;
}

/*!
 * \fn static void put_word (FILE *fp, const int w)
 *
 * \brief Write to word to file
 *
 * \param fp pointer to file
 * \param w  word to write
 *
 */
static void put_word (FILE       *fp,
                      const int   w)
{ /* big endian */
    put_byte(fp, w);
    put_byte(fp, w >> 8);
    return;
}

/*!
 * \fn static void put_dword (FILE *fp, const int d)
 *
 * \brief Write to REAL-word to file
 *
 * \param fp pointer to file
 * \param d  REAL-word to write
 *
 */
static void put_dword (FILE       *fp,
                       const int   d)
{ /* big endian */
    put_word(fp, d);
    put_word(fp, d >> 16);
    return;
}

/*!
 * \fn static int write_bmp16 (const char *fname, const int m, const int n,
 *                             const char map[])
 *
 * \brief Write to BMP file
 *
 * \param fname  char for filename
 * \param m      number of pixels for height
 * \param n      number of pixels for weight
 * \param map    picture for BMP
 * \return       1 if succeed, 0 if fail
 *
 * \note
 *  write_bmp16 - write 16-color raster image in BMP file format
 *
 *  DESCRIPTION
 *
 *  The routine write_bmp16 writes 16-color raster image in
 *  uncompressed BMP file format (Windows bitmap) to a binary file whose
 *  name is specified by the character string fname.
 *
 *  The parameters m and n specify, respectively, the number of rows and
 *  the numbers of columns (i.e. height and width) of the raster image.
 *
 *  The character array map has m*n elements. Elements map[0, ..., n-1]
 *  correspond to the first (top) scanline, elements map[n, ..., 2*n-1]
 *  correspond to the second scanline, etc.
 *
 *  Each element of the array map specifies a color of the corresponding
 *  pixel as 8-bit binary number XXXXIRGB, where four high-order bits (X)
 *  are ignored, I is high intensity bit, R is red color bit, G is green
 *  color bit, and B is blue color bit. Thus, all 16 possible colors are
 *  coded as following hexadecimal numbers:
 *
 *     0x00 = black         0x08 = dark gray
 *     0x01 = blue          0x09 = bright blue
 *     0x02 = green         0x0A = bright green
 *     0x03 = cyan          0x0B = bright cyan
 *     0x04 = red           0x0C = bright red
 *     0x05 = magenta       0x0D = bright magenta
 *     0x06 = brown         0x0E = yellow
 *     0x07 = light gray    0x0F = white
 *
 *  RETURNS
 *
 *  If no error occured, the routine returns zero; otherwise, it prints
 *  an appropriate error message and returns non-zero.
 *  This code is modified from graphical output in
 *         GLPK (GNU Linear Programming Kit).
 *
 *  Note:
 *
 *  GLPK is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  GLPK is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 *  or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with GLPK. If not, see <http://www.gnu.org/licenses/>.
 */
static int write_bmp16 (const char   *fname,
                        const int     m,
                        const int     n,
                        const char    map[])
{
    FILE *fp;
    int offset, bmsize, i, j, b, ret = 1;
    
    if (!(1 <= m && m <= 32767))
        printf("### ERROR: %s invalid height %d\n", __FUNCTION__, m);
    
    if (!(1 <= n && n <= 32767))
        printf("### ERROR: %s invalid width %d\n", __FUNCTION__, n);
    
    fp = fopen(fname, "wb");
    if (fp == NULL) {
        printf("### ERROR: %s unable to create `%s'\n", __FUNCTION__, fname);
        ret = 0;
        goto FINISH;
    }
    offset = 14 + 40 + 16 * 4;
    bmsize = (4 * n + 31) / 32;
    /* struct BMPFILEHEADER (14 bytes) */
    /* UINT bfType */          put_byte(fp, 'B'); put_byte(fp, 'M');
    /* DWORD bfSize */         put_dword(fp, offset + bmsize * 4);
    /* UINT bfReserved1 */     put_word(fp, 0);
    /* UINT bfReserved2 */     put_word(fp, 0);
    /* DWORD bfOffBits */      put_dword(fp, offset);
    /* struct BMPINFOHEADER (40 bytes) */
    /* DWORD biSize */         put_dword(fp, 40);
    /* LONG biWidth */         put_dword(fp, n);
    /* LONG biHeight */        put_dword(fp, m);
    /* WORD biPlanes */        put_word(fp, 1);
    /* WORD biBitCount */      put_word(fp, 4);
    /* DWORD biCompression */  put_dword(fp, 0 /* BI_RGB */);
    /* DWORD biSizeImage */    put_dword(fp, 0);
    /* LONG biXPelsPerMeter */ put_dword(fp, 2953 /* 75 dpi */);
    /* LONG biYPelsPerMeter */ put_dword(fp, 2953 /* 75 dpi */);
    /* DWORD biClrUsed */      put_dword(fp, 0);
    /* DWORD biClrImportant */ put_dword(fp, 0);
    /* struct RGBQUAD (16 * 4 = 64 bytes) */
    /* CGA-compatible colors: */
    /* 0x00 = black */         put_dword(fp, 0x000000);
    /* 0x01 = blue */          put_dword(fp, 0x000080);
    /* 0x02 = green */         put_dword(fp, 0x008000);
    /* 0x03 = cyan */          put_dword(fp, 0x008080);
    /* 0x04 = red */           put_dword(fp, 0x800000);
    /* 0x05 = magenta */       put_dword(fp, 0x800080);
    /* 0x06 = brown */         put_dword(fp, 0x808000);
    /* 0x07 = light gray */    put_dword(fp, 0xC0C0C0);
    /* 0x08 = dark gray */     put_dword(fp, 0x808080);
    /* 0x09 = bright blue */   put_dword(fp, 0x0000FF);
    /* 0x0A = bright green */  put_dword(fp, 0x00FF00);
    /* 0x0B = bright cyan */   put_dword(fp, 0x00FFFF);
    /* 0x0C = bright red */    put_dword(fp, 0xFF0000);
    /* 0x0D = bright magenta */put_dword(fp, 0xFF00FF);
    /* 0x0E = yellow */        put_dword(fp, 0xFFFF00);
    /* 0x0F = white */         put_dword(fp, 0xFFFFFF);
    /* pixel data bits */
    b = 0;
    for (i = m - 1; i >= 0; i--) {
        for (j = 0; j < ((n + 7) / 8) * 8; ++j) {
            b <<= 4;
            b |= (j < n ? map[i * n + j] & 15 : 0);
            if (j & 1) put_byte(fp, b);
        }
    }
    fflush(fp);
    
    if (ferror(fp)) {
        printf("### ERROR: %s write error on `%s'\n", __FUNCTION__, fname);
        ret = 0;
    }
    
FINISH: if (fp != NULL) fclose(fp);
    return ret;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
