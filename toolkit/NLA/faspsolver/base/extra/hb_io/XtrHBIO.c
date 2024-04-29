# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <string.h>

# include "hb_io.h"

/******************************************************************************/

int ch_eqi ( char ch1, char ch2 )

/******************************************************************************/
/*
  Purpose:

    CH_EQI is TRUE (1) if two characters are equal, disregarding case.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    13 June 2003

  Author:

    John Burkardt

  Parameters:

    Input, char CH1, CH2, the characters to compare.

    Output, int CH_EQI, is TRUE (1) if the two characters are equal,
    disregarding case and FALSE (0) otherwise.
*/
{
  int value;

  if ( 97 <= ch1 && ch1 <= 122 )
  {
    ch1 = ch1 - 32;
  }
  if ( 97 <= ch2 && ch2 <= 122 )
  {
    ch2 = ch2 - 32;
  }
  if ( ch1 == ch2 )
  {
    value = 1;
  }
  else
  {
    value = 0;
  }
  return value;
}
/******************************************************************************/

int ch_is_digit ( char c )

/******************************************************************************/
/*
  Purpose:

    CH_IS_DIGIT returns TRUE if a character is a decimal digit.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    22 January 2014

  Author:

    John Burkardt

  Parameters:

    Input, char C, the character to be analyzed.

    Output, int CH_IS_DIGIT, is TRUE if C is a digit.
*/
{
  int value;

  if ( '0' <= c && c <= '9' )
  {
    value = 1;
  }
  else
  {
    value = 0;
  }
  return value;
}
/******************************************************************************/

int ch_is_format_code ( char c )

/******************************************************************************/
/*
  Purpose:

    CH_IS_FORMAT_CODE returns TRUE (1) if a character is a FORTRAN format code.

  Discussion:

    The format codes accepted here are not the only legal format
    codes in FORTRAN90.  However, they are more than sufficient
    for my needs!

  Table:

    A  Character
    B  Binary digits
    D  Real number, exponential representation
    E  Real number, exponential representation
    F  Real number, fixed point
    G  General format
    I  Integer
    L  Logical variable
    O  Octal digits
    Z  Hexadecimal digits
    *  Free format

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    22 January 2014

  Author:

    John Burkardt

  Parameters:

    Input, char C, the character to be analyzed.

    Output, int CH_IS_FORMAT_CODE, is 1 if C is a FORTRAN format code.
*/
{
  int value;

  if ( ch_eqi ( c, 'A' ) ) 
  {
    value = 1;
  }
  else if ( ch_eqi ( c, 'B' ) )
  {
    value = 1;
  }
  else if ( ch_eqi ( c, 'D' ) )
  {
    value = 1;
  }
  else if ( ch_eqi ( c, 'E' ) )
  {
    value = 1;
  }
  else if ( ch_eqi ( c, 'F' ) )
  {
    value = 1;
  }
  else if ( ch_eqi ( c, 'G' ) )
  {
    value = 1;
  }
  else if ( ch_eqi ( c, 'I' ) )
  {
    value = 1;
  }
  else if ( ch_eqi ( c, 'L' ) )
  {
    value = 1;
  }
  else if ( ch_eqi ( c, 'O' ) )
  {
    value = 1;
  }
  else if ( ch_eqi ( c, 'Z' ) )
  {
    value = 1;
  }
  else if ( c == '*' )
  {
    value = 1;
  }
  else
  {
    value = 0;
  }
  return value;
}
/******************************************************************************/

int ch_to_digit ( char ch )

/******************************************************************************/
/*
  Purpose:

    CH_TO_DIGIT returns the integer value of a base 10 digit.

  Example:

     CH  DIGIT
    ---  -----
    '0'    0
    '1'    1
    ...  ...
    '9'    9
    ' '    0
    'X'   -1

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    13 June 2003

  Author:

    John Burkardt

  Parameters:

    Input, char CH, the decimal digit, '0' through '9' or blank are legal.

    Output, int CH_TO_DIGIT, the corresponding integer value.  If the
    character was 'illegal', then DIGIT is -1.
*/
{
  int digit;

  if ( '0' <= ch && ch <= '9' )
  {
    digit = ch - '0';
  }
  else if ( ch == ' ' )
  {
    digit = 0;
  }
  else
  {
    digit = -1;
  }

  return digit;
}
/******************************************************************************/

void hb_exact_read ( FILE *input, int nrow, int nrhs, int rhscrd, 
  char *rhsfmt, char *rhstyp, double exact[] )

/******************************************************************************/
/*
  Purpose:

    HB_EXACT_READ reads the exact solution vectors in an HB file.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    22 January 2014

  Author:

    John Burkardt

  Reference:

    Iain Duff, Roger Grimes, John Lewis,
    User's Guide for the Harwell-Boeing Sparse Matrix Collection,
    October 1992.

  Parameters:

    Input, FILE *INPUT, the unit from which data is read.

    Input, int NROW, the number of rows or variables.

    Input, int NRHS, the number of right hand sides.

    Input, int RHSCRD, the number of lines in the file for
    right hand sides.

    Input, char *RHSFMT, the 20 character format for reading values
    of the right hand side.

    Input, char *RHSTYP, the 3 character right hand side type.
    First character is F for full storage or M for same as matrix.
    Second character is G if starting "guess" vectors are supplied.
    Third character is X if exact solution vectors are supplied.
    Ignored if NRHS = 0.

    Output, double EXACT[NROW*NRHS], the exact solution vectors.
*/
{
# define LINE_MAX 255

  char code;
  char *error;
  int i;
  int j;
  int jhi;
  int jlo;
  int khi;
  int klo;
  char line[LINE_MAX];
  int line_len;
  int line_num;
  int m;
  int r;
  char *s;
  int w;

  if ( 0 < rhscrd )
  {
    if ( rhstyp[2] == 'X' )
    {
      s_to_format ( rhsfmt, &r, &code, &w, &m );

      line_num = 1 + ( nrow * nrhs - 1 ) / r;

      jhi = 0;
      for ( i = 1; i <= line_num; i++ )
      {
        error = fgets ( line, LINE_MAX, input );

        if ( !error )
        {
          fprintf ( stderr, "\n" );
          fprintf ( stderr, "HB_EXACT_READ - Fatal error!\n" );
          fprintf ( stderr, "  I/O error.\n" );
          exit ( 1 );
        }

        line_len = strlen ( line );
        line[line_len-1] = '\0';

        jlo = jhi + 1;
        jhi = i4_min ( jlo + r - 1, nrow * nrhs );

        khi = 0;
        for ( j = jlo; j <= jhi; j++ )
        {
          klo = khi + 1;
          khi = i4_min ( klo + w - 1, strlen ( line ) );
          s = s_substring ( line, klo, khi );
          exact[j-1] = atof ( s );
        }
      }
    }
  }

  return;
# undef LINE_MAX
}
/******************************************************************************/

void hb_exact_write ( FILE *output, int nrow, int nrhs, int rhscrd, 
  char *rhsfmt, char *rhstyp, double exact[] )

/******************************************************************************/
/*
  Purpose:

    HB_EXACT_WRITE writes the exact solution vectors to an HB file.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    22 January 2014

  Author:

    John Burkardt

  Reference:

    Iain Duff, Roger Grimes, John Lewis,
    User's Guide for the Harwell-Boeing Sparse Matrix Collection,
    October 1992.

  Parameters:

    Input, FILE *OUTPUT, the unit to which data is written.

    Input, int NROW, the number of rows or variables.

    Input, int NRHS, the number of right hand sides.

    Input, int RHSCRD, the number of lines in the file for
    right hand sides.

    Input, char *RHSFMT, the 20 character format for reading values
    of the right hand side.

    Input, char *RHSTYP, the 3 character right hand side type.
    First character is F for full storage or M for same as matrix.
    Second character is G if starting "guess" vectors are supplied.
    Third character is X if exact solution vectors are supplied.
    Ignored if NRHS = 0.

    Input, double EXACT[NROW*NRHS], the exact solution vectors.
*/
{
  char code;
  char format[80];
  int i;
  int j;
  int jhi;
  int jlo;
  int line_num;
  int m;
  int r;
  int w;

  if ( 0 < rhscrd )
  {
    if ( rhstyp[2] == 'X' )
    {
      s_to_format ( rhsfmt, &r, &code, &w, &m );

      sprintf ( format, "%%%dg", w );

      line_num = 1 + ( nrow * nrhs - 1 ) / r;

      jhi = 0;
      for ( i = 1; i <= line_num; i++ )
      {
        jlo = jhi + 1;
        jhi = i4_min ( jlo + r - 1, nrow * nrhs );

        for ( j = jlo; j <= jhi; j++ )
        {
          fprintf ( output, format, exact[j-1] );
        }
        fprintf ( output, "\n" );
      }
    }
  }
  return;
}
/******************************************************************************/

void hb_file_read ( FILE *input, char **title, char **key, int *totcrd, 
  int *ptrcrd, int *indcrd, int *valcrd, int *rhscrd, char **mxtype, int *nrow, 
  int *ncol, int *nnzero, int *neltvl, char **ptrfmt, char **indfmt, 
  char **valfmt, char **rhsfmt, char **rhstyp, int *nrhs, int *nrhsix, 
  int **colptr, int **rowind, double **values, double **rhsval, int **rhsptr, 
  int **rhsind, double **rhsvec, double **guess, double **exact )

/******************************************************************************/
/*
  Purpose:

    HB_FILE_READ reads an HB file.

  Discussion:

    This routine reads all the information from an HB file.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    22 January 2014

  Author:

    John Burkardt

  Reference:

    Iain Duff, Roger Grimes, John Lewis,
    User's Guide for the Harwell-Boeing Sparse Matrix Collection,
    October 1992.

  Parameters:

    Input, FILE *INPUT, the unit from which the data is read.

    Output, char *TITLE, a 72 character title for the matrix.

    Output, char *KEY, an 8 character identifier for the matrix.

    Output, int *TOTCRD, the total number of lines of data.

    Output, int *PTRCRD, the number of input lines for pointers.

    Output, int *INDCRD, the number of input lines for row indices.

    Output, int *VALCRD, the number of input lines for numerical values.

    Output, int *RHSCRD, the number of input lines for right hand sides.

    Output, char *MXTYPE, the 3 character matrix type.
    First character is R for Real, C for complex, P for pattern only.
    Second character is S for symmetric, U for unsymmetric, H for
      Hermitian, Z for skew symmetric, R for rectangular.
    Third character is A for assembled and E for unassembled
      finite element matrices.

    Output, int *NROW, the number of rows or variables.

    Output, int *NCOL, the number of columns or elements.

    Output, int *NNZERO.  In the case of assembled sparse matrices,
    this is the number of nonzeroes.  In the case of unassembled finite
    element matrices, in which the right hand side vectors are also
    stored as unassembled finite element vectors, this is the total
    number of entries in a single unassembled right hand side vector.

    Output, int *NELTVL, the number of finite element matrix entries,
    set to 0 in the case of assembled matrices.

    Output, char *PTRFMT, the 16 character format for reading pointers.

    Output, char *INDFMT, the 16 character format for reading indices.

    Output, char *VALFMT, the 20 character format for reading values.

    Output, char *RHSFMT, the 20 character format for reading values
    of the right hand side.

    Output, char *RHSTYP, the 3 character right hand side type.
    First character is F for full storage or M for same as matrix.
    Second character is G if starting "guess" vectors are supplied.
    Third character is X if exact solution vectors are supplied.

    Output, int *NRHS, the number of right hand sides.

    Output, int *NRHSIX, the number of entries of storage for right
    hand side values, in the case where RHSTYP[0] = 'M' and
    MXTYPE[2] = 'A'.

    Output, int COLPTR[NCOL+1], COLPTR[I-1] points to the location of
    the first entry of column I in the sparse matrix structure.

    If MXTYPE[2] == 'A':

      Output, int ROWIND[NNZERO], the row index of each item.

    If MXTYPE[2] == 'F':

      Output, int ROWIND[NELTVL], the row index of each item.

    If RHSTYP[0] == 'F':

      Output, double RHSVAL[NROW*NRHS], contains NRHS dense right hand
      side vectors.

      Output, int RHSPTR[], is not used.

      Output, int RHSIND[], is not used.

      Output, int RHSVEC[], is not used.

    If RHSTYP[0] = 'M' and MXTYPE[2] = 'A':

      Output, double RHSVAL[], is not used.

      Output, int RHSPTR[NRHS+1], RHSPTR[I-1] points to the location of
      the first entry of right hand side I in the sparse right hand
      side vector.

      Output, int RHSIND[NRHSIX], indicates, for each entry of
      RHSVEC, the corresponding row index.

      Output, double RHSVEC[NRHSIX], contains the value of the right hand
      side entries.

    If RHSTYP[0] = 'M' and MXTYPE[2] = 'E':

      Output, double RHSVAL[NNZERO*NRHS], contains NRHS unassembled
      finite element vector right hand sides.

      Output, int RHSPTR[], is not used.

      Output, int RHSIND[], is not used.

      Output, double RHSVEC[], is not used.

    Output, double GUESS[NROW*NRHS], the starting guess vectors.

    Output, double EXACT[NROW*NRHS], the exact solution vectors.
*/
{
/*
  Read the header block.
*/
  hb_header_read ( input, title, key, totcrd, ptrcrd, indcrd, 
    valcrd, rhscrd, mxtype, nrow, ncol, nnzero, neltvl, ptrfmt, indfmt, 
    valfmt, rhsfmt, rhstyp, nrhs, nrhsix );
/*
  Read the matrix structure.
*/
  if ( 0 < ptrcrd )
  {
    if ( (*colptr) )
    {
      free ( (*colptr) );
    }
    (*colptr) = ( int * ) malloc ( (*(ncol)+1) * sizeof ( int ) );
  }

  if ( 0 < indcrd )
  {
    if ( (*rowind) )
    {
      free ( (*rowind) );
    }

    if ( (*mxtype)[2] == 'A' )
    {
      (*rowind) = ( int * ) malloc ( *nnzero * sizeof ( int ) );
    }
    else if ( (*mxtype)[2] == 'E' )
    {
      (*rowind) = ( int * ) malloc ( *neltvl * sizeof ( int ) );
    }
    else
    {
      fprintf ( stderr, "\n" );
      fprintf ( stderr, "HB_FILE_READ - Fatal error!\n" );
      fprintf ( stderr, "  Illegal value of MXTYPE character 3!\n" );
      exit ( 1 );
    }

  }

  hb_structure_read ( input, *ncol, *mxtype, *nnzero, *neltvl, 
    *ptrcrd, *ptrfmt, *indcrd, *indfmt, *colptr, *rowind );
/*
  Read the matrix values.
*/
  if ( 0 < valcrd )
  {
    if ( *values )
    {
      free ( (*values) );
    }

    if ( (*mxtype)[2] == 'A' )
    {
      *values = ( double * ) malloc ( *nnzero * sizeof ( double ) );
    }
    else if ( (*mxtype)[2] == 'E' )
    {
      *values = ( double * ) malloc ( *neltvl * sizeof ( double ) );
    }
    else
    {
      fprintf ( stderr, "\n" );
      fprintf ( stderr, "HB_FILE_READ - Fatal error!\n" );
      fprintf ( stderr, "  Illegal value of MXTYPE character 3!\n" );
      exit ( 1 );
    }

    hb_values_read ( input, *valcrd, *mxtype, *nnzero, *neltvl, 
      *valfmt, *values );
  }
/*
  Read the right hand sides.
*/
  if ( 0 < rhscrd )
  {
    if ( (*rhstyp)[0] == 'F' )
    {
      if ( *rhsval )
      {
        free ( *rhsval );
      }

      *rhsval = ( double * ) malloc ( (*nrow) * (*nrhs) * sizeof ( double ) );
    }
    else if ( (*rhstyp)[0] == 'M' && (*mxtype)[2] == 'A' )
    {
      if ( *rhsptr )
      {
        free ( *rhsptr );
      }

      *rhsptr = ( int * ) malloc ( (*nrhs+1) * sizeof ( int ) );;

      if ( *rhsind )
      {
        free ( *rhsind );
      }

      *rhsind = ( int * ) malloc ( *nrhsix * sizeof ( int ) );;

      if ( *rhsvec )
      {
        free ( *rhsvec );
      }

      *rhsvec = ( double * ) malloc ( (*nrhsix) * sizeof ( double ) );
    }
    else if ( (*rhstyp)[0] == 'M' && (*mxtype)[2] == 'E' )
    {
      if ( *rhsval )
      {
        free ( *rhsval );
      }

      *rhsval = ( double * ) malloc ( (*nnzero) * (*nrhs) * sizeof ( double ) );
    }
    else
    {
      fprintf ( stderr, "\n" );
      fprintf ( stderr, "HB_FILE_READ - Fatal error!\n" );
      fprintf ( stderr, "  Illegal combination of RHSTYP character 1\n" );
      fprintf ( stderr, "  and MXTYPE character 3!\n" );
      exit ( 1 );
    }

    hb_rhs_read ( input, *nrow, *nnzero, *nrhs, *nrhsix, 
      *rhscrd, *ptrfmt, *indfmt, *rhsfmt, *mxtype, *rhstyp, *rhsval, 
      *rhsind, *rhsptr, *rhsvec );
/*
  Read the starting guesses.
*/
    if ( (*rhstyp)[1] == 'G' )
    {
      if ( *guess )
      {
        free ( *guess );
      }

      *guess = ( double * ) malloc ( (*nrow) * (*nrhs) * sizeof ( double ) );

      hb_guess_read ( input, *nrow, *nrhs, *rhscrd, *rhsfmt, *rhstyp, *guess );
    }
/*
  Read the exact solutions.
*/
    if ( (*rhstyp)[2] == 'X' )
    {
      if ( *exact )
      {
        free ( *exact );
      }

      *exact = ( double * ) malloc ( (*nrow) * (*nrhs) * sizeof ( double ) );

      hb_exact_read ( input, *nrow, *nrhs, *rhscrd, *rhsfmt, *rhstyp, *exact );
    }
  }

  return;
}
/******************************************************************************/

void hb_file_write ( FILE *output, char *title, char *key, int totcrd, 
  int ptrcrd, int indcrd, int valcrd, int rhscrd, char *mxtype, int nrow, 
  int ncol, int nnzero, int neltvl, char *ptrfmt, char *indfmt, char *valfmt, 
  char *rhsfmt, char *rhstyp, int nrhs, int nrhsix, int colptr[],
  int rowind[], double values[], double rhsval[], int rhsptr[], int rhsind[], 
  double rhsvec[], double guess[], double exact[] )

/******************************************************************************/
/*
  Purpose:

    HB_FILE_WRITE writes an HB file.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    22 January 2014

  Author:

    John Burkardt

  Reference:

    Iain Duff, Roger Grimes, John Lewis,
    User's Guide for the Harwell-Boeing Sparse Matrix Collection,
    October 1992.

  Parameters:

    Input, ofsteam &OUTPUT, the unit to which data is written.

    Input, char *TITLE, a 72 character title for the matrix.

    Input, char *KEY, an 8 character identifier for the matrix.

    Input, int TOTCRD, the total number of lines of data.

    Input, int PTRCRD, the number of input lines for pointers.

    Input, int INDCRD, the number of input lines for row indices.

    Input, int VALCRD, the number of input lines for numerical values.

    Input, int RHSCRD, the number of input lines for right hand sides.

    Input, char *MXTYPE, the 3 character matrix type.
    First character is R for Real, C for complex, P for pattern only.
    Second character is S for symmetric, U for unsymmetric, H for
      Hermitian, Z for skew symmetric, R for rectangular.
    Third character is A for assembled and E for unassembled
      finite element matrices.

    Input, int NROW, the number of rows or variables.

    Input, int NCOL, the number of columns or elements.

    Input, int NNZERO.  In the case of assembled sparse matrices,
    this is the number of nonzeroes.  In the case of unassembled finite
    element matrices, in which the right hand side vectors are also
    stored as unassembled finite element vectors, this is the total
    number of entries in a single unassembled right hand side vector.

    Input, int NELTVL, the number of finite element matrix entries,
    set to 0 in the case of assembled matrices.

    Input, char *PTRFMT, the 16 character format for reading pointers.

    Input, char *INDFMT, the 16 character format for reading indices.

    Input, char *VALFMT, the 20 character format for reading values.

    Input, char *RHSFMT, the 20 character format for reading values
    of the right hand side.

    Input, char *RHSTYP, the 3 character right hand side type.
    First character is F for full storage or M for same as matrix.
    Second character is G if starting "guess" vectors are supplied.
    Third character is X if exact solution vectors are supplied.
    Ignored if NRHS = 0.

    Input, int NRHS, the number of right hand sides.

    Input, int NRHSIX, the number of row indices (set to 0
    in the case of unassembled matrices.)  Ignored if NRHS = 0.

    Input, int COLPTR[NCOL+1], COLPTR(I) points to the location of
    the first entry of column I in the sparse matrix structure.

    If MXTYPE[2] == 'A':

      Input, int ROWIND[NNZERO], the row index of each item.

      Input, double VALUES[NNZERO], the nonzero values of the matrix.

    If MXTYPE[2] == 'E':

      Input, int ROWIND[NELTVL], the row index of each item.

      Input, double VALUES[NELTVL], the nonzero values of the matrix.

    If RHSTYP[0] == 'F':

      Input, double RHSVAL[NROW*NRHS], contains NRHS dense right hand
      side vectors.

      Input, int RHSPTR[], is not used.

      Input, int RHSIND[], is not used.

      Input, double RHSVEC[], is not used.

    If RHSTYP[0] = 'M' and MXTYPE[2] = 'A':

      Input, double RHSVAL[], is not used.

      Input, int RHSPTR[NRHS+1], RHSPTR(I) points to the location of
      the first entry of right hand side I in the sparse right hand
      side vector.

      Input, int RHSIND[NRHSIX], indicates, for each entry of
      RHSVEC, the corresponding row index.

      Input, double RHSVEC[NRHSIX], contains the value of the right hand
      side entries.

    If RHSTYP[0] = 'M' and MXTYPE[2] = 'E':

      Input, double RHSVAL[NNZERO*NRHS], contains NRHS unassembled
      finite element vector right hand sides.

      Input, int RHSPTR[], is not used.

      Input, int RHSIND[], is not used.

      Input, double RHSVEC[], is not used.

    Input, double GUESS[NROW*NRHS], the starting guess vectors.

    Input, double EXACT[NROW*NRHS], the exact solution vectors.
*/
{
/*
  Write the header block.
*/
  hb_header_write ( output, title, key, totcrd, ptrcrd, indcrd, 
    valcrd, rhscrd, mxtype, nrow, ncol, nnzero, neltvl, ptrfmt, indfmt, 
    valfmt, rhsfmt, rhstyp, nrhs, nrhsix );
/*
  Write the matrix structure.
*/
  hb_structure_write ( output, ncol, mxtype, nnzero, neltvl, 
    ptrfmt, indfmt, colptr, rowind );
/*
  Write the matrix values.
*/
  hb_values_write ( output, valcrd, mxtype, nnzero, neltvl, 
    valfmt, values );
/*
  Write the right hand sides.
*/
  hb_rhs_write ( output, nrow, nnzero, nrhs, nrhsix, 
    rhscrd, ptrfmt, indfmt, rhsfmt, mxtype, rhstyp, rhsval, 
    rhsind, rhsptr, rhsvec );
/*
  Write the starting guesses.
*/
  hb_guess_write ( output, nrow, nrhs, rhscrd, rhsfmt, rhstyp, guess );
/*
  Write the exact solutions.
*/
  hb_exact_write ( output, nrow, nrhs, rhscrd, rhsfmt, rhstyp, exact );

  return;
}
/******************************************************************************/

void hb_guess_read ( FILE *input, int nrow, int nrhs, int rhscrd, 
  char *rhsfmt, char *rhstyp, double guess[] )

/******************************************************************************/
/*
  Purpose:

    HB_GUESS_READ reads the starting guess vectors in an HB file.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    22 January 2014

  Author:

    John Burkardt

  Reference:

    Iain Duff, Roger Grimes, John Lewis,
    User's Guide for the Harwell-Boeing Sparse Matrix Collection,
    October 1992.

  Parameters:

    Input, FILE *INPUT, the unit from which data is read.

    Input, int NROW, the number of rows or variables.

    Input, int NRHS, the number of right hand sides.

    Input, int RHSCRD, the number of lines in the file for
    right hand sides.

    Input, char *RHSFMT, the 20 character format for reading values
    of the right hand side.

    Input, char *RHSTYP, the 3 character right hand side type.
    First character is F for full storage or M for same as matrix.
    Second character is G if starting "guess" vectors are supplied.
    Third character is X if exact solution vectors are supplied.
    Ignored if NRHS = 0.

    Output, double GUESS[NROW*NRHS], the starting guess vectors.
*/
{
# define LINE_MAX 255

  char code;
  char *error;
  int i;
  int j;
  int jhi;
  int jlo;
  int khi;
  int klo;
  char line[LINE_MAX];
  int line_len;
  int line_num;
  int m;
  int r;
  char *s;
  int w;

  if ( 0 < rhscrd )
  {
    if ( rhstyp[1] == 'G' )
    {
      s_to_format ( rhsfmt, &r, &code, &w, &m );

      line_num = 1 + ( nrow * nrhs - 1 ) / r;

      jhi = 0;
      for ( i = 1; i <= line_num; i++ )
      {
        error = fgets ( line, LINE_MAX, input );

        if ( !error )
        {
          fprintf ( stderr, "\n" );
          fprintf ( stderr, "HB_GUESS_READ - Fatal error!\n" );
          fprintf ( stderr, "  I/O error.\n" );
          exit ( 1 );
        }

        line_len = strlen ( line );
        line[line_len-1] = '\0';

        jlo = jhi + 1;
        jhi = i4_min ( jlo + r - 1, nrow * nrhs );

        khi = 0;
        for ( j = jlo; j <= jhi; j++ )
        {
          klo = khi + 1;
          khi = i4_min ( klo + w - 1, strlen ( line ) );
          s = s_substring ( line, klo, khi );
          guess[j-1] = atof ( s );
        }
      }
    }
  }

  return;
# undef LINE_MAX
}
/******************************************************************************/

void hb_guess_write ( FILE *output, int nrow, int nrhs, int rhscrd, 
  char *rhsfmt, char *rhstyp, double guess[] )

/******************************************************************************/
/*
  Purpose:

    HB_GUESS_WRITE writes the starting guess vectors to an HB file.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    22 January 2014

  Author:

    John Burkardt

  Reference:

    Iain Duff, Roger Grimes, John Lewis,
    User's Guide for the Harwell-Boeing Sparse Matrix Collection,
    October 1992.

  Parameters:

    Input, FILE *OUTPUT, the unit to which data is written.

    Input, int NROW, the number of rows or variables.

    Input, int NRHS, the number of right hand sides.

    Input, int RHSCRD, the number of lines in the file for
    right hand sides.

    Input, char *RHSFMT, the 20 character format for reading values
    of the right hand side.

    Input, char *RHSTYP, the 3 character right hand side type.
    First character is F for full storage or M for same as matrix.
    Second character is G if starting "guess" vectors are supplied.
    Third character is X if exact solution vectors are supplied.
    Ignored if NRHS = 0.

    Input, double GUESS[NROW*NRHS], the starting guess vectors.
*/
{
  char code;
  char format[80];
  int i;
  int j;
  int jhi;
  int jlo;
  int line_num;
  int m;
  int r;
  int w;

  if ( 0 < rhscrd )
  {
    if ( rhstyp[1] == 'G' )
    {
      s_to_format ( rhsfmt, &r, &code, &w, &m );
      line_num = 1 + ( nrow * nrhs - 1 ) / r;
      sprintf ( format, "%%%dg", w );

      jhi = 0;
      for ( i = 1; i <= line_num; i++ )
      {
        jlo = jhi + 1;
        jhi = i4_min ( jlo + r - 1, nrow * nrhs );

        for ( j = jlo; j <= jhi; j++ )
        {
          fprintf ( output, format, guess[j-1] );
        }
        fprintf ( output, "\n" );
      }
    }
  }
  return;
}
/******************************************************************************/

void hb_header_print ( char *title, char *key, int totcrd, int ptrcrd, 
  int indcrd, int valcrd, int rhscrd, char *mxtype, int nrow, int ncol, 
  int nnzero, int neltvl, char *ptrfmt, char *indfmt, char *valfmt, 
  char *rhsfmt, char *rhstyp, int nrhs, int nrhsix )

/******************************************************************************/
/*
  Purpose:

    HB_HEADER_PRINT prints the header of an HB file.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    21 January 2014

  Author:

    John Burkardt

  Reference:

    Iain Duff, Roger Grimes, John Lewis,
    User's Guide for the Harwell-Boeing Sparse Matrix Collection,
    October 1992.

  Parameters:

    Input, char *TITLE, a 72 character title for the matrix.

    Input, char *KEY, an 8 character identifier for the matrix.

    Input, int TOTCRD, the total number of lines of data.

    Input, int PTRCRD, the number of input lines for pointers.

    Input, int INDCRD, the number of input lines for row indices.

    Input, int VALCRD, the number of input lines for numerical values.

    Input, int RHSCRD, the number of input lines for right hand sides.

    Input, char *MXTYPE, the 3 character matrix type.
    First character is R for Real, C for complex, P for pattern only.
    Second character is S for symmetric, U for unsymmetric, H for
      Hermitian, Z for skew symmetric, R for rectangular.
    Third character is A for assembled and E for unassembled
      finite element matrices.

    Input, int NROW, the number of rows or variables.

    Input, int NCOL, the number of columns or elements.

    Input, int NNZERO.  In the case of assembled sparse matrices,
    this is the number of nonzeroes.  In the case of unassembled finite
    element matrices, in which the right hand side vectors are also
    stored as unassembled finite element vectors, this is the total
    number of entries in a single unassembled right hand side vector.

    Input, int NELTVL, the number of finite element matrix entries,
    set to 0 in the case of assembled matrices.

    Input, char *PTRFMT, the 16 character format for reading pointers.

    Input, char *INDFMT, the 16 character format for reading indices.

    Input, char *VALFMT, the 20 character format for reading values.

    Input, char *RHSFMT, the 20 character format for reading values
    of the right hand side.

    Input, char *RHSTYP, the 3 character right hand side type.
    First character is F for full storage or M for same as matrix.
    Second character is G if starting "guess" vectors are supplied.
    Third character is X if exact solution vectors are supplied.

    Input, int NRHS, the number of right hand sides.

    Input, int NRHSIX, the number of entries of storage for right
    hand side values, in the case where RHSTYP[0] = 'M' and
    MXTYPE[2] = 'A'.
*/
{
  printf ( "\n" );
  printf ( "  TITLE = \"%s\".\n", title );
  printf ( "\n" );
  printf ( "  TOTCRD = %d\n", totcrd );
  printf ( "  PTRCRD = %d\n", ptrcrd );
  printf ( "  INDCRD = %d\n", indcrd );
  printf ( "  VALCRD = %d\n", valcrd );
  printf ( "  RHSCRD = %d\n", rhscrd );
  printf ( "\n" );
  printf ( "  KEY =    \"%s\".\n", key );
  printf ( "  MXTYPE = \"%s\".\n", mxtype );
  printf ( "  RHSTYP = \"%s\".\n", rhstyp );
  printf ( "\n" );
  printf ( "  NROW =   %d\n", nrow );
  printf ( "  NCOL =   %d\n", ncol );
  printf ( "  NNZERO = %d\n", nnzero );
  printf ( "  NELTVL = %d\n", neltvl );
  printf ( "  NRHS =   %d\n", nrhs );
  printf ( "  NRHSIX = %d\n", nrhsix );
  printf ( "\n" );
  printf ( "  PTRFMT = \"%s\".\n", ptrfmt );
  printf ( "  INDFMT = \"%s\".\n", indfmt );
  printf ( "  VALFMT = \"%s\".\n", valfmt );
  printf ( "  RHSFMT = \"%s\".\n", rhsfmt );

  return;
}
/******************************************************************************/

void hb_header_read ( FILE *input, char **title, char **key, int *totcrd, 
  int *ptrcrd, int *indcrd, int *valcrd, int *rhscrd, char **mxtype, int *nrow, 
  int *ncol, int *nnzero, int *neltvl, char **ptrfmt, char **indfmt, char **valfmt, 
  char **rhsfmt, char **rhstyp, int *nrhs, int *nrhsix )

/******************************************************************************/
/*
  Purpose:

    HB_HEADER_READ reads the header of an HB file.

  Discussion:

    The user should already have opened the file, and positioned it
    to the first record.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    21 January 2014

  Author:

    John Burkardt

  Reference:

    Iain Duff, Roger Grimes, John Lewis,
    User's Guide for the Harwell-Boeing Sparse Matrix Collection,
    October 1992.

  Parameters:

    Input, FILE *INPUT, the unit from which data is read.

    Output, char *TITLE, a 72 character title for the matrix.

    Output, char *KEY, an 8 character identifier for the matrix.

    Output, int *TOTCRD, the total number of lines of data.

    Output, int *PTRCRD, the number of input lines for pointers.

    Output, int *INDCRD, the number of input lines for row indices.

    Output, int *VALCRD, the number of input lines for numerical values.

    Output, int *RHSCRD, the number of input lines for right hand sides.

    Output, char *MXTYPE, the 3 character matrix type.
    First character is R for Real, C for complex, P for pattern only.
    Second character is S for symmetric, U for unsymmetric, H for
      Hermitian, Z for skew symmetric, R for rectangular.
    Third character is A for assembled and E for unassembled
      finite element matrices.

    Output, int *NROW, the number of rows or variables.

    Output, int *NCOL, the number of columns or elements.

    Output, int *NNZERO.  In the case of assembled sparse matrices,
    this is the number of nonzeroes.  In the case of unassembled finite
    element matrices, in which the right hand side vectors are also
    stored as unassembled finite element vectors, this is the total
    number of entries in a single unassembled right hand side vector.

    Output, int *NELTVL, the number of finite element matrix entries,
    set to 0 in the case of assembled matrices.

    Output, char *PTRFMT, the 16 character format for reading pointers.

    Output, char *INDFMT, the 16 character format for reading indices.

    Output, char *VALFMT, the 20 character format for reading values.

    Output, char *RHSFMT, the 20 character format for reading values
    of the right hand side.

    Output, char *RHSTYP, the 3 character right hand side type.
    First character is F for full storage or M for same as matrix.
    Second character is G if starting "guess" vectors are supplied.
    Third character is X if exact solution vectors are supplied.

    Output, int *NRHS, the number of right hand sides.

    Output, int *NRHSIX, the number of entries of storage for right
    hand side values, in the case where RHSTYP[0] = 'M' and
    MXTYPE[2] = 'A'.
*/
{
# define LINE_MAX 255

  char *error;
  char *field;
  char line[LINE_MAX];
  int line_len;
/*
  Read line 1.
*/
  error = fgets ( line, LINE_MAX, input );

  if ( !error )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "HB_HEADER_READ - Fatal error!\n" );
    fprintf ( stderr, "  I/O error reading header line 1.\n" );
    exit ( 1 );
  }

  line_len = strlen ( line );
  line[line_len-1] = '\0';

  *title = s_substring ( line, 1, 72 );
  s_trim ( *title );

  *key = s_substring ( line, 73, 80 );
  s_trim ( *key );
/*
  Read line 2.
*/
  error = fgets ( line, LINE_MAX, input );

  if ( !error )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "HB_HEADER_READ - Fatal error!\n" );
    fprintf ( stderr, "  I/O error reading header line 2.\n" );
    exit ( 1 );
  }

  line_len = strlen ( line );
  line[line_len-1] = '\0';

  field = s_substring ( line,  1, 14 );
  *totcrd = atoi ( field );

  field = s_substring ( line, 15, 28 );
  *ptrcrd = atoi ( field );

  field = s_substring ( line, 29, 42 );
  *indcrd = atoi ( field );

  field = s_substring ( line, 43, 56 );
  *valcrd = atoi ( field );

  field = s_substring ( line, 57, 70 );
  *rhscrd = atoi ( field );
/*
  Read line 3.
*/
  error = fgets ( line, LINE_MAX, input );

  if ( !error )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "HB_HEADER_READ - Fatal error!\n" );
    fprintf ( stderr, "  I/O error reading header line 3.\n" );
    exit ( 1 );
  }

  line_len = strlen ( line );
  line[line_len-1] = '\0';

  *mxtype = s_substring ( line, 1, 3 );
  s_trim ( *mxtype );

  field = s_substring ( line, 15, 28 );
  *nrow = atoi ( field );

  field = s_substring ( line, 29, 42 );
  *ncol = atoi ( field );

  field = s_substring ( line, 43, 56 );
  *nnzero = atoi ( field );

  field = s_substring ( line, 57, 70 );
  *neltvl = atoi ( field );
/*
  Read line 4.
*/
  error = fgets ( line, LINE_MAX, input );

  if ( !error )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "HB_HEADER_READ - Fatal error!\n" );
    fprintf ( stderr, "  I/O error reading header line 4.\n" );
    exit ( 1 );
  }

  line_len = strlen ( line );
  line[line_len-1] = '\0';

  *ptrfmt = s_substring ( line,  1, 16 );
  s_trim ( *ptrfmt );

  *indfmt = s_substring ( line, 17, 32 );
  s_trim ( *indfmt );

  *valfmt = s_substring ( line, 33, 52 );
  s_trim ( *valfmt );

  *rhsfmt = s_substring ( line, 53, 72 );
  s_trim ( *rhsfmt );
/*
  Read line 5.
*/
  if ( 0 < rhscrd )
  {
    error = fgets ( line, LINE_MAX, input );

    if ( !error )
    {
      fprintf ( stderr, "\n" );
      fprintf ( stderr, "HB_HEADER_READ - Fatal error!\n" );
      fprintf ( stderr, "  I/O error reading header line 5.\n" );
      exit ( 1 );
    }

    line_len = strlen ( line );
    line[line_len-1] = '\0';

    *rhstyp = s_substring ( line, 1, 3 );
    s_trim ( *rhstyp );

    field = s_substring ( line, 15, 28 );
    *nrhs = atoi ( field );

    field = s_substring ( line, 29, 42 );
    *nrhsix = atoi ( field );
  }
  else
  {
    *rhstyp = NULL;
    *nrhs = 0;
    *nrhsix = 0;
  }

  return;
# undef LINE_MAX
}
/******************************************************************************/

void hb_header_write ( FILE *output, char *title, char *key, int totcrd, 
  int ptrcrd, int indcrd, int valcrd, int rhscrd, char *mxtype, int nrow, 
  int ncol, int nnzero, int neltvl, char *ptrfmt, char *indfmt, char *valfmt, 
  char *rhsfmt, char *rhstyp, int nrhs, int nrhsix )

/******************************************************************************/
/*
  Purpose:

    HB_HEADER_WRITE writes the header of an HB file.

  Discussion:

    The user should already have opened the file, and positioned it
    to the first record.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    09 April 2004

  Author:

    John Burkardt

  Reference:

    Iain Duff, Roger Grimes, John Lewis,
    User's Guide for the Harwell-Boeing Sparse Matrix Collection,
    October 1992.

  Parameters:

    Input, FILE *OUTPUT, the unit to which data is written.

    Input, char *TITLE, a 72 character title for the matrix.

    Input, char *KEY, an 8 character identifier for the matrix.

    Input, int TOTCRD, the total number of lines of data.

    Input, int PTRCRD, the number of input lines for pointers.

    Input, int INDCRD, the number of input lines for row indices.

    Input, int VALCRD, the number of input lines for numerical values.

    Input, int RHSCRD, the number of input lines for right hand sides.

    Input, char *MXTYPE, the 3 character matrix type.
    First character is R for Real, C for complex, P for pattern only.
    Second character is S for symmetric, U for unsymmetric, H for
      Hermitian, Z for skew symmetric, R for rectangular.
    Third character is A for assembled and E for unassembled
      finite element matrices.

    Input, int NROW, the number of rows or variables.

    Input, int NCOL, the number of columns or elements.

    Input, int NNZERO.  In the case of assembled sparse matrices,
    this is the number of nonzeroes.  In the case of unassembled finite
    element matrices, in which the right hand side vectors are also
    stored as unassembled finite element vectors, this is the total
    number of entries in a single unassembled right hand side vector.

    Input, int NELTVL, the number of finite element matrix entries,
    set to 0 in the case of assembled matrices.

    Input, char *PTRFMT, the 16 character format for reading pointers.

    Input, char *INDFMT, the 16 character format for reading indices.

    Input, char *VALFMT, the 20 character format for reading values.

    Input, char *RHSFMT, the 20 character format for reading values
    of the right hand side.

    Input, char *RHSTYP, the 3 character right hand side type.
    First character is F for full storage or M for same as matrix.
    Second character is G if starting "guess" vectors are supplied.
    Third character is X if exact solution vectors are supplied.
    Ignored if NRHS = 0.

    Input, int NRHS, the number of right hand sides.

    Input, int NRHSIX, the number of row indices (set to 0
    in the case of unassembled matrices.)  Ignored if NRHS = 0.
*/
{
/*
  Note that the seemingly bizarre negative signs on the character string field
  widths causes the strings to be printed left-justified.
*/
  fprintf ( output, "%-72s%-8s\n", title, key );

  fprintf ( output, "%14d%14d%14d%14d%14d\n", 
    totcrd, ptrcrd, indcrd, valcrd, rhscrd );

  fprintf ( output, "%-3s           %14d%14d%14d%14d\n", 
    mxtype, nrow, ncol, nnzero, neltvl );

  fprintf ( output, "%-16s%-16s%-20s%-20s\n", 
    ptrfmt, indfmt, valfmt, rhsfmt );

  if ( 0 < rhscrd )
  {
    fprintf ( output, "%-3s           %14d%14d\n", rhstyp, nrhs, nrhsix );
  }

  return;
}
/******************************************************************************/

double *hb_matvec_a_mem ( int nrow, int ncol, int nnzero, int nrhs, 
  int colptr[], int rowind[], double values[], double exact[] )

/******************************************************************************/
/*
  Purpose:

    HB_MATVEC_A_MEM multiplies an assembled Harwell Boeing matrix times a vector.

  Discussion:

    In this "A_MEM" version of the routine, the matrix is assumed to be in
    "assembled" form, and all the data is assumed to be small enough
    to reside completely in memory; the matrix and multiplicand vectors
    are assumed to have been read into memory before this routine is called.

    It is assumed that MXTYPE(3:3) = 'A', that is, that the matrix is
    stored in the "assembled" format.

    Also, the storage used for the vectors X and the products A*X
    corresponds to RHSTYP(1:1) = 'F', that is, the "full" storage mode
    for vectors.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    22 January 2014

  Author:

    John Burkardt

  Reference:

    Iain Duff, Roger Grimes, John Lewis,
    User's Guide for the Harwell-Boeing Sparse Matrix Collection,
    October 1992.

  Parameters:

    Input, int NROW, the number of rows or variables.

    Input, int NCOL, the number of columns or elements.

    Input, int NNZERO.  In the case of assembled sparse matrices,
    this is the number of nonzeroes.  

    Input, int NRHS, the number of right hand sides.

    Input, int COLPTR[NCOL+1], COLPTR(I) points to the location of
    the first entry of column I in the sparse matrix structure.

    Input, int ROWIND[NNZERO], the row index of each item.

    Input, double VALUES[NNZERO], the nonzero values of the matrix.

    Input, double EXACT[NCOL*NRHS], contains NRHS dense vectors.

    Output, double HB_MATVEC_A_MEM[NROW*NRHS], the product vectors A*X.
*/
{
  int column;
  int k;
  double *rhsval;
  int rhs;
  int row;

  rhsval = ( double * ) malloc ( nrow * nrhs * sizeof ( double ) );
/*
  Zero out the result vectors.
*/
  for ( rhs = 1; rhs <= nrhs; rhs++ )
  {
    for ( row = 1; row <= nrow; row++ )
    {
      rhsval[row-1+(rhs-1)*nrow] = 0.0;
    }
  }
/*
  For each column J of the matrix,
*/
  for ( column = 1; column <= ncol; column++ )
  {
/*
  For nonzero entry K
*/
    for ( k = colptr[column-1]; k <= colptr[column]-1; k++ )
    {
      row = rowind[k-1];
/*
  For each right hand side vector:

    B(I,1:NRHS) = B(I,1:NRHS) + A(I,J) * X(J,1:NRHS)
*/
      for ( rhs = 1; rhs <= nrhs; rhs++ )
      {
        rhsval[row-1+(rhs-1)*nrow] = rhsval[row-1+(rhs-1)*nrow] 
          + values[k-1] * exact[column-1+(rhs-1)*ncol];
      }
    }
  }

  return rhsval;
}
/******************************************************************************/

void hb_rhs_read ( FILE *input, int nrow, int nnzero, int nrhs, int nrhsix, 
  int rhscrd, char *ptrfmt, char *indfmt, char *rhsfmt, char *mxtype, 
  char *rhstyp, double rhsval[], int rhsind[], int rhsptr[], double rhsvec[] )

/******************************************************************************/
/*
  Purpose:

    HB_RHS_READ reads the right hand side information in an HB file.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    22 January 2014

  Author:

    John Burkardt

  Reference:

    Iain Duff, Roger Grimes, John Lewis,
    User's Guide for the Harwell-Boeing Sparse Matrix Collection,
    October 1992.

  Parameters:

    Input, FILE *INPUT, the unit from which data is read.

    Input, int NROW, the number of rows or variables.

    Input, int NNZERO.  In the case of assembled sparse matrices,
    this is the number of nonzeroes.  In the case of unassembled finite
    element matrices, in which the right hand side vectors are also
    stored as unassembled finite element vectors, this is the total
    number of entries in a single unassembled right hand side vector.

    Input, int NRHS, the number of right hand sides.

    Input, int NRHSIX, the number of entries of storage for right
    hand side values, in the case where RHSTYP[0] = 'M' and
    MXTYPE[2] = 'A'.

    Input, int RHSCRD, the number of lines in the file for
    right hand sides.

    Input, char *PTRFMT, the 16 character format for reading pointers.

    Input, char *INDFMT, the 16 character format for reading indices.

    Input, char *RHSFMT, the 20 character format for reading values
    of the right hand side.

    Input, char *MXTYPE, the 3 character matrix type.
    First character is R for Real, C for complex, P for pattern only.
    Second character is S for symmetric, U for unsymmetric, H for
      Hermitian, Z for skew symmetric, R for rectangular.
    Third character is A for assembled and E for unassembled
      finite element matrices.

    Input, char *RHSTYP, the 3 character right hand side type.
    First character is F for full storage or M for same as matrix.
    Second character is G if starting "guess" vectors are supplied.
    Third character is X if exact solution vectors are supplied.
    Ignored if NRHS = 0.

    If RHSTYP[0] == 'F':

      Output, double RHSVAL[NROW*NRHS], contains NRHS dense right hand
      side vectors.

      Output, int RHSPTR[], is not used.

      Output, int RHSIND[], is not used.

      Output, int RHSVEC[], is not used.

    If RHSTYP[0] = 'M' and MXTYPE[2] = 'A':

      Output, double RHSVAL[], is not used.

      Output, int RHSPTR[NRHS+1], RHSPTR[I-1] points to the location of
      the first entry of right hand side I in the sparse right hand
      side vector.

      Output, int RHSIND[NRHSIX], indicates, for each entry of
      RHSVEC, the corresponding row index.

      Output, double RHSVEC[NRHSIX], contains the value of the right hand
      side entries.

    If RHSTYP[0] = 'M' and MXTYPE[2] = 'E':

      Output, double RHSVAL[NNZERO*NRHS], contains NRHS unassembled
      finite element vector right hand sides.

      Output, int RHSPTR[], is not used.

      Output, int RHSIND[], is not used.

      Output, double RHSVEC[], is not used.
*/
{
# define LINE_MAX 255

  char code;
  char *error;
  int i;
  int j;
  int jhi;
  int jlo;
  int khi;
  int klo;
  char line[LINE_MAX];
  int line_len;
  int line_num;
  int m;
  int r;
  char *s;
  int w;
/*
  Read the right hand sides.
    case F                             = "full" or "dense";
    case not F + matrix storage is "A" = sparse pointer RHS
    case not F + matrix storage is "E" = finite element RHS
*/
  if ( 0 < rhscrd )
  {
/*
  Dense right hand sides:
*/
    if ( rhstyp[0] == 'F' )
    {
      s_to_format ( rhsfmt, &r, &code, &w, &m );

      line_num = 1 + ( nrow * nrhs - 1 ) / r;

      jhi = 0;
      for ( i = 1; i <= line_num; i++ )
      {
        error = fgets ( line, LINE_MAX, input );

        if ( !error )
        {
          fprintf ( stderr, "\n" );
          fprintf ( stderr, "HB_RHS_READ - Fatal error!\n" );
          fprintf ( stderr, "  I/O error.\n" );
          exit ( 1 );
        }

        line_len = strlen ( line );
        line[line_len-1] = '\0';

        jlo = jhi + 1;
        jhi = i4_min ( jlo + r - 1, nrow * nrhs );

        khi = 0;
        for ( j = jlo; j <= jhi; j++ )
        {
          klo = khi + 1;
          khi = i4_min ( klo + w - 1, strlen ( line ) );
          s = s_substring ( line, klo, khi );
          rhsval[j-1] = atof ( s );
        }
      }
    }
/*
  Sparse right-hand sides stored like the matrix.
  Read pointer array, indices, and values.
*/
    else if ( rhstyp[0] == 'M' )
    {
      if ( mxtype[2] == 'A' )
      {
        s_to_format ( ptrfmt, &r, &code, &w, &m );

        line_num = 1 + ( nrhs + 1 - 1 ) / r;

        jhi = 0;
        for ( i = 1; i <= line_num; i++ )
        {
          error = fgets ( line, LINE_MAX, input );

          if ( !error )
          {
            fprintf ( stderr, "\n" );
            fprintf ( stderr, "HB_RHS_READ - Fatal error!\n" );
            fprintf ( stderr, "  I/O error.\n" );
            exit ( 1 );
          }

          line_len = strlen ( line );
          line[line_len-1] = '\0';

          jlo = jhi + 1;
          jhi = i4_min ( jlo + r - 1, nrhs + 1 );
 
          khi = 0;
          for ( j = jlo; j <= jhi; j++ )
          { 
            klo = khi + 1;
            khi = i4_min ( klo + w - 1, strlen ( line ) );
            s = s_substring ( line, klo, khi );
            rhsptr[j-1] = atoi ( s );
          }
        }

        s_to_format ( indfmt, &r, &code, &w, &m );

        line_num = 1 + ( nrhsix - 1 ) / r;

        jhi = 0;
        for ( i = 1; i <= line_num; i++ )
        {
          error = fgets ( line, LINE_MAX, input );

          if ( !error )
          {
            fprintf ( stderr, "\n" );
            fprintf ( stderr, "HB_RHS_READ - Fatal error!\n" );
            fprintf ( stderr, "  I/O error.\n" );
            exit ( 1 );
          }

          line_len = strlen ( line );
          line[line_len-1] = '\0';

          jlo = jhi + 1;
          jhi = i4_min ( jlo + r - 1, nnzero );
 
          khi = 0;
          for ( j = jlo; j <= jhi; j++ )
          { 
            klo = khi + 1;
            khi = i4_min ( klo + w - 1, strlen ( line ) );
            s = s_substring ( line, klo, khi );
            rhsind[j-1] = atoi ( s );
          }
        }

        s_to_format ( rhsfmt, &r, &code, &w, &m );

        line_num = 1 + ( nrhsix - 1 ) / r;

        jhi = 0;
        for ( i = 1; i <= line_num; i++ )
        {
          error = fgets ( line, LINE_MAX, input );

          if ( !error )
          {
            fprintf ( stderr, "\n" );
            fprintf ( stderr, "HB_RHS_READ - Fatal error!\n" );
            fprintf ( stderr, "  I/O error.\n" );
            exit ( 1 );
          }

          line_len = strlen ( line );
          line[line_len-1] = '\0';

          jlo = jhi + 1;
          jhi = i4_min ( jlo + r - 1, nrhsix );

          khi = 0;
          for ( j = jlo; j <= jhi; j++ )
          {
            klo = khi + 1;
            khi = i4_min ( klo + w - 1, strlen ( line ) );
            s = s_substring ( line, klo, khi );
            rhsvec[j-1] = atof ( s );
          }
        }
      }
/*
  Sparse right hand sides in finite element format.
*/
      else if ( mxtype[2] == 'E' )
      {
        s_to_format ( rhsfmt, &r, &code, &w, &m );

        line_num = 1 + ( nnzero * nrhs - 1 ) / r;

        jhi = 0;
        for ( i = 1; i <= line_num; i++ )
        {
          error = fgets ( line, LINE_MAX, input );

          if ( !error )
          {
            fprintf ( stderr, "\n" );
            fprintf ( stderr, "HB_RHS_READ - Fatal error!\n" );
            fprintf ( stderr, "  I/O error.\n" );
            exit ( 1 );
          }

          line_len = strlen ( line );
          line[line_len-1] = '\0';

          jlo = jhi + 1;
          jhi = i4_min ( jlo + r - 1, nnzero * nrhs );

          khi = 0;
          for ( j = jlo; j <= jhi; j++ )
          {
            klo = khi + 1;
            khi = i4_min ( klo + w - 1, strlen ( line ) );
            s = s_substring ( line, klo, khi );
            rhsval[j-1] = atof ( s );
          }
        }
      }
      else
      {
        fprintf ( stderr, "\n" );
        fprintf ( stderr, "HB_RHS_READ - Fatal error!\n" );
        fprintf ( stderr, "  Illegal value of MXTYPE character 3!\n" );
        exit ( 1 );
      }

    }
/*
  0 < RHSCRD, but RHSTYP not recognized.
*/
    else
    {
      fprintf ( stderr, "\n" );
      fprintf ( stderr, "HB_RHS_READ - Fatal error!\n" );
      fprintf ( stderr, "  Illegal value of RHSTYP character 1!\n" );
      exit ( 1 );
    }

  }

  return;
# undef LINE_MAX
}
/******************************************************************************/

void hb_rhs_write ( FILE *output, int nrow, int nnzero, int nrhs, int nrhsix, 
  int rhscrd, char *ptrfmt, char *indfmt, char *rhsfmt, char *mxtype, 
  char *rhstyp, double rhsval[], int rhsind[], int rhsptr[], double rhsvec[] )

/******************************************************************************/
/*
  Purpose:

    HB_RHS_WRITE writes the right hand side information to an HB file.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    22 January 2014

  Author:

    John Burkardt

  Reference:

    Iain Duff, Roger Grimes, John Lewis,
    User's Guide for the Harwell-Boeing Sparse Matrix Collection,
    October 1992.

  Parameters:

    Input, FILE *OUTPUT, the unit to which data is written.

    Input, int NROW, the number of rows or variables.

    Input, int NNZERO.  In the case of assembled sparse matrices,
    this is the number of nonzeroes.  In the case of unassembled finite
    element matrices, in which the right hand side vectors are also
    stored as unassembled finite element vectors, this is the total
    number of entries in a single unassembled right hand side vector.

    Input, int NRHS, the number of right hand sides.

    Input, int NRHSIX, the number of entries of storage for right
    hand side values, in the case where RHSTYP[0] = 'M' and
    MXTYPE[2] = 'A'.

    Input, int RHSCRD, the number of lines in the file for
    right hand sides.

    Input, char *PTRFMT, the 16 character format for reading pointers.

    Input, char *INDFMT, the 16 character format for reading indices.

    Input, char *RHSFMT, the 20 character format for reading values
    of the right hand side.

    Input, char *MXTYPE, the 3 character matrix type.
    First character is R for Real, C for complex, P for pattern only.
    Second character is S for symmetric, U for unsymmetric, H for
      Hermitian, Z for skew symmetric, R for rectangular.
    Third character is A for assembled and E for unassembled
      finite element matrices.

    Input, char *RHSTYP, the 3 character right hand side type.
    First character is F for full storage or M for same as matrix.
    Second character is G if starting "guess" vectors are supplied.
    Third character is X if exact solution vectors are supplied.
    Ignored if NRHS = 0.

    If RHSTYP[0] == 'F':

      Input, double RHSVAL[NROW*NRHS], contains NRHS dense right hand
      side vectors.

      Input, int RHSPTR[], is not used.

      Input, int RHSIND[], is not used.

      Input, double RHSVEC[], is not used.

    If RHSTYP[0] = 'M' and MXTYPE[2] = 'A':

      Input, double RHSVAL[], is not used.

      Input, int RHSPTR[NRHS+1], RHSPTR(I) points to the location of
      the first entry of right hand side I in the sparse right hand
      side vector.

      Input, int RHSIND[NRHSIX], indicates, for each entry of
      RHSVEC, the corresponding row index.

      Input, double RHSVEC[NRHSIX], contains the value of the right hand
      side entries.

    If RHSTYP[0] = 'M' and MXTYPE[2] = 'E':

      Input, double RHSVAL[NNZERO*NRHS], contains NRHS unassembled
      finite element vector right hand sides.

      Input, int RHSPTR[], is not used.

      Input, int RHSIND[], is not used.

      Input, double RHSVEC[], is not used.
*/
{
  char code;
  char format[80];
  int i;
  int j;
  int jhi;
  int jlo;
  int line_num;
  int m;
  int r;
  int w;
/*
  Read the right hand sides.
    case F                             = "full" or "dense";
    case not F + matrix storage is "A" = sparse pointer RHS
    case not F + matrix storage is "E" = finite element RHS
*/
  if ( 0 < rhscrd )
  {
/*
  Dense right hand sides:
*/
    if ( rhstyp[0] == 'F' )
    {
      s_to_format ( rhsfmt, &r, &code, &w, &m );
      line_num = 1 + ( nrow * nrhs - 1 ) / r;
      sprintf ( format, "%%%dg", w );

      jhi = 0;
      for ( i = 1; i <= line_num; i++ )
      {
        jlo = jhi + 1;
        jhi = i4_min ( jlo + r - 1, nrow * nrhs );

        for ( j = jlo; j <= jhi; j++ )
        {
          fprintf ( output, format, rhsval[j-1] );
        }
        fprintf ( output, "\n" );
      }
    }
/*
  Sparse right-hand sides stored like the matrix.
  Read pointer array, indices, and values.
*/
    else if ( rhstyp[0] == 'M' )
    {
      if ( mxtype[2] == 'A' )
      {
        s_to_format ( ptrfmt, &r, &code, &w, &m );
        line_num = 1 + ( nrhs + 1 - 1 ) / r;
        sprintf ( format, "%%%dd", w );

        jhi = 0;
        for ( i = 1; i <= line_num; i++ )
        {
          jlo = jhi + 1;
          jhi = i4_min ( jlo + r - 1, nrhs + 1 );

          for ( j = jlo; j <= jhi; j++ )
          {
            fprintf ( output, format, rhsptr[j-1] );
          }
          fprintf ( output, "\n" );
        }

        s_to_format ( indfmt, &r, &code, &w, &m );
        line_num = 1 + ( nrhsix - 1 ) / r;
        sprintf ( format, "%%%dd", w );

        jhi = 0;
        for ( i = 1; i <= line_num; i++ )
        {
          jlo = jhi + 1;
          jhi = i4_min ( jlo + r - 1, nrhsix );

          for ( j = jlo; j <= jhi; j++ )
          {
            fprintf ( output, format, rhsind[j-1] );
          }
          fprintf ( output, "\n" );
        }

        s_to_format ( rhsfmt, &r, &code, &w, &m );
        line_num = 1 + ( nrhsix - 1 ) / r;
        sprintf ( format, "%%%dg", w );

        jhi = 0;
        for ( i = 1; i <= line_num; i++ )
        {
          jlo = jhi + 1;
          jhi = i4_min ( jlo + r - 1, nrhsix );

          for ( j = jlo; j <= jhi; j++ )
          {
            fprintf ( output, format, rhsvec[j-1] );
          }
          fprintf ( output, "\n" );
        }
      }
/*
  Sparse right hand sides in finite element format.
*/
      else if ( mxtype[2] == 'E' )
      {
        s_to_format ( rhsfmt, &r, &code, &w, &m );
        line_num = 1 + ( nnzero * nrhs - 1 ) / r;
        sprintf ( format, "%%%dg", w );

        jhi = 0;
        for ( i = 1; i <= line_num; i++ )
        {
          jlo = jhi + 1;
          jhi = i4_min ( jlo + r - 1, nnzero * nrhs );

          for ( j = jlo; j <= jhi; j++ )
          {
            fprintf ( output, format, rhsval[j-1] );
          }
          fprintf ( output, "\n" );
        }
      }
      else
      {
        fprintf ( stderr, "\n" );
        fprintf ( stderr, "HB_RHS_WRITE - Fatal error!\n" );
        fprintf ( stderr, "  Illegal value of MXTYPE character 3!\n" );
        exit ( 1 );
      }
    }
  }
  else
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "HB_RHS_WRITE - Fatal error!\n" );
    fprintf ( stderr, "  Illegal value of RHSTYP character 1!\n" );
    exit ( 1 );
  }

  return;
}
/******************************************************************************/

void hb_structure_print ( int ncol, char *mxtype, int nnzero, int neltvl, 
  int colptr[], int rowind[] )

/******************************************************************************/
/*
  Purpose:

    HB_STRUCTURE_PRINT prints the structure of an HB matrix.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    21 January 2014

  Author:

    John Burkardt

  Reference:

    Iain Duff, Roger Grimes, John Lewis,
    User's Guide for the Harwell-Boeing Sparse Matrix Collection,
    October 1992.

  Parameters:

    Input, int NCOL, the number of columns.

    Input, char *MXTYPE, the 3 character matrix type.
    First character is R for Real, C for complex, P for pattern only.
    Second character is S for symmetric, U for unsymmetric, H for
      Hermitian, Z for skew symmetric, R for rectangular.
    Third character is A for assembled and E for unassembled
      finite element matrices.

    Input, int NNZERO.  In the case of assembled sparse matrices,
    this is the number of nonzeroes.  In the case of unassembled finite
    element matrices, in which the right hand side vectors are also
    stored as unassembled finite element vectors, this is the total
    number of entries in a single unassembled right hand side vector.

    Input, int NELTVL, the number of finite element matrix entries,
    set to 0 in the case of assembled matrices.

    Input, int COLPTR[NCOL+1], COLPTR[I-1] points to the location of
    the first entry of column I in the sparse matrix structure.

    If MXTYPE[2] == 'A':

      Input, int ROWIND[NNZERO], the row index of each item.

    If MXTYPE[2] == 'F':

      Input, int ROWIND[NELTVL], the row index of each item.
*/
{
  int j;
  int k;
  int khi;
  int klo;

  if ( mxtype[2] == 'A' )
  {
    printf ( "\n" );
    printf ( "Column Begin   End   ----------------------------------------\n" );
    printf ( "\n" );
    for ( j = 1; j <= i4_min ( ncol, 10 ); j++ )
    {
      if ( colptr[j]-1 < colptr[j-1] )
      {
        printf ( "%6d   EMPTY\n", j );
      }
      else
      {
        for ( klo = colptr[j-1]; klo <= colptr[j]-1; klo = klo + 10 )
        {
          khi = i4_min ( klo + 9, colptr[j]-1 );
          if ( klo == colptr[j-1] )
          {
            printf ( "%6d%6d%6d   ", j, colptr[j-1], colptr[j]-1 );
          }
          for ( k = klo; k <= khi; k++ )
          {
            printf ( "%4d", rowind[k-1] );
          }
          printf ( "\n" );
        }
      }
    }
    printf ( "                     ----------------------------------------\n" );
  }
  else if ( mxtype[2] == 'E' )
  {
    printf ( "\n" );
    printf ( "Column Begin   End   ----------------------------------------\n" );
    printf ( "                     ----------------------------------------\n" );

    printf ( "\n" );
    printf ( "  I haven't thought about how to print an\n" );
    printf ( "  unassembled matrix yet!\n" );
  }
  else
  {
    printf ( "\n" );
    printf ( "HB_STRUCTURE_PRINT - Fatal error!\n" );
    printf ( "  Illegal value of MXTYPE character #3 = '%c'\n", mxtype[2] );
    exit ( 1 );
  }

  return;
}
/******************************************************************************/

void hb_structure_read ( FILE *input, int ncol, char *mxtype, int nnzero, 
  int neltvl, int ptrcrd, char *ptrfmt, int indcrd, char *indfmt, 
  int colptr[], int rowind[] )

/******************************************************************************/
/*
  Purpose:

    HB_STRUCTURE_READ reads the structure of an HB matrix.

  Discussion:

    The user should already have opened the file, and positioned it
    to just after the header records.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    22 January 2014

  Author:

    John Burkardt

  Reference:

    Iain Duff, Roger Grimes, John Lewis,
    User's Guide for the Harwell-Boeing Sparse Matrix Collection,
    October 1992.

  Parameters:

    Input, FILE *INPUT, the unit from which data is read.

    Input, int NCOL, the number of columns.

    Input, char *MXTYPE, the 3 character matrix type.
    First character is R for Real, C for complex, P for pattern only.
    Second character is S for symmetric, U for unsymmetric, H for
      Hermitian, Z for skew symmetric, R for rectangular.
    Third character is A for assembled and E for unassembled
      finite element matrices.

    Input, int NNZERO.  In the case of assembled sparse matrices,
    this is the number of nonzeroes.  In the case of unassembled finite
    element matrices, in which the right hand side vectors are also
    stored as unassembled finite element vectors, this is the total
    number of entries in a single unassembled right hand side vector.

    Input, int NELTVL, the number of finite element matrix entries,
    set to 0 in the case of assembled matrices.

    Input, int PTRCRD, the number of input lines for pointers.

    Input, char *PTRFMT, the 16 character format for reading pointers.

    Input, int INDCRD, the number of input lines for indices.

    Input, char *INDFMT, the 16 character format for reading indices.

    Output, int COLPTR[NCOL+1], COLPTR[I-1] points to the location of
    the first entry of column I in the sparse matrix structure.

    If MXTYPE[2] == 'A':

      Output, int ROWIND[NNZERO], the row index of each item.

    If MXTYPE[2] == 'F':

      Output, int ROWIND[NELTVL], the row index of each item.
*/
{
# define LINE_MAX 255

  char code;
  char *error;
  int i;
  int j;
  int jhi;
  int jlo;
  int khi;
  int klo;
  char line[LINE_MAX];
  int line_len;
  int line_num;
  int m;
  int number;
  int r;
  char *s;
  int w;

  s_to_format ( ptrfmt, &r, &code, &w, &m );

  if ( mxtype[2] == 'A' )
  {
    line_num = 1 + ( ( ncol + 1 ) - 1 ) / r;
  }
  else
  {
    line_num = 1 + ( ( ncol     ) - 1 ) / r;
  }

  jhi = 0;
  for ( i = 1; i <= line_num; i++ )
  {
    error = fgets ( line, LINE_MAX, input );

    if ( !error )
    {
      fprintf ( stderr, "\n" );
      fprintf ( stderr, "HB_STRUCTURE_READ - Fatal error!\n" );
      fprintf ( stderr, "  I/O error.\n" );
      exit ( 1 );
    }

    line_len = strlen ( line );
    line[line_len-1] = '\0';

    jlo = jhi + 1;
    if ( mxtype[2] == 'A' )
    {
      jhi = i4_min ( jlo + r - 1, ncol + 1 );
    }
    else
    {
      jhi = i4_min ( jlo + r - 1, ncol     );
    }
    khi = 0;
    for ( j = jlo; j <= jhi; j++ )
    {
      klo = khi + 1;
      khi = i4_min ( klo + w - 1, strlen ( line ) );
      s = s_substring ( line, klo, khi );
      colptr[j-1] = atoi ( s );
    }
  }

  if ( mxtype[2] == 'A' )
  {
    s_to_format ( indfmt, &r, &code, &w, &m );

    line_num = 1 + ( nnzero - 1 ) / r;

    jhi = 0;
    for ( i = 1; i <= line_num; i++ )
    {
      error = fgets ( line, LINE_MAX, input );

      if ( !error )
      {
        fprintf ( stderr, "\n" );
        fprintf ( stderr, "HB_STRUCTURE_READ - Fatal error!\n" );
        fprintf ( stderr, "  I/O error.\n" );
        exit ( 1 );
      }

      line_len = strlen ( line );
      line[line_len-1] = '\0';

      jlo = jhi + 1;
      jhi = i4_min ( jlo + r - 1, nnzero );
 
      khi = 0;
      for ( j = jlo; j <= jhi; j++ )
      { 
        klo = khi + 1;
        khi = i4_min ( klo + w - 1, strlen ( line ) );
        s = s_substring ( line, klo, khi );
        rowind[j-1] = atoi ( s );
      }
    }
  }
  else if ( mxtype[2] == 'E' )
  {
    s_to_format ( indfmt, &r, &code, &w, &m );

    number = colptr[ncol-1] - colptr[0];
    line_num = 1 + ( number - 1 ) / r;

    jhi = 0;
    for ( i = 1; i <= line_num; i++ )
    {
      error = fgets ( line, LINE_MAX, input );

      if ( !error )
      {
        fprintf ( stderr, "\n" );
        fprintf ( stderr, "HB_STRUCTURE_READ - Fatal error!\n" );
        fprintf ( stderr, "  I/O error.\n" );
        exit ( 1 );
      }

      line_len = strlen ( line );
      line[line_len-1] = '\0';

      jlo = jhi + 1;
      jhi = i4_min ( jlo + r - 1, number );
 
      khi = 0;
      for ( j = jlo; j <= jhi; j++ )
      { 
        klo = khi + 1;
        khi = i4_min ( klo + w - 1, strlen ( line ) );
        s = s_substring ( line, klo, khi );
        rowind[j-1] = atoi ( s );
      }
    }
  }
  else
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "HB_STRUCTURE_READ - Fatal error!\n" );
    fprintf ( stderr, "  Illegal value of MXTYPE character 3.\n" );
    exit ( 1 );
  }

  return;
# undef LINE_MAX
}
/******************************************************************************/

void hb_structure_write ( FILE *output, int ncol, char *mxtype, 
  int nnzero, int neltvl, char *ptrfmt, char *indfmt, int colptr[], 
  int rowind[] )

/******************************************************************************/
/*
  Purpose:

    HB_STRUCTURE_WRITE writes the structure of an HB matrix.

  Discussion:

    If the user is creating an HB file, then the user should
    already have opened the file, and written the header records.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    22 January 2014

  Author:

    John Burkardt

  Reference:

    Iain Duff, Roger Grimes, John Lewis,
    User's Guide for the Harwell-Boeing Sparse Matrix Collection,
    October 1992.

  Parameters:

    Input, FILE *OUTPUT, the unit to which data is written.

    Input, int NCOL, the number of columns.

    Input, char *MXTYPE, the 3 character matrix type.
    First character is R for Real, C for complex, P for pattern only.
    Second character is S for symmetric, U for unsymmetric, H for
      Hermitian, Z for skew symmetric, R for rectangular.
    Third character is A for assembled and E for unassembled
      finite element matrices.

    Input, int NNZERO.  In the case of assembled sparse matrices,
    this is the number of nonzeroes.  In the case of unassembled finite
    element matrices, in which the right hand side vectors are also
    stored as unassembled finite element vectors, this is the total
    number of entries in a single unassembled right hand side vector.

    Input, int NELTVL, the number of finite element matrix entries,
    set to 0 in the case of assembled matrices.

    Input, char *PTRFMT, the 16 character format for reading pointers.

    Input, char *INDFMT, the 16 character format for reading indices.

    Input, int COLPTR[NCOL+1], COLPTR[I-1] points to the location of
    the first entry of column I in the sparse matrix structure.

    If MXTYPE[2] == 'A':

      Input, int ROWIND[NNZERO], the row index of each item.

    If MXTYPE[2] == 'F':

      Input, int ROWIND[NELTVL], the row index of each item.
*/
{
  char code;
  char format[80];
  int j;
  int m;
  int number;
  int r;
  int w;

  s_to_format ( ptrfmt, &r, &code, &w, &m );
  sprintf ( format, "%%%dd", w );

  for ( j = 1; j <= ncol + 1; j++ )
  {
    fprintf ( output, format, colptr[j-1] );
    if ( j % r == 0 )
    {
      fprintf ( output, "\n" );
    }
  }
  if ( ( ncol + 1 ) % r != 0 )
  {
    fprintf ( output, "\n" );
  }

  s_to_format ( indfmt, &r, &code, &w, &m );
  sprintf ( format, "%%%dd", w );

  if ( mxtype[2] == 'A' )
  {
    for ( j = 1; j <= nnzero; j++ )
    {
      fprintf ( output, format, rowind[j-1] );
      if ( j % r == 0 )
      {
        fprintf ( output, "\n" );
      }
    }
    if ( nnzero % r != 0 )
    {
      fprintf ( output, "\n" );
    }
  }
  else if ( mxtype[2] == 'E' )
  {
    number = colptr[ncol-1] - colptr[0];

    for ( j = 1; j <= number; j++ )
    {
      fprintf ( output, format, rowind[j-1] );
      if ( j % r == 0 )
      {
        fprintf ( output, "\n" );
      }
    }
    if ( number % r != 0 )
    {
      fprintf ( output, "\n" );
    }
  }
  else
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "HB_STRUCTURE_WRITE - Fatal error!\n" );
    fprintf ( stderr, "  Illegal value of MXTYPE character 3.\n" );
    exit ( 1 );
  }

  return;
}
/******************************************************************************/

int *hb_ua_colind ( int ncol, int colptr[], int nnzero )

/******************************************************************************/
/*
  Purpose:

    HB_UA_COLUMN_INDEX creates a column index for an unsymmetric assembled matrix.

  Discussion:

    It is assumed that the input data corresponds to a Harwell-Boeing
    matrix which is unsymmetric, and assembled.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    22 September 2004

  Author:

    John Burkardt

  Reference:

    Iain Duff, Roger Grimes, John Lewis,
    User's Guide for the Harwell-Boeing Sparse Matrix Collection,
    October 1992.

  Parameters:

    Input, integer NCOL, the number of columns.

    Input, integer COLPTR[NCOL+1], COLPTR(I) points to the location of
    the first entry of column I in the sparse matrix structure.

    Input, int NNZERO, the number of nonzeros.

    Output, int HB_UA_COLIND[NNZERO], the column index of each matrix value.
*/
{
  int *colind;
  int i;
  int j;

  colind = ( int * ) malloc ( nnzero * sizeof ( int ) );

  for ( i = 1; i <= ncol; i++ )
  {
    for ( j = colptr[i-1]; j <= colptr[i] - 1; j++ )
    {
      colind[j-1] = i;
    }
  }

  return colind;
}
/******************************************************************************/

void hb_values_print ( int ncol, int colptr[], char *mxtype, int nnzero, 
  int neltvl, double values[] )

/******************************************************************************/
/*
  Purpose:

    HB_VALUES_PRINT prints the values of an HB matrix.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    22 January 2014

  Author:

    John Burkardt

  Reference:

    Iain Duff, Roger Grimes, John Lewis,
    User's Guide for the Harwell-Boeing Sparse Matrix Collection,
    October 1992.

  Parameters:

    Input, int NCOL, the number of columns.

    Input, int COLPTR[NCOL+1], COLPTR[I-1] points to the location of
    the first entry of column I in the sparse matrix structure.

    Input, char *MXTYPE, the 3 character matrix type.
    First character is R for Real, C for complex, P for pattern only.
    Second character is S for symmetric, U for unsymmetric, H for
      Hermitian, Z for skew symmetric, R for rectangular.
    Third character is A for assembled and E for unassembled
      finite element matrices.

    Input, int NNZERO.  In the case of assembled sparse matrices,
    this is the number of nonzeroes.  In the case of unassembled finite
    element matrices, in which the right hand side vectors are also
    stored as unassembled finite element vectors, this is the total
    number of entries in a single unassembled right hand side vector.

    Input, int NELTVL, the number of finite element matrix entries,
    set to 0 in the case of assembled matrices.

    If MXTYPE[2] == 'A':

      Input, double VALUES[NNZERO], the nonzero values of the matrix.

    If MXTYPE[2] == 'E':

      Input, double VALUES[NELTVL], the nonzero values of the matrix.
*/
{
  int j;
  int k;
  int khi;
  int klo;

  if ( mxtype[2] == 'A' )
  {
    printf ( "\n" );
    printf ( "Column Begin   End   ----------------------------------------\n" );
    printf ( "\n" );

    for ( j = 1; j <= ncol; j++ )
    {
      if ( 5 < j && j < ncol )
      {
        continue;
      }

      if ( j == ncol && 6 < ncol )
      {
        printf ( "Skipping intermediate columns...)\n" );
      }

      if ( colptr[j]-1 < colptr[j-1] )
      {
        printf ( "%6d   EMPTY\n", j );
      }
      else
      {
        for ( klo = colptr[j-1]; klo <= colptr[j]-1; klo = klo + 5 )
        {
          khi = i4_min ( klo + 4, colptr[j]-1 );
          if ( klo == colptr[j-1] )
          {
            printf ( "%5d  %5d  %5d   ", j, colptr[j-1], colptr[j]-1 );
          }
          for ( k = klo; k <= khi; k++ )
          {
            printf ( "%12g", values[k-1] );
          }
          printf ( "\n" );
        }
      }
    }

    printf ( "                     ----------------------------------------\n" );
  }
  else if ( mxtype[2] == 'E' )
  {
    printf ( "\n" );
    printf ( "Column Begin   End   ----------------------------------------\n" );
    printf ( "                     ----------------------------------------\n" );

    printf ( "\n" );
    printf ( "I haven't thought about how to print an\n" );
    printf ( "unassembled matrix yet!\n" );
  }
  else
  {
    printf ( "\n" );
    printf ( "HB_VALUES_PRINT - Fatal error!\n" );
    printf ( "  Illegal value of MXTYPE character 3 = '%c'\n", mxtype[2] );
    exit ( 1 );
  }

  return;
}
/******************************************************************************/

void hb_values_read ( FILE *input, int valcrd, char *mxtype, int nnzero,
  int neltvl, char *valfmt, double values[] )

/******************************************************************************/
/*
  Purpose:

    HB_VALUES_READ reads the values of an HB matrix.

  Discussion:

    The user should already have opened the file, and positioned it
    to just after the header and structure records.

    Values are contained in an HB file if the VALCRD parameter
    is nonzero.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    22 January 2014

  Author:

    John Burkardt

  Reference:

    Iain Duff, Roger Grimes, John Lewis,
    User's Guide for the Harwell-Boeing Sparse Matrix Collection,
    October 1992.

  Parameters:

    Input, FILE *INPUT, the unit from which data is read.

    Input, int VALCRD, the number of input lines for numerical values.

    Input, char *MXTYPE, the 3 character matrix type.
    First character is R for Real, C for complex, P for pattern only.
    Second character is S for symmetric, U for unsymmetric, H for
      Hermitian, Z for skew symmetric, R for rectangular.
    Third character is A for assembled and E for unassembled
      finite element matrices.

    Input, int NNZERO.  In the case of assembled sparse matrices,
    this is the number of nonzeroes.  In the case of unassembled finite
    element matrices, in which the right hand side vectors are also
    stored as unassembled finite element vectors, this is the total
    number of entries in a single unassembled right hand side vector.

    Input, int NELTVL, the number of finite element matrix entries,
    set to 0 in the case of assembled matrices.

    Input, char *VALFMT, the 20 character format for reading values.

    If MXTYPE[2] == 'A':

      Output, double VALUES[NNZERO], the nonzero values of the matrix.

    If MXTYPE[2] == 'E':

      Output, double VALUES[NELTVL], the nonzero values of the matrix.
*/
{
# define LINE_MAX 255

  char code;
  char *error;
  int i;
  int j;
  int jhi;
  int jlo;
  int khi;
  int klo;
  char line[LINE_MAX];
  int line_len;
  int line_num;
  int m;
  int r;
  char *s;
  int w;

  s_to_format ( valfmt, &r, &code, &w, &m );
/*
  Read the matrix values.
    case "A" = assembled;
    case "E" = unassembled finite element matrices.
*/
  if ( 0 < valcrd )
  {
    if ( mxtype[2] == 'A' )
    {
      line_num = 1 + ( nnzero - 1 ) / r;
    }
    else if ( mxtype[2] == 'E' )
    {
      line_num = 1 + ( neltvl - 1 ) / r;
    }
    else
    {
      fprintf ( stderr, "\n" );
      fprintf ( stderr, "HB_VALUES_READ - Fatal error!\n" );
      fprintf ( stderr, "  Illegal value of MXTYPE character 3.\n" );
      exit ( 1 );
    }

    jhi = 0;
    for ( i = 1; i <= line_num; i++ )
    {
      error = fgets ( line, LINE_MAX, input );

      if ( !error )
      {
        fprintf ( stderr, "\n" );
        fprintf ( stderr, "HB_VALUES_READ - Fatal error!\n" );
        fprintf ( stderr, "  I/O error.\n" );
        exit ( 1 );
      }

      line_len = strlen ( line );
      line[line_len-1] = '\0';

      jlo = jhi + 1;
      if ( mxtype[2] == 'A' )
      {
        jhi = i4_min ( jlo + r - 1, nnzero );
      }
      else
      {
        jhi = i4_min ( jlo + r - 1, neltvl );
      }
      khi = 0;
      for ( j = jlo; j <= jhi; j++ )
      {
        klo = khi + 1;
        khi = i4_min ( klo + w - 1, strlen ( line ) );
        s = s_substring ( line, klo, khi );
        values[j-1] = atof ( s );
      }
    }
  }

  return;
# undef LINE_MAX
}
/******************************************************************************/

void hb_values_write ( FILE *output, int valcrd, char *mxtype, 
  int nnzero, int neltvl, char *valfmt, double values[] )

/******************************************************************************/
/*
  Purpose:

    HB_VALUES_WRITE writes the values of an HB matrix.

  Discussion:

    If the user is creating an HB file, then the user should already
    have opened the file, and written the header and structure records.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    22 January 2014

  Author:

    John Burkardt

  Reference:

    Iain Duff, Roger Grimes, John Lewis,
    User's Guide for the Harwell-Boeing Sparse Matrix Collection,
    October 1992.

  Parameters:

    Input, FILE *OUTPUT, the unit to which data is written.

    Input, int VALCRD, the number of input lines for numerical values.

    Input, char *MXTYPE, the 3 character matrix type.
    First character is R for Real, C for complex, P for pattern only.
    Second character is S for symmetric, U for unsymmetric, H for
      Hermitian, Z for skew symmetric, R for rectangular.
    Third character is A for assembled and E for unassembled
      finite element matrices.

    Input, int NNZERO.  In the case of assembled sparse matrices,
    this is the number of nonzeroes.  In the case of unassembled finite
    element matrices, in which the right hand side vectors are also
    stored as unassembled finite element vectors, this is the total
    number of entries in a single unassembled right hand side vector.

    Input, int NELTVL, the number of finite element matrix entries,
    set to 0 in the case of assembled matrices.

    Input, char *VALFMT, the 20 character format for reading values.

    If MXTYPE[2] == 'A':

      Input, double VALUES[NNZERO], the nonzero values of the matrix.

    If MXTYPE[2] == 'E':

      Input, double VALUES[NELTVL], the nonzero values of the matrix.
*/
{
  char code;
  char format[80];
  int i;
  int j;
  int jhi;
  int jlo;
  int line_num;
  int m;
  int r;
  int w;

  if ( 0 < valcrd )
  {
    s_to_format ( valfmt, &r, &code, &w, &m );
    sprintf ( format, "%%%dg", w );

    if ( mxtype[2] == 'A' )
    {
      line_num = 1 + ( nnzero - 1 ) / r;
    }
    else if ( mxtype[2] == 'E' )
    {
      line_num = 1 + ( neltvl - 1 ) / r;
    }
    else
    {
      fprintf ( stderr, "\n" );
      fprintf ( stderr, "HB_VALUES_WRITE - Fatal error!\n" );
      fprintf ( stderr, "  Illegal value of MXTYPE character 3.\n" );
      exit ( 1 );
    }

    jhi = 0;
    for ( i = 1; i <= line_num; i++ )
    {
      jlo = jhi + 1;
      if ( mxtype[2] == 'A' )
      {
        jhi = i4_min ( jlo + r - 1, nnzero );
      }
      else
      {
        jhi = i4_min ( jlo + r - 1, neltvl );
      }

      for ( j = jlo; j <= jhi; j++ )
      {
        fprintf ( output, format, values[j-1] );
      }
      fprintf ( output, "\n" );
    }
  }

  return;
}
/******************************************************************************/

double *hb_vecmat_a_mem ( int nrow, int ncol, int nnzero, int nrhs, 
  int colptr[], int rowind[], double values[], double exact[] )

/******************************************************************************/
/*
  Purpose:

    HB_VECMAT_A_MEM multiplies a vector times an assembled Harwell Boeing matrix.

  Discussion:

    In this "A_MEM" version of the routine, the matrix is assumed to be in
    "assembled" form, and all the data is assumed to be small enough
    to reside completely in memory; the matrix and multiplicand vectors
    are assumed to have been read into memory before this routine is called.

    It is assumed that MXTYPE(3:3) = 'A', that is, that the matrix is
    stored in the "assembled" format.

    Also, the storage used for the vectors X and the products A*X
    corresponds to RHSTYP(1:1) = 'F', that is, the "full" storage mode
    for vectors.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    22 January 2014

  Author:

    John Burkardt

  Reference:

    Iain Duff, Roger Grimes, John Lewis,
    User's Guide for the Harwell-Boeing Sparse Matrix Collection,
    October 1992.

  Parameters:

    Input, int NROW, the number of rows or variables.

    Input, int NCOL, the number of columns or elements.

    Input, int NNZERO.  In the case of assembled sparse matrices,
    this is the number of nonzeroes.  

    Input, int NRHS, the number of right hand sides.

    Input, int COLPTR[NCOL+1], COLPTR(I) points to the location of
    the first entry of column I in the sparse matrix structure.

    Input, int ROWIND[NNZERO], the row index of each item.

    Input, double VALUES[NNZERO], the nonzero values of the matrix.

    Input, double EXACT[NROW*NRHS], contains NRHS dense vectors.

    Output, double HB_VECMAT_A_MEM[NCOL*NRHS], the product vectors A'*X.
*/
{
  int column;
  int k;
  double *rhsval;
  int rhs;
  int row;

  rhsval = ( double * ) malloc ( ncol * nrhs * sizeof ( double ) );
/*
  Zero out the result vectors.
*/
  for ( rhs = 1; rhs <= nrhs; rhs++ )
  {
    for ( column = 1; column <= ncol; column++ )
    {
      rhsval[column-1+(rhs-1)*ncol] = 0.0;
    }
  }
/*
  For each column J of the matrix,
*/
  for ( column = 1; column <= ncol; column++ )
  {
/*
  For nonzero entry K
*/
    for ( k = colptr[column-1]; k <= colptr[column]-1; k++ )
    {
      row = rowind[k-1];
/*
  For each right hand side vector:

    B(J,1:NRHS) = B(J,1:NRHS) + X(I,1:NRHS) * A(I,J)
*/
      for ( rhs = 1; rhs <= nrhs; rhs++ )
      {
        rhsval[column-1+(rhs-1)*ncol] = rhsval[column-1+(rhs-1)*ncol] 
          + values[k-1] * exact[row-1+(rhs-1)*nrow];
      }
    }
  }

  return rhsval;
}
/******************************************************************************/

int i4_max ( int i1, int i2 )

/******************************************************************************/
/*
  Purpose:

    I4_MAX returns the maximum of two I4's.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    29 August 2006

  Author:

    John Burkardt

  Parameters:

    Input, int I1, I2, are two integers to be compared.

    Output, int I4_MAX, the larger of I1 and I2.
*/
{
  int value;

  if ( i2 < i1 )
  {
    value = i1;
  }
  else
  {
    value = i2;
  }
  return value;
}
/******************************************************************************/

int i4_min ( int i1, int i2 )

/******************************************************************************/
/*
  Purpose:

    I4_MIN returns the smaller of two I4's.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    29 August 2006

  Author:

    John Burkardt

  Parameters:

    Input, int I1, I2, two integers to be compared.

    Output, int I4_MIN, the smaller of I1 and I2.
*/
{
  int value;

  if ( i1 < i2 )
  {
    value = i1;
  }
  else
  {
    value = i2;
  }
  return value;
}
/******************************************************************************/

void i4vec_print ( int n, int a[], char *title )

/******************************************************************************/
/*
  Purpose:

    I4VEC_PRINT prints an I4VEC.

  Discussion:

    An I4VEC is a vector of I4's.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    14 November 2003

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of components of the vector.

    Input, int A[N], the vector to be printed.

    Input, char *TITLE, a title.
*/
{
  int i;

  fprintf ( stdout, "\n" );
  fprintf ( stdout, "%s\n", title );
  fprintf ( stdout, "\n" );

  for ( i = 0; i < n; i++ )
  {
    fprintf ( stdout, "  %6d: %8d\n", i, a[i] );
  }
  return;
}
/******************************************************************************/

void i4vec_print_part ( int n, int a[], int max_print, char *title )

/******************************************************************************/
/*
  Purpose:

    I4VEC_PRINT_PART prints "part" of an I4VEC.

  Discussion:

    The user specifies MAX_PRINT, the maximum number of lines to print.

    If N, the size of the vector, is no more than MAX_PRINT, then
    the entire vector is printed, one entry per line.

    Otherwise, if possible, the first MAX_PRINT-2 entries are printed,
    followed by a line of periods suggesting an omission,
    and the last entry.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    27 October 2010

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of entries of the vector.

    Input, int A[N], the vector to be printed.

    Input, int MAX_PRINT, the maximum number of lines
    to print.

    Input, char *TITLE, a title.
*/
{
  int i;

  if ( max_print <= 0 )
  {
    return;
  }

  if ( n <= 0 )
  {
    return;
  }

  fprintf ( stdout, "\n" );
  fprintf ( stdout, "%s\n", title );
  fprintf ( stdout, "\n" );

  if ( n <= max_print )
  {
    for ( i = 0; i < n; i++ )
    {
      fprintf ( stdout, "  %8d: %8d\n", i, a[i] );
    }
  }
  else if ( 3 <= max_print )
  {
    for ( i = 0; i < max_print - 2; i++ )
    {
      fprintf ( stdout, "  %8d: %8d\n", i, a[i] );
    }
    fprintf ( stdout, "  ......  ........\n" );
    i = n - 1;
    fprintf ( stdout, "  %8d: %8d\n", i, a[i] );
  }
  else
  {
    for ( i= 0; i < max_print - 1; i++ )
    {
      fprintf ( stdout, "  %8d: %8d\n", i, a[i] );
    }
    i = max_print - 1;
    fprintf ( stdout, "  %8d: %8d  ...more entries...\n", i, a[i] );
  }

  return;
}
/******************************************************************************/

void r8mat_print ( int m, int n, double a[], char *title )

/******************************************************************************/
/*
  Purpose:

    R8MAT_PRINT prints an R8MAT.

  Discussion:

    An R8MAT is a doubly dimensioned array of R8 values, stored as a vector
    in column-major order.

    Entry A(I,J) is stored as A[I+J*M]

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    28 May 2008

  Author:

    John Burkardt

  Parameters:

    Input, int M, the number of rows in A.

    Input, int N, the number of columns in A.

    Input, double A[M*N], the M by N matrix.

    Input, char *TITLE, a title.
*/
{
  r8mat_print_some ( m, n, a, 1, 1, m, n, title );

  return;
}
/******************************************************************************/

void r8mat_print_some ( int m, int n, double a[], int ilo, int jlo, int ihi,
  int jhi, char *title )

/******************************************************************************/
/*
  Purpose:

    R8MAT_PRINT_SOME prints some of an R8MAT.

  Discussion:

    An R8MAT is a doubly dimensioned array of R8 values, stored as a vector
    in column-major order.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    26 June 2013

  Author:

    John Burkardt

  Parameters:

    Input, int M, the number of rows of the matrix.
    M must be positive.

    Input, int N, the number of columns of the matrix.
    N must be positive.

    Input, double A[M*N], the matrix.

    Input, int ILO, JLO, IHI, JHI, designate the first row and
    column, and the last row and column to be printed.

    Input, char *TITLE, a title.
*/
{
# define INCX 5

  int i;
  int i2hi;
  int i2lo;
  int j;
  int j2hi;
  int j2lo;

  fprintf ( stdout, "\n" );
  fprintf ( stdout, "%s\n", title );

  if ( m <= 0 || n <= 0 )
  {
    fprintf ( stdout, "\n" );
    fprintf ( stdout, "  (None)\n" );
    return;
  }
/*
  Print the columns of the matrix, in strips of 5.
*/
  for ( j2lo = jlo; j2lo <= jhi; j2lo = j2lo + INCX )
  {
    j2hi = j2lo + INCX - 1;
    if ( n < j2hi )
    {
      j2hi = n;
    }
    if ( jhi < j2hi )
    {
      j2hi = jhi;
    }

    fprintf ( stdout, "\n" );
/*
  For each column J in the current range...

  Write the header.
*/
    fprintf ( stdout, "  Col:  ");
    for ( j = j2lo; j <= j2hi; j++ )
    {
      fprintf ( stdout, "  %7d     ", j - 1 );
    }
    fprintf ( stdout, "\n" );
    fprintf ( stdout, "  Row\n" );
    fprintf ( stdout, "\n" );
/*
  Determine the range of the rows in this strip.
*/
    if ( 1 < ilo )
    {
      i2lo = ilo;
    }
    else
    {
      i2lo = 1;
    }
    if ( m < ihi )
    {
      i2hi = m;
    }
    else
    {
      i2hi = ihi;
    }

    for ( i = i2lo; i <= i2hi; i++ )
    {
/*
  Print out (up to) 5 entries in row I, that lie in the current strip.
*/
      fprintf ( stdout, "%5d:", i - 1 );
      for ( j = j2lo; j <= j2hi; j++ )
      {
        fprintf ( stdout, "  %14g", a[i-1+(j-1)*m] );
      }
      fprintf ( stdout, "\n" );
    }
  }

  return;
# undef INCX
}
/******************************************************************************/

void r8vec_print ( int n, double a[], char *title )

/******************************************************************************/
/*
  Purpose:

    R8VEC_PRINT prints an R8VEC.

  Discussion:

    An R8VEC is a vector of R8's.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    08 April 2009

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of components of the vector.

    Input, double A[N], the vector to be printed.

    Input, char *TITLE, a title.
*/
{
  int i;

  fprintf ( stdout, "\n" );
  fprintf ( stdout, "%s\n", title );
  fprintf ( stdout, "\n" );
  for ( i = 0; i < n; i++ )
  {
    fprintf ( stdout, "  %8d: %14g\n", i, a[i] );
  }

  return;
}
/******************************************************************************/

void r8vec_print_part ( int n, double a[], int max_print, char *title )

/******************************************************************************/
/*
  Purpose:

    R8VEC_PRINT_PART prints "part" of an R8VEC.

  Discussion:

    The user specifies MAX_PRINT, the maximum number of lines to print.

    If N, the size of the vector, is no more than MAX_PRINT, then
    the entire vector is printed, one entry per line.

    Otherwise, if possible, the first MAX_PRINT-2 entries are printed,
    followed by a line of periods suggesting an omission,
    and the last entry.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    27 February 2010

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of entries of the vector.

    Input, double A[N], the vector to be printed.

    Input, int MAX_PRINT, the maximum number of lines
    to print.

    Input, char *TITLE, a title.
*/
{
  int i;

  if ( max_print <= 0 )
  {
    return;
  }

  if ( n <= 0 )
  {
    return;
  }

  fprintf ( stdout, "\n" );
  fprintf ( stdout, "%s\n", title );
  fprintf ( stdout, "\n" );

  if ( n <= max_print )
  {
    for ( i = 0; i < n; i++ )
    {
      fprintf ( stdout, "  %8d: %14g\n", i, a[i] );
    }
  }
  else if ( 3 <= max_print )
  {
    for ( i = 0; i < max_print - 2; i++ )
    {
      fprintf ( stdout, "  %8d: %14g\n", i, a[i] );
    }
    fprintf ( stdout, "  ......  ..............\n" );
    i = n - 1;
    fprintf ( stdout, "  %8d: %14g\n", i, a[i] );
  }
  else
  {
    for ( i = 0; i < max_print - 1; i++ )
    {
      fprintf ( stdout, "  %8d: %14g\n", i, a[i] );
    }
    i = max_print - 1;
    fprintf ( stdout, "  %8d: %14g  ...more entries...\n", i, a[i] );
  }

  return;
}
/******************************************************************************/

int s_len_trim ( char *s )

/******************************************************************************/
/*
  Purpose:

    S_LEN_TRIM returns the length of a string to the last nonblank.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    26 April 2003

  Author:

    John Burkardt

  Parameters:

    Input, char *S, a pointer to a string.

    Output, int S_LEN_TRIM, the length of the string to the last nonblank.
    If S_LEN_TRIM is 0, then the string is entirely blank.
*/
{
  int n;
  char *t;

  n = strlen ( s );
  t = s + strlen ( s ) - 1;

  while ( 0 < n )
  {
    if ( *t != ' ' )
    {
      return n;
    }
    t--;
    n--;
  }

  return n;
}
/******************************************************************************/

char *s_substring ( char *s, int a, int b )

/******************************************************************************/
/*
  Purpose:

    S_SUBSTRING returns a substring of a given string.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    21 January 2014

  Author:

    John Burkardt

  Parameters:

    Input, char *S, a pointer to a string.

    Input, int A, B, the indices of the first and last character of S to copy.
    These are 1-based indices!  B should be 

    Output, char *S_SUBSTRING, a pointer to the substring.
*/
{
  int i;
  int j;
  char *t;

  t = ( char * ) malloc ( ( b + 2 - a ) * sizeof ( char ) );

  j = 0;
  for ( i = a; i <= b; i++ )
  {
    t[j] = s[i-1];
      
    if (t[j] == 'D') t[j] = 'E';  // need to be changed

    j = j + 1;
  }
  t[j] = '\0';

  return t;
}
/******************************************************************************/

void s_to_format ( char *s, int *r, char *code, int *w, int *m )

/******************************************************************************/
/*
  Purpose:

    S_TO_FORMAT reads a FORTRAN format from a string.

  Discussion:

    This routine will read as many characters as possible until it reaches
    the end of the string, or encounters a character which cannot be
    part of the format.  This routine is limited in its ability to
    recognize FORTRAN formats.  In particular, we are only expecting
    a single format specification, and cannot handle extra features
    such as 'ES' and 'EN' codes, '5X' spacing, and so on.

    Legal input is:

       0 nothing
       1 blanks
       2 optional '('
       3 blanks
       4 optional repeat factor R
       5 blanks
       6 CODE ( 'A', 'B', 'E', 'F', 'G', 'I', 'L', 'O', 'Z', '*' )
       7 blanks
       8 width W
       9 optional decimal point
      10 optional mantissa M
      11 blanks
      12 optional ')'
      13 blanks

  Example:

    S                 R   CODE   W    M

    'I12              1   I      12   0
    'E8.0'            1   E       8   0
    'F10.5'           1   F      10   5
    '2G14.6'          2   G      14   6
    '*'               1   *      -1  -1

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    22 January 2014

  Author:

    John Burkardt

  Parameters:

    Input, char *S, the string containing the
    data to be read.  Reading will begin at position 1 and
    terminate at the end of the string, or when no more
    characters can be read.

    Output, int *R, the repetition factor, which defaults to 1.

    Output, char *CODE, the format code.

    Output, int *W, the field width.

    Output, int *M, the mantissa width.
*/
{
  char c;
  int d;
  int debug = 0;
  int LEFT = 1;
  int paren_sum;
  int pos;
  int RIGHT = -1;
  int s_length;
  int state;

  state = 0;
  paren_sum = 0;
  pos = 0;
  s_length = s_len_trim ( s );

  *r = 0;
  *w = 0;
  *code = '?';
  *m = 0;

  while ( pos < s_length )
  {
    c = s[pos];
    pos = pos + 1;
/*
  BLANK character:
*/
    if ( c == ' ' )
    {
      if ( state == 4 )
      {
        state = 5;
      }
      else if ( state == 6 )
      {
        state = 7;
      }
      else if ( state == 10 )
      {
        state = 11;
      }
      else if ( state == 12 )
      {
        state = 13;
      }
    }
/*
  LEFT PAREN
*/
    else if ( c == '(' )
    {
      if ( state < 2 )
      {
        paren_sum = paren_sum + LEFT;
      }
      else
      {
        if ( debug )
        {
          fprintf ( stderr, "\n" );
          fprintf ( stderr, "S_TO_FORMAT - Fatal error!\n" );
          fprintf ( stderr, "  Current state = %d\n", state );
          fprintf ( stderr, "  Input character = '%c'.\n", c );
          exit ( 1 );
        }
        state = -1;
        break;
      }
    }
/*
  DIGIT (R, F, or W)
*/
    else if ( ch_is_digit ( c ) )
    {
      if ( state <= 3 )
      {
        state = 4;
        *r = ch_to_digit ( c );
      }
      else if ( state == 4 )
      {
        d = ch_to_digit ( c );
        *r = 10 * (*r) + d;
      }
      else if ( state == 6 || state == 7 )
      {
        if ( *code == '*' )
        {
          if ( debug )
          {
            fprintf ( stderr, "\n" );
            fprintf ( stderr, "S_TO_FORMAT - Fatal error!\n" );
            fprintf ( stderr, "  Current state = %d\n", state );
            fprintf ( stderr, "  Current code = '%c'.\n", *code );
            fprintf ( stderr, "  Input character = '%c'.\n", c );
            exit ( 1 );
          }
          state = -1;
          break;
        }
        state = 8;
        *w = ch_to_digit ( c );
      }
      else if ( state == 8 )
      {
        d = ch_to_digit ( c );
        *w = 10 * (*w) + d;
      }
      else if ( state == 9 )
      {
        state = 10;
        *m = ch_to_digit ( c );
      }
      else if ( state == 10 )
      {
        d = ch_to_digit ( c );
        *m = 10 * (*m) + d;
      }
      else
      {
        if ( debug )
        {
          fprintf ( stderr, "\n" );
          fprintf ( stderr, "S_TO_FORMAT - Fatal error!\n" );
          fprintf ( stderr, "  Current state = %d\n", state );
          fprintf ( stderr, "  Input character = '%c'.\n", c );
          exit ( 1 );
        }
        state = -1;
        break;
      }
    }
/*
  DECIMAL POINT
*/
    else if ( c == '.' )
    {
      if ( state == 8 )
      {
        state = 9;
      }
      else
      {
        if ( debug )
        {
          fprintf ( stderr, "\n" );
          fprintf ( stderr, "S_TO_FORMAT - Fatal error!\n" );
          fprintf ( stderr, "  Current state = %d\n", state );
          fprintf ( stderr, "  Input character = '%c'.\n", c );
          exit ( 1 );
        }
        state = -1;
        break;
      }
    }
/*
  RIGHT PAREN
*/
    else if ( c == ')' )
    {
      paren_sum = paren_sum + RIGHT;

      if ( paren_sum != 0 )
      {
        if ( debug )
        {
          fprintf ( stderr, "\n" );
          fprintf ( stderr, "S_TO_FORMAT - Fatal error!\n" );
          fprintf ( stderr, "  Current paren sum = %d\n", paren_sum );
          fprintf ( stderr, "  Input character = '%c'.\n", c );
          exit ( 1 );
        }
        state = -1;
        break;
      }

      if ( state == 6 && *code == '*' )
      {
        state = 12;
      }
      else if ( 6 <= state )
      {
        state = 12;
      }
      else
      {
        if ( debug )
        {
          fprintf ( stderr, "\n" );
          fprintf ( stderr, "S_TO_FORMAT - Fatal error!\n" );
          fprintf ( stderr, "  Current state = %d\n", state );
          fprintf ( stderr, "  Input character = '%c'.\n", c );
          exit ( 1 );
        }
        state = -1;
        break;
      }
    }
/*
  Code
*/
    else if ( ch_is_format_code ( c ) )
    {
      if ( state < 6 )
      {
        state = 6;
        *code = c;
      }
      else
      {
        if ( debug )
        {
          fprintf ( stderr, "\n" );
          fprintf ( stderr, "S_TO_FORMAT - Fatal error!\n" );
          fprintf ( stderr, "  Current state = %d\n", state );
          fprintf ( stderr, "  Input character = '%c'.\n", c );
          exit ( 1 );
        }
        state = -1;
        break;
      }
    }
/*
  Unexpected character
*/
    else
    {
      if ( debug )
      {
        fprintf ( stderr, "\n" );
        fprintf ( stderr, "S_TO_FORMAT - Fatal error!\n" );
        fprintf ( stderr, "  Current state = %d\n", state );
        fprintf ( stderr, "  Input character = '%c'.\n", c );
        exit ( 1 );
      }
      state = -1;
      break;
    }
  }

  if ( paren_sum != 0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "S_TO_FORMAT - Fatal error!\n" );
    fprintf ( stderr, "  Parentheses mismatch.\n" );
    exit ( 1 );
  }

  if ( state < 0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "S_TO_FORMAT - Fatal error!\n" );
    fprintf ( stderr, "  Parsing error.\n" );
    exit ( 1 );
  }

  if ( *r == 0 )
  {
    *r = 1;
  }

  return;
}
/******************************************************************************/

void s_trim ( char *s )

/******************************************************************************/
/*
  Purpose:

    S_TRIM promotes the final null forward through trailing blanks.

  Discussion:

    What we're trying to say is that we reposition the null character
    so that trailing blanks are no longer visible.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    10 April 2004

  Author:

    John Burkardt

  Parameters:

    Input/output, char *S, the string to be trimmed.
*/
{
  char c;
  int n;
  char *t;

  n = strlen ( s );
  t = s + strlen ( s ) - 1;

  while ( 0 < n ) 
  {
    if ( *t != ' ' )
    {
      return;
    }
    c      = *t;
    *t     = *(t+1);
    *(t+1) = c;
    t--;
    n--;
  }

  return;
}
/******************************************************************************/

void timestamp ( void )

/******************************************************************************/
/*
  Purpose:

    TIMESTAMP prints the current YMDHMS date as a time stamp.

  Example:

    31 May 2001 09:45:54 AM

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    24 September 2003
    Modified by Chensong on 09/23/2014

  Author:

    John Burkardt

  Parameters:

    None
*/
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  size_t len;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  len = strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  if ( len > 0 ) fprintf ( stdout, "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}

