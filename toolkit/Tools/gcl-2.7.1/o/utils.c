/* Copyright (C) 2024 Camm Maguire */
#include <stdarg.h>
#include "include.h"

/* The functions IisProp check the property holds, and return the
   argument.   They may in future allow resetting the argument.
*/

/* object CEerror(char *error_str, char *cont_str, int num, object arg1, */
/* 	       object arg2, object arg3, object arg4); */
object IisSymbol(object f)
{
    if (type_of(f) != t_symbol) 
	FEwrong_type_argument(sLsymbol, f);
    return f;
}

object IisArray(object f)
{
    if (!TS_MEMBER(type_of(f),TS(t_array)|TS(t_vector)|TS(t_bitvector)|TS(t_string)|
		   TS(t_simple_array)|TS(t_simple_vector)|TS(t_simple_bitvector)|TS(t_simple_string)))
	FEwrong_type_argument(sLarray, f);
    return f;
}

object Iis_fixnum(object f)
{
    if (type_of(f) != t_fixnum)
	FEwrong_type_argument(sLfixnum, f);
    return f;
}

char *lisp_copy_to_null_terminated(object string, char *buf, int n)
{
    string=coerce_to_string(string);
    if (VLEN(string) + 1 > n) {
      buf = (void *) malloc(VLEN(string) + 1);
    }
    bcopy(string->st.st_self, buf, VLEN(string));
    buf[VLEN(string)] = 0;
    return buf;
}


