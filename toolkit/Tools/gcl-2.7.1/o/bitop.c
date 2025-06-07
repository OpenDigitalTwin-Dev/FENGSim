#include "include.h"
/* static void */
/* get_mark_bit(void) */
/* {error("get_mark_bit called");} */
/* static void */
/* set_mark_bit(void) */
/* {error("set_mark_bit called");} */
/* static void */
/* get_set_mark_bit(void) */
/* {error("get_set_mark_bit called");} */


/*
  These have all been replaced by macros

extern int *mark_table;
static 
get_mark_bit(x)
int x;
{
	int y;

	y = (*(mark_table+(x/4/32)) >> (x/4%32)) & 1;
	return(y);
}
static 
set_mark_bit(x)
int x;
{
	int y;

	y = 1 << (x/4%32);
	y = (*(mark_table+(x/4/32))) | y;
	*(mark_table+ (x/4/32))=y;
}
static 
get_set_mark_bit(x)
int x;
{
	int y;

	y = get_mark_bit(x);
	set_mark_bit(x);
	return(y);
}

*/
