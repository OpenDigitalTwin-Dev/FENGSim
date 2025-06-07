/* format of a rsyms output file:
struct lsymbol_table tab;   gives number of symbols, and sum of length of 
			 strings 
addr,char[],addr,char[],...
This can be read since the addr is sizeof(int) and the char[] is null
terminated, immediately followed by and addr...
there are tab.n_symbols pairs occurring.

*/
#ifndef HEADER_SEEK
#define HEADER_SEEK(x)
#endif


typedef unsigned long addr;

struct node{
  const char *string;
  addr address;
#ifdef AIX3
  unsigned short tc_offset;
#endif  
};

struct lsymbol_table{
  unsigned int n_symbols ;
  unsigned int tot_leng;};

#define SYM_ADDRESS(table,i) table.ptable[i].address
#define SYM_STRING(table,i)  table.ptable[i].string
#define SYM_TC_OFF(table,i) ((*(table).ptable))[i].tc_offset

/* typedef struct node *TABL;  */
/* gcc does not like typedef struct node TABL[];*/

typedef struct node TABL[]; 

struct  string_address_table
{ struct node *ptable;
  unsigned int length;
  struct node *local_ptable;
  unsigned int local_length;
  unsigned int alloc_length;
};

#if !defined(HAVE_LIBBFD) && !defined(SPECIAL_RSYM)
#error Need either BFD or SPECIAL_RSYM
#endif

#ifdef SPECIAL_RSYM
struct string_address_table c_table;
#else
struct bfd_link_info link_info;
#endif
struct string_address_table combined_table;

#define PTABLE_EXTRA 20

