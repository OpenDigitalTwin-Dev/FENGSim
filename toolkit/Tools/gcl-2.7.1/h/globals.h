EXTER union lispunion Cnil_body OBJ_ALIGN;
EXTER union lispunion Ct_body   OBJ_ALIGN;

#define MULTIPLE_VALUES_LIMIT 32

struct call_data { 

object  fun;
hfixnum argd;
hfixnum nvalues;
object  values[MULTIPLE_VALUES_LIMIT];
fixnum  valp;
double  double_return;

};
EXTER struct call_data fcall;

EXTER struct character character_table[256] OBJ_ALIGN; /*FIXME, sync with char code constants above.*/
EXTER struct unadjstring character_name_table[256] OBJ_ALIGN;

EXTER object null_string;
