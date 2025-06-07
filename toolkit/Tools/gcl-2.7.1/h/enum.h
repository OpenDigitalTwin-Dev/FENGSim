enum signals_allowed_values {
  sig_none,
  sig_normal,
  sig_try_to_delay,
  sig_safe,
  sig_at_read,
  sig_use_signals_allowed_value

};

  
enum aelttype {   /*  array element type  */
 aet_ch,          /*  character  */
 aet_bit,         /*  bit  */
 aet_nnchar,      /*  non-neg char */
 aet_uchar,       /*  unsigned char */
 aet_char,        /*  signed char */
 aet_nnshort,     /*  non-neg short   */
 aet_ushort,      /*  unsigned short   */
 aet_short,       /*  signed short */
 aet_sf,          /*  short-float  */
#if SIZEOF_LONG != SIZEOF_INT
 aet_nnint,       /*  non-neg int   */
 aet_uint,        /*  unsigned int   */
 aet_int,         /*  signed int */
#endif
 aet_lf,          /*  plong-float  */
 aet_object,      /*  t  */
 aet_nnfix,       /*  non-neg fixnum  */
 aet_fix,         /*  fixnum  */
#if SIZEOF_LONG == SIZEOF_INT
 aet_nnint,       /*  non-neg int   */
 aet_uint,        /*  unsigned int   */
 aet_int,         /*  signed int */
#endif
 aet_last
};

enum aemode {
  aem_signed,
  aem_unsigned,
  aem_float,
  aem_complex,
  aem_character,
  aem_t
};
