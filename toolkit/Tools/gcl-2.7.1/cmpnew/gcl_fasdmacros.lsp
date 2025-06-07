
(in-package :compiler)
(defstruct (fasd (:type vector))
  stream
  table
  eof
  direction
  package
  index
  filepos
  table_length
  evald_forms ; list of forms eval'd. (load-time-eval)
  )

(defvar *fasd-ops*
'(  d_nil         ;/* dnil: nil */
  d_eval_skip    ;    /* deval o1: evaluate o1 after reading it */
  d_delimiter    ;/* occurs after d_listd_general and d_new_indexed_items */
  d_enter_vector ;     /* d_enter_vector o1 o2 .. on d_delimiter  make a cf_data with
		  ;  this length.   Used internally by gcl.  Just make
		  ;  an array in other lisps */
  d_cons        ; /* d_cons o1 o2: (o1 . o2) */
  d_dot         ;
  d_list    ;/* list* delimited by d_delimiter d_list,o1,o2, ... ,d_dot,on
		;for (o1 o2       . on)
		;or d_list,o1,o2, ... ,on,d_delimiter  for (o1 o2 ...  on)
	      ;*/
  d_list1   ;/* nil terminated length 1  d_list1o1   */
  d_list2   ; /* nil terminated length 2 */
  d_list3
  d_list4
  d_eval
  d_short_symbol
  d_short_string
  d_short_fixnum
  d_short_symbol_and_package
  d_bignum
  d_fixnum
  d_string
  d_objnull
  d_structure
  d_package
  d_symbol
  d_symbol_and_package
  d_end_of_file
  d_standard_character
  d_vector
  d_array
  d_begin_dump
  d_general_type
  d_sharp_equals ;              /* define a sharp */
  d_sharp_value
  d_sharp_value2
  d_new_indexed_item
  d_new_indexed_items
  d_reset_index
  d_macro
  d_reserve1
  d_reserve2
  d_reserve3
  d_reserve4
  d_indexed_item3 ;      /* d_indexed_item3 followed by 3bytes to give index */
  d_indexed_item2  ;      /* d_indexed_item2 followed by 2bytes to give index */
  d_indexed_item1 
  d_indexed_item0    ;  /* This must occur last ! */
))

(defmacro put-op (op str)
  `(write-byte ,(or (position op *fasd-ops*)
		    (error "illegal op")) ,str))

(defmacro put2 (n str)
  `(progn  (write-bytei ,n 0 ,str)
	   (write-bytei  ,n 1 ,str)))
  
(defmacro write-bytei (n i str)
  `(write-byte (the fixnum (ash (the fixnum ,n) >> ,(* i 8))) ,str))
  

(provide 'FASDMACROS)

