;; Copyright (C) 2024 Camm Maguire
(in-package :cstruct)

(export '(lisp-type defdlfun +ks+ +fl+ strcat adjustable-vector adjustable-array matrix))
(si::import-internal 'si::(\| & ^ ~ c+ c* << >> object double end-shft std-instance
			   c-object-== c-fixnum-== c-float-== c-double-== c-fcomplex-== c-dcomplex-== fcomplex dcomplex
			   string-concatenate lit seqind seqbnd fixnum-length char-length cref address nani short int
			   cnum unsigned-char unsigned-short unsigned-int
			   package-internal package-external array-dims cmp-norm-tp tp0 tp1 tp2 tp3 tp4 tp5 tp6 tp7 tp8))

(dolist (l '((:float      "make_shortfloat"      short-float     cnum);FIXME repetitive with gcl_cmpopt.lsp
	     (:double     "make_longfloat"       long-float      cnum)
	     (:character  "code_char"            character       cnum)
	     (:char       "make_fixnum"          char            cnum)
	     (:short      "make_fixnum"          short           cnum)
	     (:int        "make_fixnum"          int             cnum)
	     (:uchar      "make_fixnum"          unsigned-char   cnum)
	     (:ushort     "make_fixnum"          unsigned-short  cnum)
	     (:uint       "make_fixnum"          unsigned-int    cnum)
	     (:fixnum     "make_fixnum"          fixnum          cnum)
	     (:long       "make_fixnum"          fixnum          cnum)
	     (:fcomplex   "make_fcomplex"        fcomplex        cnum)
	     (:dcomplex   "make_dcomplex"        dcomplex        cnum)
	     (:string     "make_simple_string"   string)
	     (:object     ""                     t)

;	     (:stdesig    ""                     (or symbol string character))
	     (:strstd     ""                     (or structure std-instance))
	     (:matrix     ""                     matrix)
	     (:adjvector  ""                     adjustable-vector)
	     (:adjarray   ""                     adjustable-array)
	     (:longfloat  ""                     long-float)
	     (:shortfloat ""                     short-float)
	     (:hashtable  ""                     hash-table)
	     (:ocomplex   ""                     complex)
	     (:bitvector  ""                     bit-vector)
	     (:random     ""                     random-state)
	     (:ustring    ""                     string)
	     (:fixarray   ""                     (array fixnum))
	     (:sfarray    ""                     (array short-float))
	     (:lfarray    ""                     (array long-float))

	     (:real       ""                     real)

	     (:float*     nil                    nil             (array short-float) "->sfa.sfa_self")
	     (:double*    nil                    nil             (array long-float)  "->lfa.lfa_self")
	     (:long*      nil                    nil             (array fixnum)      "->fixa.fixa_self")
	     (:void*      nil                    nil             (array t)           "->v.v_self")));FIXME
  (setf (get (car l) 'lisp-type) (if (cadr l) (caddr l) (cadddr l))))

(si::*make-constant '+fl+ (- (integer-length fixnum-length) 4))
(si::*make-constant '+ks+ 
		 `((:char 0 0)(:uchar 0 1)(:short 1 0)(:ushort 1 1)(:int 2 0) ,@(when (member :64bit *features*) `((:uint 2 1)))
		   (:float 2 2) (:double 3 2) (:fcomplex 3 3) (:dcomplex 4 3) 
		   (:long ,+fl+ 0) (:fixnum ,+fl+ 0) (:object ,+fl+ 5)))

(eval-when 
  (compile)
  (defmacro idefun (n &rest args) `(progn (defun ,n ,@args) (si::putprop ',n t 'si::cmp-inline) (export ',n)))
  (defmacro mffe nil
    `(progn
       ,@(mapcar (lambda (z &aux (x (pop z))(s (pop z))(m (car z))(n (intern (string-concatenate "*" (string-upcase x)))))
		   `(idefun ,n (x o s y)
			    (declare (fixnum x o) ,@(unless (eq n '*object) `((boolean s))))
			    ,(cond ((when (eq n '*fixnum) (member :sparc64 *features*));Possibly unaligned access
				    `(if s
					;FIXME there does not appear any useful way to lift thie branch into lisp for possible branch elimination
					 (lit :fixnum "((" (:fixnum x) "&(sizeof(fixnum)-1)) ? "
					      "({fixnum _t=" (:fixnum y) ";unsigned char *p1=(void *)(((fixnum *)" (:fixnum x) ")+" (:fixnum o) "),*p2=(void *)&_t,*pe=p1+sizeof(fixnum);for (;p1<pe;) *p1++=*p2++;_t;}) : "
					      "({((fixnum *)" (:fixnum x) ")[" (:fixnum o) "]=" (:fixnum y) ";}))")
					 (lit :fixnum "((" (:fixnum x) "&(sizeof(fixnum)-1)) ? "
					      "({fixnum _t;unsigned char *p1=(void *)(((fixnum *)" (:fixnum x) ")+" (:fixnum o) "),*p2=(void *)&_t,*pe=p1+sizeof(fixnum);for (;p1<pe;) *p2++=*p1++;_t;}) : "
					      "((fixnum *)" (:fixnum x) ")[" (:fixnum o) "])")))
				   ((eq n '*object);sgc header touch support
				    `(if (eq s t) (lit ,x "(((" ,(strcat x) "*)" (:fixnum x) ")[" (:fixnum o) "]=" (,x y) ")")
					 (if s (lit ,x "({" (:object s) "->d.e=1;(((" ,(strcat x) "*)" (:fixnum x) ")[" (:fixnum o) "]=" (,x y) ");})")
					     (lit ,x "((" ,(strcat x) "*)" (:fixnum x) ")[" (:fixnum o) "]"))))
				   (`(if s (lit ,x "(((" ,(strcat x) "*)" (:fixnum x) ")[" (:fixnum o) "]=" (,x y) ")")
					 (lit ,x "((" ,(strcat x) "*)" (:fixnum x) ")[" (:fixnum o) "]"))))))
		 +ks+)))
  (defmacro mfff nil
   `(progn
      (idefun address (x) (lit :fixnum "((fixnum)" (:object x) ")"))
      (idefun nani (x) (declare (fixnum x)) (lit :object "((object)" (:fixnum x) ")"))
      (idefun ~ (x) (declare (fixnum x)) (lit :fixnum "(~" (:fixnum x) ")"))
      ,@(mapcar (lambda (x &aux (c (consp x))(n (if c (car x) x))(s (string (if c (cdr x) x))))
		  `(idefun ,n (x y) (declare (fixnum x y)) (lit :fixnum "(" (:fixnum x) ,s (:fixnum y) ")")))
		'(& \| ^ >> << (c+ . +) (c* . *) (c- . -) (c/ . /) %))
      (idefun tp0 (x) (lit :fixnum  "tp0(" (:object x) ")"))
      (idefun tp1 (x) (lit :fixnum  "tp1(" (:object x) ")"))
      (idefun tp2 (x) (lit :fixnum  "tp2(" (:object x) ")"))
      (idefun tp3 (x) (lit :fixnum  "tp3(" (:object x) ")"))
      (idefun tp4 (x) (lit :fixnum  "tp4(" (:object x) ")"))
      (idefun tp5 (x) (lit :fixnum  "tp5(" (:object x) ")"))
      (idefun tp6 (x) (lit :fixnum  "tp6(" (:object x) ")"))
      (idefun tp7 (x) (lit :fixnum  "tp7(" (:object x) ")"))
      (idefun tp8 (x) (lit :fixnum  "tp8(" (:object x) ")"))
      ,@(mapcan (lambda (x)
	    (mapcan (lambda (y)
		      (mapcar (lambda (z)
				(let ((n (intern (string-upcase (strcat "C-" (string x) "-" (string y) "-" (string z))))))
				  `(idefun ,n (x y) (lit :boolean "(" (,x x) ,(string z) (,y y) ")")))) 
			      '(>)))
		    '(:fixnum :float :double)))
	  '(:fixnum :float :double))
      ,@(mapcan (lambda (x)
	    (mapcan (lambda (y)
		      (mapcar (lambda (z)
				(let ((n (intern (string-upcase (strcat "C-" (string x) "-" (string y) "-" (string z))))))
				  `(idefun ,n (x y) (lit :boolean "(" (,x x) ,(string z) (,y y) ")")))) 
			      '(==)))
		    '(:fixnum :float :double :fcomplex :dcomplex)))
	  '(:fixnum :float :double :fcomplex :dcomplex))
      ,@(mapcar (lambda (x &aux (tp (intern (string x)))(tp (or (eq tp 'object) tp))(n (intern (string-upcase (strcat "C-" x "-=="))))) 
		  `(idefun ,n (x y) (declare (,tp x y))(lit :boolean (,x x) "==" (,x y))))
		'(:object :fixnum :float :double :fcomplex :dcomplex)))))

(eval-when 
  (eval)
  #.`(progn 
       ,@(mapcar #'(lambda (z &aux (x (pop z))(s (pop z))(m (car z))(n (intern (string-concatenate "*" (string-upcase x)))))
		     `(progn (defun ,n (x o s y)
			       (declare (fixnum x o)(boolean s))
			       (cref (c+ x (<< o ,s)) ,(<< 1 s) ,m (if s 1 0) y))
			     (si::putprop ',n t 'si::cmp-inline)
			     (export ',n)))
		 +ks+))
  (defun mffe nil nil)
  (defun mfff nil nil))

(mffe)
(mfff)
