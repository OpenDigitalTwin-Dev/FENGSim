;; CMPFUN  Library functions.
;;;
;; Copyright (C) 1994 M. Hagiya, W. Schelter, T. Yuasa
;; Copyright (C) 2024 Camm Maguire

;; This file is part of GNU Common Lisp, herein referred to as GCL
;;
;; GCL is free software; you can redistribute it and/or modify it under
;;  the terms of the GNU LIBRARY GENERAL PUBLIC LICENSE as published by
;; the Free Software Foundation; either version 2, or (at your option)
;; any later version.
;; 
;; GCL is distributed in the hope that it will be useful, but WITHOUT
;; ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
;; FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library General Public 
;; License for more details.
;; 
;; You should have received a copy of the GNU Library General Public License 
;; along with GCL; see the file COPYING.  If not, write to the Free Software
;; Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.


(in-package :compiler)

(si:putprop 'princ 'c1princ 'c1)
(si:putprop 'princ 'c2princ 'c2)
(si:putprop 'terpri 'c1terpri 'c1)

(si:putprop 'apply 'c1apply 'c1)
(si:putprop 'apply 'c2apply 'c2)
(si:putprop 'funcall 'c1funcall 'c1)


(defvar *princ-string-limit* 80)

(defun c1princ (args &aux stream (info (make-info :flags (iflags side-effects))))
  (when (endp args) (too-few-args 'princ 1 0))
  (unless (or (endp (cdr args)) (endp (cddr args)))
          (too-many-args 'princ 2 (length args)))
  (setq stream (if (endp (cdr args))
                   (c1nil)
                   (c1arg (cadr args) info)))
  (if (and (or (and (stringp (car args))
                    (<= (length (car args)) *princ-string-limit*))
               (characterp (car args)))
           (or (endp (cdr args))
               (and (eq (car stream) 'var)
                    (member (var-kind (caaddr stream)) '(GLOBAL SPECIAL)))))
      (list 'princ info (car args)
            (if (endp (cdr args)) nil (var-loc (caaddr stream)))
            stream)
      (list 'call-global info 'princ
            (list (c1arg (car args) info) stream))))

(defun c2princ (string vv-index stream)
  (cond ((eq *value-to-go* 'trash)
         (cond ((characterp string)
                (wt-nl "princ_char(" (char-code string))
                (if (null vv-index) (wt ",Cnil") (wt "," (vv-str vv-index)))
                (wt ");"))
               ((= (length string) 1)
                (wt-nl "princ_char(" (char-code (aref string 0)))
                (if (null vv-index) (wt ",Cnil") (wt "," (vv-str vv-index)))
                (wt ");"))
               (t
                (wt-nl "princ_str(\"")
                (dotimes (n (length string))
                  (let ((char (schar string n)))
                       (cond ((char= char #\\) (wt "\\\\"))
                             ((char= char #\") (wt "\\\""))
                             ((char= char #\Newline) (wt "\\n"))
                             ((char= char #\Return) (wt "\\r"))
                             (t (wt char)))))
                (wt "\",")
                (if (null vv-index) (wt "Cnil") (wt "" (vv-str vv-index)))
                (wt ");")))
         (unwind-exit nil))
        ((eql string #\Newline) (c2call-global 'terpri (list stream) nil t))
        (t (c2call-global
            'princ
            (list (list 'LOCATION
                        (make-info :type (cmp-norm-tp (if (characterp string) 'character 'string)))
                        (list 'VV string))
                  stream) nil t))))

(defun c1terpri (args &aux stream (info (make-info :flags (iflags side-effects))))
  (unless (or (endp args) (endp (cdr args)))
          (too-many-args 'terpri 1 (length args)))
  (setq stream (if (endp args)
                   (c1nil)
                   (c1arg (car args) info)))
  (if (or (endp args)
          (and (eq (car stream) 'var)
               (member (var-kind (caaddr stream)) '(GLOBAL SPECIAL))))
      (list 'princ info #\Newline
            (if (endp args) nil (var-loc (caaddr stream)))
            stream)
      (list 'call-global info 'terpri (list stream))))

;; (defun c1terpri (args &aux stream (info (make-info :flags (iflags side-effects))))
;;   (unless (or (endp args) (endp (cdr args)))
;;           (too-many-args 'terpri 1 (length args)))
;;   (setq stream (if (endp args)
;;                    (c1nil)
;;                    (c1expr* (car args) info)))
;;   (if (or (endp args)
;;           (and (eq (car stream) 'var)
;;                (member (var-kind (caaddr stream)) '(GLOBAL SPECIAL))))
;;       (list 'princ info #\Newline
;;             (if (endp args) nil (var-loc (caaddr stream)))
;;             stream)
;;       (list 'call-global info 'terpri (list stream))))


(defun c2apply (funob args)
  (unless (eq 'ordinary (car funob)) (baboon))
  (let* ((fn (caddr funob))
	 (all (cons fn args))
	 (*inline-blocks* 0))
    (setq *sup-used* t)
    (unwind-exit
     (get-inline-loc
      (list (make-list (length all) :initial-element t)
	    '* #.(flags ans set svt)
	    (concatenate
	     'string
	     "({fixnum _v=(fixnum)#v;object _z,_f=(#0),_l=(#1),_ll=_l;
        object _x4=Cnil,_x3=Cnil,_x2=Cnil,_x1=Cnil,_x0=Cnil;
        char _m=(#n-2),_q=_f->fun.fun_minarg>_m ? _f->fun.fun_minarg-_m : 0;
        char _n=Rset && !_f->fun.fun_argd ? _q : -1;
        fcall.fun=_f;fcall.valp=_v;fcall.argd=-(#n-1);
        switch (_n) {
          case 5: if (_l==Cnil) {_n=-1;break;} _x4=_l->c.c_car;_l=_l->c.c_cdr;
          case 4: if (_l==Cnil) {_n=-1;break;} _x3=_l->c.c_car;_l=_l->c.c_cdr;
          case 3: if (_l==Cnil) {_n=-1;break;} _x2=_l->c.c_car;_l=_l->c.c_cdr;
          case 2: if (_l==Cnil) {_n=-1;break;} _x1=_l->c.c_car;_l=_l->c.c_cdr;
          case 1: if (_l==Cnil) {_n=-1;break;} _x0=_l->c.c_car;_l=_l->c.c_cdr;
          case 0: if (_n+_m+(_l==Cnil ? 0 : 1)>_f->fun.fun_maxarg) _n=-1; else fcall.argd-=_n;
          default: break;
        }
        switch (_n) {
          case 5:  _z=_f->fun.fun_self(#*_x4,_x3,_x2,_x1,_x0,_l);break;
          case 4:  _z=_f->fun.fun_self(#*_x3,_x2,_x1,_x0,_l);break;
          case 3:  _z=_f->fun.fun_self(#*_x2,_x1,_x0,_l);break;
          case 2:  _z=_f->fun.fun_self(#*_x1,_x0,_l);break;
          case 1:  _z=_f->fun.fun_self(#*_x0,_l);break;
          case 0:  _z="
	     (if (cdr args)
		 "_f->fun.fun_self(#*_l)"
	       "(_f->fun.fun_maxarg ? _f->fun.fun_self(#*_l) : _f->fun.fun_self())")
	     ";break;
          default: _z=call_proc_cs2(#*_ll);break;
        }
        if (!(_f)->fun.fun_neval && !(_f)->fun.fun_vv) vs_top=_v ? (object *)_v : sup;
        _z;})"))
      (list* (car all) (car (last all)) (butlast (cdr all)))))
    (close-inline-blocks)))

;FIXME c1symbol-fun, eliminate mi1b
(defmacro try-provisional-functions (&rest body);ensure body has no side-effects for double eval
  `(or
    (with-restore-vars
	(let* ((*prov* t)(ops *prov-src*)(*prov-src* *prov-src*)(res (progn ,@body)))
	  (unless (iflag-p (info-flags (cadr res)) provisional)
	    (keep-vars) (mapc 'eliminate-src (ldiff *prov-src* ops)) res)))
    (progn ,@body)))

(defun c1apply (args)
  (when (or (endp args) (endp (cdr args)))
    (too-few-args 'apply 2 (length args)))
  (try-provisional-functions
   (let* ((ff (c1arg (car args)))(args (cdr args))(fid (coerce-ff ff)))
     (if (eq fid 'funcall) (c1apply args) (mi1 fid (butlast args) (car (last args)) ff)))))

(defun c1funcall (args)
  (when (endp args) (too-few-args 'funcall 1 0))
  (try-provisional-functions
   (let* ((ff (c1arg (car args)))(args (cdr args))(fid (coerce-ff ff)))
     (case fid (funcall (c1funcall args))(apply (c1apply args)) (otherwise (mi1 fid args nil ff))))))


;; (defun c1funcall-apply (args &optional last)
;;   (mi1 (if last 'apply 'funcall) args (car last)))

;; (defun c1funcall (args)
;;   (when (endp args) (too-few-args 'funcall 1 0))
;;   (c1funcall-apply args))

;; (defun c1apply (args)
;;   (when (or (endp args) (endp (cdr args)))
;;     (too-few-args 'apply 2 (length args)))
;;   (let* ((last (last args))
;; 	 (args (ldiff args last)))
;;     (c1funcall-apply args last)))


(defun eq-subtp (x y)  ;FIXME axe mult values
  (let ((s (type>= y x)))
    (values s (or s (type>= (tp-not y) x)))))

(defun eql-is-eq-tp (x)
  (eq-subtp x #teql-is-eq-tp))

(defun equal-is-eq-tp (x)
  (eq-subtp x #tequal-is-eq-tp))

(defun equalp-is-eq-tp (x)
  (eq-subtp x #tequalp-is-eq-tp))


(defun do-eq-et-al (fn args &aux (info (make-info :type #tboolean)));FIXME  pass through function inlining
  (cmpck (not (eql (length args) 2)) "Predicate ~s takes two arguments" fn)
  (let* ((nargs (c1args args info))
	 (t1 (info-type (cadar nargs)))(t2 (info-type (cadadr nargs)))
	 (a1 (atomic-tp t1))(a2 (atomic-tp t2))
	 (nfn (ecase fn
		(eq fn)
		(eql (let ((tp #teql-is-eq-tp)) (if (or (type<= t1 tp)(type<= t2 tp)) 'eq fn)))
		(equal (let ((tp #tequal-is-eq-tp)) (if (or (type<= t1 tp)(type<= t2 tp)) 'eq fn)))
		(equalp (let ((tp #tequalp-is-eq-tp)) (if (or (type<= t1 tp)(type<= t2 tp)) 'eq fn)))))
	 (nfn (if (when (eq nfn 'equal) (or (type<= t1 #tnumber) (type<= t2 #tnumber)))
		  'eql nfn)))
    (cond ((when (and t1 t2 (member nfn '(eq eql))) (not (type-and t1 t2)))
	   (c1progn (append args (list nil)) (nconc nargs (list (c1nil)))))
	  ((and a1 a2 (case nfn (eq (or (eql-is-eq (car a1)) (eql-is-eq (car a2))))(eql t)))
	   (let ((q (eql (car a1) (car a2))))
	     (c1progn (append args (list q)) (nconc nargs (list (if q (c1t) (c1nil)))))))
	  ((when (and t1 t2)
	     (let ((x (get-vbind (car nargs)))(y (get-vbind (cadr nargs))))
	       (when (or (when x (eq x y)) (and (symbolp (car args)) (eq (car args) (cadr args))))
		 (c1t)))))
	  (t (unless (and t1 t2) (setf (info-type info) nil)) `(call-global ,info ,nfn ,nargs)))))

(dolist (l `(eq eql equal equalp))
  (si::putprop l 'do-eq-et-al 'c1g))

(defun num-type-bounds (t1)
  (let ((x (tp-bnds t1)))
    (when x (list (car x) (cdr x)))))
  
(defun ntrr (x y)
  (and x y
       (list (and (car x) (car y))
	     (and (cadr x) (cadr y) (eq (car x) (car y))))))

(defun dntrr (l)
  (reduce 'ntrr (cdr l) :initial-value (car l)))

(defun num-type-rel (fn t1 t2 &optional s &aux (t1 (coerce-to-one-value t1))(t2 (coerce-to-one-value t2)))
  (let ((nop (car (rassoc fn '((>= . <) (> . <=) (= . /=)))))
	(rfn (cdr (assoc  fn '((>= . >) (> . >=))))))
    (cond (nop (let ((q (num-type-rel nop t1 t2))) 
		 (list (and (not (car q)) (cadr q)) (cadr q))))
	  ((and (consp t1) (eq (car t1) 'or))
	   (dntrr (mapcar (lambda (x) (num-type-rel fn x t2)) (cdr t1))))
	  ((and (consp t2) (eq (car t2) 'or))
	   (dntrr (mapcar (lambda (x) (num-type-rel fn t1 x)) (cdr t2))))
	  ((eq fn '=) (cond ((not (and t1 t2)) (list nil t))
			    ;; ((and (type>= #tcomplex t1) (not (type-and #tcomplex t2)))
			    ;;  (unless (type-and (cadr t1) (type-and t2 #t(real 0.0 0.0)))
			    ;;    (list nil t)))
			    ;; ((and (type>= #tcomplex t2) (not (type-and #tcomplex t1)))
			    ;;  (unless (type-and (cadr t2) (type-and t1 #t(real 0.0 0.0)))
			    ;;    (list nil t)))
			    ((let ((x (num-type-rel '>= t1 t2))(y (num-type-rel '>= t2 t1)))
			       (list (and (car x) (car y)) (and (cadr x) (cadr y)))))))
	  ((not s) (let ((f (num-type-rel fn t1 t2 t)))
		     (list f (or f (num-type-rel rfn t2 t1 t)))))
	  ((not (and t1 t2)) nil)
	  ((and (type>= #treal t1) (type>= #treal t2))
	   (let ((t1 (car  (num-type-bounds t1)))
		 (t2 (cadr (num-type-bounds t2))))
	     (and (numberp t1) (numberp t2) (values (funcall fn t1 t2))))))))


(defun do-num-relations (fn args)
  (let* ((info (make-info :type #tboolean))
	 (nargs (c1args args info))
	 (t1 (and (car args) (info-type (cadar nargs))))
	 (t2 (and (cadr args) (info-type (cadadr nargs))))
	 (fn (or (cdr (assoc fn '((si::=2 . =)(si::/=2 . /=)(si::>=2 . >=)
				  (si::>2 . >)(si::<2 . <)(si::<=2 . <=))))
		 fn))
	 (r (and t1 t2 (num-type-rel fn t1 t2))))
    (cond ((cddr args) (list 'call-global info fn nargs))
	  ((or (car r) (cadr r))
	   (let ((r (when (car r) t)))
	     (c1progn (append args (list r)) (nconc nargs (list (if r (c1t) (c1nil)))))))
	  ((let ((x (get-vbind (car nargs)))(y (get-vbind (cadr nargs))))
	     (when (or (when x (eq x y)) (and (symbolp (car args)) (eq (car args) (cadr args))))
	       (unless (type-and (type-or1 t1 t2) #t(or (short-float unordered) (long-float unordered)))
		 (if (member fn '(= >= <=)) (c1t) (c1nil))))))
	  ((list 'call-global info fn nargs)))))

(dolist (l `(>= > < <= = /=))
  (si::putprop l 'do-num-relations 'c1g))

(defun do-+- (fn args)
  (let* ((info (make-info :type #tnumber))
	 (nargs (c1args args info))
	 (i1 (info-type (cadar nargs)))
	 (t1 (car (atomic-tp i1)))
	 (i2 (info-type (cadadr nargs)))
	 (t2 (car (atomic-tp i2))))
    (cond ((and (eq fn 'number-plus) (eql t1 0) (ignorable-form (car nargs))) (cadr nargs));contagion
	  ((and (eql t2 0) (ignorable-form (cadr nargs))) (car nargs))
	  (t (setf (info-type info)
		   (type-and (info-type info)
			     (funcall (get fn 'type-propagator) fn i1 i2)))
	     `(call-global ,info ,fn ,nargs)))))
(setf (get 'number-plus 'c1g) 'do-+-)
(setf (get 'number-minus 'c1g) 'do-+-)


(defun do-*/ (fn args)
  (let* ((info (make-info :type #tnumber))
	 (nargs (c1args args info))
	 (i1 (info-type (cadar nargs)))
	 (t1 (car (atomic-tp i1)))
	 (i2 (info-type (cadadr nargs)))
	 (t2 (car (atomic-tp i2))))
    (cond ((and (eq fn 'number-times) (eql t1 1) (ignorable-form (car nargs))) (cadr nargs));contagion
	  ((and (eql t2 1) (ignorable-form (cadr nargs))) (car nargs))
	  (t (setf (info-type info)
		   (type-and (info-type info)
			     (funcall (get fn 'type-propagator) fn i1 i2)))
	     `(call-global ,info ,fn ,nargs)))))
(setf (get 'number-times 'c1g) 'do-*/)
(setf (get 'number-divide 'c1g) 'do-*/)


(dolist (l `(eq eql equal equalp > >= < <= = /= length + - / * min max;FIXME get a good list here
		car cdr caar cadr cdar cddr caaar caadr cadar cdaar caddr cdadr cddar cdddr 
		caaaar caaadr caadar cadaar cdaaar caaddr cadadr cdaadr caddar cdadar cddaar
		cadddr cdaddr cddadr cdddar cddddr logand lognot logior logxor c-type complex-real
		complex-imag ratio-numerator ratio-denominator cnum-type si::number-plus si::number-minus
		si::number-times si::number-divide ;FIXME more
		,@(mapcar (lambda (x) (cdr x)) (remove-if-not (lambda (x) (symbolp (cdr x))) +cmp-type-alist+))))
  (si::putprop l t 'c1no-side-effects))

(setf (get 'cons 'c1no-side-effects) t)
(setf (get 'make-list 'c1no-side-effects) t)
(setf (get 'si::make-vector 'c1no-side-effects) t)
(setf (get 'complex 'c1no-side-effects) t)


;;bound type comparisons
;; only boolean eval const args


(defun test-to-tf (test)
  (let ((test (if (constantp test) (cmp-eval test) test)))
    (cond ((member test `(eql ,#'eql)) '(eql-is-eq eql-is-eq-tp))
	  ((member test `(equal ,#'equal)) '(equal-is-eq equal-is-eq-tp))
	  ((member test `(equalp ,#'equalp)) '(equalp-is-eq equalp-is-eq-tp)))))

(defun cons-type-length (type)
  (cond ((and (consp type) (eq (car type) 'cons)) (the seqind (+ 1 (cons-type-length (caddr type)))))
	(0)))

(defvar *frozen-defstructs* nil)

;; Return the most particular type we can EASILY obtain
;; from x.  
(defun result-type (x)
  (cond ((symbolp x)
	 (cmp-unnorm-tp (info-type (cadr (c1arg x)))))
	((constantp x) (type-of x))
	((and (consp x) (eq (car x) 'the)) (second x))
	(t t)))

;; (defun co1schar (f args)
;;   (declare (ignore f))
;;   (and (listp (car args)) (not *safe-compile*)
;; 	(cdr args)
;; 	(eq (caar args) 'symbol-name)
;; 	(c1expr `(aref (the string ,(second (car args)))
;; 			,(second args)))))

;; (si::putprop 'schar 'co1schar 'co1)

(si::putprop 'cons 'co1cons 'co1)
;; turn repetitious cons's into a list*

(defun cons-to-listc (x)
  (typecase x
    ((cons t (cons t null))
     (let ((y (cadr x)))
       (typecase y
	 ((cons (member cons) (cons t (cons t null)))
	  (let ((d (cons-to-listc (cdr y))))
	    (when d (cons (car x) d))))
	 (otherwise x))))))

(defun limit-list-call-args (form &aux (of form)(fn (pop of))
				    (x (nthcdr (- call-arguments-limit 2) of)))
  (if (cdr x)
      `(list* ,@(ldiff of x) ,(limit-list-call-args (cons fn x)))
      form))

(defun co1cons (f args &aux (tem (cons-to-listc args)))
  (declare (ignore f))
  (when tem
    (c1expr (limit-list-call-args
	     (if (equal '(nil) (last tem))
		 (cons 'list (butlast tem))
		 (cons 'list* tem))))))

;; Facilities for faster reading and writing from file streams.
;; You must declare the stream to be :in-file
;; or :out-file

;(si::putprop 'read-byte 'co1read-byte 'co1)
#-cygwin(si::putprop 'read-char 'co1read-char 'co1)
(si::putprop 'write-byte 'co1write-byte 'co1)
(si::putprop 'write-char 'co1write-char 'co1)

(defun fast-read (args read-fun)
  (cond
    ((and (not *safe-compile*)
	  (< *space* 2)
	  (null (second args))
	  (boundp 'si::*eof*))
     (cond
       ((atom (car args))
	(or (car args) (setq args (cons '*standard-input* (cdr args))))
	(let ((stream (car args))
	      (eof (third args)))
	  `(let ((ans 0))
	     (declare (fixnum  ans))
	     (cond ((fp-okp ,stream)
		    (setq ans  (sgetc1 ,stream))
		    (cond ((and (eql ans ,si::*eof*)
				(sfeof  ,stream))
			   ,eof)
			  (,(if (eq read-fun 'read-char1) '(code-char ans) 'ans))))
		   ((,read-fun ,stream  ,eof))
		   ))))
       (`(let ((.strm. ,(car args)))
	   (declare (type ,(result-type (car args)) .strm.))
	     ,(fast-read (cons '.strm. (cdr args)) read-fun)))))))

;; (defun co1read-byte (f args &aux tem) f
;;   (let* ((s (sgen "CO1READ-BYTE"))(nargs (cons s (cdr args))))
;;   (cond ((setq tem (fast-read nargs 'read-byte1))
;; 	 (let ((*space* 10))		;prevent recursion!
;; 	   (c1expr `(let ((,s ,(car args)))
;; 		      (if (= 1 (si::get-byte-stream-nchars ,s)) ,tem ,(cons f nargs)))))))))

(defun co1read-char (f args &aux tem)
  (declare (ignore f))
  (cond ((setq tem (fast-read args 'read-char1))
	 (let ((*space* 10))		;prevent recursion!
	   (c1expr tem)))))    

(defun cfast-write (args write-fun tp)
  (when (and (not *safe-compile*)
	     (< *space* 2)
	     (boundp 'si::*eof*))
    (let* ((stream (second args))(stream (or stream '*standard-output*)))
      (if (atom stream)
	  (let ((ch (sgen "CFAST-WRITE-CH")))
	    `(let ((,ch ,(car args)))
	       (if (and (fp-okp ,stream) (typep ,ch ',tp)) (sputc ,ch ,stream) (,write-fun ,ch ,stream))
	       ,ch))
	(let ((str (sgen "CFAST-WRITE-STR")))
	  `(let ((,str ,stream))
	     (declare (type ,(result-type stream) ,str))
	     ,(cfast-write (list (car args) str) write-fun tp)))))))


(defun co1write-byte (f args)
  (declare (ignore f))
  (let ((tem (cfast-write args 'write-byte 'fixnum)))
    (when tem 
      (let ((*space* 10))
	(c1expr tem)))))


(defun co1write-char (f args)
  (declare (ignore f))
  (let* ((tem (cfast-write args 'write-char 'character)))
    (when tem 
      (let ((*space* 10))
	(c1expr tem)))))

(defun aet-c-type (type)
  (or (cdr (assoc type +c-type-string-alist+)) (baboon)))


(si:putprop 'vector-push 'co1vector-push 'co1)
(si:putprop 'vector-push-extend 'co1vector-push 'co1)
(defun co1vector-push (f args) f
  (unless
   (or *safe-compile* t
       (> *space* 3)
       (null (cdr args))
       )
   (let ((*space* 10))
     (c1expr
      (let ((val (sgen "CO1VECTOR-PUSH-VAL"))
	    (v (sgen "CO1VECTOR-PUSH-V"))
	    (i (sgen "CO1VECTOR-PUSH-I"))
	    (dim (sgen "CO1VECTOR-PUSH-DIM")))
	`(let* ((,val ,(car args))
		(,v ,(second args))
		(,i (fill-pointer ,v))
		(,dim (array-total-size ,v)))
	   (declare (fixnum ,i ,dim))
	   (declare (type ,(result-type (second args)) ,v))
	   (declare (type ,(result-type (car args)) ,val))
	   (cond ((< ,i ,dim)
		  (the fixnum (si::fill-pointer-set ,v (the fixnum (+ 1 ,i))))
		  (si::aset ,val ,v ,i)
		  ,i)
		 (t ,(cond ((eq f 'vector-push-extend)
			    `(vector-push-extend ,val
						 ,v ,@(cddr args))))))))))))

(defun constant-fold-p (x)
  (cond ((constantp x) t)
	((atom  x) nil)
	((eq (car x) 'the)
	 (constant-fold-p (third x)))
	((and 
	      (symbolp (car x))
	      (eq (get (car x) 'co1)
		  'co1constant-fold))
	 (dolist (w (cdr x))
		 (or (constant-fold-p w)
		     (return-from constant-fold-p nil)))
	 t)
	(t nil)))

(defun co1constant-fold (f args )
  (cond ((and (fboundp f)
	      (dolist (v args t)
		      (or (constant-fold-p v)
			  (return-from co1constant-fold nil))))
	 (c1expr (cmp-eval (cons f args))))))




  
(defun narg-list-type (nargs &optional dot)
  (let* ((y (mapcar (lambda (x &aux (atp (atomic-tp (info-type (cadr x)))))
		     (cond ;((get-vbind x))
			   (atp (car atp));FIXME
			   ((get-vbind x))
			   ((lit-bind x))
			   ((new-bind))))
		    nargs)))
;    (when dot (setf (cdr (last y 2)) (car (last y)))) ;FIXME bump-pcons -- get rid of pcons entirely
    (let* ((s (when dot (car (last y))))(s (when s (unless (typep s 'proper-list) s)))(tp (info-type (cadar (last nargs)))));FIXME
      (cond ((when s (type>= #tproper-list tp)) #tproper-cons)
	    ((when s (type-and #tproper-list tp)) #tcons)
	    (t (when dot (setf (cdr (last y 2)) (car (last y)))) (object-type y))))))

(defun c1list (args)
  (let* ((info (make-info))
	 (nargs (c1args args info)))
    (cond ((not nargs) (c1nil))
	  ((setf (info-type info) (narg-list-type nargs))
	   `(call-global ,info list ,nargs)))))
(si::putprop 'list 'c1list 'c1)
      
(defun c1list* (args)
  (unless args (too-few-args 'list* 1 0))
  (let* ((info (make-info))
	 (nargs (c1args args info)))
    (cond ((not nargs) (c1nil))
	  ((not (cdr nargs)) (car nargs))
	  ((setf (info-type info) (narg-list-type nargs t))
	   `(call-global ,info ,(if (cddr nargs) 'list* 'cons) ,nargs)))))
(si::putprop 'list* 'c1list* 'c1)
(si::putprop 'cons  'c1list* 'c1)

(define-compiler-macro list  (&whole form) (limit-list-call-args form))
(define-compiler-macro list* (&whole form) (limit-list-call-args form))
