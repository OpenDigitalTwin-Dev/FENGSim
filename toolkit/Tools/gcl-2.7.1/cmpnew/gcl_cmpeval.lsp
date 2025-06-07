;;; CMPEVAL  The Expression Dispatcher.
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




(export '(si::define-compiler-macro
	  si::undef-compiler-macro
          si::define-inline-function) :si)

(in-package :compiler)

(si:putprop 'progn 'c1progn 'c1special)
(si:putprop 'progn 'c2progn 'c2)

(si:putprop 'si:structure-ref 'c1structure-ref 'c1)
(si:putprop 'structure-ref 'c2structure-ref 'c2)
(si:putprop 'structure-ref 'wt-structure-ref 'wt-loc)
(si:putprop 'si:structure-set 'c1structure-set 'c1)
(si:putprop 'structure-set 'c2structure-set 'c2)

(defun c1expr* (form info)
  (setq form (c1expr form))
  (add-info info (cadr form))
  form)

(defun readable-val (val)
  (cond ((not (arrayp val)))
	((not (si::staticp val)))))

(defun setq-p (form l) 
  (cond ((eq form l)) ((atom form) nil) ((or (setq-p (car form) l) (setq-p (cdr form) l)))))

(defun atomic-type-constant-value (atp &aux (a (car atp)))
  (when atp
    (typecase 
     a
     ((or function cons array))
     (otherwise (c1constant-value a (when (symbolp a) (symbol-package a)))))))

;; (defun atomic-type-constant-value (atp &aux (a (car atp)))
;;   (when atp
;;     (typecase 
;;      a
;;      ((or function cons array))
;;      (otherwise 
;;       (unless (eq a +opaque+)
;; 	(if (when (symbolp a) (get a 'tmp)) ;FIXME cdr
;; 	    (let ((a (get-var a)))
;; 	      (when a (c1var a)))
;; 	  (c1constant-value a (when (symbolp a) (symbol-package a)))))))))

;; (defun atomic-type-constant-value (atp &aux (a (car atp)))
;;   (when atp
;;     (typecase 
;;      a
;;      ((or function cons array))
;;      (otherwise (c1constant-value a (when (symbolp a) (symbol-package a)))))))

(defun c1expr-avct (res)
  (or (when (ignorable-form res)
	(atomic-type-constant-value (atomic-tp (info-type (cadr res)))))
      res))

(defun c1expr (form)
  (catch *cmperr-tag*
    (cond ((symbolp form)
           (cond ((constantp form) 
		  (let ((val (symbol-value form)))
		    (or 
		     (c1constant-value val nil)
		     `(location ,(make-info :type (object-type val)) (VV ,(add-constant form))))))
					;                 ((c1var form))))
                 ((c1expr-avct (c1var form))))) ;FIXME pcl
          ((consp form)
           (let ((fun (car form)))
	     (c1expr-avct (cond ((symbolp fun)
				 (c1symbol-fun form))
				((and (consp fun) (eq (car fun) 'lambda))
				 (c1symbol-fun (cons 'funcall form)))
				((and (consp fun) (eq (car fun) 'si:|#,|))
				 (cmperr "Sharp-comma-macro was found in a bad place."))
				(t (cmperr "The function ~s is illegal." fun))))))
          (t (c1constant-value form t)))))

(si::putprop 'si:|#,| 'c1sharp-comma 'c1special)
(si::putprop 'load-time-value 'c1load-time-value 'c1special)

(defun c1sharp-comma (arg)
  (c1constant-value (cons 'si:|#,| arg) t))

(defun c1load-time-value (arg)
  (c1constant-value
   (cons 'si:|#,|
	 (if *compiler-compile*
	     (let ((x (cmp-eval (car arg))));FIXME double cmp-eval with c1constant-value
	       (if (and (cdr arg) (cadr arg))
		   x
		 `(si::nani ,(si::address x))))
	   (car arg)))
   t))

;; (si::putprop 'si::define-structure 'c1define-structure 't1)

;; (defun c1define-structure (arg)
;;   (eval (cons 'si::define-structure arg))
;;   (add-object2 (cons '|#,| (cons 'si::define-structure arg)))
;;   nil)

(defun flags-pos (flag &aux (i 0))
  (declare (fixnum i))
  (dolist
      (v
       '((allocates-new-storage ans)            ;; might invoke gbc
	 (side-effect-p set)                    ;; no effect on arguments
	 (constantp)                            ;; always returns same result,
	                                        ;; double eval ok.
	 (result-type-from-args rfa)            ;; if passed args of matching
					        ;; type result is of result type
         (is)                                   ;; extends the `integer stack'.
	 (inline-types-function itf)            ;; car of ii is a function returning match info
	 (sets-vs-top svt)
	 (normalized-types nt)
	 (apply-arg aa)))                
    (cond ((member flag v :test 'eq)
	   (return-from flags-pos i)))
    (setq i (+ i 1)))
  (error "unknown opt flag"))

(defmacro flag-p (n flag)
  `(logbitp ,(flags-pos  flag)  ,n))

(defmacro flag-or (n flag)
  `(logior ,(ash 1 (flags-pos  flag))  ,n))

;; old style opts had '(args ret new-storage side-effect string)
;; these new-storage and side-effect have been combined into
;; one integer, along with several other flags.

(defun fix-opt (opt)
  (let ((a (cddr opt)))
    (cmpck (not (typep (car a ) 'fixnum)) "Obsolete optimization: use fix-opt ~s"  opt)
    (when (listp (car opt))
      (unless (flag-p (caddr opt) nt)
	(let ((s (uniq-sig (list (mapcar 'cmp-norm-tp (car opt)) (cmp-norm-tp (cadr opt))))))
	  (setf (car opt) (car s)
		(cadr opt) (cadr s)
		(caddr opt) (logior (caddr opt) (flags nt))))))
    opt))

;; some hacks for revising a list of optimizers.
#+revise
(progn
(defun output-opt (opt sym flag)
  (fix-opt opt)
  (format t "(push '(~(~s ~s #.(flags~)" (car opt) (second opt))
  (let ((o (third opt)))
    (if (flag-p o set) (princ " set"))
    (if (flag-p o ans) (princ " ans"))
    (if (flag-p o rfa) (princ " rfa"))
    (if (flag-p o constantp) (princ "constantp ")))
  (format t ")")
  (if (and (stringp (nth 3 opt))
	   (> (length (nth 3 opt)) 40))
      (format t "~%  "))
  (prin1 (nth 3 opt))
  (format t ")~%   ~((get '~s '~s)~))~%"  sym flag))

(defun output-all-opts (&aux  lis did)
  (sloop::sloop
   for v in ;(list (find-package "LISP"))
					(list-all-packages)
   do
   (setq lis
	 (sloop::sloop
	  for sym in-package (package-name v)
	  when (or (get sym 'inline-always)
		   (get sym 'inline-safe)
		   (get sym 'inline-unsafe))
	  collect sym))
   (setq lis (sort lis #'(lambda (x y) (string-lessp (symbol-name x)
						      (symbol-name y)))))
   do
   (sloop::sloop for sym in lis do
		 (format t "~%;;~s~% " sym)
       (sloop::sloop for u in '(inline-always inline-safe inline-unsafe)
		     do (sloop::sloop
			 for w in (nreverse (remove-duplicates
					    (copy-list (get sym u))
					    :test 'equal))
			 do (output-opt w  sym u)))))))				      
				

(defun result-type-from-args (f args)
  (when (and (or (not *compiler-new-safety*) (member f '(unbox box))));FIXME 
    (let* ((be (get f 'type-propagator))
	   (ba (and be ;(si::dt-apply be (cons f (mapcar 'coerce-to-one-valuea args))))));FIXME
		    (apply be (cons f (mapcar 'coerce-to-one-value args))))));FIXME
      (when ba
	(return-from result-type-from-args ba)))
    (dolist (v '(inline-always inline-unsafe))
      (let* ((w (get f v)))
	(if (and w (symbolp (caar w)) (flag-p (third (car w)) itf))
	    (return-from result-type-from-args (cadr (apply (caar w) args)))
	  (dolist (w w)
	    (fix-opt w)
	    (when (and
		   (flag-p (third w) result-type-from-args)
		   (>= (length args) (- (length (car w)) (length (member '* (car w)))))
		   (do ((a args (cdr a)) 
			(b (car w) (if (and (eq (cadr b) '*) (endp (cddr b))) b (cdr b))))
		       ((null a) t)
		       (unless (and (car a) (car b) (type>= (car b) (car a)))
			 (return nil))))
	      (return-from result-type-from-args (second w)))))))))

;; (defun result-type-from-args (f args)
;;   (when (and (or (not *compiler-new-safety*) (member f '(unbox box))));FIXME 
;;     (let* ((be (get f 'type-propagator))
;; 	   (ba (and be ;(si::dt-apply be (cons f (mapcar 'coerce-to-one-valuea args))))));FIXME
;; 		    (apply be (cons f (mapcar 'coerce-to-one-value args))))));FIXME
;;       (when ba
;; 	(return-from result-type-from-args (cmp-norm-tp ba))))
;;     (dolist (v '(inline-always inline-unsafe))
;;       (let* ((w (get f v)))
;; 	(if (and w (symbolp (caar w)) (flag-p (third (car w)) itf))
;; 	    (return-from result-type-from-args (cadr (apply (caar w) args)))
;; 	  (dolist (w w)
;; 	    (fix-opt w)
;; 	    (when (and
;; 		   (flag-p (third w) result-type-from-args)
;; 		   (>= (length args) (- (length (car w)) (length (member '* (car w)))))
;; 		   (do ((a args (cdr a)) 
;; 			(b (car w) (if (and (eq (cadr b) '*) (endp (cddr b))) b (cdr b))))
;; 		       ((null a) t)
;; 		       (unless (and (car a) (car b) (type>= (car b) (car a)))
;; 			 (return nil))))
;; 	      (return-from result-type-from-args (second w)))))))))
	

;; omitting a flag means it is set to nil.
(defmacro flags (&rest lis &aux (i 0))
  (dolist (v lis)
    (setq i (logior  i (ash 1 (flags-pos v)))))
  i)

;; Usage:
; (flagp-p (caddr ii) side-effect-p)
; (push '((integer integer) integer #.(flags const raf) "addii(#0,#1)")
;         (get '+ 'inline-always))

;(defun arg-appears (x y dep)
;  (cond ((atom y) nil)
;	((consp (car y))
;	 (or (arg-appears x (cdar y) t) (arg-appears x (cdr y) dep)))
;	(t
;	 (or (and (eq x (car y)) dep)
;	     (arg-appears x (cdr y) dep)))))

(defun cons-to-right (x)
  (and x (or (consp (car x)) (cons-to-right (cdr x)))))

(defun needs-pre-eval (x)
  (or (and (consp (car x)) (not (eq (caar x) 'quote)))
      (and (atom (car x))
	   (not (constantp (car x)))
	   (cons-to-right (cdr x)))))
;	   (arg-appears (car x) (cdr x) nil))))

(defun bind-before-cons (x y)
  (and y (consp (car y)) (atom (cadar y))
       (if (eq x (cadar y)) (caar y)
	 (bind-before-cons x (cdr y)))))
  
(defun pull-evals-int (x form lets)
  (if (atom x)
      (list (nreverse form) (nreverse lets))
    (let* ((s (if (needs-pre-eval x) (bind-before-cons (car x) lets) (car x)))
	   (lets (if s lets (cons (list (tmpsym) (car x)) lets)))
	   (s (or s (caar lets))))
      (pull-evals-int (cdr x) (cons s form) lets))))

(defun pull-evals (form)
  (let ((form (pull-evals-int (cdr form) (list (car form)) nil)))
    (values (car form) (cadr form))))

(defun binary-nest-int (form len)
  (declare (fixnum len) (list form))
  (if (> len 3)
      (binary-nest-int
       (cons (car form)
	     (cons (list (car form) (cadr form) (caddr form))
		   (cdddr form)))
       (1- len))
    form))

(defmacro let-wrap (lets form)
  `(if ,lets
       (list 'let* ,lets ,form)
     ,form))

(defun binary-nest (form env)
  (declare (ignore env))
  (let ((len (length form)))
    (declare (fixnum len))
    (if (> len 3)
	(let-wrap nil (binary-nest-int form len))
      ;; (multiple-value-bind (form lets) (values form nil);(pull-evals form)
      ;; 	  (let-wrap lets (binary-nest-int form len)))
      form)))

(si::putprop '* 'binary-nest 'si::compiler-macro-prop)
(si::putprop '+ 'binary-nest 'si::compiler-macro-prop)

(si::putprop 'logand 'binary-nest 'si::compiler-macro-prop)
(si::putprop 'logior 'binary-nest 'si::compiler-macro-prop)
(si::putprop 'logxor 'binary-nest 'si::compiler-macro-prop)

(si::putprop 'max 'binary-nest 'si::compiler-macro-prop)
(si::putprop 'min 'binary-nest 'si::compiler-macro-prop)

(si::putprop 'gcd 'binary-nest 'si::compiler-macro-prop)
(si::putprop 'lcm 'binary-nest 'si::compiler-macro-prop)

(si::putprop '- 'binary-nest 'si::compiler-macro-prop)
(si::putprop '/ 'binary-nest 'si::compiler-macro-prop)


(defun multiple-value-bind-expander (form env)
  (declare (ignore env))
  (if (and (consp (caddr form)) (eq (caaddr form) 'values))
      (let ((l1 (length (cadr form))) (l2 (length (cdaddr form))))
      `(let (,@(mapcar 'list (cadr form) (cdaddr form))
	       ,@(when (> l1 l2)
		   (nthcdr l2 (cadr form))))
	 ,@(when (> l2 l1) (nthcdr l1 (cdaddr form)))
	 ,@(cdddr form)))
    form))
(si::putprop 'multiple-value-bind 'multiple-value-bind-expander 'si::compiler-macro-prop)

;FIXME apply-expander
;; (defun funcall-expander (form env &aux x);FIXME inlinable-fn?
;;   (declare (ignore env))
;;   (cond ((and (consp (cadr form)) (eq (caadr form) 'lambda)) (cdr form))
;; 	((and (consp (cadr form)) (eq (caadr form) 'function)
;; 	      (setq x (si::funid-p (cadadr form))))
;; 	 `(,x ,@(cddr form)))
;; 	((constantp (cadr form)) `(,(cmp-eval (cadr form)) ,@(cddr form)))
;; 	(form)))
;; (si::putprop 'funcall 'funcall-expander 'si::compiler-macro-prop)

(defun logical-binary-nest (form env)
  (declare (ignore env))
  (if (> (length form) 3)
      (multiple-value-bind (form lets) (pull-evals form)
	(let (r)
	  (do ((f (cdr form) (cdr f)))
	      ((null (cdr f))
	       (let-wrap lets (cons 'and (nreverse r))))
	    (push (list (car form) (car f) (cadr f)) r))))
	form))

(si::putprop '> 'logical-binary-nest 'si::compiler-macro-prop)
(si::putprop '>= 'logical-binary-nest 'si::compiler-macro-prop)
(si::putprop '< 'logical-binary-nest 'si::compiler-macro-prop)
(si::putprop '<= 'logical-binary-nest 'si::compiler-macro-prop)
(si::putprop '= 'logical-binary-nest 'si::compiler-macro-prop)

(si::putprop 'char> 'logical-binary-nest 'si::compiler-macro-prop)
(si::putprop 'char>= 'logical-binary-nest 'si::compiler-macro-prop)
(si::putprop 'char< 'logical-binary-nest 'si::compiler-macro-prop)
(si::putprop 'char<= 'logical-binary-nest 'si::compiler-macro-prop)
(si::putprop 'char= 'logical-binary-nest 'si::compiler-macro-prop)

(defun logical-outer-nest (form env)
  (declare (ignore env))
  (if (> (length form) 3)
      (multiple-value-bind (form lets) (pull-evals form)
	(let (r)
	  (do ((f (cdr form) (cdr f)))
	      ((null (cdr f))
	       (let-wrap lets (cons 'and (nreverse r))))
	    (do ((g (cdr f) (cdr g))) ((null g))
	    (push (list (car form) (car f) (car g)) r)))))
    form))

(si::putprop '/= 'logical-outer-nest 'si::compiler-macro-prop)
(si::putprop 'char/= 'logical-outer-nest 'si::compiler-macro-prop)

(defun incr-to-plus (form env)
  (declare (ignore env))
  `(+ ,(cadr form) 1))

(defun decr-to-minus (form env)
  (declare (ignore env))
  `(- ,(cadr form) 1))

(si::putprop '1+ 'incr-to-plus 'si::compiler-macro-prop)
(si::putprop '1- 'decr-to-minus 'si::compiler-macro-prop)

(defun plusp-compiler-macro (form env)
  (declare (ignore env))
  (if (and (cdr form) (endp (cddr form)))
      `(> ,(cadr form) 0)
    form))
(si::putprop 'plusp 'plusp-compiler-macro 'si::compiler-macro-prop)

(defun minusp-compiler-macro (form env)
  (declare (ignore env))
  (if (and (cdr form) (endp (cddr form)))
      `(< ,(cadr form) 0)
    form))
(si::putprop 'minusp 'minusp-compiler-macro 'si::compiler-macro-prop)

(defun zerop-compiler-macro (form env)
  (declare (ignore env))
  (if (and (cdr form) (endp (cddr form)))
      `(= ,(cadr form) 0)
    form))
(si::putprop 'zerop 'zerop-compiler-macro 'si::compiler-macro-prop)

(defun local-aliases (var excl &aux (bind (get-vbind var)) res)
  (when bind
    (let ((e (member-if-not 'var-p *vars*)))
      (do ((x *vars* (cdr x))) ((eq x e) res)
	(let ((cx (car x)))
	  (unless (member cx excl)
	    (when (eq bind (get-vbind cx))
	      (push cx res))))))))

(defun c1infer-tp (args)
  (let* ((n (pop args))
	 (v (c1vref n))
	 (x (car v))
	 (tpi (ensure-known-type (pop args)))
	 (tp (type-and (var-type x) tpi))
	 (l (local-aliases x nil))
	 (tp (reduce 'type-and l :key 'var-type :initial-value tp))
	 (l (mapc (lambda (x) (do-setq-tp x nil tp)) l))
	 (res (c1expr (car args)))
	 (ri (cadr res)))
    (if (exit-to-fmla-p)
	(let ((info (make-info)))
	  (add-info info ri)
	  (setf (info-type info) (info-type ri))
	  `(infer-tp ,info ,l ,tpi ,res))
	res)))


(defun c2infer-tp (x tp fm)
  (declare (ignore x tp))
  (c2expr fm))
(si::putprop 'infer-tp 'c1infer-tp 'c1)
(si::putprop 'infer-tp 'c2infer-tp 'c2)












(defconstant +cnum-tp-alist+ `((,#tfixnum . ,(c-type 0))
			       (,#tbignum . ,(c-type (1+ most-positive-fixnum)))
			       (,#tratio  . ,(c-type 1/2))
			       (,#tshort-float . ,(c-type 0.0s0))
			       (,#tlong-float  . ,(c-type 0.0))
			       (,#tfcomplex  . ,(1+ si::c-type-max))
			       (,#tdcomplex  . ,(+ 2 si::c-type-max))
			       (,#t(complex rational) . ,(c-type #c(0 1)))))

(defconstant +hash-index-type+ #t(or (integer -1 -1) seqind))


(defun identity-expander (form env)
  (declare (ignore env))
  (if (cddr form) form (cadr form)))
(si::putprop 'identity 'identity-expander 'si::compiler-macro-prop)

;; (defun seqind-wrap (form)
;;   (if *safe-compile*
;;       form
;;     `(the seqind ,form)))

(defun fboundp-expander (form env)
  (declare (ignore env))
  `(si::fboundp-sym (si::funid-sym ,(cadr form))))
(si::putprop 'fboundp 'fboundp-expander 'si::compiler-macro-prop)

;; (defun maphash-expander (form env)
;;   (declare (ignore env))
;;   (let ((block (tmpsym))(tag (gensym)) (ind (gensym)) (key (gensym)) (val (gensym)))
;;     `(block 
;;       ,block
;;       (let ((,ind -1))
;; 	(declare (,+hash-index-type+ ,ind))
;; 	(tagbody 
;; 	 ,tag
;; 	 (when (< (setq ,ind (si::next-hash-table-index ,(caddr form) (1+ ,ind))) 0)
;; 	   (return-from ,block))
;; 	 (let ((,key (si::hash-key-by-index ,(caddr form) ,ind))
;; 	       (,val (si::hash-entry-by-index ,(caddr form) ,ind)))
;; 	   (funcall ,(cadr form) ,key ,val))
;; 	 (go ,tag))))))
;; (si::putprop 'maphash 'maphash-expander 'si::compiler-macro-prop)
	
;; (defun array-row-major-index-expander (form env &optional (it 0))
;;   (declare (fixnum it)(ignorable env))
;;   (let ((l (length form)))
;;     (cond ((= l 2) 0)
;; 	  ((= l 3) (seqind-wrap (caddr form)))
;; 	  (t (let ((it (1+ it))
;; 		   (fn (car form))
;; 		   (ar (cadr form))
;; 		   (first (seqind-wrap (caddr form)))
;; 		   (second (seqind-wrap (cadddr form)))
;; 		   (rest (cddddr form)))
;; 	       (array-row-major-index-expander
;; 		`(,fn ,ar ,(seqind-wrap
;; 			    `(+
;; 			      ,(seqind-wrap
;; 				`(* ,first (array-dimension ,ar ,it))) ,second)) ,@rest)
;; 		nil it))))))

;;(si::putprop 'array-row-major-index 'array-row-major-index-expander 'si::compiler-macro-prop)

;; (defmacro with-pulled-array (bindings form &body body) ;FIXME
;;   `(let ((,(car bindings) (cadr ,form)))
;;      (let ((,(cadr bindings) `((,(tmpsym) ,,(car bindings)))))
;;        (let ((,(caddr bindings) (or (caar ,(cadr bindings)) ,(car bindings))))
;; 	 ,@body))))
	

;; (defun aref-expander (form env)
;;   (declare (ignore env))
;;   (with-pulled-array
;;    (ar lets sym) form
;;    (let ((isym (tmpsym)))
;;      (let ((lets (append lets `((,isym (array-row-major-index ,sym ,@(cddr form)))))))
;;        (let-wrap lets `(compiler::cmp-aref ,sym ,isym))))))

;; (si::putprop 'aref 'aref-expander 'si::compiler-macro-prop)
;; (si::putprop 'row-major-aref 'aref-expander 'si::compiler-macro-prop)

;; (defun aset-expander (form env)
;;   (declare (ignore env))
;;   (let ((form (if (eq (car form) 'si::aset-wrap) form 
;; 		(cons (car form) (append (cddr form) (list (cadr form)))))));FIXME
;;     (with-pulled-array
;;      (ar lets sym) form
;;      (let ((isym (tmpsym)))
;;        (let ((lets (append lets `((,isym (array-row-major-index ,sym ,@(butlast (cddr form))))))))
;; 	 (let-wrap lets `(compiler::cmp-aset ,sym ,isym ,(car (last form)))))))))

;; (si::putprop 'si::aset 'aset-expander 'si::compiler-macro-prop)
;; (si::putprop 'si::aset-wrap 'aset-expander 'si::compiler-macro-prop)
;FIXME -- test and install this and svref, CM 20050106
;(si::putprop 'svset 'aset-expander 'si::compiler-macro-prop)

;; (defun array-dimension-expander (form env)
;;   (declare (ignore env))
;;   (with-pulled-array
;;    (ar lets sym) form
;;    (let-wrap lets `(compiler::cmp-array-dimension ,sym ,(caddr form)))))

;;(si::putprop 'array-dimension 'array-dimension-expander 'si::compiler-macro-prop)

(defmacro inlinable-fn (a) 
  `(or (constantp ,a) (and (consp ,a) (member (car ,a) '(function lambda)))))

(define-compiler-macro or (&whole form)
  (cond ((endp (cdr form)) nil)
	((endp (cddr form)) (cadr form))
	((cmp-macroexpand `(,(pop form) ,(pop form) (or ,@form))))))

(defvar *basic-inlines* nil)

(defun comment (x) x)
(defun c1comment (args)
  (list 'comment (make-info :type t  :flags (iflags side-effects))
	(let ((x (car args)))
	  (if (constantp x) (cmp-eval x) x))))
(defun c2comment (comment &aux (comment-string (comment-string comment)))
  (when *annotate*
    (wt-nl "/*")(princ comment-string *compiler-output1*)(wt "*/")))
(si::putprop 'comment 'c1comment 'c1)
(si::putprop 'comment 'c2comment 'c2)



(defvar *inl-hash* (make-hash-table :test 'eq))

(defun ibtp (t1 t2 &aux (a1 (atomic-tp t1))(a2 (atomic-tp t2)))
  (if (unless (type-and t1 t2) (and a1 a2 (listp t1) (listp t2) (equal (car t1) (car t2))))
      (car t1) (type-or1 t1 t2)))

(defun coalesce-inl (cl inl tps rt &aux (lev (this-safety-level)))
  (when (> lev (third inl))
    (keyed-cmpnote (list (car cl) 'inl-hash 'inl-hash-coalesce)
		   "Coalescing safety ~s: ~s ~s" (car cl) (third inl) lev)
    (setf (third inl) lev))
  (unless (type<= rt (cdr (fifth inl)))
    (let ((n (ibtp (cdr (fifth inl)) rt)))
      (keyed-cmpnote (list (car cl) 'inl-hash 'inl-hash-coalesce)
		     "Coalescing return-type ~s: ~s ~s" (car cl) (cdr (fifth inl)) n)
      (setf (cdr (fifth inl)) n)))
  (mapl (lambda (x y &aux (cx (car x))(cy (car y)))
	  (unless (type<= cy cx)
	    (let ((n (ibtp cx cy)))
	      (keyed-cmpnote (list (car cl) 'inl-hash 'inl-hash-coalesce)
			     "Coalescing arg-type ~s: ~s ~s" (car cl) cx n)
	      (setf (car x) n))))
	(car inl) tps))

(defun can-coalesce (x tr inl tps)
  (and (equal tr (second x))
       (string= (car (last inl)) (car (last x)))
       (>= (car inl) (third x))
       (eql (length tps) (length (car x)))
       (every 'type>= tps (car x))))

(defun remove-comment (s &aux (b (string-match #v"/\\*" s))(e (string-match #v"\\*/" s)))
  (if (< -1 b e) (string-concatenate (subseq s 0 b) (remove-comment (subseq s (+ e 2)))) s))

(defun lit-inl2 (form &aux (lf (eq 'lit (car form))))
  (list (this-safety-level)
	(mapcar (lambda (x) (assert (eq (car x) 'ub)) (third x)) (when lf (fifth form)))
	(cons (when lf (third form)) (info-type (cadr form)))
	(if lf (remove-comment (fourth form)) "")))

(defun cl-to-fn (cl)
  (when (null (cdr (last cl)))
    (let ((fn (car cl)))
      (when (symbolp fn)
	(unless (local-fun-p fn)
	  fn)))))

(defun get-inl-list (cl &optional set &aux (fn (cl-to-fn cl)))
  (when fn
    (or (gethash fn *inl-hash*)
	(when set
	  (setf (gethash fn *inl-hash*) (list nil))))))

(defun inls-match (cl fms &aux (lev (this-safety-level))
			    (tps (mapcar (lambda (x) (info-type (caddr x))) fms)))
  (when (member-if-not 'atomic-tp tps)
    (car (member tps (car (get-inl-list cl))
		 :test (lambda (x y &aux (cy (car y)))
			 (when (<= lev (third y))
			   (when (eql (length x) (length cy))
			     (every 'type<= x cy))))))))

(defun ?add-inl (cl fms fm)
  (unless (or (member-if 'atomic-tp fms :key (lambda (x) (info-type (caddr x))))
	      (atomic-tp (info-type (cadr fm))) (exit-to-fmla-p)); (inls-match cl fms)
    (let* ((tps (mapcar (lambda (x) (info-type (caddr x))) fms))
	   (tr (mapcar (lambda (x &aux (v (car (last x))))
			 (when (and (consp v) (eq (car v) 'var))
			   (position (cddr v) fms :key 'cdddr :test 'equalp)));FIXME
		       (if (eq (car fm) 'var) (list (list fm)) (fifth fm))))
	   (nat (let ((i -1)) (mapcan (lambda (x &aux (y (incf i))) (unless (atomic-tp x) (list y))) tps))))
      (unless (or (member nil tr) (set-difference nat tr))
	(let* ((pl (get-inl-list cl t))
	       (inl (lit-inl2 fm))
	       (z (member-if (lambda (x) (can-coalesce x tr inl tps)) (car pl))))
	  (cond (z (coalesce-inl cl (car z) tps (cdr (third inl)))
		   (setf (cdr z) (remove-if (lambda (x) (can-coalesce x tr inl tps)) (cdr z))))
		(pl
		 (let ((x (list* tps tr inl)))
		   (keyed-cmpnote (list (car cl) 'inl-hash 'inl-hash-add)
				  "Adding inl-hash ~s: ~s" (car cl) x)
		   (push x (car pl))))))))))

(defun prepend-comment (form s)
  (if *annotate*
      (si::string-concatenate "/* " (prin1-to-string form) " */" (remove-comment s))
      s))

(defun apply-inl (cl fms &aux (inl (inls-match cl fms)))
  (when inl
    (let* ((c1fms (mapcar (lambda (x) (cdr (nth x fms))) (second inl))))
      (unless (member-if-not (lambda (x)
			       (case (car x)
				 (var (eq (var-kind (caaddr x)) 'lexical))
				 ((lit location) t)))
			     c1fms)
	(cond ((zerop (length (car (last inl))))
	       (let* ((x (car c1fms))(h (pop x))
		      (i (copy-info (pop x))))
		 (setf (info-type i) (type-and (cdr (fifth inl)) (info-type i)))
		 (keyed-cmpnote (list (car cl) 'inl-hash 'inl-hash-apply)
				"Applying var inl-hash ~s" (car cl))
		 (list* h i x)))
	      ((let ((x (c1lit (list (car (fifth inl)) (prepend-comment (cons 'applied cl) (car (last inl)))) (mapcar 'list  (fourth inl) c1fms))))
		 (setf (info-type (cadr x)) (type-and (cdr (fifth inl)) (info-type (cadr x))))
		 (keyed-cmpnote (list (car cl) 'inl-hash 'inl-hash-apply)
				"Applying inl-hash ~s: ~s: ~s" (car cl) (fourth x))
		 x)))))))

(defun dump-inl-hash (f)
  (with-open-file (s f :direction :output)
    (prin1 '(in-package :compiler) s)
    (terpri s)
    (maphash (lambda (x y)
	       (prin1
		`(setf (gethash ',x *inl-hash*)
		       (list
			(list
			 ,@(mapcar (lambda (z)
				     `(list (mapcar 'uniq-tp ',(mapcar 'export-type (pop z)))
					    ',(pop z) ',(pop z) ',(pop z)
					    (cons ',(caar z) (uniq-tp ',(cdar z)))
					    ,(cadr z)))
				   (car y)))))
		      s)
	       (terpri s))
	     *inl-hash*))
  nil)

(defun show-inls (fn)
  (mapcar (lambda (x) (list (mapcar 'cmp-unnorm-tp (car x)) (third x) (car (last x))))
	  (car (gethash fn *inl-hash*))))

(defun c1inline (args env inls)
  (let* ((cl (pop args))(fm (pop args)))
    (or (apply-inl cl inls)
	(let* ((nargs (under-env env (c1let-* (cdr fm) t inls))))
	  (case (car nargs)
	    ((var lit)
	     (?add-inl cl inls nargs)
	     (when (stringp (fourth nargs)) (setf (fourth nargs) (prepend-comment cl (fourth nargs))))
	     nargs)
	    (otherwise (list 'inline (copy-info (cadr nargs)) cl nargs)))))))

(defvar *annotate* nil)

(defun comment-string (comment)
  (when *annotate*
    (mysub (mysub (write-to-string comment :length 3 :level 3) "/*" "_*") "*/" "*_")))

(defun c2inline (comment expr &aux (comment-string (comment-string comment)))
  (when *annotate* (wt-nl "/*")(princ comment-string *compiler-output1*)(wt "*/"))
  (c2expr expr)
  (when *annotate* (wt-nl "/* END ")(princ comment-string *compiler-output1*)(wt "*/")))
(si::putprop 'inline 'c1inline 'c1)
(si::putprop 'inline 'c2inline 'c2)

;; (defun c1size (form)
;;   (cond ((atom form) 0)
;; 	((1+ (+ (c1size (car form)) (c1size (cdr form)))))))


;; (defvar *inline-forms* nil)

;; (defun copy-vars (form)
;;   (cond ((var-p form) (setf (var-store form) (var-kind form)))
;; 	((consp form) (copy-vars (car form)) (copy-vars (cdr form)))))

;; (defun set-vars (form)
;;   (cond ((var-p form) (setf (var-kind form) (var-store form)))
;; 	((consp form) (set-vars (car form)) (set-vars (cdr form)))))

;; (defun global-ref-p (form)
;;   (cond ((and (var-p form) (member (var-kind form) '(global special))))
;; 	((atom form) nil)
;; 	((or (global-ref-p (car form)) (global-ref-p (cdr form))))))

;; (defun closure-p (form)
;;   (and (eq (car form) 'function)
;;        (eq (caaddr form) 'lambda)
;;        (or (do-referred (s (cadr (caddr form)))
;; 			(unless (member s (caaddr (caddr form))) (return t)))
;; 	   (global-ref-p form))))

;; (defun vv-p (form)
;;   (cond ((atom form) nil)
;; 	((and (eq (car form) 'location) (listp (caddr form))
;; 	      (or (eq (caaddr form) 'vv)
;; 		  (and (member (caaddr form) '(fixnum-value character-value long-float-value short-float-value fcomplex-value dcomplex-value))
;; 		       (cadr (caddr form))))))
;; 	((or (vv-p (car form)) (vv-p (cdr form))))))

;;FIXME
;(dolist (l '(typep coerce constantly complement open load delete-package import compile compile-file
;		  error cerror warn break get-setf-method make-list))
;  (si::putprop l t 'cmp-no-src-inline))

;; (defvar *prop-hash* nil)
					; (make-hash-table :test 'equal))
(defvar *src-inline-recursion* nil)
(defvar *prev-sri* nil)

(defvar *src-hash* (make-hash-table :test 'eq))

;; (defun src-inlineable (form)
;;   (let ((n (car form)))
;;     (and (symbolp n)
;; 	 (not (get n 'cmp-no-src-inline))
;; 	 (fboundp n)
;; 	 (or (gethash n *src-hash*)
;; 	     (setf (gethash n *src-hash*)
;; 		   (let ((fn (symbol-function n))) (when (functionp fn) (function-lambda-expression fn)))))
;; 	 (or (inline-asserted n)
;; 	     (eq (symbol-package n) (load-time-value (find-package 'c)))
;; 	     (multiple-value-bind (s k) (find-symbol (symbol-name n) 'lisp)
;; 				  (when (eq n s) (eq k :external)))))))

;; (defun mark-for-hash-inlining (fms)
;;   (let ((i 0)
;; 	(c1t (c1t))
;; 	(c1nil (c1nil)))
;;     (mapl (lambda (x)
;; 	    (when (car x)
;; 	      (when (or (eq (car x) c1t) (eq (car x) c1nil))
;; 		(setf (car x) (list (caar x) (copy-info (cadar x)) (caddar x))))
;; 	      (setf (info-unused1 (cadar x)) (incf i)))) fms)))

;; (defun inline-hasheable (form fms c1)
;;   (let ((cp (member-if 'closure-p fms))
;; 	(vvp (vv-p (if (eq (car (fourth c1)) 'let*) (cddddr (fourth c1)) c1)))
;; 	(rec (and (boundp '*recursion-detected*) (eq *recursion-detected* t))))
;;     (when cp (keyed-cmpnote 'inline-hash "not hashing ~s due to closure~%" form))
;;     (when vvp (keyed-cmpnote 'inline-hash "not hashing ~s due to vv objs~%" form))
;;     (when rec (keyed-cmpnote 'inline-hash "not hashing ~s due to recursion~%" form))
;;     (not (or cp vvp rec))))

	   
;; (defun info-form-alist (o n)
;;   (mapcan (lambda (o)
;; 	    (when o
;; 	      (let ((n (car (member (info-unused1 (cadr o)) n :key (lambda (x) (when x (info-unused1 (cadr x))))))))
;; 		(when n (list (cons o n)))))) o))

;; (defun array-replace (x y z)
;;   (do ((i 0 (1+ i))) ((>= i (length x)))
;;     (when (eq y (aref x i))
;;       (setf (aref x i) z))))

;; (defun info-replace-var (x y z)
;;   (array-replace (info-referred-array x) y z)
;;   (array-replace (info-changed-array x) y z))

;; (defun info-replace-var (x y z)
;;   (nsubst z y (info-ref x))
;;   (nsubst z y (info-ch x)))

;; (defun info-var-match (i v)
;;   (or (is-referred v i) (is-changed v i)))

;; (defun collect-matching-vars (ov f)
;;   (cond ((var-p f) (when (or (member f ov) (list-split (var-aliases f) ov)) (list f)))
;; 	((info-p f) (let (r)
;; 		      (dolist (ov ov r)
;; 			(when (info-var-match f ov) (push ov r)))))
;; 	((atom f) nil)
;; 	((nunion (collect-matching-vars ov (car f)) (collect-matching-vars ov (cdr f))))))

;; (defun collect-matching-info (ov f)
;;   (cond ((info-p f) (when (member-if (lambda (x) (info-var-match f x)) ov) (list f)))
;; 	((atom f) nil)
;; 	((nunion (collect-matching-info ov (car f)) (collect-matching-info ov (cdr f))))))

;; (defun fms-fix (f fms)
;;   (let* ((vv (collect-matching-vars (third f) fms))
;; 	 (ii (collect-matching-info vv fms))
;; 	 (nv (mapcar 'copy-var vv))
;; 	 (a (mapcar 'cons vv nv))
;; 	 (nv (mapc (lambda (x) (setf (var-aliases x) (sublis a (var-aliases x)))) nv))
;; 	 (ni (mapcar 'copy-info ii))
;; 	 (ni (mapc (lambda (x) (mapc (lambda (y z) (info-replace-var x y z)) vv nv)) ni)))
;;     (sublis (nconc a (mapcar 'cons ii ni)) fms)))
  
;; (defun get-inline-h (form prop fms)

;;   (let ((h (when *prop-hash* (gethash prop *prop-hash*))))

;;     (when h

;;       (unless (acceptable-inline h form (cddr prop))
;; 	(return-from get-inline-h (cons nil (cdr h))))

;;       (let* ((f (car h))
;; 	     (fms (fms-fix (fourth f) fms))
;; 	     (al (info-form-alist (car (last h)) fms))
;; 	     (nfs (mapcar 'cdr al))
;; 	     (oi (cadr f))
;; 	     (info (make-info))
;; 	     (al (cons (cons oi info) al))
;; 	     (al (cons (cons (caddr f) (with-output-to-string (s) (princ form s))) al)))

;; 	(set-vars f)
;; 	(setf (info-type info) (info-type oi))
;; 	(dolist (l nfs) (add-info info (cadr l)))

;; 	(cons (sublis al f) (cdr h))))))
  

;; (defun acceptable-inline (h form tpis)
;;   (let* ((c1 (car h))
;; 	 (sz (cadr h))
;; 	 (d (and c1
;; 		 (inline-possible (car form))
;; 		 (or (< sz (* 1000 (- 3 (max 0 *space*))))
;; 		     (and (< *space* 3) (member-if (lambda (x) (and (atomic-tp (car x)) (functionp (cadar x)))) tpis))))))
;;     (if d
;; 	(keyed-cmpnote 'inline "inlining ~s ~s~%" form (not (not h)))
;;       (keyed-cmpnote 'inline "not inlining ~s ~s ~s ~s~%" form sz (* 1000 (- 3 (max 0 *space*))) tpis))
;;     d))


;; (defun fms-callees (fms)
;;   (mapcan
;;    (lambda (x) 
;;      (when (eq (car x) 'function) 
;;        (let ((fun (caaddr x))) 
;; 	 (when (fun-p fun)
;; 	   (cadr (fun-call fun)))))) fms))

;; (defun push-callees (fms)
;;   (let ((fc (fms-callees fms)))
;;     (setq *callees* (nunion *callees* fc :test 'eq :key 'car))))

;; (defun bind-all-vars-int (form nf bindings)
;;   (cond ((null form)
;; 	 (list bindings (nreverse nf)))
;; 	((consp (car form))
;; 	 (let ((lwf (bind-all-vars-int (cdar form) (list (caar form)) bindings)))
;; 	   (bind-all-vars-int (cdr form) (cons (cadr lwf) nf) (car lwf))))
;; 	(t
;; 	 (let* ((sym (if (symbolp (car form)) (cdr (assoc (car form) bindings)) (car form)))
;; 		(bindings (if sym bindings (cons (cons (car form) (tmpsym)) bindings)))
;; 		(sym (or sym (cdar bindings))))
;; 	   (bind-all-vars-int (cdr form) (cons sym nf) bindings)))))

;; (defun bind-all-vars (form)
;;   (if (atom form) form
;;     (let ((res (bind-all-vars-int (cdr form) (list (car form)) nil)))
;;       (if (car res)
;; 	  (list 'let* (mapcar (lambda (x) (list (cdr x) (car x))) (nreverse (car res)))
;; 		(cadr res))
;; 	(cadr res)))))


;; (defun if-protect-fun-inf (form env)
;;   (declare (ignore env))
;;   (cons (car form)
;; 	(cons (cadr form)
;; 	      (cons (bind-all-vars (caddr form))
;; 		    (if (cadddr form) (list (bind-all-vars (cadddr form))))))))
		
;(defvar *callees* nil)

(defun maybe-reverse-type-prop (dt f)
  (unless *safe-compile*;FIXME push-vbind/c1var copy  (when (consp f) (eq (car f) 'lit))
    (set-form-type f (coerce-to-one-value dt))))

;; (defun maybe-reverse-type-prop (dt f)
;;   (unless *safe-compile*
;;     (set-form-type f dt)))

(defun cll (fn)
  (car (member (sir-name fn) *src-inline-recursion* :key 'caar)))

(defun inline-sym-src (n)
  (and (inline-possible n)
       (or (inline-asserted n)
	   (get n 'consider-inline)
	   (multiple-value-bind (s k) (find-symbol (symbol-name n) :cl) 
				(when (eq n s) (eq k :external))))
       (or (local-fun-src n)
	   (let ((fn (when (fboundp n) (symbol-function n))))
	     (when (functionp fn) 
	       (unless (typep fn 'funcallable-std-instance);FIXME really just need to check/handle for closure
		 (values (or (gethash fn *src-hash*) (setf (gethash fn *src-hash*) (function-lambda-expression fn))))))))))

;; (defun inline-sym-src (n)
;;   (and (inline-possible n)
;;        (or (inline-asserted n)
;; 	   (eq (symbol-package n) (load-time-value (find-package :c)))
;; 	   (eq (symbol-package n) (load-time-value (find-package :libm)))
;; 	   (eq (symbol-package n) (load-time-value (find-package :libc)))
;; 	   (multiple-value-bind (s k) (find-symbol (symbol-name n) :cl) 
;; 				(when (eq n s) (eq k :external))))
;;        (or (local-fun-src n)
;; 	   (let ((fn (when (fboundp n) (symbol-function n))))
;; 	     (when (functionp fn) 
;; 	       (unless (typep fn 'generic-function)
;; 		 (values (or (gethash fn *src-hash*) (setf (gethash fn *src-hash*) (function-lambda-expression fn))))))))))

;; (defun inline-sym-src (n)
;;   (and (inline-possible n)
;;        (or (inline-asserted n)
;; 	   (eq (symbol-package n) (load-time-value (find-package 'c)))
;; 	   (eq (symbol-package n) (load-time-value (find-package "libm")))
;; 	   (eq (symbol-package n) (load-time-value (find-package "libc")))
;; 	   (multiple-value-bind (s k) (find-symbol (symbol-name n) 'lisp) 
;; 				(when (eq n s) (eq k :external))))
;;        (or (local-fun-src n)
;; 	   (let ((fn (when (fboundp n) (symbol-function n))))
;; 	     (when (functionp fn) (values (function-lambda-expression fn)))))))

;; (defun inline-sym-src (n)
;;   (and (inline-possible n)
;;        (or (inline-asserted n)
;; 	   (eq (symbol-package n) (load-time-value (find-package 'c)))
;; 	   (multiple-value-bind (s k) (find-symbol (symbol-name n) 'lisp) 
;; 				(when (eq n s) (eq k :external))))
;;        (or (local-fun-src n)
;; 	   (gethash n *src-hash*) 
;; 	   (setf (gethash n *src-hash*)
;; 		 (let ((fn (when (fboundp n) (symbol-function n))))
;; 		   (when (functionp fn) (function-lambda-expression fn)))))))

(defun inline-src (fn)
  (unless *compiler-new-safety*
    (when (> *speed* 0)
      (cond ((symbolp fn) (inline-sym-src fn))
	    ((functionp fn) (local-fun-src fn))
	    ((and (consp fn) (eq (car fn) 'lambda)) fn)))))

(defun ttl-tag-src (src tag &optional block &aux (h (pop src)) (ll (pop src)))
  (multiple-value-bind
   (doc decls ctps body)
   (parse-body-header src)
   (let* ((aux (member '&aux ll));FIXME centralize with new-defun-args
	  (ll (ldiff ll aux))
 	  (non-aux (mapcan (lambda (x &aux (lp (listp x)))
			  (cons (if lp (if (listp (car x)) (cadar x) (car x)) x)
				(when (when lp (cddr x)) (list (caddr x))))) ll))
	  (non-aux (set-difference non-aux '(&optional &rest &key &allow-other-keys)))
	  (od (split-decls non-aux decls))
	  (rd (cons `(declare (optimize (safety ,(decl-safety decls)))) (pop od)))
	  (oc (split-ctps non-aux ctps))
	  (rc (pop oc))
	  (n (blocked-body-name body))
	  (body (if n (cddar body) body))
	  (n (or n block))
	  ;rebind args beneath ttl tag for tail recursion with closures
	  (bind (when block (mapcar 'list non-aux non-aux)))
	  (bind (nconc bind (cdr aux)))
;	  (bind (nconc (mapcar 'list non-aux non-aux) (cdr aux)))
	  (body `(block ,n (tagbody ,tag (return-from ,n (let* ,bind ,@(when block rd) ,@(car od) ,@(when block rc) ,@(car oc) ,@body))))))
     `(,h ,ll ,@(when doc (list doc)) ,@rd ,@rc ,body))))

;; (defun ttl-tag-src (src &optional (tag (tmpsym)) (block (tmpsym)) &aux (h (pop src)) (ll (pop src)))
;;   (setf (get tag 'ttl-tag) t)
;;   (multiple-value-bind
;;    (doc decls ctps body)
;;    (parse-body-header src)
;;    (let* ((aux (member '&aux ll));FIXME centralize with new-defun-args
;; 	  (ll (ldiff ll aux))
;; 	  (regs (mapcar (lambda (x) (cond ((symbolp x) x) ((symbolp (car x)) (car x)) ((cadar x)))) ll))
;; 	  (regs (set-difference regs '(&optional &rest &key &allow-other-keys)))
;; 	  (od (split-decls regs decls))
;; 	  (rd (cons `(declare (optimize (safety ,(decl-safety decls)))) (pop od)))
;; 	  (oc (split-ctps regs ctps))
;; 	  (rc (pop oc))
;; 	  (n (blocked-body-name body))
;; 	  (body (if n (cddar body) body))
;; 	  (n (or n block))
;; 	  (body `(block ,n (tagbody ,tag (return-from ,n (let* ,(cdr aux) ,@(car od) ,@(car oc) ,@body))))))
;;      `(,h ,ll ,@(when doc (list doc)) ,@rd ,@rc ,body))))

;; (defun ttl-tag-src (src &optional (tag (tmpsym)) block &aux (h (pop src)) (ll (pop src)))
;;   (setf (get tag 'ttl-tag) t)
;;   (multiple-value-bind
;;    (doc decls ctps body)
;;    (parse-body-header src)
;;    (let* ((aux (member '&aux ll))
;; 	  (ll (ldiff ll aux))
;; 	  (aux (cdr aux))
;; 	  (auxv (mapcar (lambda (x) (if (consp x) (car x) x)) aux))
;; 	  (ad (split-decls auxv decls))
;; 	  (od (cadr ad))
;; 	  (ad (car ad))
;; 	  (ac (split-ctps auxv ctps))
;; 	  (oc (cadr ac))
;; 	  (ac (car ac))
;; 	  (n (blocked-body-name body))
;; 	  (body (if n (cddar body) body))
;; 	  (n (or n block))
;; 	  (body `(block ,n (tagbody ,tag (return-from ,n (let* ,aux ,@ad ,@ac ,@body))))))
;;      `(,h ,ll ,@(when doc (list doc)) ,@od ,@oc ,body))))

(defvar *int* nil)
(defmacro ttm (fn &body body)
  `(let* ((st (get-internal-real-time))
	  (res ,@body)
	  (end (- (get-internal-real-time) st))
	  (dd (or (cdr (assoc ,fn *int*)) (cdar (push (list ,fn 0 0) *int*)))))
     (incf (car dd))
     (incf (cadr dd) end)
     res))
     
(defun mi4 (fn args la src env inls)
  (c1inline (list (cons fn (append args la)) (blla (cadr src) args la (cddr src))) env inls))

(defun sir-tag (sir)
  (cadar (member-if (lambda (x) (and (eq (caar x) (car sir)) (cdddr x)))
		    (reverse *src-inline-recursion*))))

(defun discrete-tp (tp &optional (i 0))
  (when (< i 5);FIXME
    (cond ((atomic-tp tp))
	  ((when (consp tp) (eq (car tp) 'or))
	   (not (member-if-not (lambda (x) (discrete-tp x (incf i))) (cdr tp)))))))

;; This function regulates the heuristic balance between compiler
;; speed and type precision primarily via tagbody iteration.  The
;; algorithm is essentially a guess at a surrounding type which might
;; not be overflowed on subsequent compilation iteration.  More
;; sophisticated ideas include bumping based on the type increment
;; instead of the type-or, and collecting bounding information during
;; the compilation pass.  Type inferencing on the branch pivot is not
;; currently effective mostly because they are frequently not
;; available on the first pass e.g. with atomic integer types.
;; Several GCL programs nest tagbodys very deeply (e.g. axiom/fricas),
;; so even a single extra iteration can be exponentially expensive.
(defun bbump-tp (tp)
  (cond ((car (member tp '(#tnull
			   #t(and fixnum (integer 1))
			   #t(and fixnum (integer 0))
			   #t(or null (and fixnum (integer 1)))
			   #t(or null (and fixnum (integer 0))))
		      :test 'type<=)))
	((discrete-tp tp) tp)
	((bump-tp tp))))

(defun cln (x &optional (i 0))
  (if (atom x) i (cln (cdr x) (1+ i))))

(defun new-type-p (a b)
  (cond ((binding-p a) nil);;FIXME ????
	((binding-p b) nil)
	((eql a b) nil)
	((atom a))
	((atom b))
	((or (new-type-p (car a) (car b)) (new-type-p (cdr a) (cdr b))))))

(defun tm (a b &aux (ca (cons-count a)))
  (when (< ca (if (< ca (cons-count b)) 50 32));FIXME, catch si::+array-typep-alist+
    (new-type-p a b)))

;; (defun arg-types-match (tps sir &optional ctp)
;;   (if tps
;;       (and (= (length tps) (length sir));FIXME unroll strategy	       
;; 	   (every (lambda (x y) 
;; 		    (or (type>= x y)
;; 			(and (type>= #tinteger x) (type>= #tinteger y))
;; 			(when ctp 
;; 			  (let ((ax (car (atomic-tp x)))(ay (car (atomic-tp y))))
;; 			    (when (consp ay) ;(setq aax ax aay ay) ;(print (list aax aay))(break)
;; 			      (not 
;; 			       (tm ay ax)
;; ;			       (when (and (consp ax) (<= (length ax) 15)) (tailp ay ax))
;; 			       )))))) tps sir))
;;     (not (member-if 'atomic-tp sir))))

;; (defun top-tagged-sir (sir &aux tagged-sir)
;;   (mapc (lambda (x) (when (eq (caar x) (car sir)) (when (cdddr x) (setq tagged-sir x))))
;; 	*src-inline-recursion*)
;;   tagged-sir)

;; (defun prev-sir (sir &aux (f (name-sir sir))(tp sir)(n (pop tp))
;; 		     (p (member n *src-inline-recursion* :key 'caar)))
;;   (when p
;;     (when (or (arg-types-match (cdaar p) tp)
;; 	      (member-if (lambda (x) (when (eq n (caar x)) (arg-types-match (cdar x) tp t))) (cdr p)))
;;       (let ((tagged-sir (unless (or (tail-recursion-possible f) (member-if 'atomic-tp tp))
;; 			  (top-tagged-sir sir))))
;; 	(if tagged-sir
;; 	    (throw tagged-sir *src-inline-recursion*)
;; 	  t)))))

;; (defun prev-sir (sir &aux (f (name-sir sir))(tp sir)(n (pop tp)) sub)

;;   (let ((p (member-if (lambda (x)
;; 			(when (eq n (caar x))
;; 			  (when (cdddr x)
;; 			    (arg-types-match (cdar x) tp (prog1 sub (setq sub t))))))
;; 		      *src-inline-recursion*)))
;;     (when p
;;       (cond ((tail-recursion-possible f) t)
;; 	    ((member-if 'atomic-tp tp) t)
;; 	    ((throw (car p) *src-inline-recursion*))))))

;; (defun arg-types-match (tps sir &optional ctp)
;;   (if t;tps
;;       (and (= (length tps) (length sir));FIXME unroll strategy	       
;; 	   (every (lambda (x y) 
;; 		    (or (si::type= x y)
;; 			(and (type>= #tinteger x) (type>= #tinteger y))
;; ;; 			(when ctp 
;; ;; 			  (let ((ax (car (atomic-tp x)))(ay (car (atomic-tp y))))
;; ;; 			    (when (consp ay) ;(setq aax ax aay ay) ;(print (list aax aay))(break)
;; ;; 			      (not 
;; ;; 			       (tm ay ax)
;; ;; ;			       (when (and (consp ax) (<= (length ax) 15)) (tailp ay ax))
;; ;; 			       ))))
;; 			)) tps sir))
;;       (progn (break "foo")(not (member-if 'atomic-tp sir)))))

;; (defun too-complicated-p (sir)
;;   (>
;;    (max (count (car sir) *src-inline-recursion* :key 'caar)
;; 	(reduce (lambda (y x &aux (x (car (atomic-tp x))))
;; 		  (max y (if (listp x) (length x) 0)))
;; 		(cdr sir) :initial-value 0))
;;    20))

;; (defun prev-sir (sir &aux (f (name-sir sir))(tp sir)(n (pop tp)) sub)
;; ;  (print (list n (count n *src-inline-recursion* :key 'caar)))
;;   ;; (let ((x (mapcan (lambda (x) (when (consp x) (list x (length x))))
;;   ;; 		   (remove nil (mapcar (lambda (x) (car (atomic-tp x))) tp)))))
;;   ;;   (when x (print x)))
;;   (let* ((p (member-if (lambda (x)
;; 			(when (eq n (caar x))
;; 			  (when (cdddr x)
;; 			    (arg-types-match (cdar x) tp)))); (prog1 sub (setq sub t))
;; 		       *src-inline-recursion*))
;; 	 (ts (top-tagged-sir sir))
;; 	 (c (when ts (when (too-complicated-p sir) (list ts))))
;; ;	 (c (when ts (when (member-if 'complicated-cons-type-p tp) (list ts))))
;; 	 (p (or p c)))
;;     (when p
;;       (cond ((unless c (tail-recursion-possible f)) t)
;; 	    ((unless c (member-if 'atomic-tp tp)) (break "bar") t)
;; 	    ((throw (car p) *src-inline-recursion*))))))


;; (defun arg-types-match (tps sir)
;;   (and (= (length tps) (length sir))
;;        (every (lambda (x y) 
;; 		(or (si::type= x y)
;; 		    (and (type>= #tinteger x) (type>= #tinteger y))))
;; 		  tps sir)))

;; (defun too-complicated-p (sir)
;;   (mapc (lambda (x) (when (eq (car sir) (caar x))
;; 		      (when (cddr x)
;; 			(when (some (lambda (x y &aux (x (car (atomic-tp x)))(y (car (atomic-tp y))))
;; 				      (and (consp x) (consp y) (tailp x y) (> (length y) 20)))
;; 				    (cdr sir) (cdar x))
;; ;			  (print sir)(break)
;; 			  (return-from too-complicated-p t)))))
;; 	*src-inline-recursion*)
;;   (>
;;    (count (car sir) *src-inline-recursion* :key 'caar)
;;    20))

;; (defun top-tagged-sir (sir &aux tagged-sir tts)
;;   (mapc (lambda (x) (when (eq (caar x) (car sir)) (when (cdddr x) (setq tts tagged-sir tagged-sir x))))
;; 	*src-inline-recursion*)
;;   tagged-sir)

;; (defun top-tagged-sir (sir &aux tagged-sir tts)
;;   (mapc (lambda (x) (when (eq (caar x) (car sir)) (when (cdddr x) (setq tts tagged-sir tagged-sir x))))
;; 	*src-inline-recursion*)
;;   tts)

;; (defun top-tagged-sir (sir &aux tagged-sir tts)
;;   (mapc (lambda (x) (when (eq (caar x) (car sir)) (when (cdddr x) (setq tts tagged-sir tagged-sir x))))
;; 	*src-inline-recursion*)
;;   (if (member-if 'atomic-tp tagged-sir) tts tagged-sir))

;; (defun top-tagged-sir (sir &aux last-tagged-sir penultimate-tagged-sir)
;;   (mapc (lambda (x)
;; 	  (when (eq (caar x) (car sir))
;; 	    (when (cdddr x)
;; 	      (setq penultimate-tagged-sir last-tagged-sir last-tagged-sir x))))
;; 	*src-inline-recursion*)
;;   (or (unless (member-if 'atomic-tp (cdr last-tagged-sir)) last-tagged-sir)
;;       penultimate-tagged-sir))

;; (defun prev-sir (sir &aux (f (name-sir sir))(tp sir)(n (pop tp)) sub)
;;   (let* ((p (member-if (lambda (x)
;; 			(when (eq n (caar x))
;; 			  (when (cdddr x)
;; 			    (arg-types-match (cdar x) tp))))
;; 		       *src-inline-recursion*))
;; 	 (ts (top-tagged-sir sir))
;; 	 (c (when ts (when (too-complicated-p sir) (list ts))))
;; 	 (p (or p c)))
;;     (when p
;;       (cond ((unless c (tail-recursion-possible f)) t)
;; 	    ((unless c (member-if 'atomic-tp tp)) t)
;; 	    ((throw (car p) *src-inline-recursion*))))))

;; (defun last-or-penultimate (sir filter &aux (n (car sir)) last penultimate)
;;   (mapc (lambda (x) (when (and (eq n (caar x)) (cdddr x) (funcall filter x))
;; 		      (setq penultimate last last x)))
;; 	*src-inline-recursion*)
;;   (or last ;(unless (member-if 'atomic-tp last) last) ;inline at least one of these
;;       penultimate))

;; (defun prev-sir (sir &aux (f (name-sir sir))(tp sir)(n (pop tp)) sub)
;;   (let* ((p (last-or-penultimate sir (lambda (x) (arg-types-match (cdar x) tp))))
;; 	 (c (unless p
;; 	      (when (too-complicated-p sir)
;; 		(last-or-penultimate sir 'identity))))
;; 	 (p (or p c)))
;;     (when p
;;       (or (unless c (tail-recursion-possible f))
;; 	  (unless c (member-if 'atomic-tp tp))
;; 	  (throw p *src-inline-recursion*)))))



;; (defun top-tagged-sir (sir &aux last penul)
;;   (mapc (lambda (x) (when (eq (caar x) (car sir)) (when (cdddr x) (setq penul last last x))))
;; 	*src-inline-recursion*)
;;   (cond ((member-if 'atomic-tp (car last)) penul)
;; 	((eql (length (car last)) (length (car penul))) last);types t?
;; 	(penul)))

;; (defun top-tagged-sir (sir &aux last penul)
;;   (mapc (lambda (x) (when (eq (caar x) (car sir)) (when (cdddr x) (setq penul last last x))))
;; 	*src-inline-recursion*)
;;   (cond ;((member-if 'atomic-tp (car last)) penul)
;; 	;((eql (length (car last)) (length (car penul))) last);types t?
;; 	(penul)))

;; (defun prev-sir (sir &aux (f (name-sir sir))(tp sir)(n (pop tp)))
;;   (let* ((p (car (member-if
;; 		  (lambda (x)
;; 		    (when (eq n (caar x))
;; 		      (when (cdddr x)
;; 			(arg-types-match (cdar x) tp))))
;; 		  *src-inline-recursion*)))
;; 	 (c (unless p (when (too-complicated-p sir) (top-tagged-sir sir))))
;; 	 (p (or p c)))
;;     (when p
;;       (cond ((unless c (tail-recursion-possible f)) t)
;; 	    ((unless c (member-if 'atomic-tp tp)) t)
;; 	    ((throw p *src-inline-recursion*))))))

(defvar *src-loop-unroll-limit* 20)

(defun arg-types-match (tps sir)
  (and (= (length tps) (length sir))
       (every (lambda (x y) 
		(or (type= x y)
		    (and (type>= #tinteger x) (type>= #tinteger y))
		    (let ((cx (car (atomic-tp x)))(cy (car (atomic-tp y))))
		      (and (consp cx) (consp cy)
			   (if (tailp cy cx)
			       (> (labels ((l (x i) (if (consp x) (l (cdr x) (1+ i)) i))) (l cx 0)) *src-loop-unroll-limit*)
			       (tailp cx cy))))))
	      tps sir)))

(defun prior-inline-similar-types (n tp)
    (car (member-if
	(lambda (x)
	  (when (eq n (caar x))
	    (when (cdddr x)
	      (arg-types-match (cdar x) tp))))
	*src-inline-recursion*)))
  

(defun inline-too-complex (sir list &aux (i 0) last penul)
  (mapc (lambda (x) (when (eq (caar x) (car sir)) (when (cdddr x) (incf i) (setq penul last last x))))
	list)
  (when (> i *src-loop-unroll-limit*)
    (let ((p (cond
               ;(last)
	       ((member-if 'atomic-tp (cdar last)) penul)
	       ((eql (length (car last)) (length (car penul))) last);types t?
	       (penul))))
      (if p (throw p list) t))))

(defun prev-sir (sir &aux (f (name-sir sir))(tp sir)(n (pop tp)) p)
  (cond ((setq p (prior-inline-similar-types n tp))
	 (or (tail-recursion-possible f) (throw p *src-inline-recursion*)))
	((inline-too-complex sir *src-inline-recursion*))
	((inline-too-complex sir *prev-sri*))))

;; (let* ((p (car (member-if
;; 		  (lambda (x)
;; 		    (when (eq n (caar x))
;; 		      (when (cdddr x)
;; 			(arg-types-match (cdar x) tp t))))
;; 		  *src-inline-recursion*)))
;; ;	 (p (when p (or (top-tagged-sir sir) p)));ldiff
;; 	 (c (unless p (when (too-complicated-p sir) (top-tagged-sir sir))))
;; 	 (p (or p c)))
;;     (when p
;;       ;; (print (list n (caar c) (count (car sir) *src-inline-recursion* :key 'caar) (length *src-inline-recursion*)
;;       ;; 		   (or (unless c (tail-recursion-possible f)) (unless c (member-if 'atomic-tp tp))) ))
;;       (cond ((unless c (tail-recursion-possible f)) t)
;; ;	    ((unless c (member-if 'atomic-tp tp)) t)
;; 	    ((throw p *src-inline-recursion*))))))









(defun make-tagged-sir (sir tag ll &optional (ttag nil ttag-p))
  (list* sir tag ll (when ttag-p (list ttag))))

(defun maybe-cons-tagged-sir (tagged-sir src env &aux (id (name-sir (car tagged-sir))))
  (cond ((and (eq src (local-fun-src id))
	      (not (let ((*funs* (if env (fifth env) *funs*)));FIXME?
			 (eq src (local-fun-src id))))); flet not labels
	 *src-inline-recursion*)
	((cons tagged-sir *src-inline-recursion*))))

(defun maybe-cons-sir (sir tag ttag src env &aux (id (name-sir sir)))
  (cond ((and (eq src (local-fun-src id))
	      (not (let ((*funs* (if env (fifth env) *funs*)));FIXME?
			 (eq src (local-fun-src id)))))
	 *src-inline-recursion*)
	((cons (list sir tag (cadr src) ttag) *src-inline-recursion*))))

(defun sir-name (id)
  (cond ((local-fun-p id)) ((symbolp id) id) ((alloc-spice))));FIXME, do not push anonymous?

(defun name-sir (sir &aux (f (car sir)))
  (if (fun-p f)
      (fun-name f)
    f))

(defun infer-tp-p (f)
  (cond ((eq f 'infer-tp))
	((atom f) nil)
	((or (infer-tp-p (car f)) (infer-tp-p (cdr f))))))

(defun cons-count (f)
  (cond ((atom f) 0)
	((+ 1 (cons-count (car f)) (cons-count (cdr f))))))

(defun type-fm (fun fms)
  (case fun
	((si::tpi typep coerce) (cadr fms))
	(si::num-comp (caddr fms))
	(make-sequence (car fms))))

(defun constant-type-p (tp)
  (typecase
   tp
   (symbol t)
   (binding nil)
   (atom t)
   (cons (and (constant-type-p (car tp)) (constant-type-p (cdr tp))))))

(defun known-type-p (fm)
  (let ((tp (atomic-tp (info-type (cadr fm)))))
    (when tp (constant-type-p (car tp)))))

(defun maybe-inline-src (fun fms src &aux fm)
  (when src
    (cond ((member fun *inline*))
	  ((setq fm (type-fm fun fms)) (known-type-p fm))
	  ((member fun '(row-major-aref
			 si::row-major-aset
			 si::row-major-aref-int
			 si::set-array
			 array-element-type
			 si::0-byte-array-self
			 si::set-0-byte-array-self));FIXME
	   (flet ((tst (tp) (not (or (type>= tp #tarray) (type>= tp #tvector)))))
	     (tst (info-type (if (eq fun 'si::row-major-aset) (cadadr fms) (cadar fms))))))
;	  ((< (cons-count src) 30))
	  ((not (symbolp fun)))
	  ((let* ((n (symbol-package fun))(n (when n (package-name n)))(p (find-package :lib))) 
	     (when n (or (when p (find-symbol n p)) (string-equal "CSTRUCT" n)))));FIXME
	  ((local-fun-p fun))
	  ((intersection-p '(&key &rest) (cadr src)))
	  ((member-if-not (lambda (x) (type>= (car x) (cdr x))) 
			  (mapcar (lambda (x y) (cons (info-type (cadr x)) (coerce-to-one-value y))) fms (get-arg-types fun))))
	  ((when (exit-to-fmla-p) (infer-tp-p src)))
	  ((< (cons-count src) 50)))));100

(dolist (l '(upgraded-array-element-type row-major-aref row-major-aset si::set-array array-element-type))
  (setf (get l 'consider-inline) t))

;; (defun maybe-inline-src (fun fms src)
;;   (when src
;;     (or
;;      (not (symbolp fun))
;;      (inline-asserted fun)
;;      (not (get fun 'consider-inline))
;;      (let* ((y (get-arg-types fun))
;; 	    (y (or (car y) #tt))
;; 	    (y (if (eq y '*) #tt y))
;; 	    (x (info-type (cadar fms)))
;; 	    (x (if (eq x #tvector) #tarray x))
;; 	    (x (if (or (type>= #tarray x) (atomic-tp x)) x #tt)));FIXME
;;        (not (type>= x y))))))

(defun mi3a (env fun fms)
  (under-env 
   env
   (let ((src (inline-src fun)))
     (when (maybe-inline-src fun fms src)
       src))))
	     

(defun mi3 (fun args la fms ttag envl inls &aux (src (mi3a (pop envl) fun fms)) (env (car envl)))
  (when src
    (let ((sir (cons (sir-name fun) (mapcar (lambda (x) (when x (info-type (cadr x)))) fms))))
      (unless (prev-sir sir)
	(let* ((tag (make-ttl-tag));(tmpsym)
	       (tsrc (ttl-tag-src src tag))
	       (tagged-sir (make-tagged-sir sir tag (cadr src) ttag))
	       (*src-inline-recursion* (maybe-cons-tagged-sir tagged-sir src env))
	       (*top-level-src-p* (member src *top-level-src*)))
	  (catch tagged-sir (mi4 fun args la tsrc env inls)))))))


(defun mod-env (e l)
  (setq *lexical-env-mask* (nconc (remove-if (lambda (x) (or (symbolp x) (is-fun-var x))) (ldiff l e)) *lexical-env-mask*));FIXME
  l)



(defvar *lexical-env-mask* nil)

(defmacro under-env (env &rest forms &aux (e (tmpsym)))
  `(let* ((,e ,env)
	  (*lexical-env-mask* (pop ,e))
	  (*vars*   (mod-env (pop ,e) *vars*))
	  (*blocks* (mod-env (pop ,e) *blocks*))
	  (*tags*   (mod-env (pop ,e) *tags*))
	  (*funs*   (mod-env (pop ,e) *funs*)))
     ,@forms))


(defun barrier-cross-p (fun &aux (f (local-fun-p fun)))
  (not (tailp (member-if-not 'fun-p *funs*)
	      (member f *funs*))))

(defun tail-recursion-possible (fun &aux (f (assoc fun *c1exit*)))
  (when f
    (unless (barrier-cross-p fun)
      (do ((l *vars* (cdr l))(e (caddr f)))
	  ((eq l e) t)
	(let ((v (car l)))
	  (when (var-p v)
	    (unless (eq 'lexical (var-kind v))
	      (unless (member v *lexical-env-mask*)
		(return nil)))))))))

(defun mi2 (fun args la fms envl)
  (let* ((sir (cll fun))
	 (tag (cadr sir))
	 (targs (if la (append args (list la)) args))
	 (inls (mapcar 'cons targs fms))
	 (inl (mi3 fun args la fms tag envl inls)))
    (cond ((info-p (cadr inl))
	   (keyed-cmpnote (list 'inline (if (fun-p fun) (fun-name fun) fun))
			  "inlining ~s ~s ~s" fun (mapcar (lambda (x) (info-type (cadr x))) fms) la)
	   inl)
	  (inl
	   (setq inl (mapcar (lambda (x) (name-sir (car x))) (ldiff inl *src-inline-recursion*)))
	   (keyed-cmpnote (list* 'inline 'inline-abort inl) "aborting inline of ~s" inl)
	   (setq *notinline* (nunion inl *notinline*));FIXME too extreme?
	   nil)
	  ((and sir (tail-recursion-possible fun))
	   (keyed-cmpnote (list 'tail-recursion fun) "tail recursive call to ~s replaced with iteration" fun)
	   (c1let-* (cdr (blla-recur tag (caddr sir) args la)) t inls)))))

;; (defun mi2 (fun args la fms envl)
;;   (let* ((sir (cll fun))
;; 	 (tag (cadr sir))
;; 	 (targs (if la (append args (list la)) args))
;; 	 (*inline-forms* (mapcar 'cons targs fms))
;; 	 (inl (mi3 fun args la fms tag envl)))
;;     (cond (inl
;; 	   (mapc (lambda (x) (add-info (cadr inl) (cadr x))) fms);FIXME
;; 	   (when (eq (car (fifth inl)) 'let*)
;; 	     (setf (cadr (fifth inl)) (copy-info (cadr inl))))
;; 	   (keyed-cmpnote (list 'inline fun) "inlining ~s ~s ~s" fun args la)
;; 	   inl)
;; 	  ((and sir (member fun *c1exit*))
;; 	   (keyed-cmpnote (list 'tail-recursion fun)
;; 			  "tail recursive call to ~s replaced with iteration" fun)
;; 	   (c1expr (blla-recur tag (caddr sir) args la))))))

;; (defun mi2 (fun args la fms envl)
;;   (let* ((sir (cll fun))
;; 	 (tag (cadr sir))
;; 	 (targs (if la (append args (list la)) args))
;; 	 (*inline-forms* (mapcar 'cons targs fms))
;; 	 (inl (mi3 fun args la fms tag envl)))
;;     (cond (inl
;; 	   (mapc (lambda (x) (add-info (cadr inl) (cadr x))) fms);FIXME
;; 	   (when (eq (car (fifth inl)) 'let*)
;; 	     (setf (cadr (fifth inl)) (copy-info (cadr inl))))
;; 	   (keyed-cmpnote (list 'inline fun) "inlining ~s ~s ~s" fun args la)
;; 	   inl)
;; 	  ((and sir (member fun *c1exit*))
;; 	   (keyed-cmpnote (list 'tail-recursion fun)
;; 			  "tail recursive call to ~s replaced with iteration" fun)
;; 	   (c1expr (blla-recur tag (caddr sir) args la))))))

;(defvar *provisional-inline* nil)
(defun make-c1forms (fn args last info)
  (let* ((at (get-arg-types fn))
	 (nargs (c1args args info))
	 (c1l (when last (c1arg last info)))
	 (nargs (if (when last (not (type>= #tnull (info-type (cadr c1l)))))
		    (progn (add-info info (cadr c1l)) (nconc nargs (list c1l)))
		  nargs))
	 (nat (mapcar (lambda (x) (info-type (cadr x))) nargs))
	 (ss (gethash fn *sigs*));FIXME?
	 (at (if (and ss (not (car ss))) nat at)))

    (mapc (lambda (x) (setf (info-type (cadr x)) (coerce-to-one-value (info-type (cadr x))))) nargs)

    (unless (or last (local-fun-p fn) (eq fn (when (consp *current-form*) (cadr *current-form*))));FIXME
      (let* (p
	     (m (do ((a at (if (eq (car a) '*) a (cdr a)))
		     (r args (cdr r))
		     (f nargs (cdr f)))
		    ((or (endp f) (endp a))
		     (or f (and a (not (eq (car a) '*)))))
		  (unless (or (eq '* (car a)) (type-and (car a) (info-type (cadar f))))
		    (setq p t)))))
	(when m
	  (funcall (if (eq (symbol-package fn) #.(find-package 'cl)) 'cmpwarn 'cmpstyle-warn)
		   "Wrong number of args in call to ~s:~% ~a ~a ~a~%"
		   fn (cons fn args) (mapcar 'cmp-unnorm-tp at) (mapcar 'cmp-unnorm-tp nat)))
	(when p
	  (keyed-cmpnote
	   (list fn 'inline)
	   "inlining of ~a prevented due to argument type mismatch:~% ~a ~a ~a~%"
	   fn (cons fn args) (mapcar 'cmp-unnorm-tp at) (mapcar 'cmp-unnorm-tp nat)))
	(when (or p m)
	  (setf (info-type info) nil))))

    (do ((a at (if (eq '* (car a)) a (cdr a)))
	 (r args (cdr r))
	 (f (if last (butlast nargs) nargs) (cdr f)))
	((or (endp f) (endp a)) nargs)
	(maybe-reverse-type-prop (car a) (car f)))))


(defun make-ordinary (fn &aux *c1exit*);FIXME *c1exit*
  (let* ((s (sgen "ORDS"))(g (sgen "ORDG"))
	 (e (c1let-* `(((,s ,g)) 
		       ;(check-type ,s (not list)) FIXME bootstrap
		       (if (functionp ,s) ,s (funcallable-symbol-function ,s))
;		       (coerce ,s 'function)
		       ) t (list (cons g fn)))); (coerce ,s 'function)
;	 (e (c1let-* `(((,s ,g)) (etypecase ,s ((and symbol (not boolean)) (fsf ,s)) (function ,s))) t (list (cons g fn)))); (coerce ,s 'function)
	 (info (make-info)))
    (add-info info (cadr e))
    (list 'ordinary info e)))

;; (defun make-ordinary (fn)
;;   (let* ((s (tmpsym))(g (tmpsym))
;; 	 (e (c1let-* `(((,s ,g)) (etypecase ,s (symbol (fsf ,s)) (function ,s))) t (list (cons g fn))))
;; 	 (info (make-info)))
;;     (add-info info (cadr e))
;;     (list 'ordinary info e)))

;; (defun make-ordinary (fn)
;;   (let* ((s (tmpsym))(g (tmpsym))
;; 	 (e (c1let-* `(((,s ,g)) (etypecase ,s (symbol (fsf ,s)) (function ,s))) t nil (list (cons g fn))))
;; 	 (info (make-info)))
;;     (add-info info (cadr e))
;;     (list 'ordinary info e)))

;; (defun make-ordinary (fn)
;;   (let* ((s (tmpsym))(g (tmpsym))
;; 	 (*inline-forms* (list (cons g fn)))
;; 	 (e (c1expr `(let* ((,s ,g)) (etypecase ,s (symbol (fsf ,s)) (function ,s))))))
;;     (list 'ordinary (cadr e) e)))

;; (defun make-ordinary (fn)
;;   (let* ((s (tmpsym))(g (tmpsym))
;; 	 (*inline-forms* (list (cons g fn)))
;; 	 (e (c1expr `(let* ((,s ,g)) (if (symbolp ,s) (fsf ,s) ,s)))))
;;     (list 'ordinary (cadr e) e)))

;; (defun or-ccb-assignments (fms)
;;   (mapc (lambda (v)
;; 	  (when (var-p v) 
;; 	    (let ((tp (get (var-store v) 'ccb-tp)));FIXME setq tp nil?
;; 	      (when tp
;; 		(do-setq-tp v '(ccb-ref) (type-or1 (var-type v) (get (var-store v) 'ccb-tp)))
;; 		(setf (var-store v) +opaque+))))) *vars*))

(defun bump-cons-tp (tp &aux (c (type-and tp #tcons))(p (type-and tp #tproper-cons)))
  (type-or1 tp (if (type>= p c) #tproper-cons #tcons)))

(defun do-ccb-ch (ccb-ch)
  (mapc (lambda (x &aux (v (pop x)))
	  (do-setq-tp v '(ccb-ch) (type-or1 (var-type v) (bump-cons-tp (info-type (cadr x)))))
	  (push-vbind v x t))
	ccb-ch))

(defun or-ccb-assignments (fms)
  (mapc (lambda (x)
	  (do-ccb-ch (info-ch-ccb (cadr x))))
	fms))

(defun mi6 (fn fms)
  (or-ccb-assignments fms)
  (unless (and (symbolp fn) (get fn 'c1no-side-effects))
    (dolist (f fms)
      (when (and (consp f) (eq (car f) 'var))
	(let* ((ft (info-type (cadr f)))
	       (p (when (and ft (type>= #tcons ft)) #tcons))
	       (p (when (and p (type>= #tproper-cons ft)) #tproper-cons)))
	  (when (and p (not (type>= ft p)))
	    (bump-pcons (caaddr f) p)))))))

;; (defun mi6 (fn fms)
;;   (unless (and (symbolp fn) (get fn 'c1no-side-effects))
;;     (dolist (f fms)
;;       (when (and (consp f) (eq (car f) 'var))
;; 	(let* ((ft (info-type (cadr f)))
;; 	       (p (when (and ft (type>= #tcons ft)) #tcons))
;; 	       (p (when (and p (type>= #tproper-cons ft)) #tproper-cons)))
;; 	  (when (and p (not (type>= ft p)))
;; 	    (bump-pcons (caaddr f) p)))))))


(defun binding-forms (st)
  (mapcan (lambda (x &aux (z (binding-form x))) (when z (list z))) st))

(defun global-var-stores (&aux z)
  (reduce (lambda (y x)
	    (or-binds
	     (when (var-p x)
	      (unless (eq (var-kind x) 'lexical)
		(var-store x)))
	     y)) *vars* :initial-value z))


(defun mi5 (fn info fms la &aux (ll (when la (list (length fms)))) fd)
  (when (iflag-p (info-flags info) side-effects)
    (c1side-effects nil))
  (mi6 fn fms)
  (let ((r (assoc fn *recursion-detected*))) (when r (setf (cdr r) t)))
  (cond	((consp fn) 
	 (let ((ord (make-ordinary fn)))
	   (add-info info (cadr ord))
	   (or-ccb-assignments (list fn))
	   `(,(if la 'apply 'funcall) ,info ,ord ,fms)))
	((setq fd (c1local-fun fn))
	 (add-info info (cadr fd))
	 (setf (info-type info) (info-type (cadr fd)))
	 (let ((fm (fifth fd)))
	   (when fm (or-ccb-assignments (list fm)))
	   `(call-local ,info ,(nconc (caddr fd) ll) ,(cadddr fd) ,fm ,fms)));FIXME
	(t
	 (or-ccb-assignments (binding-forms (global-var-stores)))
	 (push fn (info-ref info))
	 `(call-global ,info ,fn ,fms nil ,@ll))))

;; (defun mi5 (fn info fms la &aux (ll (when la (list (length fms)))) fd)
;;   (mi6 fn fms)
;;   (when (eq fn (cadr *current-form*)) (setq *recursion-detected* t))
;;   (cond	((consp fn) 
;; 	 (let ((ord (make-ordinary fn)))
;; 	   (add-info info (cadr ord))
;; 	   `(,(if la 'apply 'funcall) ,info ,ord ,fms)))
;; 	((setq fd (c1local-fun fn))
;; 	 (add-info info (cadr fd))
;; 	 (setf (info-type info) (if (eq (info-type (cadr fd)) 'boolean) #tboolean (info-type (cadr fd))));FIXME
;; 	 `(call-local ,info ,(nconc (caddr fd) ll) ,(cadddr fd) ,(fifth fd) ,fms));FIXME
;; 	(`(call-global ,info ,fn ,fms nil ,@ll))))

;; (defun mi5 (fn info fms la 
;; 	       &aux (nlast (when la (type>= #tnull (info-type (cadr (car (last fms)))))))
;; 	       (fms (if nlast (butlast fms) fms))
;; 	       (la (unless nlast la))
;; 	       (ll (when la (list (length fms)))))
;;   (mi6 fn fms)
;;   (when (eq fn (cadr *current-form*)) (setq *recursion-detected* t))
;;   (cond	((consp fn) `(,(if la 'apply 'funcall) ,info ,(make-ordinary fn) ,fms))
;; 	((let ((fd (c1local-fun fn)))
;; 	   (when fd
;; 	     (add-info info (cadr fd))
;; 	     (setf (info-type info) (if (eq (info-type (cadr fd)) 'boolean) #tboolean (info-type (cadr fd))))
;; 	     `(call-local ,info ,(append (caddr fd) ll) ,fms))))
;; 	(`(call-global ,info ,fn ,fms nil ,@ll))))

;; (defun mi5 (fn info fms la &aux (ll (when la (list (length fms)))))
;;   (mi6 fn fms)
;;   (when (eq fn (cadr *current-form*)) (setq *recursion-detected* t))
;;   (cond	((consp fn) `(,(if la 'apply 'funcall) ,info ,(make-ordinary fn) ,fms))
;; 	((let ((fd (c1local-fun fn)))
;; 	   (when fd
;; 	     (add-info info (cadr fd))
;; 	     (setf (info-type info) (if (eq (info-type (cadr fd)) 'boolean) #tboolean (info-type (cadr fd))))
;; 	     `(call-local ,info ,(append (caddr fd) ll) ,fms))))
;; 	(`(call-global ,info ,fn ,fms nil ,@ll))))


;; Objects when read are not eql
(declaim (inline unreadable-individual-p))
(defun unreadable-individual-p (x)
  (typecase x (number)(symbol (not (symbol-package x)))(otherwise t)))

(defun bump-unreadable-individuals (tp)
  (bump-individuals 'unreadable-individual-p tp))



(defun type-from-args (fun fms last info &aux x)
  (when (symbolp fun)
    (unless (get fun 'c1no-side-effects)
      (setf (info-flags info) (logior (info-flags info) (iflags side-effects)))));FIXME
  (cond ((setq x (member-if-not 'identity fms :key (lambda (x) (info-type (cadr x)))))
	 (keyed-cmpnote (list fun 'nil-arg)
			"Setting return type on call to ~s to nil due to nil-typed form:~%~s"
			fun x)
	 (setf (info-type info) nil))
	(last)
	((and (symbolp fun) (not (local-fun-p fun)))
	 (let ((tp (result-type-from-args fun (mapcar (lambda (x) (info-type (cadr x))) fms))))
	   (when tp
	     (setf (info-type info) (type-and (info-type info) tp))))))
  ;;FIXME inline functions from source with static data
  ;; (when (unreadable-individuals-p (info-type info))
  ;;   (keyed-cmpnote (list fun 'unreadable-individuals)
  ;; 		   "~<;; ~@;Setting return type on call to ~s to nil due to unreadable individuals in~%~s~;~:>"
  ;; 		   (list fun (cmp-unnorm-tp (info-type info))))
  ;;   (setf (info-type info) nil))
  (info-type info))

(defun coerce-ff (ff)
  (coerce-to-funid (car (atomic-tp (info-type (cadr ff))))));(when (member (car ff) '(foo location var)) ))

(defun coerce-to-local-fn (ob)
  (if (functionp ob) ob (local-fun-fn ob)))

(defun ff-env (ff)
  (cond ((not ff) nil)
	((symbolp ff) (ff-env (local-fun-fn ff)))
	((consp ff) (let ((x (car (atomic-tp (info-type (cadr ff)))))) (unless (consp x) (ff-env x))));FIXME
	((functionp ff) (list (or (fn-get ff 'ce) (current-env)) (fn-get ff 'df)))))
	
  ;; (let* ((fn (when ff (coerce-to-local-fn (car (atomic-tp (info-type (cadr ff))))))))
  ;;   (when fn
  ;;     (let* ((ce (fn-get fn 'ce))
  ;; 	     (df (fn-get fn 'df)))
  ;; 	(list ce df)))))

;; (defun ff-env (ff)
;;   (when ff
;;     (values (gethash (coerce-to-local-fn (car (atomic-tp (info-type (cadr ff))))) *fun-ev-hash*))))

;; (defun coerce-to-local-fun (ob)
;;   (if (functionp ob) ob (local-fun-fun ob)))

;; (defun ff-env (ff)
;;   (when ff
;;     (gethash (coerce-to-local-fun (car (atomic-tp (info-type (cadr ff))))) *fun-ev-hash*)))
;;   (case (car ff)
;; 	(location (gethash (local-fun-fun (car (atomic-tp (info-type (cadr ff))))) *fun-ev-hash*))
;; 	(foo (gethash (car (atomic-tp (info-type (cadr ff)))) *fun-ev-hash*))))
;  (when (member (car ff) '(foo location)) (gethash (car (atomic-tp (info-type (cadr ff)))) *fun-ev-hash*)))

(defun mi1c (fun args last info &optional ff prov &aux (*prov* prov))

  (let* ((otp (info-type info))
	 (fms (make-c1forms fun args last info))
	 (last (when (and last (nth (length args) fms)) last))
	 (tp (type-from-args fun fms last info)))
    (or
     (when (or tp (eq otp tp))
       (mi2 fun args last fms (ff-env (or ff fun))))
     (when (member-if-not 'identity fms :key (lambda (x) (info-type (cadr x))))
       (c1progn args fms))
     (mi5 (or (when (symbolp fun) fun) ff) info fms last))))


(defvar *prov-src* nil)

(defun mi1b (fun args last info &optional ff &aux (ops *prov-src*)(*prov-src* *prov-src*))
  (with-restore-vars
   (let ((res (mi1c fun args last info ff t)))
     (cond ((iflag-p (info-flags (cadr res)) provisional)
	    (keyed-cmpnote 'provisional "~s has provisional functions, res address ~s" fun (address res)))
	   (t (keep-vars) (mapc 'eliminate-src (ldiff *prov-src* ops)) res)))))

(defun mi1a (fun args last info &optional ff &aux (i1 (copy-info info)));FIXME side-effects on info
  (or (mi1b fun args last info ff)
      (prog1 (mi1c fun args last i1 ff)
	(setf (info-type info) (info-type i1)))))


(defun current-env nil (list *lexical-env-mask* *vars* *blocks* *tags* *funs*))



(defun mi1 (fn args &optional last ff)
  (let* ((tp (get-return-type fn))
	 (sp (if (when (symbolp fn) (get fn 'no-sp-change)) 0 1))
	 (info (make-info :type (bump-unreadable-individuals tp) :flags (if sp (iflags sp-change) 0)))
 	 (res (mi1a fn args last info ff)))
    (when tp 
      (let ((t1 (info-type (cadr res)))(t2 (info-type info)))
	(when (exit-to-fmla-p)
	  (labels ((tb (tp) (type-or1 (when (type-and #tnull tp) #tnull)
				      (when (type-and #t(not null) tp) #ttrue))))
	    (setq t1 (tb t1) t2 (tb t2) tp (tb tp))))
	(setf (info-type (cadr res)) (type-and t1 (if (type= t1 t2) tp t2)))))
    res))


(defun local-fun-obj (fname)
  (typecase fname
    (function (fn-get fname 'fun))
    (fun fname)
    (symbol (car (member-if (lambda (x)
			      (when (fun-p x)
				(unless (member x *lexical-env-mask*)
				  (eq fname (fun-name x)))))
			    *funs*)))))

(defun local-fun-p (fname &aux (fun (local-fun-obj fname)))
  (when (and fun (fun-src fun)) fun))

(defun local-macro-p (fname &aux (fun (local-fun-obj fname)))
  (when fun (unless (fun-src fun) fun)))

(defun funs-to-macrolet-env nil
  `(nil ,(mapcan (lambda (x)
		   (when (fun-p x)
		     (unless (member x *lexical-env-mask*)
		       `(,(if (fun-src x) `(,(fun-name x) function ,(lambda (&rest r) (declare (ignore r)) nil)) `(,(fun-name x) macro ,(fun-fn x)))))))
		 *funs*)
	nil))

(defun c1symbol-fun (whole &aux (fname (car whole)) (args (cdr whole)) fd)
  (values
   (cond ((setq fd (get fname 'c1special)) (funcall fd args))
	 ((and (setq fd (get fname 'co1special)) (funcall fd fname args)))
	 ((setq fd (local-macro-p fname))
	  (c1expr (cmp-expand-macro-w (fun-fn fd) whole)))
	 ((local-fun-p fname) (mi1 fname args))
	 ((unless (member fname *notinline*)
	    (let* ((fn (compiler-macro-function fname))
		   (res (if fn (funcall fn whole nil) whole)));FIXME cmp-expand-macro-w?
	      (unless (eq whole res) (c1expr res)))))
	 ((and (setq fd (get fname 'co1))
	       (inline-possible fname)
	       (funcall fd fname args)))
	 ((and (setq fd (get fname 'c1)) (inline-possible fname))
	  (funcall fd args))
	 ((and (setq fd (get fname 'c1g)) (inline-possible fname))
	  (funcall fd fname args))
	 ((setq fd (macro-function fname))
	  (c1expr (cmp-expand-macro-w fd whole)))
	 ((eq fname 'si:|#,|)
	  (cmperr "Sharp-comma-macro was found in a bad place."))
	 ((mi1 fname args)))))


;; (defun remove-doc-string (body)
;;   (nconc (do (d doc) ((or (not body) (if (stringp (car body)) 
;; 					 (or (endp (cdr body)) doc)
;; 				       (or (not (consp (car body))) (not (eq 'declare (caar body))))))
;; 		      (nreverse d))
;; 	     (let ((x (pop body))) (if (stringp x) (unless doc (push x doc)) (push x d)))) body))



(defun c1funcallable-symbol-function (args &aux a)
  (let* ((info (make-info :type #tfunction))
	 (nargs (c1args args info)))
    (cond ((setq a (atomic-tp (info-type (cadar nargs))))
	   (c1expr `(function ,(let ((x (coerce-to-funid (car a))))
				 (if (functionp x) (fn-get x 'id) x)))))
	  ((list 'call-global info 'funcallable-symbol-function nargs)))))
(si::putprop 'funcallable-symbol-function 'c1funcallable-symbol-function 'c1)

;; (defun c1lambda-fun (lambda-expr args)
;;   (c1expr (blla (car lambda-expr) args nil (cdr lambda-expr))))

(defun c2expr (form)
  (values
   (if (eq (car form) 'call-global)
       (c2call-global (caddr form) (cadddr form) nil (info-type (cadr form)) (sixth form))
     (if (or (eq (car form) 'let)
	     (eq (car form) 'let*))
	 (let ((*volatile* (volatile (cadr form))))
	   (declare (special *volatile*))
	   (apply (get (car form) 'c2) (cddr form)))
       (let ((tem (get (car form) 'c2)))
	 (cond (tem (apply tem (cddr form)))
	       ((setq tem (get (car form) 'wholec2))
		(funcall tem form))
	       (t (baboon))))))))

(defun c2expr* (form)
  (let* ((*exit* (next-label))
         (*unwind-exit* (cons *exit* *unwind-exit*)))
        (c2expr form)
        (wt-label *exit*)))

(defun c2expr-top (form top &aux (*vs* 0) (*max-vs* 0) (*level* (1+ *level*))
                                 (*reservation-cmacro* (next-cmacro)))
  (wt-nl "{register object *base" (1- *level*) "=base;")
  (base-used)
  (wt-nl "{register object *base=V" top ";")
  (wt-nl "register object *sup=vs_base+VM" *reservation-cmacro* ";")
  ;;; Dummy assignments for lint
  (wt-nl "base" (1- *level*) "[0]=base" (1- *level*) "[0];")
  (wt-nl "base[0]=base[0];")
  (if *safe-compile*
      (wt-nl "vs_reserve(VM" *reservation-cmacro* ");")
      (wt-nl "vs_check;"))
  (let* ((cm *reservation-cmacro*)
	 (vstu (if *mv-var*
		   (let ((loc (write-to-string (var-loc *mv-var*))))
		     (concatenate
		      'string
		      " if ((b_)>=-1) vs_top=V" loc " ? (object *)V" loc "+(b_) : base;"))
		   " vs_top=base;")))
    (wt-h "#define VMRV" cm "(a_,b_)" vstu " return(a_);")
    (wt-h "#define VMR" cm "(a_) VMRV" cm "(a_,0);"))
  (wt-nl) (reset-top)
  (c2expr form)
  (push (cons *reservation-cmacro* *max-vs*) *reservations*)
  (wt-nl "}}"))

(defun c2expr-top* (form top)
  (let* ((*exit* (next-label))
         (*unwind-exit* (cons *exit* *unwind-exit*)))
        (c2expr-top form top)
        (wt-label *exit*)))

;; (defun c1progn (forms &aux (fl nil))
;;   (cond ((endp forms) (c1nil))
;;         ((endp (cdr forms)) (c1expr (car forms)))
;;         ((let ((info (make-info)))
;; 	   (do ((forms forms (cdr forms))) ((not forms))
;; 	       (let* ((*c1exit* (unless (cdr forms) *c1exit*))
;; 		      (form (c1expr (car forms))))
;; 		 (push form fl)
;; 		 (add-info info (cadr form))))
;; 	   (setf (info-type info) (info-type (cadar fl)))
;; 	   (list 'progn info (nreverse fl))))))


(defun truncate-progn-at-nil-return-p (rp forms c1forms)
  (when (and rp (not (info-type (cadar rp))))
    (keyed-cmpnote 'nil-return "progn truncated at nil return, eliminating ~s" forms)
    (eliminate-src (cons 'progn (nthcdr (length c1forms) forms)))
    t))


(defun c1progn (forms &optional c1forms &aux r rp (info (make-info)))
  (when c1forms (assert (eql (length forms) (length c1forms))))
  (flet ((collect (f i)
	   (setq rp (last (if rp (rplacd rp f) (setq r f))))
	   (add-info info i)))
    (do ((forms forms (cdr forms))) ((or (not forms) (truncate-progn-at-nil-return-p rp forms c1forms)))
      (let ((form (or (pop c1forms) (if (cdr forms) (c1arg (car forms)) (c1expr (car forms))))))
	(cond ((and (cdr forms) (ignorable-form form)))
	      ((eq (car form) 'progn) (collect (third form) (cadr form)))
	      ((collect (cons form nil) (cadr form))))))
    (cond ((cdr r)
	   (setf (info-type info) (info-type (cadar rp)))
	   (list 'progn info r))
	  ((the list (car r)));FIXME
	  ((c1nil)))))


;;; Should be deleted.
(defun c1progn* (forms info)
  (setq forms (c1progn forms))
  (add-info info (cadr forms))
  forms)

(defun c2progn (forms)
  ;;; The length of forms may not be less than 1.
  (do ((l forms (cdr l)))
      ((endp (cdr l))
       (when l (c2expr (car l))))
      (let* ((*value-to-go* 'trash)
             (*exit* (next-label))
             (*unwind-exit* (cons *exit* *unwind-exit*)))
            (c2expr (car l))
            (wt-label *exit*))))

(defun c1arg (form &optional (info (make-info)) &aux *c1exit*)
  (c1expr* form info))

(defun c1args (forms info)
  (mapcar (lambda (form) (c1arg form info)) forms))

;; (defun c1args (forms info &aux *c1exit*)
;;   (mapcar (lambda (form) (c1expr* form info)) forms))

;;; Structures

(defun c1structure-ref (args)
  (if (and (not *safe-compile*)
	   (not (endp args))
           (not (endp (cdr args)))
           (consp (cadr args))
           (eq (caadr args) 'quote)
           (not (endp (cdadr args)))
           (symbolp (cadadr args))
           (endp (cddadr args))
           (not (endp (cddr args)))
           (si:fixnump (caddr args))
           (endp (cdddr args)))
      (c1structure-ref1 (car args)  (cadadr args) (caddr args))
      (let ((info (make-info)))
        (list 'call-global info 'si:structure-ref (c1args args info)))))

(defun c1structure-ref1 (form name index &aux (info (make-info)))
  ;;; Explicitly called from c1expr and c1structure-ref.
  (cond (*safe-compile* (c1expr `(si::structure-ref ,form ',name ,index)))
	((let* ((sd (get name 'si::s-data))
		(aet-type (aref (si::s-data-raw sd) index))
		(sym (find-symbol (si::string-concatenate
				   (or (si::s-data-conc-name sd) "")
				   (car (nth index (si::s-data-slot-descriptions sd))))))
		(tp (if sym (get-return-type sym) '*))
		(tp (type-and tp (nth aet-type +cmp-array-types+)))) 

	   (setf (info-type info) (if (and (eq name 'si::s-data) (= index 2));;FIXME -- this belongs somewhere else.  CM 20050106
				      #t(vector unsigned-char)
				      tp))
	   (list 'structure-ref info
		 (c1arg form info)
		 name
		 index sd)))))

;; (defun c1structure-ref1 (form name index &aux (info (make-info)))
;;   ;;; Explicitly called from c1expr and c1structure-ref.
;;   (cond (*safe-compile* (c1expr `(si::structure-ref ,form ',name ,index)))
;; 	((let* ((sd (get name 'si::s-data))
;; 		(aet-type (aref (si::s-data-raw sd) index))
;; 		(sym (find-symbol (si::string-concatenate
;; 				   (or (si::s-data-conc-name sd) "")
;; 				   (car (nth index (si::s-data-slot-descriptions sd))))))
;; 		(tp (if sym (get-return-type sym) '*))
;; 		(tp (type-and tp (nth aet-type +cmp-array-types+)))) 

;; 	   (setf (info-type info) (if (and (eq name 'si::s-data) (= index 2));;FIXME -- this belongs somewhere else.  CM 20050106
;; 				      #t(vector unsigned-char)
;; 				      tp))
;; 	   (list 'structure-ref info
;; 		 (c1expr* form info)
;; 		 (add-symbol name)
;; 		 index sd)))))

(defun coerce-loc-structure-ref (arg type-wanted &aux (form (cdr arg)))
  (let* ((sd (fourth form))
	 (index (caddr form)))
    (cond (sd
	    (let* ((aet-type (aref (si::s-data-raw sd) index))
		   (type (nth aet-type +cmp-array-types+)))
	      (cond ((eq (inline-type type) 'inline)
		     (or (= aet-type +aet-type-object+) (error "bad type ~a" type))))
	      (setf (info-type (car arg)) type)
	      (coerce-loc
		      (list (inline-type
			     type)
		           (flags)
			    'my-call
			    (list
			     (car
			      (inline-args (list (car form))
					   '(t)))
			     'joe index sd))
		      type-wanted))
		)
	  (t (wfs-error)))))


(defun c2structure-ref (form name-vv index sd
                             &aux (*vs* *vs*) (*inline-blocks* 0))
  (let ((loc (car (inline-args (list form) '(t))))
	(type (nth (aref (si::s-data-raw sd) index) +cmp-array-types+)))
       (unwind-exit
	 (list (inline-type type)
			  (flags) 'my-call
			  (list  loc  name-vv
				 index sd))))
  (close-inline-blocks)
  )

(defun c1str-ref (args)
  (let* ((info (make-info))
	 (nargs (c1args args info)))
    (list* 'str-ref info nargs)))
(setf (get 'str-ref 'c1) 'c1str-ref)

(defun sinline-type (tp);FIXME STREF STSET handled as aref
  (if (type= tp #tcharacter) 'inline-character (inline-type tp)))

(defun c2str-ref (loc nm off)
  (let* ((nm (car (atomic-tp (info-type (cadr nm)))))
	 (sd (get nm 'si::s-data))
	 (loc (car (inline-args (list loc) '(t))))
	 (off (car (atomic-tp (info-type (cadr off))))))
    (unless (and off sd (not *compiler-push-events*)) (baboon))
    (unwind-exit
     (list (sinline-type (nth (aref (si::s-data-raw sd) off) +cmp-array-types+));FIXME STREF STSET handled as aref
	   (flags) 'my-call (list loc nil off sd)))
    (close-inline-blocks)))
(setf (get 'str-ref 'c2) 'c2str-ref)



(defun my-call (loc name-vv ind sd);FIXME get-inline-loc above
  (declare (ignore name-vv))
  (let* ((raw (si::s-data-raw sd))
	 (spos (si::s-data-slot-position sd)))
    (if *compiler-push-events* (wfs-error)
      (wt "STREF("  (aet-c-type (nth (aref raw ind) +cmp-array-types+) )
	  "," loc "," (aref spos ind) ")"))))


(defun c1structure-set (args &aux (info (make-info :flags (iflags side-effects))))
  (if (and (not (endp args)) (not *safe-compile*)
           (not (endp (cdr args)))
           (consp (cadr args))
           (eq (caadr args) 'quote)
           (not (endp (cdadr args)))
           (symbolp (cadadr args))
           (endp (cddadr args))
           (not (endp (cddr args)))
           (si:fixnump (caddr args))
           (not (endp (cdddr args)))
           (endp (cddddr args)))
      (let* ((x (c1arg (car args) info))
	     (sd (get (cadadr args) 'si::s-data))
	     (raw (si::s-data-raw sd))
	     (type (nth (aref raw (caddr args)) +cmp-array-types+))
             (y (c1arg (if (type= #tcharacter type) `(char-code ,(cadddr args)) (cadddr args)) info)));FIXME STREF STSET handled as aref
        (setf (info-type info) (info-type (cadr y)))
        (list 'structure-set info x
              (cadadr args) ;;; remove QUOTE.
              (caddr args) y (get (cadadr args) 'si::s-data)))
      (list 'call-global info 'si:structure-set (c1args args info))))

;; (defun c1structure-set (args &aux (info (make-info :flags (iflags side-effects))))
;;   (if (and (not (endp args)) (not *safe-compile*)
;;            (not (endp (cdr args)))
;;            (consp (cadr args))
;;            (eq (caadr args) 'quote)
;;            (not (endp (cdadr args)))
;;            (symbolp (cadadr args))
;;            (endp (cddadr args))
;;            (not (endp (cddr args)))
;;            (si:fixnump (caddr args))
;;            (not (endp (cdddr args)))
;;            (endp (cddddr args)))
;;       (let ((x (c1expr (car args)))
;;             (y (c1expr (cadddr args))))
;;         (add-info info (cadr x))
;;         (add-info info (cadr y))
;;         (setf (info-type info) (info-type (cadr y)))
;;         (list 'structure-set info x
;;               (add-symbol (cadadr args)) ;;; remove QUOTE.
;;               (caddr args) y (get (cadadr args) 'si::s-data)))
;;       (list 'call-global info 'si:structure-set (c1args args info))))


;; The following (side-effects) exists for putting at the end of an
;; argument list to force all previous arguments to be stored in
;; variables, when computing inline-args.


(push '(() t #.(flags ans set) "Ct")  (get 'side-effects  'inline-always))

(defun c2structure-set (x name-vv ind y sd 
                          &aux locs (*vs* *vs*) (*inline-blocks* 0))
  (declare (ignore name-vv))
  (let* ((raw (si::s-data-raw sd))
	 (type (nth (aref raw ind) +cmp-array-types+))
	 (type (if (type= #tcharacter type) (car (assoc #tchar +c-type-string-alist+ :test 'type=)) type));FIXME STREF STSET handled as aref)
	 (spos (si::s-data-slot-position sd))
	 (tftype type)
	 ix iy)

    (setq locs (inline-args
		(list x y (list 'call-global (make-info) 'side-effects nil))
		(if (eq type t) '(t t t)
		  `(t ,tftype t))))

    (setq ix (car locs))
    (setq iy (cadr locs))
    (if *safe-compile* (wfs-error))
    (wt-nl "STSET(" (aet-c-type type) "," ix "," (aref spos ind) ", " iy ");");FIXME STREF STSET handled as aref
    (unwind-exit (list (sinline-type tftype) (flags) 'wt-loc (list iy)))
    (close-inline-blocks)))

(defun sv-wrap (x) `(symbol-value ',x))

(defun infinite-val-symbol (val)
  (or (car (member val '(+inf -inf nan +sinf -sinf snan) :key 'symbol-value))
      (baboon)))

(defun printable-long-float (val)
  (labels ((scl (val s) `(* ,(/ val (symbol-value s)) ,s)))
	  (let ((nval
		 (cond ((not (isfinite val)) `(symbol-value ',(infinite-val-symbol val)))
		       ((> (abs val) (/ most-positive-long-float 2)) (scl val 'most-positive-long-float))
		       ((< 0.0 (abs val) (* least-positive-normalized-long-float 1.0d20)) (scl val 'least-positive-normalized-long-float)))))
	    (if nval (cons '|#,| nval) val))))
  

(defun printable-short-float (val)
  (labels ((scl (val s) `(* ,(/ val (symbol-value s)) ,s)))
	  (let ((nval
		 (cond ((not (isfinite val)) `(symbol-value ',(infinite-val-symbol val)))
		       ((> (abs val) (/ most-positive-short-float 2)) (scl val 'most-positive-short-float))
		       ((< 0.0 (abs val) (* least-positive-normalized-short-float 1.0d20)) (scl val 'least-positive-normalized-short-float)))))
	    (if nval (cons '|#,| nval) val))))


(defun ltvp (val)
  (when (consp val) (eq (car val) '|#,|)))

(defun c1constant-value-object (val always)
  (typecase
   val
   (char                               `(char-value nil ,val))
   (immfix                             `(fixnum-value nil ,val))
   (character                          `(character-value nil ,(char-code val)))
   (long-float                         `(vv ,(printable-long-float val)))
   (short-float                        `(vv ,(printable-short-float val)));FIXME
   ((or fixnum complex)                `(vv ,val))
   (otherwise                          (when (or always (ltvp val))
					 `(vv ,val)))))

(defun c1constant-value (val always &aux (val (if (exit-to-fmla-p) (not (not val)) val)))
  (case 
   val
   ((nil) (c1nil))
   ((t)   (c1t))
   (otherwise
    (let ((l (c1constant-value-object val (or always (when *compiler-compile* (not *keep-gaz*))))))
      (when l 
	`(location 
	  ,(make-info :type (or (ltvp val)
				(object-type
				 (typecase val
				   (function  (afe (cons 'df nil) (mf (fle val))))
				   (list (copy-tree val))
				   (t val)))))
	  ,l))))))

(defvar *compiler-temps*
        '(tmp0 tmp1 tmp2 tmp3 tmp4 tmp5 tmp6 tmp7 tmp8 tmp9))

(defmacro si:define-inline-function (name vars &body body)
  (let ((temps nil)
        (*compiler-temps* *compiler-temps*))
    (dolist (var vars)
      (if (and (symbolp var)
               (not (member var '(&optional &rest &key &aux))))
          (push (or (pop *compiler-temps*)
                    (gentemp "TMP" (find-package 'compiler)))
                temps)
          (error "The parameter ~s for the inline function ~s is illegal."
                 var name)))
    (let ((binding (cons 'list (mapcar
                                #'(lambda (var temp) `(list ',var ,temp))
                                vars temps))))
      `(progn
         (defun ,name ,vars ,@body)
         (si:define-compiler-macro ,name ,temps
           (list* 'let ,binding ',body))))))


(defun co1structure-predicate (f args &aux tem)
  (cond ((and (symbolp f)
	      (setq tem (get f 'si::struct-predicate))
	      args (not (cdr args)))
	 (c1expr `(typep ,(car args) ',tem)))))


;;New C ffi
;
(defmacro defdlfun ((crt name &optional (lib "")) &rest tps
		    &aux (tsyms (load-time-value (mapl (lambda (x) (setf (car x) (gensym "DEFDLFUN")))
						       (make-list call-arguments-limit)))))
  (unless (>= (length tsyms) (length tps))
    (baboon))
  (flet ((cc (x) (if (consp x) (car x) x)))
	(let* ((sym  (mdlsym name lib))
	       (dls  (strcat "DL" name))
	       (ttps (mapcan (lambda (x) (if (atom x) (list x) (list (list (car x)) (cadr x)))) tps))
	       (args (mapcar (lambda (x) (declare (ignore x)) (pop tsyms)) ttps))
	       (cast (apply 'strcat (maplist (lambda (x) (strcat (cc (car x)) (if (cdr x) "," ""))) tps)))
	       (cast (strcat "(" crt "(*)(" cast "))")))
	  `(defun ,sym ,args
	     (declare (optimize (safety 2)))
	     ,@(mapcar (lambda (x y) `(check-type ,x ,(get (cc y) 'lisp-type))) args ttps)
	     (cadd-dladdress ,dls ,sym)
	     (lit ,crt
		  ,@(when (eq crt :void) `("("))
		  "(" ,cast "(" ,dls "))("
		  ,@(mapcon (lambda (x y) `((,(cc (car x)) ,(car y))
					    ,(if (cdr x) (if (consp (car x)) "+" ",") ""))) ttps args)
		  ")"
		  ,@(when (eq crt :void) `(",Cnil)")))))))

(defun c1cadd-dladdress (args)
  (list 'cadd-dladdress (make-info :type #tnull) args))
(defun c2cadd-dladdress (args)
  (apply 'add-dladdress args))
(si::putprop 'cadd-dladdress 'c1cadd-dladdress 'c1)
(si::putprop 'cadd-dladdress 'c2cadd-dladdress 'c2)

(defun c1clines (args)
  (list 'clines (make-info :type nil) (with-output-to-string (s) (princ (car args) s))))
(defun c2clines (clines)
  (wt-nl clines))
(si::putprop 'clines 'c1clines 'c1)
(si::putprop 'clines 'c2clines 'c2)


;; (define-compiler-macro typep (&whole form &rest args &aux (info (make-info))(nargs (c1args args info)))
;;   (let* ((info (make-info))
;; 	 (nargs (with-restore-vars (c1args args info)))
;; 	 (tp (info-type (cadar nargs)))
;; 	 (a (atomic-tp (info-type (cadadr nargs))))
;; 	 (c (cmp-norm-tp (car a))))
;;     (if (when a (constant-type-p (car a)))
;; 	(cond ((type>= c tp) (print (list c tp t)) t)
;; 	      ((not (type-and c tp)) (print (list c tp nil)) nil)
;; 	      (form));FIXME hash here
;;       form)))


(define-compiler-macro fset (&whole form &rest args)
  (when *sig-discovery*
    (let* ((info (make-info))
	   (nargs (with-restore-vars (c1args args info)))
	   (ff (cadr nargs))
	   (fun (when (eq (car ff) 'function) (caaddr ff)))
	   (fun (when (fun-p fun) fun))
	   (sym (car (atomic-tp (info-type (cadar nargs))))))
      (when (and sym fun);FIXME
	(push (cons sym (apply 'si::make-function-plist (fun-call fun))) si::*sig-discovery-props*))))
  form)


(define-compiler-macro typep (&whole form &rest args);FIXME compiler-in-use
  (with-restore-vars
   (let* ((info (make-info))
	  (nargs (c1args args info))
	  (tp (info-type (cadar nargs)))
	  (a (atomic-tp (info-type (cadadr nargs))))
	  (c (if (when a (constant-type-p (car a))) (cmp-norm-tp (car a)) '*)))
     (cond ((eq c '*) form)
	   ((member-if-not 'ignorable-form nargs) form)
	   ((type>= c tp) (keep-vars) t)
	   ((not (type-and c tp)) (keep-vars) nil)
	   ((when (consp (car a)) (eq (caar a) 'or))
	    `(typecase ,(car args) (,(car a) t)))
	   (form)))));FIXME hash here


(define-compiler-macro vector-push-extend (&whole form &rest args);FIXME compiler-in-use
  (let* ((vref (when (symbolp (cadr args)) (c1vref (cadr args))));FIXME local-aliases
	 (var (car vref)))
    (when vref
      (do-setq-tp var form (reduce (lambda (y x) (if (type-and y x) (type-or1 y x) y))
				    '#.(mapcar (lambda (x) (cmp-norm-tp `(,(cdr x) 1))) si::*all-array-types*)
				    :initial-value (var-type var)))))
  form)


