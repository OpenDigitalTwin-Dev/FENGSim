;;; CMPCALL  Function call.
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

(defvar *ifuncall* nil)


(defun link-arg-p (x)
  (or (is-global-arg-type x) (not (is-local-arg-type x))))

(defun fast-link-proclaimed-type-p (fname &optional args)
  (and 
   (symbolp fname)
;   (not (get fname 'lfun))
   (and (< (length args) 64)
	(or  (and (get fname 'fixed-args) (listp args))
	     (and (link-arg-p (get-return-type fname))
		  (not (member-if-not 'link-arg-p (get-arg-types fname))))))))

(si::putprop 'funcall 'c2funcall-aux 'wholec2)
(si:putprop 'call-global 'c2call-global 'c2)

;;Like macro-function except it searches the lexical environment,
;;to determine if the macro is shadowed by a function or a macro.
(defun cmp-macro-function (name &aux (fun (local-fun-obj name)))
  (if fun (unless (fun-src fun) (fun-fn fun)) (macro-function name)))


(defun c2funcall-aux(form &aux (funob (caddr form)) (args (cadddr form)))
  (c2funcall funob args))

(defvar  *use-sfuncall* t)
(defvar *super-funcall* nil)

(defun c2funcall (funob args &optional loc)

  (unless (listp args)
    (if *compiler-push-events*
	(wt-nl "super_funcall(" loc ");")
      (if *super-funcall*
	  (funcall *super-funcall* loc)
	(wt-nl "super_funcall_no_event(" loc ");")))
    (unwind-exit 'fun-val)
    (return-from c2funcall nil))

  (unless (eq 'ordinary (car funob)) (baboon))

  (let* ((fn (caddr funob))
	 (all (cons fn args))
	 (*inline-blocks* 0))
    (setq *sup-used* t)
    (unwind-exit 
     (get-inline-loc
      (list (make-list (length all) :initial-element t)
	    '* #.(flags ans set svt) 
	    (concatenate 'string
	    "({object _z,_f=#0;fixnum _v=(fixnum)#v;
        fcall.fun=_f;fcall.valp=_v;fcall.argd=#n-1;
        _z=Rset && !(_f)->fun.fun_argd &&
        fcall.argd>=(_f)->fun.fun_minarg && fcall.argd<=((_f)->fun.fun_maxarg) ?
        "
	    (if args
		"(_f)->fun.fun_self(#*)"
	      "((_f)->fun.fun_maxarg ? (_f)->fun.fun_self(#?) : (_f)->fun.fun_self(#*))")
	    " : call_proc_cs2(#?);
           if (!(_f)->fun.fun_neval && !(_f)->fun.fun_vv) vs_top=_v ? (object *)_v : sup;
           _z;})")) all))
    (close-inline-blocks)))


(defun save-avma (fd)
  (when (and (not *restore-avma*)
	     (setq *restore-avma*
		 (or 
		  (member 'integer (car fd))
		  (eq (cadr fd) 'integer)
		  (flag-p (caddr fd) is))))
    (wt-nl "{ save_avma;")
    (inc-inline-blocks)
    (or (consp *inline-blocks*)
	(setq *inline-blocks* (cons  *inline-blocks* 'restore-avma)))))

    
(defun find-var (n x &optional f)
  (cond ((not f) (find-var n x t))
	((var-p x) (when (eq n (var-name x)) x))
	((atom x) nil)
	((or (find-var n (car x) f) (find-var n (cdr x) f)))))

(defun ori-p (x)
  (and (consp x) 
       (eq (car x) 'var) 
       (char= #\Z (aref (symbol-name (var-name (caaddr x))) 0))))

(defun kp (x y)
  (setf (get y 'kp) x)
  (cons x y))

(defun ll-sym (x &optional kn)
  (cond ((atom x) x)
	((atom (car x)) (car x))
	(kn (caar x))
	((cadar x))))

(defun ll-alist (l &aux k (a "G"))
  (mapcan (lambda (x) 
	    (cond ((member x lambda-list-keywords) (setq k x a (string (aref (symbol-name k) 1))) nil)
		  (`(,@(when (and (consp x) (caddr x)) (list (kp (caddr x) (gensym "P"))))
		     ,(kp (ll-sym x) (gensym a)))))) l))

;; (defun ll-alist (l &aux k)
;;   (mapcan (lambda (x) 
;; 	    (cond ((member x lambda-list-keywords) (setq k x) nil)
;; 		  (`(,@(when (and (consp x) (caddr x)) (list (kp (caddr x) (gensym "P"))))
;; 		     ,(kp (ll-sym x) (gensym (if k (string (aref (symbol-name k) 1)) "G"))))))) l))

(defun name-keys (l &aux k)
  (mapcar (lambda (x)
	    (cond ((member x lambda-list-keywords) (setq k x) x)
		  ((eq k '&key)
		   (cond ((atom x) (list (list (intern (symbol-name x) 'keyword) x)))
			 ((atom (car x)) 
			  (list* (list (intern (symbol-name (car x)) 'keyword) (car x)) (cdr x)))
			 (x)))
		  (x))) l))


(defun blla-recur (tag ll args last) 
  (let* ((ll (ldiff ll (member '&aux ll)));FIXME ? impossible check?
	 (ll (name-keys ll))
	 (s (ll-alist ll))
	 (sl (sublis s ll)))
    (blla sl args last `((tail-recur ,tag ,s)))))

(defmacro tail-recur (&rest r) (declare (ignore r)))

(defun c1tail-recur (args)
  (let* ((s (cadr args))
	 (ts (or (car (member (car args) *ttl-tags* :key 'car)) (baboon)))
	 (ttl-tag (pop ts))
	 (nv (mapcar (lambda (x) (car (member (cdr x) *vars* :key (lambda (x) (when (var-p x) (var-name x)))))) s))
	 (ov (mapcar (lambda (x) (car (member (car x) (car ts) :key (lambda (x) (when (var-p x) (var-name x)))))) s))
	 (v (mapc (lambda (x) (set-var-noreplace x)) (append nv ov)))
	 (*vars* (append v *vars*))
	 (*tags* (cons ttl-tag *tags*))
	 (*lexical-env-mask* (remove ttl-tag (set-difference *lexical-env-mask* v))))
    (c1expr `(progn
	       (setq ,@(mapcan (lambda (x) (list (car x) (cdr x))) s))
	       (go ,(tag-name ttl-tag))))))

    
(setf (get 'tail-recur 'c1) 'c1tail-recur)

(defun c2call-global (fname args loc return-type &optional lastp &aux fd (*vs* *vs*))
  (assert (not (special-operator-p fname)))
  (assert (not (macro-function fname)))
  (assert (listp args))
  (assert (null loc))
  (assert (setq fd (get-inline-info fname args return-type (when lastp (length args))))) 
  (let ((*inline-blocks* 0)
	(*restore-avma*  *restore-avma*)) 
    (save-avma fd)
    (unwind-exit (get-inline-loc fd args) nil fname)
    (close-inline-blocks)))

(defun link-rt (tp global)
  (cond ((cmpt tp) `(,(car tp) ,@(mapcar (lambda (x) (link-rt x global)) (cdr tp))))
	((not tp) #tnull)
	((type>= #tboolean tp) #tt);FIXME
	((car (member tp `(,@(if global +c-global-arg-types+ +c-local-var-types+) t *) :test 'type<=)))))

(defun ldiffn (list tail)
  (if tail (ldiff list tail) list))
(declaim (inline ldiffn))

(defun commasep (x)
  (mapcon (lambda (x) (if (cdr x) (list (car x) ",") (list (car x)))) x))

(defun ms (&rest r) 
  (apply 'concatenate 'string 
	 (mapcar (lambda (x) 
		   (cond ((listp x) (apply 'ms x))
			 ((stringp x) x)
			 ((write-to-string x)))) r)))

(defun nords (n &aux (i -1))
  (mapl (lambda (x) (setf (car x) (incf i))) (make-list n)))

(defun nobs (n &optional (p "_x"))
  (mapcar (lambda (x) (ms p x)) (nords n)))

(defun bind-str (nreq nsup nl)
  (let* ((unroll (nobs (- nreq nsup)))
	 (decl (commasep (cons (list "_l=#" nsup) unroll)))
	 (unroll (mapcar (lambda (x) (list nl x "=_l->c.c_car;if (_l!=Cnil) _n--;_l=_l->c.c_cdr;")) unroll))
	 (ndecl (unless (= nreq nsup) (list "fixnum _n=" (- (1+ nsup)) ";"))))
    (ms ndecl "object " decl ";" unroll)))

(defun cond-str (nreq nsup st)
  (ms "(" 
      (unless (= nreq nsup) (list "_n==" (- (1+ nreq)) (unless st "&&")))
      (unless st "_l==Cnil") ")"))

(defun mod-argstr (n call st nsup)
  (let* ((x (commasep (append (nobs nsup "#") (nobs (- n nsup)) (when st (list "_l")))))
	 (s (or (position #\# call) (length call))))
    (ms (subseq call 0 s) x)))

(defun nvfun-wrap (cname argstr sig clp ap)
  (vfun-wrap (ms cname "(" argstr ")") sig clp ap))

(defun wrong-number-args (&rest r)
  (error 'program-error :format-control "Wrong number of arguments to anonymous function: ~a" :format-arguments (list r)))

(defun insufficient-arg-str (fnstr nreq nsup sig st
				   &aux (sig (if st sig (cons '(*) (cdr sig)))) ;(st nil)(nreq 0)
				   (fnstr (or fnstr (ms (vv-str 'wrong-number-args) "->s.s_gfdef"))))
  (ms (cdr (assoc (cadr sig) +to-c-var-alist+))
      "("
      (nvfun-wrap "call_proc_cs2" 
		  (ms (commasep (append (nobs nsup "#") (nobs (- nreq nsup)) `(("#" ,nsup))))) 
		  sig fnstr (1+ nreq)) 
      ")"));FIXME better way?

;;FIXME can unroll in lisp only?
;; (defun lisp-unroll (sig args)
;;   (let* ((at (car sig))
;; 	 (st (member '* at))
;; 	 (regs (ldiffn at st))
;; 	 (nr (length regs))
;; 	 (la (1- (length args)))
;; 	 (nd (- nr la))
;; 	 (binds (mapc (lambda (x) (setf (car x) (tmpsym))) (make-list la)))
;; 	 (l (tmpsym))
;; 	 (unrolls (mapc (lambda (x) (setf (car x) (tmpsym))) (make-list nd))))
;;     `(let (,@(mapcar 'list binds args)
;; 	   (,l (car (last args)))
;; 	   ,@(mapcar (lambda (x) (list x `(pop ,l))) unrolls))
;;        (if (,l)
;; 	   (apply ',fn ,@binds ,@unrolls ,l)
;; 	 (funcall ',fn ,@binds ,@unrolls)))))

(defun maybe-unroll (argstr cname sig ap clp fnstr)
  (let* ((at (car sig))
	 (st (member '* at))
	 (nreq (length (ldiffn at st)))
	 (nsup (if ap (1- ap) nreq)))
    (when (or (< nsup nreq) (and ap (= nsup nreq) (not st)))
      (let ((nl (list (string #\Newline) "        ")))
	(ms (list "@" (nords (1+ nsup)) ";") 
	    "({" (bind-str nreq nsup nl) nl (cond-str nreq nsup st)  " ? " nl
	    (nvfun-wrap cname (mod-argstr nreq argstr st nsup) sig clp ap) " : " nl
	    (insufficient-arg-str fnstr nreq nsup sig st) ";})")))))


(defun g1 (fnstr cname sig ap clp &optional (lev -1))
  (let* ((x (make-inline-arg-str sig lev)))
    (or (maybe-unroll x cname sig ap clp fnstr)
	(nvfun-wrap cname x sig clp ap))))

;; (defun g0 (cname sig apnarg clp &optional (lev -1))
;;   (let* ((at (car sig))
;; 	 (st (member '* at))
;; 	 (nreg (length (ldiff at st)))
;; 	 (apreg (if apnarg (1- apnarg) nreg))
;; 	 (u (when (< apreg nreg) (- nreg apreg)))
;; 	 (x (make-inline-arg-str sig lev))
;; 	 (ss (when u (search (strcat "#" (write-to-string apreg)) x)))
;; 	 (x (if ss (subseq x 0 ss) x))
;; 	 (yy (when u (let (r) (dotimes (i u (nreverse r)) (push i r)))))
;; 	 (yy (mapcar (lambda (x) (strcat "_x" (write-to-string x))) yy))
;; 	 (y (append yy (when (when st u) (list "_l"))))
;; 	 (y (mapcon (lambda (x) (if (cdr x) (list (car x) ",") (list (car x)))) y))
;; 	 (y (apply 'strcat y))
;; 	 (z (length x))(w (length y))
;; 	 (s (if (or (= w 0) (= z 0) 
;; 		    (char= (char x (1- z)) #\,) (char= (char x (1- z)) #\*)) "" ","))
;; 	 (x (strcat x s y))
;; 	 (x (format nil "(~a(~a))" cname x))
;; 	 (x (vfun-wrap x sig clp))
;; 	 (ss (when apnarg (search "#n" x)))
;; 	 (x (if ss (progn (setf (aref x (1- ss)) #\-) 
;; 			  (when u
;; 			    (setf (aref x (+ 2 ss)) #\-)
;; 			    (setf (aref x (+ 3 ss)) (code-char (+ (truncate u 10) (char-code #\0))))
;; 			    (setf (aref x (+ 4 ss)) (code-char (+ (mod u 10) (char-code #\0)))))
;; 			  x) x))
;; 	 (nx (apply 'strcat (mapcar (lambda (x) (strcat x "=_l->c.c_car;_l=_l->c.c_cdr;")) yy)))
;; 	 (nx (strcat "object _l=#" (write-to-string apreg) 
;; 		     (apply 'strcat (mapcar (lambda (x) (strcat "," x)) yy)) ";" nx))
;; 	 (x (if (> w 0) (concatenate 'string "({" nx x ";})") x)))
;;     x))

(defun g (fname n sig &optional apnarg (clp t)
	  &aux (cname (format nil "/* ~a */(~a)(*LnkLI~d)" (function-string fname) (rep-type (cadr sig)) n))
	    (fnstr (ms (vv-str fname) "->s.s_gfdef"))
	    (clp (when clp fnstr)))
  (g1 fnstr cname sig apnarg clp))

;; (defun g (fname n sig &optional apnarg (clp t)
;; 		&aux (cname (format nil "/* ~a */(*LnkLI~d)" (function-string fname) n))
;; 		(clp (when clp (concatenate 'string (vv-str (add-object fname)) "->s.s_gfdef"))))
;;   (g0 cname sig apnarg clp))

(defun call-arg-types (at la apnarg)
  (let* ((st (member '* at))
	 (reg (ldiff at st))
	 (nr (length reg))
	 (la (if apnarg (max nr (1- la)) la))
	 (ns (- nr la)))
    (cond ((> ns 0) (butlast reg ns));funcall too few args
	  (st at)
	  ((= ns 0) at)
	  ((append at '(*))))));let call_proc_new foil fast linking and catch errors

(defun add-fast-link (fname la &optional apnarg
			    &aux n
			    (at (call-arg-types (mapcar (lambda (x) (link-rt x t)) (get-arg-types fname)) la apnarg))
			    (rt (link-rt (get-return-type fname) t))
			    (clp (cclosure-p fname))
			    (tail (list rt at clp apnarg)))
  
  (cond ((setq n (caddar (member-if 
			  (lambda (x) 
			    (and (eq (car x) fname) 
				 (equal (cdddr x) tail))) *function-links*)))
	 (car (member-if
	       (lambda (x) 
		 (let ((x (last x 2))) 
		   (when (eq 'link-call (car x)) 
		     (eql n (cadr x))))) *inline-functions*)))
	((let* ((n (progn (add-object2 fname) (next-cfun)))
		(f (flags ans set))
		(f (if (single-type-p rt) f (flag-or f svt)))
		(f (if apnarg (flag-or f aa) f)))
	   (push (list* fname (format nil "LI~d" n) n tail) *function-links*)
	   (car (push (list fname at rt f
		       (g fname n (list at rt) apnarg clp)
		       'link-call n)
		      *inline-functions*))))))

;; (defun add-fast-link (fname &optional apnarg
;; 			    &aux n
;; 			    (at (mapcar (lambda (x) (link-rt x t)) (get-arg-types fname)))
;; 			    (rt (link-rt (get-return-type fname) t))
;; 			    (clp (cclosure-p fname))
;; 			    (tail (list rt at clp apnarg)))
  
;;   (cond ((setq n (caddar (member-if 
;; 			  (lambda (x) 
;; 			    (and (eq (car x) fname) 
;; 				 (equal (cdddr x) tail))) *function-links*)))
;; 	 (car (member-if
;; 	       (lambda (x) 
;; 		 (let ((x (last x 2))) 
;; 		   (when (eq 'link-call (car x)) 
;; 		     (eql n (cadr x))))) *inline-functions*)))
;; 	((let* ((n (next-cfun))
;; 		(f (flags ans set))
;; 		(f (if (single-type-p rt) f (flag-or f svt)))
;; 		(f (if apnarg (flag-or f aa) f)))
;; 	   (push (list* fname (format nil "LI~d" n) n tail) *function-links*)
;; 	   (car (push (list fname at rt f
;; 		       (g fname n (list at rt) apnarg clp)
;; 		       'link-call n)
;; 		      *inline-functions*))))))



;;make a function which will be called hopefully only once,
;;and will establish the link.
(defun wt-function-link (x)
  (let* ((name (pop x))
	 (num (pop x))
	 (n (pop x))
	 (type (pop x))
	 (type (or type t));FIXME
	 (args (pop x))
	 (clp (pop x)))
    (declare (ignore n))
    (cond
      (t
       ;;change later to include above.
       ;;(setq type (cdr (assoc type '((t . "object")(:btpr . "bptr")))))
       (wt-nl1 "static " (declaration-type (rep-type type)) " LnkT" num)
       (let ((d (declaration-type (rep-type (if (link-arg-p type) type t)))));FIXME
	 (if (or args (not (eq t type)))
	     (wt "(object first,...){" d "V1;va_list ap;va_start(ap,first);V1=(" d ")"
		 "call_proc_new(" (vv-str name) "," (if clp "1" "0") ","
		 (write-to-string (argsizes args type 0));FIXME
		 ",(void **)(void *)&Lnk" num "," (new-proclaimed-argd args type)
		 ",first,ap);va_end(ap);return V1;}")
	   (wt "(){" d "V1=(" d ")call_proc_new_nval(" (vv-str name) "," (if clp "1" "0") ","
	       (write-to-string (argsizes args type 0));FIXME
	       ",(void **)(void *)&Lnk" num "," (new-proclaimed-argd args type)
	       ",0);return V1;}")))))
    (setq name (function-string name))
    (if (find #\/ name) (setq name (remove #\/ name)))
    (wt " /* " name " */")))
      


;;For funcalling when the argument is guaranteed to be a compiled-function.
;;For (funcall-c he 3 4), he being a compiled function. (not a symbol)!
;; (defun wt-funcall-c (args)
;;   (let ((fun (car args))
;; 	(real-args (cdr args))
;; 	loc)
;;     (cond ((eql (car fun) 'var)
;;            (let ((fun-loc (cons (car fun) (third fun))))
;; 	     (when *safe-compile*
;; 		   (wt-nl "(type_of(")
;; 		   (wt-loc fun-loc)
;; 		   (wt ")==t_cfun)||FEinvalid_function(")
;; 		   (wt-loc fun-loc)(wt ");"))
;; 	   (push-args real-args)
;; 	   (wt-nl "(")  
;; 	   (wt-loc  fun-loc)))
;; 	  (t
;; 	   (setq loc (list 'cvar (cs-push t t)))
;; 	   (let ((*value-to-go* loc))
;; 	     (wt-nl 
;; 	      "{object V" (second loc) ";")
;; 	     (c2expr* (car args))
;; 	     (push-args (cdr args))
;; 	     (wt "(V" (second loc)))))
;;     (wt ")->cf.cf_self ();")
;;     (and loc (wt "}")))
;;   (unwind-exit 'fun-val))



(si:putprop 'simple-call 'wt-simple-call 'wt-loc)

(defun wt-simple-call (cfun base n &optional (vv-index nil))
  (wt "simple_" cfun "(")
  (when vv-index (wt (vv-str vv-index) ","))
  (wt "base+" base "," n ")")
  (base-used))

;;; Functions that use SAVE-FUNOB should reset *vs*.
(defun save-funob (funob &aux (temp (list 'vs (vs-push))))
  (let ((*value-to-go* temp))
    (c2expr* funob)
    temp))
;; (defun save-funob (funob &optional force)
;;   (case (car funob)
;;         ((call-quote-lambda call-local))
;;         (call-global
;;          (unless (and (not force)
;; 		      (inline-possible (caddr funob))
;; 		      (or (get (caddr funob) 'Lfun)
;; 			  (get (caddr funob) 'Ufun)
;; 			  (assoc (caddr funob) *global-funs*)))
		     
;;            (let ((temp (list 'vs (vs-push))))
;;                 (if *safe-compile*
;;                     (wt-nl
;;                      temp
;;                      "=symbol_function(" (vv-str (add-symbol (caddr funob))) ");")
;;                     (wt-nl temp
;;                            "=" (vv-str (add-symbol (caddr funob))) "->s.s_gfdef;"))
;;                 temp)))
;;         (ordinary (let* ((temp (list 'vs (vs-push)))
;;                          (*value-to-go* temp))
;;                         (c2expr* (caddr funob))
;;                         temp))
;;         (otherwise (baboon))
;;         ))

(defun push-args (args &optional lastp)
  (cond ((null args) (wt-nl "vs_base=vs_top;"))
        ((consp args)
         (let ((*vs* *vs*) (base *vs*))
           (dolist (arg args)
             (let ((*value-to-go* (list 'vs (vs-push))))
               (c2expr* arg)))
           (wt-nl "vs_top=(vs_base=base+" base ")+" (- *vs* base) ";")
	   (when lastp
	     (wt-nl "{object _x=*--vs_top;for (;_x!=Cnil;_x=_x->c.c_cdr) *vs_top++=_x->c.c_car;}"))
           (base-used)))))

(defun push-args-lispcall (args)
  (dolist (arg args)
    (let ((*value-to-go* (list 'vs (vs-push))))
      (c2expr* arg))))

