;;; CMPVAR  Variables.
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

(si:putprop 'var 'c2var 'c2)
(si:putprop 'location 'c2location 'c2)
(si:putprop 'setq 'c1setq 'c1special)
(si:putprop 'setq 'c2setq 'c2)
(si:putprop 'progv 'c1progv 'c1special)
(si:putprop 'progv 'c2progv 'c2)
;; (si:putprop 'psetq 'c1psetq 'c1)
;; (si:putprop 'psetq 'c2psetq 'c2)

(si:putprop 'var 'set-var 'set-loc)
(si:putprop 'cvar 'set-cvar 'set-loc)
(si:putprop 'var 'wt-var 'wt-loc)

(defstruct (var (:print-function (lambda (x s i) (s-print 'var (var-name x) (si::address x) s))))
  name		;;; Variable name.
  kind		;;; One of LEXICAL, SPECIAL, GLOBAL, REPLACED, FIXNUM,
  		;;; CHARACTER, LONG-FLOAT, SHORT-FLOAT, and OBJECT.
  ref		;;; Referenced or not.
  		;;; During Pass1, T, NIL, or IGNORE.
  		;;; During Pass2, the vs-address for the variable.
  ref-ccb	;;; Cross closure reference.
  		;;; During Pass1, T or NIL.
  		;;; During Pass2, the ccb-vs for the variable, or NIL.
  loc		;;; For SPECIAL and GLOBAL, the vv-index for variable name.
		;;; For others, this field is used to indicate whether
		;;; to be allocated on the value-stack: OBJECT means
		;;; the variable is declared as OBJECT, and CLB means
		;;; the variable is referenced across Level Boundary and thus
		;;; cannot be allocated on the C stack.  Note that OBJECT is
		;;; set during variable binding and CLB is set when the
		;;; variable is used later, and therefore CLB may supersede
		;;; OBJECT.
  		;;; For REPLACED, the actual location of the variable.
  		;;; For FIXNUM, CHARACTER, LONG-FLOAT, SHORT-FLOAT, and
  		;;; OBJECT, the cvar for the C variable that holds the value.
  		;;; Not used for LEXICAL.
  (dt t)	;;; Declared Type of the variable.
  (type t)	;;; Current Type of the variable.
  (mt t)	;;; Maximum type of the life of this binding
  tag           ;;; Inner tag (to binding) being analyzed if any
  (register 0 :type unsigned-char)  ;;; If greater than specified am't this goes into register.
  (flags    0 :type unsigned-char)  ;;; If variable is declared dynamic-extent
  (space    0 :type char)           ;;; If variable is declared as an object array of this size
  (known-init -1 :type char)        ;;; Number of above known to be implicitly initialized
  store         ;;; keep kind in hashed c1forms
  aliases
  )

(si::freeze-defstruct 'var)

(defun var-dynamic (v);FIXME
  (/= 0 (logand 1 (var-flags v))))
(defun var-reffed (v)
  (/= 0 (logand 2 (var-flags v))))
(defun var-noreplace (v)
  (/= 0 (logand 4 (var-flags v))))
(defun var-set (v)
  (/= 0 (logand 8 (var-flags v))))
(defun var-aliased (v)
  (/= 0 (logand 16 (var-flags v))))

(defun set-var-dynamic (v)
  (setf (var-flags v) (logior 1 (var-flags v))))
(defun set-var-reffed (v)
  (setf (var-flags v) (logior 2 (var-flags v))))
(defun set-var-noreplace (v)
  (setf (var-flags v) (logior 4 (var-flags v))))
(defun set-var-set (v)
  (setf (var-flags v) (logior 8 (var-flags v))))
(defun set-var-aliased (v)
  (setf (var-flags v) (logior 16 (var-flags v))))

(defun unset-var-set (v)
  (setf (var-flags v) (logandc2 (var-flags v) 8)))
(defun unset-var-aliased (v)
  (setf (var-flags v) (logandc2 (var-flags v) 16)))

;;; A special binding creates a var object with the kind field SPECIAL,
;;; whereas a special declaration without binding creates a var object with
;;; the kind field GLOBAL.  Thus a reference to GLOBAL may need to make sure
;;; that the variable has a value.

(defvar *vars* nil)
(defvar *register-min* 4) ;criteria for putting in register.
(defvar *undefined-vars* nil)
(defvar *special-binding* nil)

;;; During Pass 1, *vars* holds a list of var objects and the symbols 'CB'
;;; (Closure Boundary) and 'LB' (Level Boundary).  'CB' will be pushed on
;;; *vars* when the compiler begins to process a closure.  'LB' will be pushed
;;; on *vars* when *level* is incremented.
;;; *GLOBALS* holds a list of var objects for those variables that are
;;; not defined.  This list is used only to suppress duplicated warnings when
;;; undefined variables are detected.

(defun is-rep-referred (var info)
  (let ((rx (var-rep-loc var)))
    (do-referred (v info)
     (let ((ry (var-rep-loc v)))
       (when (or (eql-not-nil (var-loc var) ry)
		 (eql-not-nil (var-loc v) rx)
		 (eql-not-nil rx ry))
	 (return-from is-rep-referred t))))))

(defun ens-k-tp (tp)
  (or (third tp)
      (member-if (lambda (x)
		   (when (member (car x) '(proper-cons si::improper-cons))
		     (member-if (lambda (x)
				  (when (listp x)
				    (or (ens-k-tp (car x)) (ens-k-tp (cadr x)))))
				(cdr x))))
		 (car tp))))

(defun ensure-known-type (tp)
  (if (when (listp tp) (ens-k-tp (third tp)))
      (car tp)
    tp))

(defun c1make-var (name specials ignores types &aux x)

  (let ((var (make-var :name name)))

    (cmpck (not (symbolp name)) "The variable ~s is not a symbol." name)
    (cmpck (constantp name)     "The constant ~s is being bound." name)
    
    (dolist (v types)
      (when (eq (car v) name)
	(case (cdr v)
	      (object (setf (var-loc var) 'object))
	      (register (setf (var-register var) (+ (var-register var) 100)))
	      (dynamic-extent #+dynamic-extent (set-var-dynamic var))
	      (t (unless (and (not (get (var-name var) 'tmp));FIXME
			      *compiler-new-safety*) 
		   (setf (var-type var) (ensure-known-type (nil-to-t (type-and (var-type var) (cdr v))))))))))
    
    (cond ((or (member name specials) (si:specialp name))
	   (setf (var-kind var) 'SPECIAL)
	   (setf (var-loc var) name)
	   (when (and (not *compiler-new-safety*) (not (assoc name types)) (setq x (get name 'cmp-type)))
	     (setf (var-type var) (ensure-known-type x)))
	   (setq *special-binding* t))
	  (t
	   (and (boundp '*c-gc*) *c-gc*
		(or (null (var-type var))
		    (eq t (var-type var)))
		(setf (var-loc var) 'object))
	   (setf (var-kind var) 'LEXICAL)))
    (let ((ign (member name ignores)))
      (when ign
	(setf (var-ref var) (if (eq (cadr ign) 'ignorable) 'IGNORABLE 'IGNORE))))
    
    (setf (var-mt var) (var-type var))
    (setf (var-dt var) (var-type var))
    var))

(defvar *top-level-src* nil)
(defvar *top-level-src-p* t)

(defun mark-toplevel-src (src)
  (when *top-level-src-p*
    (pushnew src *top-level-src*))
  src)

(defun check-vref (var)
  (when *top-level-src-p*
    (when (and (eq (var-kind var) 'LEXICAL)
	       (not (var-reffed var))
	       (not (var-ref var)));;; This field may be IGNORE or IGNORABLE here.
      (cmpstyle-warn "The variable ~s is not used." (var-name var)))))

(defun var-cb (v)
  (or (var-ref-ccb v) (eq 'clb (var-loc v))))

(defun add-vref (vref info &optional setq)
  (cond ((cadr vref)  (push (car vref) (info-ref-ccb info)))
	((caddr vref) (push (car vref) (info-ref-clb info)))
	((not setq)   (push (car vref) (info-ref     info)))))

(defun make-vs (info) (mapcan (lambda (x) (when (var-p x) (list (cons x (var-bind x))))) (info-ref info)))

(defun check-vs (vs &aux (b (member-if-not 'var-p *vars*)))
  (not (member-if-not (lambda (x &aux (v (pop x))(vv (member v *vars*)))
			(when vv
			  (when (tailp b vv)
			    (bind-match x v))))
		      vs)))

(defun find-vs (form)
  (case (car form)
    ((var lit) (car (last form)))))

(defun c1var (name)
  (let* ((info (make-info))
	 (vref (c1vref name))
	 (tmp (get-var (local-var vref)))
	 (tmp (unless (eq tmp (car vref)) tmp))
	 (vref (if tmp (c1vref tmp) vref))
	 (c1fv (when (cadr vref) (c1inner-fun-var))))
    (setf (info-type info) (if (or (cadr vref) (caddr vref)) (var-dt (car vref)) (var-type (car vref)))
	  (var-mt (car vref)) (type-or1 (info-type info) (var-mt (car vref))))
    (add-vref vref info)
    (when c1fv
      (add-info info (cadr c1fv)))
    (mapc (lambda (x) (setf (info-ch-ccb info) (nunion (info-ch-ccb info) (info-ch-ccb (cadr x)))));FIXME nunion asym
	  (binding-forms (var-store (car vref))))
    (let ((fmla (exit-to-fmla-p)))
      (cond ((when fmla (type>= #tnull (info-type info))) (c1nil))
	    ((when fmla (type>= #t(not null) (info-type info))) (c1t))
	    ((let ((tmp (get-vbind-form (local-var vref))))
	       (when (and tmp );FIXME (type>= (var-mt (car vref)) (var-mt (caaddr tmp)))
		 (when (check-vs (find-vs tmp));(when (eq 'var (car tmp)) (car (last tmp)))
		   (let* ((f (pop tmp))(i (copy-info (pop tmp))))
;		     (setf (info-type i) (if (eq f 'var) (var-type (caar tmp)) (type-and (info-type i) (info-type info))));FIXME
		     (setf (info-type i) (type-and (info-type i) (info-type info)))
		     (when (eq f 'var)
		       (setf (info-type i) (type-and (info-type i) (var-type (caar tmp)))))
		     (list* f i tmp))))))
	    ((list 'var info vref c1fv (make-vs info)))))))

(defun ref-obs (form obs sccb sclb s &aux (i (cadr form)))
  (mapc (lambda (x)
	  (when (member x (info-ref-ccb i))
	    (funcall sccb x))
	  (when (member x (info-ref-clb i))
	    (funcall sclb x))
	  (when (member x (info-ref i))
	    (funcall s x)))
	obs))
(declaim (inline ref-obs))

(defun ref-vars (form vars)
  (ref-obs form vars 
	   (lambda (x) (when (eq (var-kind x) 'lexical) (setf (var-ref-ccb x) t)))
	   (lambda (x) (when (eq (var-kind x) 'lexical) (setf (var-loc x) 'clb)) (setf (var-ref x) t))
	   (lambda (x) (setf (var-ref x) t (var-register x) (1+ (var-register x))))))

(defun inner-fun-var (&optional (v *vars*) f &aux (y v) (x (pop v)))
  (cond ((atom v) nil)
	((is-fun-var x) (inner-fun-var v y))
	((eq x 'cb) f)
	((inner-fun-var v f))))

(defun c1inner-fun-var nil
  (let ((*vars* (inner-fun-var)))
    (c1var (var-name (car *vars*)))))
    

(defun local-var (vref &aux (v (pop vref)))
  (unless (or (car vref) (cadr vref))
    v))

(defun get-vbind-form (form &aux (binding (get-vbind form)))
  (when binding
    (when (binding-repeatable binding)
      (binding-form binding))))

(defun var-bind (var &aux (st (when (var-p var) (when (eq 'lexical (var-kind var)) (var-store var)))))
  (unless (cdr st)
    (car st)))

(defun get-vbind (form)
  (var-bind
   (typecase
    form
    ((cons (eql var) t) (when (check-vs (car (last form)))  (local-var (caddr form))))
    (var form))))

(defun lit-bind (x)
  (case (car x)
    (lit (sixth x))))

(defun get-bind (x)
  (typecase
   x
   ((cons (eql var) t) (when (check-vs (car (last x))) (var-bind (local-var (caddr x)))))
   ((cons (eql lit) t) (when (check-vs (car (last x))) (lit-bind x)))
   (var (var-bind x))
   (binding x)))

(defun repeatable-var-binding (form)
  (case (car form)
	((var location lit) form)))

(defun repeatable-binding-p (form &aux (i (cadr (repeatable-var-binding form))))
  (when i
    (when (info-type i)
      (unless (iflag-p (info-flags i) side-effects)
	(unless (or (info-ref-clb i) (info-ref-ccb i))
	  t)))))


(defun new-bind (&optional form)
  (make-binding :form form :repeatable (repeatable-binding-p form)))

(defun or-bind (b l &aux (bi (cadr (binding-form b))))
  (cond ((when (cdr l) (when bi (not (info-ch-ccb bi))));FIXME coalesce anonymous too?
	 (pushnew b l :test (lambda (x y)
			      (or (eq x y)
				  (when (binding-form y)
				    (type<= (info-type bi) (info-type (cadr (binding-form y)))))))))
	((pushnew b l))))

(defun or-binds (l1 l2)
  (reduce (lambda (y x) (or-bind x y)) l1 :initial-value l2))

(defun bind-block (name)
  (or (eq name +mv+); FIXME c1 *mv-var*
;      (eq name +first+)
;      (eq name +fun+)
;      (get name 'tmp)
;      (eq name +nargs+) ;FIXME invalidate on call
      ))

(defun push-vbind (var form &optional or)
  (unless (bind-block (var-name var))
    (setf (var-store var)
	  (or-bind
	   (or (get-bind form) (new-bind form))
	   (when or (var-store var))))))

(defun push-vbinds (var forms); &optional or
  (mapc (lambda (x) (push-vbind var x t)) forms))

(defun bind-match (f1 f2 &aux (b1 (get-bind f1)))
  (when b1
    (eq b1 (get-bind f2))))


(defun get-top-var-binding (bind)
  (labels ((f (l) (member bind l :key 'var-bind))
	   (r (l) (let* ((var (car l))
			 (nl  (f (cdr l)))
			 (nl  (when (eq nl (member (car nl) *vars*)) nl)));FIXME impossible?
		    (if (tailp nl (member-if-not 'var-p l)) var (r nl)))))
	  (when bind ;FIXME defvar
	    (r (f *vars*)))))

(defun get-var (o &aux (vp (var-p o)))
  (or (get-top-var-binding (if vp (get-vbind o) o)) (when vp o)))

(defun c1vref (name &optional setq &aux ccb clb)
  (dolist (var *vars*
               (let ((var (sch-global name)))
                 (unless var
		   (unless (symbolp name) (baboon))
                   (unless (or (si:specialp name) (constantp name)) (undefined-variable name))
                   (setq var (make-var :name name
                                       :kind 'GLOBAL
                                       :loc name
                                       :type (or (get name 'cmp-type) t)
				       :ref t));FIXME
                   (push var *undefined-vars*))
                 (list var ccb)))
      (cond ((eq var 'cb) (setq ccb t))
            ((eq var 'lb) (setq clb t))
            ((or (when (eq (var-name var) name) (not (member var *lexical-env-mask*))) (eq var name))
	     (unless setq
	       (when (eq (var-ref var) 'IGNORE)
		 (unless (var-reffed var)
		   (cmpstyle-warn "The ignored variable ~s is used." name))))
	     (set-var-reffed var)
	     (keyed-cmpnote (list 'var-ref (var-name var))
			    "Making variable ~s reference with barrier ~s" (var-name var) (if ccb 'cb (if clb 'lb)))
	     (return-from c1vref (list* var (if (eq (var-kind var) 'lexical) (list ccb clb) '(nil nil))))))))

(defun c2var-kind (var)
  (when (and (eq (var-kind var) 'LEXICAL)
           (not (var-ref-ccb var))
           (not (eq (var-loc var) 'clb)))
    (cond ((eq (var-loc var) 'object) (setf (var-type var) #tt) (var-loc var)) ;FIXME check ok; need *c-vars* and kind to agree
	  ((car (member (var-type var) +c-local-var-types+ :test 'type<=)))
	  ((and (boundp '*c-gc*) *c-gc* 'OBJECT)))))


(defun c2var (vref c1fv stores) (declare (ignore c1fv stores)) (unwind-exit (cons 'var vref) nil 'single-value))

(defun c2location (loc) (unwind-exit loc nil 'single-value))


(defun wt-var (var ccb &optional clb)
  (declare (ignorable clb));FIXME
  (case (var-kind var)
        (LEXICAL (cond (ccb (wt-ccb-vs (var-ref-ccb var)))
                       ((var-ref-ccb var) (wt-vs* (var-ref var)))
		       ((and (eq t (var-ref var)) 
			     (si:fixnump (var-loc var))
			     *c-gc*
			     (eq t (var-type var)))
			(setf (var-kind var) 'object)
			(wt-var var ccb))
                       (t (wt-vs (var-ref var)))))
        (SPECIAL (wt "(" (vv-str (var-loc var)) "->s.s_dbind)"))
        (REPLACED (wt (var-loc var)))
;        (REPLACED (cond ((and (consp (var-loc var)) (info-p (cadr (var-loc var))))FIXME
;			 (let* ((*inline-blocks* 0)(v (c2expr (var-loc var))))(print v)(break)
;			   (unwind-exit (get-inline-loc `((t) t #.(flags) "(#0)") (list v))
;					nil 'single-value)
;			   (close-inline-blocks)))
;			((wt (var-loc var)))))
	(DOWN  (wt-down (var-loc var)))
        (GLOBAL (if *safe-compile*
                    (wt "symbol_value(" (vv-str (var-loc var)) ")")
		  (wt "(" (vv-str (var-loc var)) "->s.s_dbind)")))
        (t (let ((z (cdr (assoc (var-kind var) +wt-c-var-alist+))))
	     (unless z (baboon))
	     (when (and (equal #tfixnum (var-kind var)) (zerop *space*))
	       (wt "CMP"))
	     (wt z)
           (wt "(V" (var-loc var) ")")))
        ))

;; When setting bignums across setjmps, cannot use alloca as longjmp
;; restores the C stack.  FIXME -- only need malloc when reading variable
;; outside frame.  CM 20031201
(defmacro bignum-expansion-storage ()
  `(if (and (boundp '*unwind-exit*) (member 'frame *unwind-exit*))
       "gcl_gmp_alloc"
     "alloca"))

(defun set-var (loc var ccb &optional clb)
  (declare (ignore clb))
  (unless (and (consp loc)
               (eq (car loc) 'var)
               (eq (cadr loc) var)
               (eq (caddr loc) ccb))
          (case (var-kind var)
            (LEXICAL (wt-nl)
                     (cond (ccb (wt-ccb-vs (var-ref-ccb var)))
                           ((var-ref-ccb var) (wt-vs* (var-ref var)))
                           (t (wt-vs (var-ref var))))
                     (wt "= " loc ";"))
            (SPECIAL (wt-nl "(" (vv-str (var-loc var)) "->s.s_dbind)= " loc ";"))
            (GLOBAL
             (if *safe-compile*
                 (wt-nl "setq(" (vv-str (var-loc var)) "," loc ");")
                 (wt-nl "(" (vv-str (var-loc var)) "->s.s_dbind)= " loc ";")))
	    (DOWN
	      (wt-nl "") (wt-down (var-loc var))
	      (wt "=" loc ";"))
            (t
	     (wt-nl "V" (var-loc var) "= ")
	     (funcall (or (cdr (assoc (var-kind var) +wt-loc-alist+)) (baboon)) loc)
	     (wt ";")))))

(defun set-cvar (loc cvar)
  (wt-nl "V" cvar "= ")
  (let* ((fn (or (car (rassoc cvar *c-vars*)) (cdr (assoc cvar *c-vars*)) t))
	 (fn (or (car (member fn +c-local-var-types+ :test 'type<=)) 'object))
	 (fn (cdr (assoc fn +wt-loc-alist+))))
    (unless fn (baboon))
    (funcall fn loc))
  (wt ";"))

(defun sch-global (name)
  (dolist (var *undefined-vars* nil)
    (when (or (eq var name) (eq (var-name var) name)) (return-from sch-global var))))

(defun c1add-globals (globals)
  (dolist (name globals)
    (push (make-var :name name
                    :kind 'GLOBAL
                    :loc name
                    :type (or (get name 'cmp-type) t))
          *vars*)))

(defun c1setq (args)
  (cond ((endp args) (c1nil))
        ((endp (cdr args)) (too-few-args 'setq 2 1))
        ((endp (cddr args)) (c1setq1 (car args) (cadr args)))
        ((do ((pairs args) forms)
             ((endp pairs) (c1expr (cons 'progn (nreverse forms))))
             (cmpck (endp (cdr pairs)) "No form was given for the value of ~s." (car pairs))
             (push (list 'setq (pop pairs) (pop pairs)) forms)))))

(defun llvar-p (v)
  (when (eq (var-kind v) 'lexical)
    (let ((x (member v *vars*)))
      (when x
	(tailp (member-if-not 'var-p *vars*) x)))))

(defun do-setq-tp (v form t1)
  (unless nil ; *compiler-new-safety* FIXME
    (when (llvar-p v)
      (setq t1 (ensure-known-type (coerce-to-one-value t1)))
      (let* ((tp (type-and (var-dt v) t1)))
	(unless (or tp (not (and (var-dt v) t1)))
	  (cmpwarn "Type mismatches setting declared ~s variable ~s to type ~s from form ~s."
	       (cmp-unnorm-tp (var-dt v)) (var-name v) (cmp-unnorm-tp t1) (car form)))
	(keyed-cmpnote (list (var-name v) 'type-propagation 'type)
		       "Setting var-type on ~s from ~s to ~s, form ~s, max ~s" 
		       (var-name v) (cmp-unnorm-tp (var-type v)) (cmp-unnorm-tp tp) (car form) (cmp-unnorm-tp (var-mt v)))
	(when (member v *restore-vars-env*)
	  (pushnew (list v (var-type v) (var-store v)) *restore-vars* :key 'car))

	(setf (var-type v) tp)
	(unless (type>= (var-mt v) tp)
	  (setf (var-mt v) (type-and (bbump-tp (type-or1 (var-mt v) tp)) (var-dt v))))))))

(defun set-form-type (form type &optional no-recur) (sft form type no-recur))

;; (defun set-form-type (form type) (setf (info-type (cadr form)) (type-and type (info-type (cadr form)))))
;  (sft form type))  FIXME cannot handle nil return types such as tail recursive calls

(defun sft-block (form block type)
  (cond ((atom form))
	((and (eq (car form) 'return-from) (eq (third form) block))
	 (sft (car (last form)) type))
	(t (sft-block (car form) block type) (sft-block (cdr form) block type))))

(defun sft (form type &optional no-recur);FIXME sft-block labels avoid mutual recursion
  (let ((it (info-type (cadr form))))
    (unless (type>= type it)
      (let ((nt (type-and type it)))
	(unless nt
	  (keyed-cmpnote (list 'nil-arg) "Setting form type ~s to nil" (cmp-unnorm-tp it)))
	(when (or (eq form (c1nil)) (eq form (c1t)));FIXME
	  (unless (type= it nt)
	    (return-from sft nil)))
	(setf (info-type (cadr form)) nt)
	(unless no-recur
	  (case (car form)
	    (block (sft-block (fourth form) (third form) type))
	    ((decl-body inline) (sft (car (last form)) type))
	    ((let let*)
	     (sft (car (last form)) type)
	     (mapc (lambda (x y) (sft y (var-type x)))
		   (caddr form) (cadddr form)))
	    (lit (mapc (lambda (x) (do-setq-tp x nil (type-and nt (var-type x))))
		       (local-aliases (get-top-var-binding (lit-bind form)) nil)))
	    (var (do-setq-tp (caaddr form) nil (type-and nt (var-type (caaddr form)))))
	    (progn (sft (car (last (third form))) type))))))))
	  ;; (if
	  ;;     (when (ignorable-form (third form));FIXME put third form into progn
	  ;;       (let ((tt (type-and type (nil-to-t (info-type (cadr (fourth form))))))
	  ;; 	    (ft (type-and type (nil-to-t (info-type (cadr (fifth form)))))))
	  ;; 	(unless tt
	  ;; 	  (sft (fifth form) type)
	  ;; 	  (setf (car form) 'progn (cadr form) (cadr (fifth form)) (caddr form)
	  ;; 		(list (fifth form)) (cdddr form) nil))
	  ;; 	(unless ft
	  ;; 	  (sft (fourth form) type)
	  ;; 	  (setf (car form) 'progn (cadr form) (cadr (fourth form)) (caddr form)
	  ;; 		(list (fourth form)) (cdddr form) nil)))))

(defun c1setq1 (name form &aux (info (make-info)) type form1 name1)
  (cmpck (not (symbolp name)) "The variable ~s is not a symbol." name)
  (cmpck (constantp name) "The constant ~s is being assigned a value." name)
  (setq name1 (c1vref name t))
  (when (member (var-kind (car name1)) '(special global));FIXME
    (setf (info-flags info) (logior (iflags side-effects) (info-flags info))))
;  (push-changed (car name1) info)
  (add-vref name1 info t)
  (setq form1 (c1arg form info))

  (when (and (eq (car form1) 'var)
	     (or (eq (car name1) (caaddr form1))
		 (bind-match form1 (car name1))))
    (return-from c1setq1 form1))

  (unless (and (eq (car form1) 'var) (eq (car name1) (caaddr form1)))
    (push-changed (car name1) info))

  (when (eq (car form1) 'var)
    (unless (eq (caaddr form1) (car name1))
      (pushnew (caaddr form1) (var-aliases (car name1)))))

  (let* ((v (car name1))(st (var-bind v)))
    (cond ((and (eq (var-kind v) 'lexical) (or (cadr name1) (caddr name1)))
	   (setq type (info-type (cadr form1)))
	   (push (cons (car name1) form1) (info-ch-ccb info)))
	  (t 
	   (do-setq-tp v (list form form1) (info-type (cadr form1)))
	   (setq type (var-type (car name1)))
	   (push-vbind v form1)
	   (keyed-cmpnote (list (var-name v) 'var-bind) "~s store set from ~s to ~s" v st (var-bind v)))))

  (unless (eq type (info-type (cadr form1)))
    (let ((info1 (copy-info (cadr form1))))
         (setf (info-type info1) type)
         (setq form1 (list* (car form1) info1 (cddr form1)))))

  (setf (info-type info) type)
  (maybe-reverse-type-prop type form1)
  (let ((c1fv (when (cadr name1) (c1inner-fun-var))))
    (when c1fv (add-info info (cadr c1fv)))
    (list 'setq info name1 form1 c1fv)))


(defun untrimmed-var-p (v)
  (or (eq t (var-ref v)) (consp (var-ref v)) (var-cb v) (member (var-kind v) '(special global))))


(defun c2setq (vref form c1fv &aux (v (car vref)))
  (declare (ignore c1fv))
  (cond ((untrimmed-var-p v)
	 (let ((*value-to-go* (push 'var vref)))
	   (cond
	    ((member (var-kind v) '(special global));FIXME
	     (let ((loc `(cvar ,(cs-push (var-type v)))))
	       (let ((*value-to-go* loc)) (c2expr* form))
	       (set-loc loc)))
	    ((c2expr* form))))
	 (case (car form)
	       (LOCATION (c2location (caddr form)))
	       (otherwise (unwind-exit vref))))
	((c2expr form))))

(defun c1progv (args &aux (info (make-info)))
  (when (or (endp args) (endp (cdr args)))
        (too-few-args 'progv 2 (length args)))
  (list 'progv info (c1arg (pop args) info) (c1arg (pop args) info) (c1progn* args info)))

(defun c2progv (symbols values body
			&aux (cvar (cs-push t t))
			(*unwind-exit* *unwind-exit*))
  
  (wt-nl "{object " *volatile* "symbols,values;")
  (wt-nl "bds_ptr " *volatile* "V" cvar "=bds_top;")
  (wt-nl "V" cvar "=V" cvar ";");FIXME lintian unused var
  (push cvar *unwind-exit*)
  
  (let ((*vs* *vs*))
    (let ((*value-to-go* (list 'vs (vs-push))))
      (c2expr* symbols)
      (wt-nl "symbols= " *value-to-go* ";"))
    
    (let ((*value-to-go* (list 'vs (vs-push))))
      (c2expr* values)
      (wt-nl "values= " *value-to-go* ";"))
    
    (wt-nl "while(!endp(symbols)){")
    (when *safe-compile*
      (wt-nl "if(type_of(symbols->c.c_car)!=t_symbol)")
      (wt-nl
       "not_a_symbol(symbols->c.c_car);"))
    (wt-nl "if(endp(values))bds_bind(symbols->c.c_car,OBJNULL);")
    (wt-nl "else{bds_bind(symbols->c.c_car,values->c.c_car);")
    (wt-nl "values=values->c.c_cdr;}")
    (wt-nl "symbols=symbols->c.c_cdr;}")
    (setq *bds-used* t))
  (c2expr body)
  (wt "}"))

(defun wt-var-decl (var)
  (cond ((var-p var)
	 (let ((n (var-loc var)))
	   (wt *volatile* (register var) (rep-type (var-kind var)) "V" n )
	   (wt ";")))
        (t (wfs-error))))
