;;; CMPTAG  Tagbody and Go.
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


(si:putprop 'tagbody 'c1tagbody 'c1special)
(si:putprop 'tagbody 'c2tagbody 'c2)

(si:putprop 'go 'c1go 'c1special)
(si:putprop 'go 'c2go 'c2)

(defstruct (tag (:print-function (lambda (x s i) (s-print 'tag (tag-name x) (si::address x) s))))
           name			;;; Tag name.
           ref			;;; Referenced or not.  T or NIL.
           ref-clb		;;; Cross local function reference.
           			;;; During Pass1, T or NIL.
           			;;; During Pass2, the vs-address for the
           			;;; tagbody id, or NIL.
           ref-ccb		;;; Cross closure reference.
           			;;; During Pass1, T or NIL.
           			;;; During Pass2, the vs-address for the
           			;;; block id, or NIL.
           label		;;; Where to jump.  A label.
           unwind-exit		;;; Where to unwind-no-exit.
           var			;;; The tag-name holder.  A VV index.
	   switch               ;;; tag for switch.  A fixnum or 'default
           )

(si::freeze-defstruct 'tag)

(defvar *tags* nil)

;;; During Pass 1, *tags* holds a list of tag objects and the symbols 'CB'
;;; (Closure Boundary) and 'LB' (Level Boundary).  'CB' will be pushed on
;;; *tags* when the compiler begins to process a closure.  'LB' will be pushed
;;; on *tags* when *level* is incremented.

(defvar *reg-amount* 60)
;;amount to increase var-register for each variable reference in side a loop

(defun add-reg1 (form)
  (unless (tag-p form)
    (mapc (lambda (x)
	    (when (var-p x)
	      (incf (var-register x) (the fixnum *reg-amount*))))
	  (info-ref (cadr form)))))

(defun intersection-p (l1 l2)
  (member-if (lambda (x) (member x l2)) l1))
(setf (get 'intersection-p 'cmp-inline) t)

(defun add-loop-registers (tagbody &aux (first (member-if 'tag-p tagbody))
				   (tags (cons (pop first) (remove-if-not 'tag-p first)))
				   (end first))

  (mapl (lambda (x) (unless (tag-p (car x))
		      (when (intersection-p tags (info-ref (cadar x)))
			(setf end (cdr x))))) first)
  (do ((form first (cdr form)))
      ((eq form end))
      (add-reg1 (car form))))

(defun ref-tags (form tags)
  (ref-obs form tags 
	   (lambda (x) (setf (tag-ref-ccb x) t))
	   (lambda (x) (setf (tag-ref-clb x) t))
	   (lambda (x) (setf (tag-ref x) t))))


;FIXME separate pass with no repetitive allocation
(defvar *ft* nil)
(defvar *bt* nil)

(defun tst (y x &aux (z (eq (car x) y))) 
  (unless z
    (keyed-cmpnote (list 'tagbody-iteration) "Iterating tagbody at ~s ~x on ~s conflicts" (tag-name y) (address y) (length x))
    (mapc (lambda (x &aux (v (pop x))) 
	    (keyed-cmpnote (list (var-name v) 'tagbody-iteration)
			   "    Iterating tagbody: setting ~s type ~s to ~s, store ~s to ~s"
			   v (cmp-unnorm-tp (var-type v)) (cmp-unnorm-tp (car x)) (var-store v) (cadr x))
	    (setf (var-type v) (car x));FIXME do-setq-tp ?
	    (push-vbinds v (cadr x)))
	  x))
  (when z x));FIXME return type

(defun pt (y x) (or (tst y (with-restore-vars (catch y (prog1 (cons y (pr x)) (keep-vars))))) (pt y x)))

;; (defun pt (y x &optional (ws *warning-note-stack*))
;;   (or (tst y (with-restore-vars (catch y (prog1 (cons y (pr x)) (keep-vars))))) (pt y x (setq *warning-note-stack* ws))))

(defun lvars (&aux (v (member-if-not 'var-p *vars*)))
  (if v (ldiff *vars* v) *vars*))

(defun mch nil
  (mapcan (lambda (x) (when (var-p x) `((,x ,(var-type x) ,(var-store x) ,(mcpt (var-type x)))))) *vars*))

(defun or-mch (l &optional b)
  (mapc (lambda (x &aux (y x)(v (pop x))(tp (pop x))(st (pop x))(m (car x))
		     (m (unless (equal tp m) m))
		     (t1 (type-or1 (var-type v) (or m tp)));FIXME check expensive
		     (t1 (if b (type-and (var-dt v) (bbump-tp t1)) t1)))
	  (setf (cadr y) t1
		(caddr y) (or-binds (var-store v) st);FIXME union?
		(cadddr y) (mcpt t1)))
	l))

(defun mch-set (z l)
  (mapc (lambda (x)
	  (keyed-cmpnote (list (var-name (car x)) 'tagbody-label)
			 "Initializing ~s at label ~s:~%   type from ~s to ~s,~%   store from ~s to ~s"
			 (car x) (tag-name z) (var-type (car x)) (cadr x)
			 (var-store (car x)) (if (eq (var-store (car x)) (caddr x)) (caddr x) +opaque+))
	  (do-setq-tp (car x) 'mch-set (cadr x));FIXME too prolix
	  (push-vbinds (car x) (caddr x)))
	l))

(defun mch-z (z i &aux (f (cdr (assoc z *ft*))))
  (declare (ignore i));FIXME
  (if f (mch-set z (or-mch f)) (mch)));FIXME ccb-ch (if i (or-mch f) f)
;; The right way to do this is to throw ccb assignments via tag-throw on go into something like *ft*

(defun pr (x &aux (y (member-if 'tag-p x))(z (mapcar 'c1arg (ldiff x y)))(i (when z (info-type (cadar (last z))))))
  (nconc z
	 (when y
	   (let* ((z (pop y))
		  (*bt* (cons (cons z (mch-z z i)) *bt*)))
	     (pt z y)))))

(defconstant +ttl-tag-name+ (gensym "TTL"))

(defun make-ttl-tag nil (make-tag :name +ttl-tag-name+))
(defun is-ttl-tag (tag) (when (tag-p tag) (eq (tag-name tag) +ttl-tag-name+)))

(defvar *ttl-tags* nil)

(defun nttl-tags (body &aux (x (car body)))
  (if (is-ttl-tag x)
      (cons (list x *vars*) *ttl-tags*)
    *ttl-tags*))

(defun c1tagbody (body &aux (info (make-info :type #tnull)))

  (let* ((body (mapcar (lambda (x) (if (or (symbolp x) (integerp x)) (make-tag :name x) x)) body))
	 (tags (remove-if-not 'tag-p body))
	 (body (let* ((*tags* (append tags *tags*))
		      (*ft* (nconc (mapcar 'list tags) *ft*));FIXME
		      (*ttl-tags* (nttl-tags body)))
		  (pr body)))
	 (body (mapc (lambda (x) (unless (tag-p x) (ref-tags x tags))) body))
	 (ref-clb (remove-if-not 'tag-ref-clb tags))
	 (ref-ccb (remove-if-not 'tag-ref-ccb tags))
	 (tagsc (union ref-clb ref-ccb))
	 (tags (union (remove-if-not 'tag-ref tags) tagsc))
	 (body (remove-if-not (lambda (x) (if (tag-p x) (member x tags) t)) body)))
	 
    (mapc (lambda (x) (setf (tag-var x) (tag-name x))) tagsc)
    (if tagsc (set-volatile info) (add-loop-registers body))
    (when ref-ccb (mapc (lambda (x) (setf (tag-ref-ccb x) t)) ref-clb));FIXME?
    (mapc (lambda (x) (unless (tag-p x) (add-info info (cadr x)))) body)
    (let ((x (car (last body))))
      (unless (or (not x) (tag-p x) (info-type (cadr x)))
	(setf (info-type info) nil)))

    (if tags
	`(tagbody ,info ,(when ref-clb t) ,(when ref-ccb t) ,body)
      (let* ((v (car (last body)))
	     (v (if (when v (not (info-type (cadr v)))) body (nconc body (list (c1nil))))))
	(if (cdr v) `(progn ,info ,v) (car v))))))


(defun c2tagbody (ref-clb ref-ccb body)
  (cond (ref-ccb (c2tagbody-ccb body))
	(ref-clb (c2tagbody-clb body))
	((c2tagbody-local body))))

(defun c2tagbody-local (body &aux (label (next-label)))
  (dolist (x body)
    (when (typep x 'tag)
          (setf (tag-label x) (next-label*))
          (setf (tag-unwind-exit x) label)))
  (let ((*unwind-exit* (cons label *unwind-exit*)))
    (c2tagbody-body body)))

(defun c2tagbody-body (body)
  (do ((l body (cdr l)) (written nil))
      ((endp (cdr l))
       (cond (written (unwind-exit nil))
             ((typep (car l) 'tag)
	      (wt-switch-case (tag-switch (car l)))
              (wt-label (tag-label (car l)))
              (unwind-exit nil))
             (t (let* ((*exit* (next-label))
                       (*unwind-exit* (cons *exit* *unwind-exit*))
                       (*value-to-go* 'trash))
                  (c2expr (car l))
                  (wt-label *exit*))
		;gcc lintian, lacking noreturn attributes prevents
		;(unless (type>= #tnil (info-type (cadar l))) (unwind-exit nil))
		(unwind-exit nil))))
    (cond (written (setq written nil))
          ((typep (car l) 'tag)
	   (wt-switch-case (tag-switch (car l)))
	   (wt-label (tag-label (car l))))
          (t (let* ((*exit* (if (typep (cadr l) 'tag)
                                (progn (setq written t) (tag-label (cadr l)))
                                (next-label)))
                    (*unwind-exit* (cons *exit* *unwind-exit*))
                    (*value-to-go* 'trash))
               (c2expr (car l))
	       (and (typep (cadr l) 'tag)
		    (wt-switch-case (tag-switch (cadr l))))
               (wt-label *exit*))))))
  
(defun c2tagbody-clb (body &aux (label (next-label)) (*vs* *vs*))
  (let ((*unwind-exit* (cons 'frame *unwind-exit*))
        (ref-clb (vs-push)))
    (wt-nl) (wt-vs ref-clb) (wt "=alloc_frame_id();")
    (add-libc "setjmp")
    (setq *frame-used* t)
    (wt-nl "frs_push(FRS_CATCH,") (wt-vs ref-clb) (wt ");")
    (wt-nl "if(nlj_active){")
    (wt-nl "nlj_active=FALSE;")
    ;;; Allocate labels.
    (dolist (tag body)
      (when (typep tag 'tag)
        (setf (tag-label tag) (next-label*))
        (setf (tag-unwind-exit tag) label)
        (when (tag-ref-clb tag)
          (setf (tag-ref-clb tag) ref-clb)
          (wt-nl "if(eql(nlj_tag," (vv-str (tag-var tag)) ")) {")
  	  (wt-nl "   ")
	  (reset-top)
	  (wt-nl "   ")
          (wt-go (tag-label tag))
	  (wt-nl "}"))))
    (wt-nl "FEerror(\"The GO tag ~s is not established.\",1,nlj_tag);")
    (wt-nl "}")
    (let ((*unwind-exit* (cons label *unwind-exit*)))
      (c2tagbody-body body))))

(defun c2tagbody-ccb (body &aux (label (next-label))
                           (*vs* *vs*) (*clink* *clink*) (*ccb-vs* *ccb-vs*))
  (let ((*unwind-exit* (cons 'frame *unwind-exit*))
        (ref-clb (vs-push)) ref-ccb)
    (wt-nl) (wt-vs ref-clb) (wt "=alloc_frame_id();")
    (wt-nl) 
    (clink ref-clb)
    (setq ref-ccb (ccb-vs-push))
    (add-libc "setjmp")
    (setq *frame-used* t)
    (wt-nl "frs_push(FRS_CATCH,") (wt-vs* ref-clb) (wt ");")
    (wt-nl "if(nlj_active){")
    (wt-nl "nlj_active=FALSE;")
    ;;; Allocate labels.
    (dolist (tag body)
      (when (typep tag 'tag)
        (setf (tag-label tag) (next-label*))
        (setf (tag-unwind-exit tag) label)
        (when (or (tag-ref-clb tag) (tag-ref-ccb tag))
          (setf (tag-ref-clb tag) ref-clb)
          (when (tag-ref-ccb tag) (setf (tag-ref-ccb tag) ref-ccb))
          (wt-nl "if(eql(nlj_tag," (vv-str (tag-var tag)) ")) {")
	  (wt-nl "   ")
	  (reset-top)
	  (wt-nl "   ")
          (wt-go (tag-label tag))
	  (wt-nl "}"))))
    (wt-nl "FEerror(\"The GO tag ~s is not established.\",1,nlj_tag);")
    (wt-nl "}")
    (let ((*unwind-exit* (cons label *unwind-exit*)))
      (c2tagbody-body body))))

(defun mcpt (tp &aux (a (car (atomic-tp tp))))
  (when (consp a)
    (subst (copy-list a) a tp)));rplacd, etc.

(defun tag-throw (tag &aux (b (assoc tag *bt*)))
  (if b
      (let ((v (prune-mch (cdr b) t)))
	(when v (throw tag (or-mch v t))))
    (let ((f (assoc tag *ft*)))
      (or (or-mch (cdr f)) (setf (cdr f) (mch))))))

(defun c1go (args &aux (name (car args)) ccb clb inner)
  (cond ((endp args) (too-few-args 'go 1 0))
        ((not (endp (cdr args))) (too-many-args 'go 1 (length args)))
        ((not (or (symbolp name) (integerp name)))
         "The tag name ~s is not a symbol nor an integer." name))
  (dolist (tag *tags* (cmperr "The tag ~s is undefined." name))
    (case tag
	  (cb (setq ccb t inner (or inner 'cb)))
	  (lb (setq clb t inner (or inner 'lb)))
	  (t (when (when (eq (tag-name tag) name) (not (member tag *lexical-env-mask*)))
	       (tag-throw tag)
	       (let* ((ltag (list tag))
		      (info (make-info :type nil))
		      (c1fv (when ccb (c1inner-fun-var))))
		 (cond (ccb (setf (info-ref-ccb info) ltag))
		       (clb (setf (info-ref-clb info) ltag))
		       ((setf (info-ref info) ltag)))
		 (when c1fv (add-info info (cadr c1fv)))
		 (return (list 'go info tag ccb clb c1fv))))))))

(defun c2go (tag ccb clb c1fv)
  (declare (ignore c1fv))
  (cond (ccb (c2go-ccb tag))
        (clb (c2go-clb tag))
        (t (c2go-local tag))))

(defun c2go-local (tag)
  (unwind-no-exit (tag-unwind-exit tag))
  (wt-nl) (wt-go (tag-label tag)))

(defun c2go-clb (tag)
  (wt-nl "vs_base=vs_top;")
  (wt-nl "unwind(frs_sch(")
  (if (tag-ref-ccb tag)
    (wt-vs* (tag-ref-clb tag))
  (wt-vs (tag-ref-clb tag)))
  (wt ")," (vv-str (tag-var tag)) ");")
  (unwind-exit nil))

(defun c2go-ccb (tag)
  (wt-nl "{frame_ptr fr;")
  (wt-nl "fr=frs_sch(") (wt-ccb-vs (tag-ref-ccb tag)) (wt ");")
  (wt-nl "if(fr==NULL)FEerror(\"The GO tag ~s is missing.\",1," (vv-str (tag-var tag)) ");")
  (wt-nl "vs_base=vs_top;")
  (wt-nl "unwind(fr," (vv-str (tag-var tag)) ");}")
  (unwind-exit nil))


(defun wt-switch-case (x)
  (cond (x (wt-nl (if (typep x 'fixnum) "case " "") x ":"))))

(defun or-branches (trv)
  (mapc (lambda (x &aux (v (pop x))(tp (pop x))(st (car x)))
	  (cond ((var-p v)
		 (unless (subsetp st (var-store v))
		   (keyed-cmpnote
		    (list (var-name v) 'var-store 'binding '+opaque+)
		    "~s store set to +opaque+ from ~s/~s across if branches"
		    (var-name v) st (var-store v))
		   (push-vbinds v st))
		 (do-setq-tp v (list 'or-branches nil) (type-or1 (var-type v) tp)))
		(t
		 (keyed-cmpnote (list 'type-mod-unwind) "Unwinding type ~s ~s" v tp)
		 (repl-tp v tp t))))
	trv))

(defun c1switch (body)
  (flet ((tgs-p (x) (or (symbolp x) (integerp x)))) 
	(let* ((switch-op (pop body))
	       (info (make-info :type #tnil))
	       (switch-op-1 (c1arg switch-op info))
	       (st (coerce-to-one-value (info-type (cadr switch-op-1))))
	       (st (if (type>= #tfixnum st) st (baboon)))
	       tags
	       (body (remove-if (lambda (x) (when (tgs-p x) (prog1 (member x tags) (push x tags)))) body))
	       skip cs dfp rt
	       (body (remove-if-not (lambda (b)
				      (cond ((tgs-p b)
					     (unless cs (setq cs t skip t rt nil))
					     (let* ((e (object-type b))(df (member b '(t otherwise)))(e (if df st e)))
					       (cond ((and df dfp) (cmperr "default tag must be last~%"))
						     ((type-and st e) (setq skip nil dfp (or df dfp) rt (type-or1 rt e)))
						     ((keyed-cmpnote 'branch-elimination "Eliminating unreachable switch ~s" b)))))
					    ((not skip)
					     (when cs
					       (setq st (type-and st (tp-not rt)) cs nil))
					     t)
					    (t (eliminate-src b) nil)))
				    body))
	       (body (mapcar (lambda (x) (if (tgs-p x) (make-tag :name x :ref t :switch (if (typep x 'fixnum) x "default")) x)) body))
	       trv
	       (body (mapcar (lambda (x) (if (tag-p x) x (let ((x (c1branch t nil (list nil x) info))) 
							   (prog1 (pop x) (setq trv (append trv (car x))))))) body))
	       (ls (member-if 'consp body)))
	  (or-branches trv)
	  (when st (baboon))
	  (mapc (lambda (x) (assert (or (tag-p x) (not (info-type (cadr x)))))) body)
	  (if (unless (cdr ls) (ignorable-form switch-op-1))
	      (car ls)
	    (list 'switch info switch-op-1 body)))))


(defun c2switch (op body &aux (*inline-blocks* 0)(*vs* *vs*))
  (let ((args (inline-args (list op) `(,#tfixnum))))
    (wt-nl "")
    (wt-inline-loc "switch(#0){" args)
    (c2tagbody-local body)
    (wt "}")
    (unwind-exit nil)
    (close-inline-blocks)))


;; SWITCH construct for Common Lisp. (TEST &body BODY) (in package SI)

;; TEST must evaluate to something of INTEGER TYPE.  If test matches one
;; of the labels (ie integers) in the body of switch, control will jump
;; to that point.  It is an error to have two or more constants which are
;; eql in the the same switch.  If none of the constants match the value,
;; then control moves to a label T.  If there is no label T, control
;; flows as if the last term in the switch were a T.  It is an error
;; however if TEST were declared to be in a given integer range, and at
;; runtime a value outside that range were provided.  The value of a
;; switch construct is undefined.  If you wish to return a value use a
;; block construct outside the switch and a return-from.  `GO' may also
;; be used to jump to labels in the SWITCH.

;; Control falls through from case to case, just as if the cases were
;; labels in a tagbody.  To jump to the end of the switch, use
;; (switch-finish).

;; The reason for using a new construct rather than building on CASE, is
;; that CASE does not allow the user to use invoke a `GO' if necessary.
;; to switch from one case to another.  Also CASE does not allow sharing
;; of parts of code between different cases.  They have to be either the
;; same or disjoint.

;; The SWITCH may be implemented very efficiently using a jump table, if
;; the range of cases is not too much larger than the number of cases.
;; If the range is much larger than the number of cases, a binary
;; splitting of cases might be used.

;; Sample usage:
;; (defun goo (x)
;;  (switch x
;;    1 (princ "x is one, ")
;;    2 (princ "x is one or two, ")
;;    (switch-finish)
;;    3 (princ "x is three, ")
;;    (switch-finish)	 
;;    t (princ "none")))

;; We provide a Common Lisp macro for implementing the above construct:


(defmacro switch (test &body body &aux cases)
  (dolist  (v body)
    (cond ((integerp v) (push `(if (eql ,v ,test) (go ,v) nil) cases))))
  `(tagbody
     ,@(nreverse cases)
     (go t)
     ,@ body
     ,@ (if (member t body) nil '(t))
     switch-finish-label))

(defmacro switch-finish nil '(go switch-finish-label))

  
(si::putprop 'switch 'c1switch 'c1special)
(si::putprop 'switch 'c2switch 'c2)
