;; Copyright (C) 2024 Camm Maguire

;; Modified data base including return values types
;; and making the arglists correct if they have optional args.
;;

(in-package :compiler)

(dolist (l '((((stream) string) . get-output-stream-string)
	     (((simple-vector seqind) t) . svref)
	     (((t *) string) . print)
	     (((t *) string) . prin1)
	     (((t *) string) . princ)
	     (((si::function-identifier) boolean) . fboundp)
	     (((structure) structure) . si::structure-def)
	     (((t t t t t t t) pathname) . si::init-pathname)
	     (((t t *) (or (integer -1 -1 ) seqind)) . si::string-match)
;	     (((t) t) . si::type-of-c)
;	     (((list) t) . si::cons-car)
;	     (((list) t) . si::cons-cdr)
	     (((t t) cons) . cons)
	     (((*) si::proper-list) . list)
	     (((t *) list) . list*)
	     (((fixnum) t) . si::nani)
	     (((t) fixnum) . si::address);FIXME
;	     (((integer) fixnum) . si::mpz_bitlength)
;	     (((integer fixnum) integer) . si::shft)
	     (((number number) number) . si::number-plus)
	     (((number number) number) . si::number-minus)
	     (((number number) number) . si::number-times)
	     (((number number) number) . si::number-divide)
	     (((real *) (returns-exactly real real)) . floor)
	     (((real *) (returns-exactly real real)) . ceiling)
	     (((real *) (returns-exactly real real)) . truncate)
	     (((real *) (returns-exactly real real)) . round)
;	     (((cons t) cons) . rplaca)
;	     (((cons t) cons) . rplacd)
;	     (((symbol) boolean) . boundp)
;	     (((symbol) (or null package)) . symbol-package)
;	     (((symbol) string) . symbol-name)
;	     (((symbol) t) . symbol-value)
	     (((symbol t t) t) . si::sputprop)
;	     (((symbol) (or cons function)) . symbol-function);fixme
	     ;; (((array rnkind)  seqind) . array-dimension)
	     ;; (((array)  seqind) . array-total-size)
	     ;; (((array)  symbol) . array-element-type)
	     ;; (((array)  rnkind) . array-rank)
;	     (((vector) seqind) . si::fill-pointer-internal)
	     (((string) symbol) . make-symbol)
;	     (((integer integer) integer) . ash)
	     (((float) (returns-exactly (integer 0) fixnum (member 1 -1))) . integer-decode-float);fixme
;	     (((t *) nil) . error);fixme
	     (((*) string) . si::string-concatenate)))
  (let ((x (si::call (cdr l) t)))
    (cond (x (setf (car x) (list (mapcar 'cmp-norm-tp (caar l)) (cmp-norm-tp (cadar l))))
	     (si::normalize-function-plist x))
	  ((print (cdr l))))))

(dolist (l '(ceiling truncate round floor));FIXME
  (c-set-function-vv (symbol-function l) 0)
  (c-set-function-neval (symbol-function l) 1)
  )

(dolist (l '(eq eql equal equalp ldb-test logtest))
  (setf (get l 'predicate) t))

(dolist (l '(ldb-test logtest))
  (setf (get l 'predicate) t))


(declaim (notinline compile compile-file load open truename translate-pathname translate-logical-pathname probe-file))

