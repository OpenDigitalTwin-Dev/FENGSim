;-*- Mode:     Lisp -*-
;;;; Author:   Paul Dietz
;;;; Created:  Sat Mar 28 07:36:01 1998
;;;; Contains: Testing of CL Features related to "CONS", part 8

(in-package :cl-test)

(declaim (optimize (safety 3)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Error checking car, cdr, list-length

(deftest car.1
  (car '(a))
  a)

(deftest car-nil
  (car nil)
  nil)

(deftest car-symbol-error
  (classify-error (car 'a))
  type-error)

(deftest car-symbol-error.2
  (classify-error (locally (car 'a) t))
  type-error)

(deftest car.order.1
  (let ((i 0))
    (values (car (progn (incf i) '(a b))) i))
  a 1)

(deftest cdr.1
  (cdr '(a b))
  (b))

(deftest cdr-nil
  (cdr ())
  nil)

(deftest cdr.order.1
  (let ((i 0))
    (values (cdr (progn (incf i) '(a b))) i))
  (b) 1)

(deftest cdr-symbol-error
  (classify-error (cdr 'a))
  type-error)

(deftest cdr-symbol-error.2
  (classify-error (locally (cdr 'a) t))
  type-error)

(deftest list-length.4
  (list-length (copy-tree '(a b c)))
  3)

(deftest list-length-symbol
  (classify-error (list-length 'a))
  type-error)

(deftest list-length-dotted-list
  (classify-error (list-length (copy-tree '(a b c d . e))))
  type-error)

;;; Error checking of c*r functions

(deftest caar.error.1
  (classify-error (caar 'a))
  type-error)

(deftest caar.error.2
  (classify-error (caar '(a)))
  type-error)

(deftest cadr.error.1
  (classify-error (cadr 'a))
  type-error)

(deftest cadr.error.2
  (classify-error (cadr '(a . b)))
  type-error)

(deftest cdar.error.1
  (classify-error (cdar 'a))
  type-error)

(deftest cdar.error.2
  (classify-error (cdar '(a . b)))
  type-error)

(deftest cddr.error.1
  (classify-error (cddr 'a))
  type-error)

(deftest cddr.error.2
  (classify-error (cddr '(a . b)))
  type-error)

(deftest caaar.error.1
  (classify-error (caaar 'a))
  type-error)

(deftest caaar.error.2
  (classify-error (caaar '(a)))
  type-error)

(deftest caaar.error.3
  (classify-error (caaar '((a))))
  type-error)

(deftest caadr.error.1
  (classify-error (caadr 'a))
  type-error)

(deftest caadr.error.2
  (classify-error (caadr '(a . b)))
  type-error)

(deftest caadr.error.3
  (classify-error (caadr '(a . (b))))
  type-error)

(deftest cadar.error.1
  (classify-error (cadar 'a))
  type-error)

(deftest cadar.error.2
  (classify-error (cadar '(a . b)))
  type-error)

(deftest cadar.error.3
  (classify-error (cadar '((a . c) . b)))
  type-error)

(deftest caddr.error.1
  (classify-error (caddr 'a))
  type-error)

(deftest caddr.error.2
  (classify-error (caddr '(a . b)))
  type-error)

(deftest caddr.error.3
  (classify-error (caddr '(a c . b)))
  type-error)

(deftest cdaar.error.1
  (classify-error (cdaar 'a))
  type-error)

(deftest cdaar.error.2
  (classify-error (cdaar '(a)))
  type-error)

(deftest cdaar.error.3
  (classify-error (cdaar '((a . b))))
  type-error)

(deftest cdadr.error.1
  (classify-error (cdadr 'a))
  type-error)

(deftest cdadr.error.2
  (classify-error (cdadr '(a . b)))
  type-error)

(deftest cdadr.error.3
  (classify-error (cdadr '(a b . c)))
  type-error)

(deftest cddar.error.1
  (classify-error (cddar 'a))
  type-error)

(deftest cddar.error.2
  (classify-error (cddar '(a . b)))
  type-error)

(deftest cddar.error.3
  (classify-error (cddar '((a . b) . b)))
  type-error)

(deftest cdddr.error.1
  (classify-error (cdddr 'a))
  type-error)

(deftest cdddr.error.2
  (classify-error (cdddr '(a . b)))
  type-error)

(deftest cdddr.error.3
  (classify-error (cdddr '(a c . b)))
  type-error)

;;

(deftest caaaar.error.1
  (classify-error (caaaar 'a))
  type-error)

(deftest caaaar.error.2
  (classify-error (caaaar '(a)))
  type-error)

(deftest caaaar.error.3
  (classify-error (caaaar '((a))))
  type-error)

(deftest caaaar.error.4
  (classify-error (caaaar '(((a)))))
  type-error)

(deftest caaadr.error.1
  (classify-error (caaadr 'a))
  type-error)

(deftest caaadr.error.2
  (classify-error (caaadr '(a . b)))
  type-error)

(deftest caaadr.error.3
  (classify-error (caaadr '(a . (b))))
  type-error)

(deftest caaadr.error.4
  (classify-error (caaadr '(a . ((b)))))
  type-error)

(deftest caadar.error.1
  (classify-error (caadar 'a))
  type-error)

(deftest caadar.error.2
  (classify-error (caadar '(a . b)))
  type-error)

(deftest caadar.error.3
  (classify-error (caadar '((a . c) . b)))
  type-error)

(deftest caadar.error.4
  (classify-error (caadar '((a . (c)) . b)))
  type-error)

(deftest caaddr.error.1
  (classify-error (caaddr 'a))
  type-error)

(deftest caaddr.error.2
  (classify-error (caaddr '(a . b)))
  type-error)

(deftest caaddr.error.3
  (classify-error (caaddr '(a c . b)))
  type-error)

(deftest caaddr.error.4
  (classify-error (caaddr '(a c . (b))))
  type-error)

(deftest cadaar.error.1
  (classify-error (cadaar 'a))
  type-error)

(deftest cadaar.error.2
  (classify-error (cadaar '(a)))
  type-error)

(deftest cadaar.error.3
  (classify-error (cadaar '((a . b))))
  type-error)

(deftest cadaar.error.4
  (classify-error (cadaar '((a . (b)))))
  type-error)

(deftest cadadr.error.1
  (classify-error (cadadr 'a))
  type-error)

(deftest cadadr.error.2
  (classify-error (cadadr '(a . b)))
  type-error)

(deftest cadadr.error.3
  (classify-error (cadadr '(a b . c)))
  type-error)

(deftest cadadr.error.4
  (classify-error (cadadr '(a (b . e) . c)))
  type-error)

(deftest caddar.error.1
  (classify-error (caddar 'a))
  type-error)

(deftest caddar.error.2
  (classify-error (caddar '(a . b)))
  type-error)

(deftest caddar.error.3
  (classify-error (caddar '((a . b) . b)))
  type-error)

(deftest caddar.error.4
  (classify-error (caddar '((a  b . c) . b)))
  type-error)

(deftest cadddr.error.1
  (classify-error (cadddr 'a))
  type-error)

(deftest cadddr.error.2
  (classify-error (cadddr '(a . b)))
  type-error)

(deftest cadddr.error.3
  (classify-error (cadddr '(a c . b)))
  type-error)

(deftest cadddr.error.4
  (classify-error (cadddr '(a c e . b)))
  type-error)

(deftest cdaaar.error.1
  (classify-error (cdaaar 'a))
  type-error)

(deftest cdaaar.error.2
  (classify-error (cdaaar '(a)))
  type-error)

(deftest cdaaar.error.3
  (classify-error (cdaaar '((a))))
  type-error)

(deftest cdaaar.error.4
  (classify-error (cdaaar '(((a . b)))))
  type-error)

(deftest cdaadr.error.1
  (classify-error (cdaadr 'a))
  type-error)

(deftest cdaadr.error.2
  (classify-error (cdaadr '(a . b)))
  type-error)

(deftest cdaadr.error.3
  (classify-error (cdaadr '(a . (b))))
  type-error)

(deftest cdaadr.error.4
  (classify-error (cdaadr '(a . ((b . c)))))
  type-error)

(deftest cdadar.error.1
  (classify-error (cdadar 'a))
  type-error)

(deftest cdadar.error.2
  (classify-error (cdadar '(a . b)))
  type-error)

(deftest cdadar.error.3
  (classify-error (cdadar '((a . c) . b)))
  type-error)

(deftest cdadar.error.4
  (classify-error (cdadar '((a . (c . d)) . b)))
  type-error)

(deftest cdaddr.error.1
  (classify-error (cdaddr 'a))
  type-error)

(deftest cdaddr.error.2
  (classify-error (cdaddr '(a . b)))
  type-error)

(deftest cdaddr.error.3
  (classify-error (cdaddr '(a c . b)))
  type-error)

(deftest cdaddr.error.4
  (classify-error (cdaddr '(a c b . d)))
  type-error)

(deftest cddaar.error.1
  (classify-error (cddaar 'a))
  type-error)

(deftest cddaar.error.2
  (classify-error (cddaar '(a)))
  type-error)

(deftest cddaar.error.3
  (classify-error (cddaar '((a . b))))
  type-error)

(deftest cddaar.error.4
  (classify-error (cddaar '((a . (b)))))
  type-error)

(deftest cddadr.error.1
  (classify-error (cddadr 'a))
  type-error)

(deftest cddadr.error.2
  (classify-error (cddadr '(a . b)))
  type-error)

(deftest cddadr.error.3
  (classify-error (cddadr '(a b . c)))
  type-error)

(deftest cddadr.error.4
  (classify-error (cddadr '(a (b . e) . c)))
  type-error)

(deftest cdddar.error.1
  (classify-error (cdddar 'a))
  type-error)

(deftest cdddar.error.2
  (classify-error (cdddar '(a . b)))
  type-error)

(deftest cdddar.error.3
  (classify-error (cdddar '((a . b) . b)))
  type-error)

(deftest cdddar.error.4
  (classify-error (cdddar '((a  b . c) . b)))
  type-error)

(deftest cddddr.error.1
  (classify-error (cddddr 'a))
  type-error)

(deftest cddddr.error.2
  (classify-error (cddddr '(a . b)))
  type-error)

(deftest cddddr.error.3
  (classify-error (cddddr '(a c . b)))
  type-error)

(deftest cddddr.error.4
  (classify-error (cddddr '(a c e . b)))
  type-error)

;;; Need to add 'locally' wrapped forms of these
