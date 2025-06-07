(in-package :compiler)
(cdebug)
(setq *compile-print* nil si::*notify-gbc* t *annotate* nil *optimize-maximum-pages* nil)

(multiple-value-bind
 (x ps) (si::heap-report)
  (si::allocate 'structure (max 1 (truncate (* 4096 200) ps)) t))

(room t)


#+pre-gcl
(progn
  (declaim (optimize (safety 3)))
  (unless (fboundp 'logandc2) (defun logandc2 (x y) (boole boole-andc2 x y)))
  (unless (fboundp 'lognot) (defun lognot (x) (boole boole-c1 x 0)))
  (unless (fboundp 'abs) (defun abs (x) (if (< x 0) (- x) x))))

(mapc 'compile (nconc
		#+pre-gcl '(listp si::real-simple-typep-fn si::array-simple-typep-fn)
		#+pre-gcl (progn 'si::(s-data-raw s-data-slot-position s-data-slot-descriptions))
		#-pre-gcl '(sbit si::aset si::improper-consp mapcar mapcan mapc mapl)
					;maplist member member-if member-if-not
					;assoc assoc-if assoc-if-not
					;rassoc rassoc-if rassoc-if-not
		'(info-p info-ref info-type info-flags info-ch info-ref-ccb info-ref-clb)
		'(var-p var-name var-flags var-kind var-ref var-ref-ccb var-loc var-dt
		  var-type var-mt var-tag var-store)
		#-pre-gcl
		'(bit-andc2 bit-and bit-ior bit-xor bit-orc2 bit-not)
		#-pre-gcl
		(progn 'si::(copy-btp btp-equal one-bit-btp btp-count
				      new-tp4 btp-type2 btp-bnds< btp-bnds>
				      tp-and tp-or cmp-tp-not tp-not
				      tp= tp-p))
		#-pre-gcl
		'(naltp explode-nalt needs-explode ctp-and ctp<=
		  type-and type-or1 type<= type>= type=)

		'(c-array-rank c-array-dim c-array-elttype c-array-self c-array-hasfillp c-array-eltsize)
		'(c-structure-def c-structure-self c-strstd-sself)
		'(array-dimension array-row-major-index row-major-aref si::row-major-aset
		  si::row-major-aref-int aref array-rank array-total-size
		  array-has-fill-pointer-p length)
		'(typep infer-tp check-type)))

(in-package :compiler)
#-pre-gcl
(progn
  ;FIXME safety 2
  (dolist (l '(sbit svref schar char));ensure in *inl-hash*
    (compile nil `(lambda (x y) (declare (optimize (safety 1))) (,l x y)))
    (compile nil `(lambda (x y z) (declare (optimize (safety 1))) (setf (,l x y) z))))
  
  (dolist (l si::+array-types+)
    (compile nil `(lambda (x y) (declare (optimize (safety 1))((vector ,l) x)) (aref x y)))
    (compile nil `(lambda (x y z) (declare (optimize (safety 1))((vector ,l) x)(,l z)) (setf (aref x y) z)))
    (compile nil `(lambda (x y z) (declare (optimize (safety 1))((vector ,l) x)) (setf (aref x y) z))))
  
  (compile nil `(lambda (x) (declare (optimize (safety 1))((or simple-vector simple-string simple-bit-vector) x)) (length x)))
  
  (compile nil `(lambda (x) (declare (optimize (safety 2))) (address x)))
  (compile nil `(lambda (x) (declare (optimize (safety 2))) (nani x))))

(setq *optimize-maximum-pages* t)
