;; Copyright (C) 2024 Camm Maguire
(in-package :si)

(let (fso (h (make-hash-table :test 'eq)))
  (defun funcallable-class-p (x)
    (multiple-value-bind
	  (r f) (gethash x h)
      (if f r
	  (let ((y (si-cpl-or-nil x)))
	    (when y
	      (setf (gethash x h)
		    (member (or fso (setq fso (si-find-class (find-symbol "FUNCALLABLE-STANDARD-OBJECT" "PCL") nil)))
			    y))))))))

(defun normalize-instance (c)
  (cond ((funcallable-class-p c) `(funcallable-std-instance ,c))
	((member-if 'funcallable-class-p (si-subclasses c)) `(or (std-instance ,c) (funcallable-std-instance ,c)))
	(`(std-instance ,c))))

(defun just-expand-deftype (type &aux tem
			      (atp (listp type))
			      (ctp (if atp (car type) type)))
  (cond
   ((setq tem (when (symbolp ctp) (macro-function (get ctp 'deftype-definition))))
    (funcall tem (if atp type (list type)) nil))
   ((setq tem (coerce-to-standard-class ctp)) (normalize-instance tem));FIXME don't want to normalize a nil type, redundant code
   ((si-classp ctp) (si-class-name ctp));built-in
   ((setq tem (get ctp 's-data)) (or (sdata-type tem) `(structure ,ctp)))
   (t (warn 'warning :format-control "Expanding unknown type ~s to nil:" :format-arguments (list type)) nil)))

(defun expand-deftype (type &aux (e (just-expand-deftype type)))
  (unless (eq type e)
    e))

(eval-when (compile eval load)

  (defvar *array-types* (cons (cons nil 'array-nil) +array-type-alist+))

  (defvar *simple-array-types*
    (mapcar (lambda (x)
	      (cons x (intern (string-concatenate "SIMPLE-ARRAY-" (string x)))))
	    (cons nil +array-types+)))

  (defvar *non-simple-array-types*
    (mapcan (lambda (x)
	      (when (member x '(character bit t));FIXME
		(list (cons x (intern (string-concatenate "NON-SIMPLE-ARRAY-" (string x)))))))
	    (cons nil +array-types+)))

  (defvar *all-array-types* (append *simple-array-types* *non-simple-array-types*))

  (defconstant +atps+ (mapcar (lambda (x) (list x (intern (string-concatenate "ARRAY-"   (string x))))) +array-types+));FIXME

  (defconstant +k-ops+
    `((integer int^ int~ urng-recon)
      (ratio   rng^ rng~ urng-recon)
      ((short-float long-float) urng^ urng~ urng-recon)
      (complex-integer       cmpi^  cmpi~  cmp-recon)
      (complex-integer-ratio cmpir^ cmpir~ cmp-recon)
      (complex-ratio-integer cmpri^ cmpri~ cmp-recon)
      (complex-ratio         cmpr^  cmpr~  cmp-recon)
      ((complex-short-float complex-long-float) cmp^ cmp~ cmp-recon)
      ((std-instance structure funcallable-std-instance) std^ std~ std-recon)
      ((proper-cons improper-cons) cns^ cns~ cns-recon)
      (,(mapcar 'cdr *all-array-types*) ar^ ar~ ar-recon)
      (,+singleton-types+  sing^ sing~ sing-recon)))

  (defconstant +k-len+ (lreduce (lambda (xx x &aux (x (car x)))
				  (+ (if (listp x) (length x) 1) xx))
				+k-ops+ :initial-value 0)))

(defmacro negate (lst)
  (let ((l (gensym)))
    `(let ((,l ,lst))
       (cond ((not ,l))
	     ((eq ,l t) nil)
	     ((and (consp ,l) (eq (car ,l) 'not)) (cadr ,l))
	     (`(not ,,l))))))


;;; ARRAY


(defun rnki (x)
  (when (listp x)
    (when (eq (car x) 'rank)
      (cdr x))))

(defun rd^ (x y)
  (cond ((eql x y) x);eq
	((atom x) (when (listp y) (unless (member x y) x)));memq
	((atom y) (rd^ y x))
	((union x y))));test

(defun dim^ (x y)
  (cond ((eq x '*) y)
	((eq y '*) x)
	((rd^ x y))))

(defun dims^ (x y)
  (cond ((and x y)
	 (let* ((a (dim^ (car x) (car y)))
		(d (when a (dims^ (cdr x) (cdr y)))))
	   (when d
	     (list (cons a (car d))))))
	((or x y) nil)
	((list nil))))

(defun adims (x)
  (cond ((arrayp x) (array-dimensions x))
	((when (listp x) (arrayp (car x))) (array-dimensions (car x)))
	(x)))

(defun ar^ (x y &aux (rx (rnki x))(ry (rnki y))(ax (adims x))(ay (adims y)))
  (cond ((and rx ry) (let ((d (rd^ rx ry))) (when d (cons 'rank d))))
	(rx (when (rd^ rx (length (adims y))) y))
	(ry (ar^ y x))
	((and (eq x ax) (eq y ay)) (car (dims^ x y)))
	((eq x ax) (when (ar^ x ay) y))
	((eq y ay) (ar^ y x))
	((ar^ ax ay) (rd^ x y))))

(defun disu (x)
  (when x
    (let* ((cx (pop x))(cx (if (atom cx) (list cx) cx)))
      (mapcan (lambda (y) (mapcar (lambda (x) (cons x y)) cx))
	      (if x (disu x) (list x))))))

(defun ar-recon (x &optional (s (rassoc (car x) *simple-array-types*))
		   (tp (car (rassoc (pop x) *all-array-types*)) tpp)
		   &aux (ax (adims x))(ar (if s 'simple-array 'non-simple-array)))
  (cond ((not tpp) (?or (mapcar (lambda (z) (ar-recon z s tp)) x)))
	((eq x t) `(,ar ,tp *));'*
	((consp (rnki x))
	 `(and ,(ar-recon t s tp)
	       (not ,(?or (mapcar (lambda (x) (ar-recon (cons 'rank x) s tp)) (rnki x))))))
	((rnki x) `(,ar ,tp ,(rnki x)))
	((when (eq x ax) (member-if 'listp x))
	 `(and ,(ar-recon (mapcar (lambda (x) (if (atom x) x '*)) x) s tp)
	       (not ,(?or (mapcar (lambda (x) (ar-recon x s tp)) (disu x))))))
	((eq ax x) `(,ar ,tp ,x))
	((consp x) `(and ,(ar-recon ax s tp) (not (member ,@x))))
	(`(member ,x))))

(defun onot (x)
  (when x
      (let ((d (mapcar (lambda (y) (cons (car x) y)) (onot (cdr x)))))
	(if (eq (car x) '*) d
	    (cons (cons (list (car x)) (make-list (length (cdr x)) :initial-element '*)) d)))))

(defun ar~ (x &aux (ax (adims x)))
  (cond ((consp (rnki x)) (mapcar (lambda (x) (cons 'rank x)) (rnki x)))
	((rnki x) `((rank ,(rnki x))))
	((when (eq x ax) (member-if 'listp x))
	 (nconc (ar~ (substitute-if '* 'listp x)) (disu x)))
	((eq x ax) (nconc (ar~ (cons 'rank (length x))) (onot x)))
	((listp x) (nconc (ar~ (array-dimensions (car x))) x));FIXME
	((nconc (ar~ (array-dimensions x)) `((,x))))))

(defun ar-ld (type &aux (s (eq (car type) 'simple-array)))
  `(,(cdr (assoc (cadr type) (if s *simple-array-types* *non-simple-array-types*)))
     ,(let ((x (caddr type)))
	(cond ((eq x '*) t)
	      ((integerp x) (cons 'rank x));FIXME
	      (x)))))

;;; SINGETON



(defun sing^ (x y) (rd^ y x))

(defun sing~ (x)
  (cond ((listp x) x)
	((list (list x)))))

(defun sing-ld (type) (cons (car type) '(t)))

(defun sing-recon (x &aux (c (pop x)))
  (cond ((eq (car x) t) (case c (null `(member nil)) (true `(member t)) (otherwise c)));FIXME
	((listp (car x)) `(and ,c (not (member ,@(car x)))))
	(`(member ,@x))))


;;; INTEGER



(defun intcmp (x y f)
  (cond ((eq x '*) y)((eq y '*) x)((funcall f x y) y)(x)))

(defun icons (a d)
  (when (or (eq a '*) (eq d '*) (<= a d))
    (if (and (eq a '*) (eq d '*)) t (cons a d))))

(defun int^ (x y)
  (icons (intcmp (car x) (car y) '<) (intcmp (cdr x) (cdr y) '>)))

(defun int~ (x)
  (cond ((eq (car x) '*) (list (cons (1+ (cdr x)) '*)))
	((eq (cdr x) '*) (list (cons '* (1- (car x)))))
	((nconc (int~ (cons (car x) '*)) (int~ (cons '* (cdr x)))))))


;;; RANGES



(defun range-cons (range &aux (a (pop range))(a (or (eq a 'unordered) a))(d (car range)))
  (if (and (eq '* a) (eq '* d)) t (cons a d)))

(defun rngnum (x)
  (if (listp x) (car x) x))

(defun rngcmp (x y f &aux (nx (rngnum x))(ny (rngnum y)))
  (cond ((eq x '*) y)((eq y '*) x)
	((funcall f nx ny) y)((funcall f ny nx) x)
	((eq y ny) x)(y)))

(defun ncons (a d &aux (na (rngnum a))(nd (rngnum d)))
  (when (or (eq a '*) (eq d '*) (< na nd) (and (eql a d) (eql na nd) (eql a na)))
    (cons a d)))

(defun rng^ (x y)
  (ncons (rngcmp (car x) (car y) '<) (rngcmp (cdr x) (cdr y) '>)))

(defun unord^ (x y &aux (cx (car x))(cy (car y))
		     (z (if (eq cx t) cy (if (eq cy t) cx (rd^ cx cy)))))
  (when z (list z)))

(defun urng^ (x y)
  (cond ((and (cdr x) (cdr y)) (rng^ x y))
	((and (null (cdr x)) (null (cdr y))) (unord^ x y))))

(defun rngi (x)
  (if (listp x) (if (integerp (car x)) x (car x)) (list x)))

(defun rng~ (x)
  (cond ((eq (car x) '*) (list (cons (rngi (cdr x)) '*)))
	((eq (cdr x) '*) (list (cons '* (rngi (car x)))))
	((nconc (rng~ (cons (car x) '*)) (rng~ (cons '* (cdr x)))))))

(defun unord~ (x &aux (cx (car x)))
  (unless (eq cx t)
    (if (listp cx) (mapcar (lambda (x) (list x)) cx) (list (list x)))))

(defun urng~ (x)
  (cond ((null (cdr x)) (nconc (unord~ x) (list (cons '* '*))))
	((and (eq (car x) '*) (eq (cdr x) '*)) (list (cons t nil)))
	((nconc (list (cons t nil)) (rng~ x)))))

(defun rng-ld (type) (list (pop type) (range-cons type)))

(defun urng-recon (x &aux (c (pop x)))
  (if (eq (car x) t)
      (list c '* '*)
      (?or (mapcar (lambda (x &aux (o (list c (car x) (cdr x))))
		     (cond ((and (eq (car x) '*) (eq (cdr x) '*)) `(and ,o (not (,c unordered))))
			   ((cdr x) o)
			   ((listp (car x)) `(and (,c unordered) (not (member ,@(car x)))))
			   ((eq (car x) t) `(,c unordered))
			   (`(member ,(car x)))))
		   x))))
  

;;; COMPLEX






(defun sking (tp &aux (tp (nprocess-type tp)))
  (unless (or (cadr tp) (caddr tp) (cdar tp))
    (caar tp)))

(defun lookup-cmp-k (rk ik)
  (cadr (assoc (if (eq rk ik) rk (list rk ik))	+ctps+ :test 'equal)))

(defun cmp-k (x y)
  (lookup-cmp-k (car (sking x)) (car (sking y))))

(defun cmp-ld (type &aux (r (sking (cadr type)))(rk (pop r))
		      (i (sking (or (caddr type) (cadr type))))(ik (pop i)))
  (let ((k (lookup-cmp-k rk ik)))
    (when k
      `(,k ,(if (and (eq (car r) t) (eq (car i) t)) t (cons r i))))))

(defun irange (x)
  (if (isnan x) (list x) (cons x x)))

(defun cmp-irange (x &aux (r (realpart x))(i (imagpart x)))
  `((,(irange r)) . (,(irange i))))


(defun rng-ip (x)
  (unless (cdr x)
    (when (consp (car x))
      (when (realp (caar x))
	(eql (caar x) (cdar x))))))

(defun cmp-cons (a d)
  (when (and a d)
    (if (and (rng-ip a) (rng-ip d))
	(list (complex (caar a) (caar d)))
      (list (cons a d)))))

(defun cmpg~ (kr ki x)
  (cond ((consp x)
	 (let ((a (kop-not kr (car x)))
	       (d (kop-not ki (cdr x))))
	   (nconc (cmp-cons a (cdr x)) (cmp-cons (car x) d) (cmp-cons a d))))
	((cmpg~ kr ki (cmp-irange x)))))

(defun cmpi~  (x) (cmpg~ 'integer    'integer x))
(defun cmpir~ (x) (cmpg~ 'integer    'ratio   x))
(defun cmpri~ (x) (cmpg~ 'ratio      'integer x))
(defun cmpr~  (x) (cmpg~ 'ratio      'ratio   x))
(defun cmp~   (x) (cmpg~ 'long-float 'long-float x))

(defun cmpg^ (kr ki x y)
  (cond ((and (consp x) (consp y))
	 (car (cmp-cons (kop-and kr (car x) (car y)) (kop-and ki (cdr x) (cdr y)))))
	((consp x) (when (cmpg^ kr ki x (cmp-irange y)) y))
	((consp y) (cmpg^ kr ki y x))
	((rd^ x y))))


(defun cmpi^  (x y) (cmpg^ 'integer    'integer x y))
(defun cmpir^ (x y) (cmpg^ 'integer    'ratio x y))
(defun cmpri^ (x y) (cmpg^ 'ratio      'integer x y))
(defun cmpr^  (x y) (cmpg^ 'ratio      'ratio x y))
(defun cmp^   (x y) (cmpg^ 'long-float 'long-float x y))


(defun cmp-recon (x &optional (c (car (rassoc (pop x) +ctps+ :key 'car)) cp))

  (cond ((not cp) (?or (mapcar (lambda (x) (cmp-recon x c)) x)))
	((eq x t) (if (consp c) `(complex* (,(pop c) * *) (,(car c) * *))
		    `(complex (,c * *))))
	((consp x)
	 (let* ((rx (k-recon (cons (if (consp c) (pop c) c) (car x))))
		(ry (k-recon (cons (if (consp c) (car c) c) (cdr x)))))
	   (if (equal rx ry)
	       `(complex ,rx)
	     `(complex* ,rx ,ry))))
	(`(member ,x))))


;;; CONS


(defun cns-list (a d &optional m n &aux (mn (or m n)))
  (when (and (or (car a) (cadr a) (unless a mn)) (or (car d) (cadr d) (unless d mn)))
    (if (and (unless (car a) (cadr a)) (unless (car d) (cadr d))); (not m) (not n)
	`(t)
      `((,a ,d ,m ,n)))))

(defun cns-and (x y)
  (if (and x y) (ntp-and x y) (or x y)))

;(defun cns-type (a d) `(cons ,(fourth a) ,(fourth d)))
(defun cns-type (a d)
  `(cons ,(nreconstruct-type-int a) ,(nreconstruct-type-int d)));FIXME separage car cdr

(defun cns-match (a d &rest r &aux (tp (when a (cns-type a d))))
  (if tp (lremove-if-not (lambda (x) (typep x tp)) r) r))

(defun cns^ (x y)
  (let* ((a (cns-and (car x) (car y)))(d (cns-and (cadr x) (cadr y)))
	 (mx (caddr x))(nx (cadddr x))(my (caddr y))(ny (cadddr y)))
    (cond (mx (cond (my (when (eql mx my) x))
		    ((member mx ny) nil)
		    ((when a (not (typep mx (cns-type a d)))) nil)
		    (x)))
	  (my (cns^ y x))
	  (nx (car (cns-list a d nil (apply 'cns-match a d (union nx ny)))))
	  (ny (cns^ y x))
	  ((car (cns-list a d))))))

(defun cns~ (x)
  (cond ((let ((a (when (car x) (ntp-not (car x))))
	       (d (when (cadr x) (ntp-not (cadr x)))))
	   (nconc (cns-list a (cadr x))
		  (cns-list (car x) d)
		  (cns-list a d)
		  (when (caddr x)
		    (cns-list (car x) (cadr x) nil (list (caddr x))))
		  (mapcan (lambda (y) (cns-list nil nil y)) (cadddr x)))))))

(defconstant +tp-nil+ `(nil nil nil))
(defconstant +tp-t+   `(nil t nil))

(defun mntp (x)
  (case x
    ((t) +tp-t+)
    ((nil) +tp-nil+)
    (otherwise
     (if (eq (car x) 'satisfies)
	 (list (list +tp-t+ +tp-nil+ x) nil t)
       (list (if (consp (car x)) x (list x)) nil nil)))))

(defvar *pcnsk* '(proper-cons null))
(defvar *pcns-ntp* (mntp (mapcar (lambda (x) (list x t)) *pcnsk*) ))
(defvar *pcns-nntp* (mntp (mapcar (lambda (x) (list x t))
				  (set-difference
				   (lreduce (lambda (xx x &aux (x (car x)))
					      (if (listp x) (append x xx) (cons x xx)))
					    +k-ops+ :initial-value nil)
				   *pcnsk*))))

(defun pcdr (u type &aux (z (ntp-and u (nprocess-type type))))
  (ntp-prune (pop z) (pop z) (car z) (length (car u))))

(defun pcns (u type)
  (k-mk (pop type) nil (cns-list (nprocess-type (pop type)) (pcdr u (car type)))))

(defun cns-ld (type)
  (pcns (if (eq (car type) 'proper-cons) *pcns-ntp* *pcns-nntp*) type))


(defun cns-recon (x &optional (c (pop x) cp))
  (cond ((not cp) (?or (mapcar (lambda (x) (cns-recon x c)) x)))
	((eq x t) c)
	((caddr x) `(member ,(caddr x)))
	((cadddr x)
	 (let ((y `(not (member ,@(cadddr x)))))
	   (if (car x) `(and ,(cns-recon (list (car x) (cadr x)) c) ,y) y)))
	(x `(,c ,(nreconstruct-type-int (pop x))
		,(nreconstruct-type-int (car x))))))


;;; STRUCTURE and CLASS

(defun gen-def (x)
  (cond ((or (symbolp x) (si-classp x)) 'top)
	((structurep x) (sdata-name (c-structure-def x)))
	((si-class-of x))))

(defun std-car (x c)
  (if (s-class-p x) (list c) (gen-get-included (std-def x))))

(defun orthog-to-and-not (x c)
  (cond
   ((eq x t) (list c))
   ((listp x) (nconc (std-car (car x) c) x));FIXME
   ((s-class-p x) (gen-get-included x))
   (`((member ,x)))))

(defun std-def (x) (gen-def (if (listp x) (car x) x)))

(defun s-class-p (x &aux (x (std-def x)))
  (eq x (std-def x)))

(defun std~ (x)
  (cond ((s-class-p x) (if (listp x) x `((,x))))
	((nconc (std~ (std-def x)) (if (listp x) x `((,x)))))))

(defun std^ (x y)
  (cond ((eq (std-def x) (std-def y)) (rd^ x y))
	((s-class-p x) (when (std^ x (std-def y)) y))
	((s-class-p y) (std^ y x))))

(defun si-subclasses (c)
  (when c
    (cons c (lreduce
	     (lambda (y x) (lreduce (lambda (y x) (adjoin x y)) (si-subclasses x) :initial-value y))
	     (si-class-direct-subclasses c) :initial-value nil))))

(defun gen-get-included (x)
  (if (symbolp x) (get-included x) (si-subclasses x)))

(defun filter-included (c x)
  (case c
	(std-instance (lremove-if 'funcallable-class-p x))
	(funcallable-std-instance (lremove-if-not 'funcallable-class-p x))
	(otherwise x)))

(defun std-ld (x &aux (c (pop x)))
  (cons c (if x (filter-included c (gen-get-included (car x))) '(t))))

(defun std-matches (x)
  (lremove-if-not (lambda (y) (member (car y) x :test 'member :key 'cdr)) x))

(defun std-recon (x &optional (c (pop x)) &aux
		    (x (mapcar (lambda (x) (orthog-to-and-not x c)) x))
		    (m (std-matches x)))
  (?or
   (mapcar (lambda (x &aux (h (pop x)))
	     (if x `(and ,h (not ,(std-recon x c))) h))
	   (mapcar (lambda (x)
		     (lreduce
		      (lambda (y x &aux (m (member x m :key 'car)))
			(nconc y (if m (lremove-if 's-class-p (car m)) (list x))))
		      x :initial-value nil))
		   (set-difference x m)))))


;;; INDIVIDUALS


(defun kktype-of (x)
  (cond ((atom x) (cond ((coerce-to-standard-class x) 'std-instance) ((when (symbolp x) (get x 's-data)) 'structure)(x)));FIXME
	((eq (car x) 'simple-array) (cdr (assoc (cadr x) *simple-array-types*)))
	((eq (car x) 'non-simple-array) (cdr (assoc (cadr x) *non-simple-array-types*)))
	((member (car x) '(complex complex*))
	 (cmp-k (cadr x) (or (caddr x) (cadr x))))
	((car x))))

(defun ktype-of (x)
  (or (kktype-of (type-of x))
      (error "unknown type")))

(defun cons-to-cns-list (x) (cns-list nil nil x))

(defun mcns-ld (c x)
  (cons c (cons-to-cns-list x)))

#.`(defun kmem (x &aux (z (ktype-of x)))
     (case z
	   ((proper-cons improper-cons) (mcns-ld z x))
	   (,+range-types+ `(,z (,x . ,(unless (isnan x) x))))
	   (null `(,z t))
	   (otherwise `(,z ,x))))

(defun member-ld (type)
  (car (ntp-not
	(lreduce (lambda (y x) (ntp-and y (ntp-not (mntp (list (kmem x))))))
		(cdr type) :initial-value +tp-t+))))


#.`(defun k^ (k x y)
     (case k
       ,@(mapcar (lambda (x) `(,(car x) (,(cadr x) x y))) (butlast +k-ops+))
       (otherwise (,(cadr (car (last +k-ops+))) x y))))

(defun kop-and (k x y)
  (cond ((eq (car x) t) y)
	((eq (car y) t) x)
	((lreduce (lambda (xx x)
	     (lreduce (lambda (yy y) (?cns (k^ k x y) yy))
		      y :initial-value xx))
		  x :initial-value nil))))


#.`(defun k~ (k x)
     (unless (eq x t)
       (case k
	 ,@(mapcar (lambda (x) `(,(car x) (,(caddr x) x))) (butlast +k-ops+))
	 (otherwise (,(caddr (car (last +k-ops+))) x)))))

(defun kop-not (k x)
  (lreduce (lambda (xx x) (when xx (kop-and k (k~ k x) xx))) x :initial-value '(t)))

(defun kop-or (k x y)
  (kop-not k (kop-not k (append x y))))

(defun k-mk (k d x)
  (unless (eq d (car x))
    (cons k x)))

(defun k-op (op x y d &aux (k (car x)))
  (k-mk
   k d
   (case op
     (and (kop-and k (cdr x) (cdr y)))
     (or  (kop-or  k (cdr x) (cdr y)))
     (not (kop-not k (cdr x))))))


(defun ntp-prune (x y z &rest u)
  (cond
    ((not (or x z u)) (if y +tp-t+ +tp-nil+))
    ((unless (member (not y) x :test-not 'eq :key 'cadr);FIXME? shortest of list and complement?
       (eql (length x) (or (car u) +k-len+)))
     (apply 'ntp-prune nil (not y) z u))
    ((list* x y z u))))

(defun ?cns (x y) (if x (cons x y) y))

(defun ntp-not (x &aux (l (pop x)) (d (not (pop x))) (u (pop x)))
  (if u
      (list (list (ntp-not (cadr l)) (ntp-not (car l)) `(not ,(caddr l))) nil u)
    (apply 'ntp-prune
	   (lreduce (lambda (ll l) (?cns (k-op 'not l nil d) ll)) l :initial-value nil)
	   d u x)))

(defun ntp-list (op lx ly d dx dy)
  (lreduce (lambda (ll l &aux (ny (assoc (car l) ly)))
	     (?cns (cond (ny (k-op op l ny d)) (dy l)) ll))
	   lx :initial-value
	   (when dx
	     (lreduce (lambda (ll l) (?cns (unless (assoc (car l) lx) l) ll))
		      ly :initial-value nil))))

(defun ntp-subtp (x y) (ntp-and?c2-nil-p x y t))

(defun ntp-and-unknown (ox lx ux oy ly uy d)
  (let* ((xx (if ux (pop lx) ox))(xy (if uy (pop ly) oy))(x (ntp-and xx xy))
	 (mx (if ux (pop lx) ox))(my (if uy (pop ly) oy))(m (ntp-and mx my)))
    (cond ((ntp-subtp x m) x)
	  ((unless ux (ntp-subtp xy xx)) oy)
	  ((unless uy (ntp-subtp xx xy)) ox)
	  ((list (list x m `(and ,(if ux (car lx) (car (nreconstruct-type ox)))
				 ,(if uy (car ly) (car (nreconstruct-type oy)))))
		 d t)))))

(defun ntp-or-unknown (ox lx ux oy ly uy d)
  (let* ((xx (if ux (pop lx) ox))(xy (if uy (pop ly) oy))(x (ntp-or xx xy))
	 (mx (if ux (pop lx) ox))(my (if uy (pop ly) oy))(m (ntp-or mx my)))
    (cond ((ntp-subtp x m) x)
	  ((unless ux (ntp-subtp mx my)) oy)
	  ((unless uy (ntp-subtp my mx)) ox)
	  ((list (list x m `(or ,(if ux (car lx) (car (nreconstruct-type ox)))
				,(if uy (car ly) (car (nreconstruct-type oy)))))
		 d t)))))

(defun ntp-and?c2-nil-p (x y ?c2)
  (let* ((x (if (caddr x) (caar x) x))(y (if (caddr y) (if ?c2 (cadar y) (caar y)) y))
	 (lx (pop x))(ly (pop y))(dx (pop x))(dy (pop y))(dy (if ?c2 (not dy) dy))(i 0)(d (and dx dy))
	 (lk (or (car x) +k-len+)))
    (not
     (or (when dx (member-if-not (lambda (x) (or (eq (cadr x) ?c2) (assoc (car x) lx))) ly))
	 (member-if (lambda (x &aux (y (assoc (car x) ly)))
		      (cadr (cond (y (k-op 'and x (if ?c2 (k-op 'not y nil d) y) d));FIXME remove last consing from this
				  (dy (incf i) x))))
		    lx)
	 (when d (not (eql lk (+ i (length ly)))))))))

(defun ntp-and (&rest xy)
  (when xy
    (let* ((x (car xy)) (y (cadr xy))
	   (ox x)(oy y)(lx (pop x))(ly (pop y))(dx (pop x))(dy (pop y))
	   (d (and dx dy))(ux (pop x))(uy (pop y)))
      (cond ((or ux uy) (ntp-and-unknown ox lx ux oy ly uy d))
	    ((not lx) (if dx oy ox))
	    ((not ly) (if dy ox oy))
	    ((apply 'ntp-prune
		    (ntp-list 'and lx ly d dx dy) d
		    nil x))))))

(defun ntp-or (&rest xy)
  (when xy
    (let* ((x (car xy)) (y (cadr xy))
	   (ox x)(oy y)(lx (pop x))(ly (pop y))(dx (pop x))(dy (pop y))
	   (d (or dx dy))(ux (pop x))(uy (pop y)))
      (cond ((or ux uy) (ntp-or-unknown ox lx ux oy ly uy d))
	    ((not (or (car x) lx)) (if dx ox oy))
	    ((not (or (car y) ly)) (if dy oy ox))
	    ((apply 'ntp-prune
		    (ntp-list 'or lx ly d (not dx) (not dy)) d
		    nil x))))))

(defconstant +ntypes+ `(,@+singleton-types+
			std-instance structure funcallable-std-instance
			t nil))

(defconstant +dtypes+ '(or and not member satisfies
			integer ratio short-float long-float
			complex* simple-array non-simple-array
			proper-cons improper-cons))

(defun normalize-type (type &aux e
			      (type (if (listp type) type (list type)));FIXME
			      (ctp (car type)))
  (cond ((eq ctp 'structure-object) `(structure));FIXME
	((member ctp +ntypes+) type)
	((member ctp +dtypes+)
	 (funcall (macro-function (get ctp 'deftype-definition)) type nil));FIXME
	((eq type (setq e (just-expand-deftype type))) type)
	((normalize-type e))))


#.`(defun ntp-load (type)
     (mntp
      (ecase (car type)
	 ((t nil) (car type))
	 (,+range-types+ (rng-ld type))
	 (complex* (cmp-ld type))
	 ((cons proper-cons improper-cons) (cns-ld type))
	 ((std-instance structure funcallable-std-instance) (std-ld type))
	 ((simple-array non-simple-array) (ar-ld type))
	 (,+singleton-types+ (sing-ld type))
	 (member (member-ld type))
	 (satisfies type))))

(defun nprocess-type (type)
  (case (car type)
    (and (lreduce 'ntp-and (mapcar 'nprocess-type (cdr type))))
    (or (lreduce  'ntp-or  (mapcar 'nprocess-type (cdr type))))
    (not (ntp-not (nprocess-type (cadr type))))
    (otherwise (ntp-load type))))

#.`(defun k-recon (x)
     (case (car x)
       ,@(mapcar (lambda (x) `(,(car x) (,(cadddr x) x))) (butlast +k-ops+))
       (otherwise (,(cadddr (car (last +k-ops+))) x))))

(defun nreconstruct-type-int (x)
  (cond ((caddr x) (caddr (car x)))
	((cadr x)
	 (let* ((x (ntp-not x)))
	   (let ((z (nreconstruct-type-int x)))
	     (or (not z) `(not ,z)))))
	((?or (mapcar 'k-recon (car x))))))

(defun nreconstruct-type (x)
  (list (nreconstruct-type-int x) (caddr x)))


(defun resolve-type (type)
  (nreconstruct-type (nprocess-type (normalize-type type))))

(defun subtypep (t1 t2 &optional env)
  (declare (ignore env) (optimize (safety 1)))
  (check-type t1 full-type-spec)
  (check-type t2 full-type-spec)
  (if (or (not t1) (eq t2 t))
      (values t t)
      (let* ((n1 (nprocess-type (normalize-type t1)))
	     (n2 (nprocess-type (normalize-type t2))))
	(values (ntp-subtp n1 n2) (not (or (caddr n1) (caddr n2)))))))



