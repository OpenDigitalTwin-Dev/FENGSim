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


;;;;    DEFSTRUCT.LSP
;;;;
;;;;        The structure routines.


;; (in-package 'lisp)
;; (export 'defstruct)


(in-package :system)


(defvar *accessors* (make-array 10 :adjustable t))
(defvar *list-accessors* (make-array 2 :adjustable t))
(defvar *vector-accessors* (make-array 2 :adjustable t))

(defun record-fn (&rest l) (declare (ignore l)) nil)

(defun make-access-function (name conc-name no-conc type named include no-fun
				  ;; from apply
				  slot-name default-init slot-type read-only
				  offset &optional predicate ost)
  (declare (ignore named default-init predicate no-fun ost))
  (let ((access-function (if no-conc slot-name
			   (intern (si:string-concatenate conc-name slot-name)))))
    (record-fn access-function 'defun '(t) slot-type)
    (cond (read-only
	   (remprop access-function 'structure-access)
	   (setf (get access-function 'struct-read-only) t))
	  (t (remprop access-function 'setf-update-fn)
	     (remprop access-function 'setf-lambda)
	     (remprop access-function 'setf-documentation)
	     (let ((tem (get access-function 'structure-access)))
	       (unless (and (consp tem) include
			    (subtypep include (car tem))
			    (eql (cdr tem) offset))
		 (setf (get access-function 'structure-access) (cons (if type type name) offset)))))))
  nil)

(defmacro key-name (key prior-keyword)
  `(cond
   ((not (consp ,key))
    ,key)
   (t 
    (unless (endp (cdddr ,key))
      (error "Bad key ~S~%" ,key))
    (cond 
     ((not (consp (car ,key)))
      (car ,key))
     ((and (eq ,prior-keyword '&key) (not (consp (caar ,key))))
      (unless (endp (cddar ,key))
	(error "Bad key ~S~%" ,key))
      (cadar ,key))
     (t
      (error "Bad key ~S~%" ,key))))))

(defmacro maybe-add-keydef (key keydefs prior-keyword)
  `(let ((def (cadar 
	       (member (key-name ,key ,prior-keyword) ,keydefs
		       :key (lambda (k)
			      (declare (optimize (safety 2)))
			      (when (consp k) (car k)))))))
     (if def
	 (cond ((not (consp ,key))
		(list ,key def))
	       (t
		(if (cdr ,key) ,key (list (car ,key) def))))
       ,key)))

(defun parse-boa-lambda-list (lambda-list keydefs)
  (let ((keywords '(none &optional &rest &key &allow-other-keys &aux))
	vs res tk restvar seen-keys)
    (do ((ll lambda-list (cdr ll))) ((endp ll))
      (let ((key (car ll)))
	(cond ((setq tk (member key keywords))
	       (setq keywords tk)
	       (push key res)
	       (push key seen-keys))
	      ((member key lambda-list-keywords)
	       (error "Keyword ~S appeared in a bad place in BOA lambda list" key))
	      (t
	       (let ((prior-keyword (car keywords)))
		 (case prior-keyword
		   ((none &rest)
		    (unless (symbolp key)
		      (error "non-symbol appeared in bad place in BOA lambda list" key))
		    (push key res)
		    (push key vs)
		    (when (eq prior-keyword '&rest)
		      (when restvar
			(error "Multiple variables after &rest in BOA lambda list"))
		      (setq restvar t)))
		   ((&optional &key)
		    (push (maybe-add-keydef key keydefs prior-keyword) res)
		    (push (key-name key prior-keyword) vs))
		   (&allow-other-keys
		    (error "Variable ~S appeared after &allow-other-keys in BOA list" key))
		   (&aux
		    (push key res)
		    (push (key-name key prior-keyword) vs))))))))
    (when (and (member '&rest seen-keys) (not restvar))
      (error "Missing &rest variable in BOA list"))
    (unless (member '&aux seen-keys)
      (push '&aux res))
    (do ((ll keydefs (cdr ll))) ((endp ll))
      (let* ((keydef (car ll))
	     (keydef-name (if (atom keydef) keydef (car keydef))))
	(unless (member keydef-name vs)
	  (push keydef res))))
    (nreverse res)))

(defun maybe-cons-keyname (x &optional y)
  (unless (consp x)
    (error 'program-error :format-control "x ~S is not a list~%" :format-arguments (list x)))
  (let ((sn (sixth x)))
    (if sn
	(if y
	    (list (list (car x) sn) y)
	  (list (list (car x) sn)))
      (if y (list (car x) y) (car x)))))

(defun make-constructor (name constructor type named
                         slot-descriptions)
  (declare (ignore named))
  (let ((slot-names
         ;; Collect the slot-names.
         (mapcar (lambda (x)
                     (cond ((null x)
                            ;; If the slot-description is NIL,
                            ;;  it is in the padding of initial-offset.
                            nil)
                           ((null (car x))
                            ;; If the slot name is NIL,
                            ;;  it is the structure name.
                            ;;  This is for typed structures with names.
                            (list 'quote (cadr x)))
                           (t (let ((sn (sixth x))) (if sn sn (car x))))))
                 slot-descriptions))
        (keys
         ;; Make the keyword parameters.
         (mapcan (lambda (x)
                     (cond ((null x) nil)
                           ((null (car x)) nil)
                           ((null (cadr x)) (list (maybe-cons-keyname x)))
                           (t (list (maybe-cons-keyname x (cadr x))))))
                 slot-descriptions)))
    (cond ((consp constructor)
	   (setq keys (parse-boa-lambda-list (cadr constructor) keys))
           (setq constructor (car constructor)))
          (t
           ;; If not a BOA constructor, just cons &KEY.
           (setq keys (cons '&key keys))))
     (cond ((null type)
	    `(defun ,constructor ,keys
	       (the ,name (si:make-structure ',name ,@slot-names))))
	   ((eq type 'vector)
	    `(defun ,constructor ,keys
	       (vector ,@slot-names)))
	   ((and (consp type) (eq (car type) 'vector))
	    (if (endp (cdr type))
		`(defun ,constructor ,keys
		   (vector ,@slot-names)))
	      `(defun ,constructor ,keys
		 (make-array ,(length slot-names)
			     :element-type ',(cadr type)
			     :initial-contents (list ,@slot-names))))
	   ((eq type 'list)
	    `(defun ,constructor ,keys
	       (list ,@slot-names)))
	   ((error "~S is an illegal structure type" type)))))
  

;;; PARSE-SLOT-DESCRIPTION parses the given slot-description
;;;  and returns a list of the form:
;;;        (slot-name default-init slot-type read-only offset)

(defun parse-slot-description (slot-description offset)
  (let (slot-name default-init slot-type read-only)
    (cond ((atom slot-description)
           (setq slot-name slot-description))
          ((endp (cdr slot-description))
           (setq slot-name (car slot-description)))
          (t
           (setq slot-name (car slot-description))
           (setq default-init (cadr slot-description))
           (do ((os (cddr slot-description) (cddr os)) (o) (v))
               ((endp os))
             (setq o (car os))
             (when (endp (cdr os))
                   (error "~S is an illegal structure slot option."
                          os))
             (setq v (cadr os))
             (case o
               (:type (setq slot-type v))
               (:read-only (setq read-only v))
               (t
                (error "~S is an illegal structure slot option."
                         os))))))
    (list slot-name default-init slot-type read-only offset nil slot-type)))


;;; OVERWRITE-SLOT-DESCRIPTIONS overwrites the old slot-descriptions
;;;  with the new descriptions which are specified in the
;;;  :include defstruct option.

(defun overwrite-slot-descriptions (news olds)
  (if (null olds)
      nil
      (let ((sds (member (caar olds) news :key #'car)))
        (cond (sds
               (when (and (null (cadddr (car sds)))
                          (cadddr (car olds)))
                     ;; If read-only is true in the old
                     ;;  and false in the new, signal an error.
                     (error "~S is an illegal include slot-description."
                            sds))
	       ;; If
	       (setf (caddr (car sds))
		     (upgraded-array-element-type (caddr (car sds))))
	       (when (not  (equal (normalize-type (or (caddr (car sds)) t))
				 (normalize-type (or (caddr (car olds)) t))))
		     (error "Type mismmatch for included slot ~a" (car sds)))
		     (cons (list* (caar sds)
				  (cadar sds)
				  (caddar sds)
				  (cadddr (car sds))
				  ;; The rest from the old.
				  (cddddr (car olds)))
                     (overwrite-slot-descriptions news (cdr olds))))
              (t
               (cons (car olds)
                     (overwrite-slot-descriptions news (cdr olds))))))))

(defconstant +aet-type-object+ (aet-type nil))
(defconstant +all-t-s-type+ 
  (make-array 50 :element-type 'unsigned-char :static t :initial-element +aet-type-object+))
(defconstant +alignment-t+ (alignment t))

(defun make-t-type (n include slot-descriptions &aux i)
  (let ((res  (make-array n :element-type 'unsigned-char :static t)))
    (when include
	  (let ((tem (get include 's-data)) raw)
	    (or tem (error "Included structure undefined ~a" include))
	    (setq raw (s-data-raw tem))
	  (dotimes (i (min n (length raw)))
		   (setf (aref res i) (aref raw i)))))
    (dolist (v slot-descriptions)
	    (setq i (nth 4 v))
	    (let ((type (third v)))
	      (cond ((<= (the fixnum (alignment type)) +alignment-t+)
		     (setf (aref res i) (aet-type type))))))
    (cond ((< n (length +all-t-s-type+))
	   (let ((def +aet-type-object+))
	     (dotimes (i n)
	       (cond ((not (= (the fixnum (aref res i)) def))
		      (return-from make-t-type res)))))
	   +all-t-s-type+)
	  (t res))))

(defvar *standard-slot-positions*
  (let ((ar (make-array 50 :element-type 'unsigned-short :static t))) 
    (dotimes (i 50)
	     (declare (fixnum i))
	     (setf (aref ar i)(*  (size-of t) i)))
    ar))

(defun round-up (a b)
  (declare (fixnum a b))
  (setq a (ceiling a b))
  (the fixnum (* a b)))


(defun get-slot-pos (leng include slot-descriptions &aux type small-types
			  has-holes) 
  (declare (ignore include) (special *standard-slot-positions*)) 
  (dolist (v slot-descriptions)
	  (when (and v (car v))
		(setf type 
		      (upgraded-array-element-type (or (caddr v) t))
		      (caddr v) type)
		(let ((val (second v)))
		  (unless (typep val type)
			  (if (and (symbolp val)
				   (constantp val))
			      (setf val (symbol-value val)))
			  (and (constantp val)
			       (setf (cadr v) (coerce val type)))))
		(cond ((member type '(signed-char unsigned-char
						short unsigned-short
					 long-float
					 bit))
		       (setq small-types t)))))
  (cond ((and (null small-types)
	      (< leng (length *standard-slot-positions*))
	      (list  *standard-slot-positions* (* leng  (size-of t)) nil)))
	(t (let ((ar (make-array leng :element-type 'unsigned-short
				 :static t))
		 (pos 0)(i 0)(align 0)type (next-pos 0))
	     (declare (fixnum pos i align next-pos))
	     ;; A default array.
		   
	     (dolist
	       (v slot-descriptions)
	       (setq type (caddr v))
	       (setq align (alignment type))
	       (unless (<= align +alignment-t+)
		       (setq type t)
		       (setf (caddr v) t)
		       (setq align +alignment-t+)
		       (setq v (nconc v '(t))))
	       (setq next-pos (round-up pos align))	
	       (or (eql pos next-pos) (setq has-holes t))
	       (setq pos next-pos)
	       (setf (aref ar i) pos)
	       (incf pos (size-of type))
	       (incf i))
	     (list ar (round-up pos (size-of t)) has-holes)
	     ))))

;FIXME reconsider holding on to computed structure types
(defun update-sdata-included (name &aux r (i (sdata-includes (get name 's-data))))
  (when i
    (let ((to (cmp-norm-tp `(and ,(sdata-name i) (not (or ,@(sdata-included i))))))
	  (tn (cmp-norm-tp name)))
      (labels ((find-updates (x &aux (tp (car x)))
		 (when (unless (tp<= #tstructure tp) (tp-and #tstructure tp))
		   (let ((ntp (if (tp-and to tp) (tp-or tn tp) tp)));FIXME negative
		     (unless (tp= tp ntp)
		       (setf (car x) ntp)
		       (push (cons tp ntp) r)))))
	       (update-sig (x &aux (y (assoc (car x) r)))
		 (when y (setf (car x) (cdr y)))))
	(mapl #'find-updates (gethash (tsrch #tstructure) *uniq-tp*));FIXME more systematic
	(mapl #'find-updates (gethash t *uniq-tp*))
	(maphash (lambda (x y)
		   (declare (ignore y))
		   (mapl #'update-sig (car x))
		   (if (cmpt (cadr x)) (mapl #'update-sig (cdadr x)) (update-sig (cdr x))))
		 *uniq-sig*)))
    (pushnew name (s-data-included i))))

;FIXME function-src for all functions, sigs for constructor and copier
(defun define-structure (name conc-name no-conc type named slot-descriptions copier
			      static include print-function constructors
			      offset predicate &optional documentation no-funs
			      &aux def leng)
  (declare (ignore copier))
  (and (consp type) (eq (car type) 'vector)(setq type 'vector))
  (setq leng (length slot-descriptions))
  (dolist (x slot-descriptions)
    (and x (car x)
	 (apply 'make-access-function
		name conc-name no-conc type named include no-funs x)))

  (cond ((and (null type)
	      (eq name 's-data))
	 ;bootstrapping code!
	 (setq def (make-s-data-structure
		     (make-array (* leng (size-of t))
				 :element-type 'unsigned-char :static t :initial-element +aet-type-object+)
		     (make-t-type leng nil slot-descriptions)
		     *standard-slot-positions*
		     slot-descriptions
		     t
		     ))
	 )
	(t
	  (let (slot-position
		 (size 0) has-holes
		 (include-str (and include
				   (get include 's-data))))
	    (when include-str
		  (cond ((and (s-data-frozen include-str)
			      (or (not (s-data-included include-str))
				  (not (let ((te (get name 's-data)))
					 (and te
					      (eq (s-data-includes 
						    te)
						  include-str))))))
			 (warn " ~a was frozen but now included"
			       include))))
	    (when (null type)
		 (setf slot-position
		       (get-slot-pos leng include slot-descriptions))
		 (setf size (cadr slot-position)
		       has-holes (caddr slot-position)
		       slot-position (car slot-position)
		       ))
	  (setf def (make-s-data
		       :name name
		       :length leng
		       :raw
		       (and (null type)
			    (make-t-type leng include slot-descriptions))
		       :slot-position slot-position
		       :size size
		       :has-holes has-holes
		       :staticp static
		       :includes include-str
		       :print-function print-function
		       :slot-descriptions slot-descriptions
		       :constructors constructors
		       :offset offset
		       :type type
		       :named named
		       :documentation documentation
		       :conc-name conc-name)))))
  (let ((tem (get name 's-data)))
    (cond ((eq name 's-data)
	   (if tem (warn "not replacing s-data property"))
	   (or tem (setf (get name 's-data) def)))
	  (tem 
	   (check-s-data tem def name))
	  (t (setf (get name 's-data) def) (update-sdata-included name)))
    (when documentation
	  (setf (get name 'structure-documentation) documentation))
    (when (and (null type) predicate)
	  (record-fn predicate 'defun '(t) t)
	  (setf (get predicate 'compiler::co1)'compiler::co1structure-predicate)
	  (setf (get predicate 'struct-predicate) name)
	  (setf (get predicate 'predicate-type) name)))
  nil)

		  
(defun str-ref (x y z) (declare (ignore y)) (structure-ref1 x z))
(export 'str-ref)

(defmacro defstruct (name &rest slots)
  (declare (optimize (safety 2)))
  (let ((slot-descriptions slots)
        options
        conc-name
        constructors default-constructor no-constructor
        copier
        predicate predicate-specified
        include
        print-function print-object type named initial-offset
        offset name-offset
        documentation
	static
	(no-conc nil))

    (when (consp name)
	  ;; The defstruct options are supplied.
          (setq options (cdr name))
          (setq name (car name)))
    
    ;; The default conc-name.
    (setq conc-name (si:string-concatenate (string name) "-"))

    ;; The default constructor.
    (setq default-constructor
          (intern (si:string-concatenate "MAKE-" (string name))))

    ;; The default copier and predicate.
    (setq copier
          (intern (si:string-concatenate "COPY-" (string name)))
          predicate
          (intern (si:string-concatenate (string name) "-P")))

    ;; Parse the defstruct options.
    (do ((os options (cdr os)) (o) (v))
        ((endp os))
	(cond ((and (consp (car os)) (not (endp (cdar os))))
	       (setq o (caar os) v (cadar os))
	       (case o
		 (:conc-name
		   (if (null v) 
		       (progn
			 (setq conc-name "")
			 (setq no-conc t))
		     (setq conc-name v)))
		 (:constructor
		   (if (null v)
		       (setq no-constructor t)
		     (if (endp (cddar os))
			 (setq constructors (cons v constructors))
		       (setq constructors (cons (cdar os) constructors)))))
		 (:copier (setq copier v))
		 (:static (setq static v))
		 (:predicate
		   (setq predicate (or v (gensym)))
		   (setq predicate-specified t))
		 (:include
		   (setq include (cdar os))
		   (unless (get v 's-data)
			   (error "~S is an illegal included structure." v)))
		 (:print-object
		  (and (consp v) (eq (car v) 'function)
		       (setq v (second v)))
		  (setq print-object v))
		 (:print-function
		  (and (consp v) (eq (car v) 'function)
		       (setq v (second v)))
		  (setq print-function v))
		 (:type (setq type v))
		 (:initial-offset (setq initial-offset v))
		 (t (error "~S is an illegal defstruct option." o))))
	      (t
		(if (consp (car os))
		    (setq o (caar os))
		  (setq o (car os)))
		(case o
		  (:constructor
		    (setq constructors
			  (cons default-constructor constructors)))
		  ((:copier :predicate :print-function))
		  (:conc-name
		   (progn
		     (setq conc-name "")
		     (setq no-conc t)))
		  (:named (setq named t))
		  (t (error "~S is an illegal defstruct option." o))))))

    (setq conc-name (intern (string conc-name)))

    (when (and print-function print-object)
      (error "Cannot specify both :print-function and :print-object."))
    (when print-object
      (setq print-function (lambda (x y z) 
			     (declare (optimize (safety 2)))
			     (declare (ignore z)) (funcall print-object x y))))

    (and include (not print-function)
	 (setq print-function (s-data-print-function (get (car include)  's-data))))

    ;; Skip the documentation string.
    (when (and (not (endp slot-descriptions))
               (stringp (car slot-descriptions)))
          (setq documentation (car slot-descriptions))
          (setq slot-descriptions (cdr slot-descriptions)))
    
    ;; Check the include option.
    (when include
          (unless (equal type
			 (s-data-type (get  (car include) 's-data)))
                  (error "~S is an illegal structure include."
                         (car include))))

    ;; Set OFFSET.
    (cond ((null include)
           (setq offset 0))
          (t 
	    (setq offset (s-data-offset (get (car include) 's-data)))))

    ;; Increment OFFSET.
    (when (and type initial-offset)
          (setq offset (+ offset initial-offset)))
    (when (and type named)
          (setq name-offset offset)
          (setq offset (1+ offset)))

    ;; Parse slot-descriptions, incrementing OFFSET for each one.
    (do ((ds slot-descriptions (cdr ds))
         (sds nil))
        ((endp ds)
         (setq slot-descriptions (nreverse sds)))
	(setq sds (cons (parse-slot-description (car ds) offset) sds))
	(setq offset (1+ offset)))

    ;; If TYPE is non-NIL and structure is named,
    ;;  add the slot for the structure-name to the slot-descriptions.
    (when (and type named)
          (setq slot-descriptions
                (cons (list nil name) slot-descriptions)))

    ;; Pad the slot-descriptions with the initial-offset number of NILs.
    (when (and type initial-offset)
          (setq slot-descriptions
                (append (make-list initial-offset) slot-descriptions)))

    ;; Append the slot-descriptions of the included structure.
    ;; The slot-descriptions in the include option are also counted.
    (cond ((null include))
          ((endp (cdr include))
           (setq slot-descriptions
                 (append (s-data-slot-descriptions
			   (get (car include) 's-data))
                         slot-descriptions)))
          (t
	    (setq slot-descriptions
		  (append (overwrite-slot-descriptions
			    (mapcar (lambda (sd)
					(parse-slot-description sd 0))
				    (cdr include))
			    (s-data-slot-descriptions
			      (get (car include) 's-data)
                              ))
			  slot-descriptions))))

    (cond (no-constructor
	    ;; If a constructor option is NIL,
	    ;;  no constructor should have been specified.
	    (when constructors
		  (error "Contradictory constructor options.")))
          ((null constructors)
	   ;; If no constructor is specified,
	   ;;  the default-constructor is made.
           (setq constructors (list default-constructor))))

    ;; We need a default constructor for the sharp-s-reader
    (or (member t (mapcar 'symbolp  constructors))
	(push (intern (string-concatenate "__si::" default-constructor))
		      constructors))

    ;; Check the named option and set the predicate.
    (when (and type (not named))
          (when predicate-specified
                (error "~S is an illegal structure predicate."
                       predicate))
          (setq predicate nil))

    (when include (setq include (car include)))

    ;; Check the print-function.
    (when (and print-function type)
          (error "A print function is supplied to a typed structure."))


    (let* ((tp (cond ((not type) nil) 
		     ((subtypep type 'list) 'list)
		     ((subtypep type 'vector) 'vector)))
	   (ctp (cond ((or (not type) named) name) (tp)))
	   new-slot-descriptions
	   (new-slot-descriptions ;(copy-list slot-descriptions)))
	    (dolist (sd slot-descriptions (nreverse new-slot-descriptions))
	      (if (and (consp sd) (eql (length sd) 7))
		(let* ((csd (car sd))
		       (sym (when (or (constantp csd) (keywordp csd) (si::specialp csd)) 
			      (make-symbol (symbol-name csd))))
		       (nsd (if (or (constantp csd) (si::specialp csd))
				(cons (intern (symbol-name csd) 'keyword) (cdr sd))
			      sd)))
		  (push (append (butlast nsd 2) (list sym (car (last nsd)))) new-slot-descriptions)
		  (when sym
		    (setf (car sd) sym)))
		(push sd new-slot-descriptions)))))

      `(progn
	 (define-structure ',name  ',conc-name ',no-conc ',type
	   ',named ',slot-descriptions ',copier ',static ',include 
	   ',print-function ',constructors 
	   ',offset ',predicate ',documentation)
	 ,@(mapcar (lambda (constructor)
		       (make-constructor name constructor type named new-slot-descriptions))
		   constructors)
	 ,@(when copier
	     `((defun ,copier (x) 
		 (declare (optimize (safety 1)))
		 (check-type x ,ctp)
		 (the ,ctp 
		      ,(ecase tp
			      ((nil) `(copy-structure x))
			      (list `(copy-list x))
			      (vector `(copy-seq x)))))))
	 ,@(mapcar (lambda (y) 
		     (let* ((sn (pop y))
			    (nm (if no-conc sn
				    (intern (si:string-concatenate (string conc-name) (string sn)))))
			    (di (pop y))
			    (st (pop y))
			    (ro (pop y))
			    (offset (pop y)))
		       (declare (ignore di ro))
		       `(defun ,nm (x)
			   (declare (optimize (safety 2)))
			   (check-type x ,ctp)
			   (the ,(or (not st) st)
				,(ecase tp
					((nil) `(str-refset x ',name ,offset));FIXME possibly macroexpand here, include?
					(list `(let ((c (nthcdr ,offset x))) (check-type c cons) (car c)));(list-nth ,offset x))
					(vector `(aref x ,offset)))))))
		   slot-descriptions)
	 ,@(mapcar (lambda (y) 
		     (let* ((sn (car y))
			    (y (if no-conc sn
				 (intern (si:string-concatenate (string conc-name) (string sn))))))
		       `(si::putprop ',y t 'compiler::cmp-inline))) slot-descriptions);FIXME
	 ,@(when predicate
	     `((defun ,predicate (x) 
		 (declare (optimize (safety 2)))
		 (the boolean 
		      ,(ecase tp
			      ((nil) `(typecase x (,name t)));`(structure-subtype-p x ',name)
			      (list
			       (unless named (error "The structure should be named."))
			       `(let ((x (when (listp x) (nthcdr ,name-offset x)))) (when x (eq (car x) ',name))))
			      (vector
			       (unless named (error "The structure should be named."))
			       `(and (typep x '(vector t))
				     (> (length x) ,name-offset)
				     (eq (aref x ,name-offset) ',name))))))
	       (si::putprop ',predicate t 'compiler::cmp-inline)))
	 ',name))))

;; First several fields of this must coincide with the C structure
;; s_data (see object.h).


(defstruct s-data (name nil :type symbol)
		 (length 0 :type fixnum)
		 raw
		 included
		 includes
		 staticp
		 print-function
		 slot-descriptions
		 slot-position 
		 (size 0 :type fixnum)
		 has-holes
		 frozen
		 documentation
		 constructors
		 offset
		 named
		 type
		 conc-name
		 )


(defun check-s-data (tem def name)
  (cond ((s-data-included tem)
	 (setf (s-data-included def)(s-data-included tem))))
  (cond ((s-data-frozen tem)
	 (setf (s-data-frozen def) t)))
  (unless (equalp def tem)
	  (warn "structure ~a is changing" name)
	  (setf (get name 's-data) def)))
(defun freeze-defstruct (name)
  (let ((tem (and (symbolp name) (get name 's-data))))
    (if tem (setf (s-data-frozen tem) t))))


;;; The #S reader.

(defun sharp-s-reader (stream subchar arg)
  (declare (ignore subchar))
  (when (and arg (null *read-suppress*))
        (error "An extra argument was supplied for the #S readmacro."))
  (let* ((l (prog1 (read stream t nil t)
	      (if *read-suppress*
		  (return-from sharp-s-reader nil))))
	 (sd
	   (or (get (car l) 's-data)
	       
	       (error "~S is not a structure." (car l)))))
    
    ;; Intern keywords in the keyword package.
    (do ((ll (cdr l) (cddr ll)))
        ((endp ll)
         ;; Find an appropriate construtor.
         (do ((cs (s-data-constructors sd) (cdr cs)))
             ((endp cs)
              (error "The structure ~S has no structure constructor."
                     (car l)))
           (when (symbolp (car cs))
                 (return (apply (car cs) (cdr l))))))
      (rplaca ll (intern (string (car ll)) 'keyword)))))


;; Set the dispatch macro.
(set-dispatch-macro-character #\# #\s 'sharp-s-reader)
(set-dispatch-macro-character #\# #\S 'sharp-s-reader)

;; Examples from Common Lisp Reference Manual.

#|
(defstruct ship
  x-position
  y-position
  x-velocity
  y-velocity
  mass)

(defstruct person name (age 20 :type signed-char) (eyes 2 :type signed-char)
							sex)
(defstruct person name (age 20 :type signed-char) (eyes 2 :type signed-char)
							sex)
(defstruct person1 name (age 20 :type fixnum)
							sex)

(defstruct joe a (a1 0 :type (mod  30)) (a2 0 :type (mod  30))
  (a3 0 :type (mod  30)) (a4 0 :type (mod 30)) )

;(defstruct person name age sex)

(defstruct (astronaut (:include person (age 45 :type fixnum))
                      (:conc-name astro-))
  helmet-size
  (favorite-beverage 'tang))

(defstruct (foo (:constructor create-foo (a
                                          &optional b (c 'sea)
                                          &rest d
                                          &aux e (f 'eff))))
  a (b 'bee) c d e f)

(defstruct (binop (:type list) :named (:initial-offset 2))
  (operator '?)
  operand-1
  operand-2)

(defstruct (annotated-binop (:type list)
                            (:initial-offset 3)
                            (:include binop))
  commutative
  associative
  identity)

|#
