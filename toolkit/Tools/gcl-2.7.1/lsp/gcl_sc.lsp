;; Copyright (C) 2024 Camm Maguire
;to-do fast link defentry, sig propagation

;; (in-package 'lisp)
;; (export '(string char schar string= string/= string> string>= 
;; 		 string< string<= string-equal string-not-equal
;; 		 string-greaterp string-not-lessp string-lessp
;; 		 string-not-greaterp  char-code  code-char  char-upcase
;; 		 char-downcase  char=  char/=  char>  char>=  char<
;; 		 char<=  char-equal  char-not-equal  char-greaterp
;; 		 char-lessp  char-not-greaterp  char-not-lessp
;; 		 upper-case-p  lower-case-p  both-case-p
;; 		 string-upcase string-downcase nstring-upcase nstring-downcase
;; 		 string-trim string-left-trim string-right-trim))

(in-package :si)

(defun symbol-name-length-one-p (x)
  (eql 1 (length (symbol-name x))))

(deftype character-designator nil `(or character (integer 0 255) (array character (1));FIXME deftype.lsp
				       (and symbol (satisfies symbol-name-length-one-p))))

(eval-when
 (compile eval)

 (defmacro with-aref-shadow (&body body)
   `(labels ((lower-case-p (x) (<= #.(char-code #\a) x #.(char-code #\z)))
	     (upper-case-p (x) (<= #.(char-code #\A) x #.(char-code #\Z)))
	     (char-upcase (x) 
	       (if (lower-case-p x)
		   (+ x #.(- (char-code #\A) (char-code #\a))) x))
	     (char-downcase (x) 
	       (if (upper-case-p x)
		   (+ x #.(- (char-code #\a) (char-code #\A))) x))
	     (aref (s i) (*uchar (c-array-self s) i nil nil))
	     (aset (v s i) (*uchar (c-array-self s) i t v))
	     (char= (x z) (= x z))
	     (char< (x z) (< x z))
	     (char> (x z) (> x z))
	     (char-equal (x z) (or (= x z) (= (char-upcase x) (char-upcase z))))
	     (char-greaterp (x z) (> (char-upcase x) (char-upcase z)))
	     (char-lessp    (x z) (< (char-upcase x) (char-upcase z))))
      (declare (ignorable #'lower-case-p #'upper-case-p #'char-upcase #'char-downcase #'aref #'aset
			  #'char= #'char< #'char> #'char-equal #'char-greaterp #'char-lessp))
      ,@body))

(defmacro defstr (name (s1 s2) = &body body)
   `(defun ,name (,s1 ,s2  &key (start1 0) end1 (start2 0) end2)
      (declare (optimize (safety 1)))
      (check-type s1 string-designator)
      (check-type s2 string-designator)
      (check-type start1 seqind)
      (check-type end1 (or null seqind))
      (check-type start2 seqind)
      (check-type end2 (or null seqind))
      (with-aref-shadow
       (let* ((s1 (string s1))
	      (s2 (string s2))
	      (l1 (length s1))
	      (l2 (length s2))
	      (e1 end1)(c1 0)
	      (e2 end2)(c2 0)
	      (end1 (or end1 l1))
	      (end2 (or end2 l2)))
	 (declare (ignorable c1 c2))
	 (unless (if e1 (<= start1 end1 l1) (<= start1 l1))
	   (error 'type-error "Bad array bounds"))
	 (unless (if e2 (<= start2 end2 l2) (<= start2 l2))
	   (error 'type-error "Bad array bounds"))
	 (do ((i1 start1 (1+ i1))
	      (i2 start2 (1+ i2)))
	     ((or (>= i1 end1) (>= i2 end2) (not (,= (setq c1 (aref s1 i1)) (setq c2 (aref s2 i2)))))
	      ,@body)
	   (declare (seqbnd i1 i2)))))));FIXME


 (defmacro defchr (n (comp key))
   `(defun ,n (c1 &optional (c2 c1 c2p) &rest r) 
      (declare (optimize (safety 1)) (list r) (dynamic-extent r));fixme
      (check-type c1 character)
      (or (not c2p)
	  (when (,comp (,key c1) (,key c2))
	    (or (null r) (apply ',n c2 r))))))
 
 (defmacro defnchr (n (test key))
   `(defun ,n (c1 &rest r) 
      (declare (optimize (safety 1)) (list r) (dynamic-extent r));fixme
      (check-type c1 character)
      (cond ((null r))
	    ((member (,key c1) r :test ',test :key ',key) nil)
	    ((apply ',n r)))))


 (defmacro defstr1 (n query case &optional copy)
   `(defun ,n (s &key (start 0) end)
      (declare (optimize (safety 1)))
      (check-type s ,(if copy 'string-designator 'string))
      (check-type start seqind)
      (check-type end (or null seqind))
      (with-aref-shadow
       (flet ((cpy (s l)
		   (let ((n (make-array l :element-type 'character)))
		     (do ((j 0 (1+ j))) ((>= j l) n)
			 (aset (aref s j) n j)))))
	    (let* ((s (string s))
		   (l (length s))
		   (e end)
		   (end (or end l))
		   (n ,(let ((x `(cpy s l))) (if copy x `(if (stringp s) s ,x)))))
	      (unless (if e (<= start end l) (<= start l)) 
		(error 'type-error "Bad sequence bounds"))
	      (do ((i start (1+ i))) ((>= i end) n)
		  (let ((ch (aref s i)))
		    (unless (,query ch)
		      (aset (,case ch) n i))))))))))

(defun character (c)
  (declare (optimize (safety 1)))
  (check-type c character-designator)
  (typecase
   c
   (character c)
   (unsigned-char (code-char c))
   (otherwise (char (string c) 0))))


(defun char-int (c)
  (declare (optimize (safety 1)))
  (check-type c character-designator)
  (char-code c))

(defun int-char (c)
  (declare (optimize (safety 1)))
  (check-type c character-designator)
  (code-char c))

(defun char-name (c)
  (declare (optimize (safety 1)))
  (check-type c character)
  (let ((c (char-code c)))
    (case c
	  (#.(char-code #\Return) "Return")
	  (#.(char-code #\Space) "Space")
	  (#.(char-code #\Rubout) "Rubout")
	  (#.(char-code #\Page) "Page")
	  (#.(char-code #\Tab) "Tab")
	  (#.(char-code #\Backspace) "Backspace")
	  (#.(char-code #\Newline) "Newline")
	  (otherwise
	   (let ((ch (code-char c)))
	     (unless (graphic-char-p ch)
	       (subseq (with-output-to-string (s) (prin1 ch s)) 2)))))))


(defun name-char (sd &aux (s (string sd)))
  (declare (optimize (safety 1)))
  (check-type sd string-designator)
  (cond ((cdr (assoc s '(("Return" . #\Return)
			 ("Space" . #\Space)
			 ("Rubout" . #\Rubout)
			 ("Page" . #\Page)
			 ("Tab" . #\Tab)
			 ("Backspace" . #\Backspace)
			 ("Newline" . #\Newline)
			 ("Linefeed" . #\Newline))
		     :test 'string-equal)))
	((let ((l (length s)))
	   (case l
	     (1 (aref s 0))
	     (2 (when (char= #\^ (aref s 0))
		  (code-char (- (char-code (aref s 1)) #.(- (char-code #\A) 1)))))
	     (3 (when (and (char= #\^ (aref s 0)) (char= #\\ (aref s 2)))
		      (code-char (- (char-code (aref s 1)) #.(- (char-code #\A) 1)))))
	     (4 (when (char= #\\ (aref s 0))
		  (code-char
		   (+ (* 64 (- (char-code (aref s 1)) #.(char-code #\0)))
		      (* 8 (- (char-code (aref s 2)) #.(char-code #\0)))
		      (- (char-code (aref s 3)) #.(char-code #\0)))))))))))

		
   

(setf (symbol-function 'char-code) (symbol-function 'c-character-code))

(defun code-char (d)
;  (declare (optimize (safety 1)))
  (typecase d
	    (unsigned-char
	     (let ((b #.(1- (integer-length (- (address #\^A) (address #\^@))))))
	       (the character (nani (c+ (address #\^@) (ash d b))))))));FIXME


(defchr char=  (= address))
(defchr char>  (> address))
(defchr char>= (>= address))
(defchr char<  (< address))
(defchr char<= (<= address))

(defchr char-equal        (char= char-upcase))
(defchr char-greaterp     (char> char-upcase))
(defchr char-lessp        (char< char-upcase))
(defchr char-not-greaterp (char<= char-upcase))
(defchr char-not-lessp    (char>= char-upcase))

(defnchr char/=         (= address))
(defnchr char-not-equal (char-equal identity))

(defun upper-case-p (c)
  (declare (optimize (safety 1)))
  (check-type c character)
  (char>= #\Z c #\A))

(defun lower-case-p (c)
  (declare (optimize (safety 1)))
  (check-type c character)
  (char>= #\z c #\a))

(defun both-case-p (c)
  (declare (optimize (safety 1)))
  (check-type c character)
  (or (upper-case-p c) (lower-case-p c)))

(defun char-upcase (c)
  (declare (optimize (safety 1)))
  (check-type c character)
  (if (lower-case-p c)
      (nani (+ (address c) #.(- (address #\A) (address #\a))))
    c))

(defun char-downcase (c)
  (declare (optimize (safety 1)))
  (check-type c character)
  (if (upper-case-p c)
      (nani (+ (address c) #.(- (address #\a) (address #\A))))
    c))

(defun alphanumericp (c)
  (declare (optimize (safety 1)))
  (check-type c character)
  (or (char<= #\0 c #\9)
      (alpha-char-p c)))

(defun alpha-char-p (c)
  (declare (optimize (safety 1)))
  (check-type c character)
  (both-case-p c))

(defun digit-char-p (c &optional (r 10))
  (declare (optimize (safety 1)))
  (check-type c character)
  (check-type r (integer 0))
  (when (typep r 'fixnum)
    (let* ((r r)(r (1- r))(i (char-code c))
	   (j (- i #.(char-code #\0)))
	   (k (- i #.(- (char-code #\a) 10)))
	   (l (- i #.(- (char-code #\A) 10))))
      (cond ((and (<=  0 j r) (<= j 9)) j);FIXME infer across inlines
	    ((<= 10 k r 36) k)
	    ((<= 10 l r 36) l)))))

(defun digit-char (w &optional (r 10))
  (declare (optimize (safety 1)))
  (check-type w (integer 0))
  (check-type r (integer 0))
  (when (and (typep w 'fixnum) (typep r 'fixnum))
    (let ((w w)(r r))
      (when (< w r)
	(code-char (if (< w 10) (+ w #.(char-code #\0)) (+ w #.(- (char-code #\A) 10))))))))

(defun graphic-char-p (c)
  (declare (optimize (safety 1)))
  (check-type c character)
  (char<= #\Space c #\~))

(defun standard-char-p (c)
  (declare (optimize (safety 1)))
  (check-type c character)
  (or (graphic-char-p c) (char= c #\Newline)))

(defun string (x)
  (declare (optimize (safety 1)))
  (check-type x string-designator)
  (typecase 
   x
   (string x)
   (symbol (symbol-name x))
   (character (c-character-name x))
   ((integer 0 255) (string (code-char x)))))



(defstr1  string-upcase   upper-case-p char-upcase   t)
(defstr1  string-downcase lower-case-p char-downcase t)
(defstr1 nstring-upcase   upper-case-p char-upcase)
(defstr1 nstring-downcase lower-case-p char-downcase)

(defstr string= (s1 s2) char=
  (and (>= i1 end1) (>= i2 end2)))

(defstr string/= (s1 s2) char=
  (unless (and (>= i1 end1) (>= i2 end2)) i1))

(defstr string> (s1 s2) char=
  (cond ((>= i1 end1) nil)
	((>= i2 end2) i1)
	((char> c1 c2) i1)))

(defstr string>= (s1 s2) char=
  (cond ((>= i2 end2) i1)
	((>= i1 end1) nil)
	((char> c1 c2) i1)))

(defstr string< (s1 s2) char=
  (cond ((>= i2 end2) nil)
	((>= i1 end1) i1)
	((char< c1 c2) i1)))

(defstr string<= (s1 s2) char=
  (cond ((>= i1 end1) i1)
	((>= i2 end2) nil)
	((char< c1 c2) i1)))


(defstr string-equal (s1 s2) char-equal
  (and (>= i1 end1) (>= i2 end2)))

(defstr string-not-equal (s1 s2) char-equal
  (unless (and (>= i1 end1) (>= i2 end2)) i1))

(defstr string-greaterp (s1 s2) char-equal
  (cond ((>= i1 end1) nil)
	((>= i2 end2) i1)
	((char-greaterp c1 c2) i1)))

(defstr string-not-lessp (s1 s2) char-equal
  (cond ((>= i2 end2) i1)
	((>= i1 end1) nil)
	((char-greaterp c1 c2) i1)))

(defstr string-lessp (s1 s2) char-equal
  (cond ((>= i2 end2) nil)
	((>= i1 end1) i1)
	((char-lessp c1 c2) i1)))

(defstr string-not-greaterp (s1 s2) char-equal
  (cond ((>= i1 end1) i1)
	((>= i2 end2) nil)
	((char-lessp c1 c2) i1)))



(defun string-left-trim (b s)
  (declare (optimize (safety 1)))
  (check-type b sequence)
  (let ((s (string s)))
    (do ((l (length s))
	 (i 0 (1+ i)))
	((or (>= i l) (not (find (aref s i) b)))
	 (if (= i 0) s (subseq s i))))))
      

(defun string-right-trim (b s)
  (declare (optimize (safety 1)))
  (check-type b sequence)
  (let* ((s (string s))
	 (l (length s)))
    (do ((i (1- l) (1- i)))
	((or (< i 0) (not (find (aref s i) b)))
	 (if (= i l) s (subseq s 0 (1+ i)))))))

(defun string-trim (b s)
  (declare (optimize (safety 1)))
  (check-type b sequence)
  (let* ((s (string s))
	 (l (length s)))
    (do ((i 0 (1+ i)))
	((or (>= i l) (not (find (aref s i) b)))
	 (do ((j (1- l) (1- j)))
	     ((or (< j i) (not (find (aref s j) b)))
	      (if (and (= i 0) (= j l)) s (subseq s i (1+ j)))))))))


;FIXME
;; (defun interpreted-function-p (x) 
;;   (typecase x (interpreted-function t)))

;; (defun seqindp (x)
;;   (typecase x (seqind t)))

(declaim (inline fixnump))
(defun fixnump (x)
  (typecase x (fixnum t)))

(declaim (inline spicep))
(defun spicep (x)
  (typecase x (spice t)))


(defun constantp (x &optional env)
  (declare (ignore env))
  (typecase 
   x
   (symbol (= 1 (c-symbol-stype x)))
   (cons (eq 'quote (car x)))
   (otherwise t)))

;; FIXME these functions cannot be loaded interpreted, cause an infinite loop on typep/fsf

(defun functionp (x)
  (typecase x (function t)))

(defun compiled-function-p (x) 
  (typecase x (compiled-function t)))

(defun stringp (x)
  (typecase x (string t)))
