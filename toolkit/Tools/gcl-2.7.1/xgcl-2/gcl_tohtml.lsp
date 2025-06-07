;  tohtml.lsp        Gordon S. Novak Jr.        ; 13 Jan 06

; Translate LaTex file to HTML web pages

; Make table of contents for LaTex files of slides

; Copyright (c) 2006 Gordon S. Novak Jr. and The University of Texas at Austin.

; This program is free software; you can redistribute it and/or modify
; it under the terms of the GNU General Public License as published by
; the Free Software Foundation; either version 2 of the License, or
; (at your option) any later version.

; This program is distributed in the hope that it will be useful,
; but WITHOUT ANY WARRANTY; without even the implied warranty of
; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
; GNU General Public License for more details.

; You should have received a copy of the GNU General Public License
; along with this program; if not, write to the Free Software
; Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

; 21 Aug 00; 07 Sep 00; 11 Sep 00; 07 Dec 00; 24 Jul 02; 25 Jul 02; 29 Jul 02
; 12 Feb 03; 28 Aug 03; 29 Aug 03; 15 Jan 04; 11 May 04; 29 Aug 05

; This program converts a LaTeX file into one or more HTML files.
; The HTML file may need some minor hand editing.

; The program produces a new file in response to \pagebreak
; and puts in links to other pages.

; I have used it to put class lecture slides on the web;
; see http://www.cs.utexas.edu/users/novak/cs375contents.html
; See the README for notes on how this is all created.
; See also the file index.lsp for making indexes.

; To use:
; Start Lisp: e.g. /p/bin/gcl
; (load "tohtml.lsp")

; To translate LaTeX to HTML web pages:
; (tohtml "myfile.tex" "myprefix" <pagenumber>)
; where "myfile.tex"  = LaTeX file
;       "myprefix"    = file name prefix for HTML files
;       <pagenumber>  = number of first page if not 1
;                       \setcounter{page} will override this

; To make contents:
; The contents program looks for header lines, which
; in my files look something like:
;      \begin{center} {\bf Lexical Analysis} \end{center}

; (makecont "myfile.tex" <pagenumber> <html>)
; where "myfile.tex"  = LaTeX file
;       <pagenumber>  = number of first page if not 1
;       <html>        = t for html output, nil for LaTeX output


; 22 Aug 97; 28 Apr 00; 07 Aug 00; 08 Aug 00; 17 Aug 00; 18 Aug 00; 07 Dec 00
; 24 Jul 02; 26 Aug 03; 28 Aug 03; 11 Jan 05
; Make a contents list for a file of LaTex slides
; n is first page number: required if first page is not 1.
; html is prefix string to make html contents
(in-package "XLIB")

(defvar *line*)
(defvar *ptr*)
(defvar *lng*)

(defun makecont (filenm &optional (n 1) html)
  (let (line ptr lng done depth pagebr lastbr doit (first t))
    (with-open-file (infile filenm :direction :input
				:if-does-not-exist nil)
      (while (not (or (null infile)
		      (eq (setq line (read-line infile nil 'zzeofvalue))
			  'zzeofvalue) ))
	(setq lng (length line))
	(setq lastbr pagebr)
	(setq pagebr
	      (and (>= lng 10)
		   (string= line "\\pagebreak" :end1 10)))
	(if (and pagebr (not first)) (incf n))
	(when (and (> lng 18)
		   (string= line "\\setcounter{page}{" :end1 18))
	  (setq *line* line)
	  (setq *lng* lng)
	  (setq *ptr* 18)
	  (setq n (parse-int)))
	(when (and (> lng 20)
		   (string= line "\\addtocounter{page}{" :end1 20))
	  (setq *line* line)
	  (setq *lng* lng)
	  (setq *ptr* 20)
	  (setq n (+ n (parse-int))) )
	(setq doit nil)
	(if (and (> lng 30)
		 (or (string= line "\\begin{center} {\\bf " :end1 20)
		     (string= line "\\begin{center}  {\\bf " :end1 21)))
	    (progn (setq doit t) (setq ptr 20)) )
	(if (and (> lng 6) lastbr
		 (string= line "{\\bf " :end1 5))
	    (progn (setq doit t) (setq ptr 5)) )
	(when doit
	  (setq first nil)
	  (if html
	      (format t "<a href=\"~A~D.html\">~D. " html n n))
	  (setq lng (length line))
	  (setq done nil)
	  (setq depth 0)
	  (if (char= (char line ptr) #\Space) (incf ptr))
	  (while (and (< ptr lng) (not done))
	    (if (char= (char line ptr) #\\)
		(if (string= line "\\index" :start1 ptr :end1 (min lng (+ ptr 6)))
		    (progn (while (and (< ptr lng)
				       (not (char= (char line ptr) #\})))
			     (incf ptr))
			   (incf ptr))))
	    (if (char= (char line ptr) #\{)
		(progn (incf depth) (princ (char line ptr)))
	        (if (char= (char line ptr) #\})
		    (if (> depth 0)
			(progn (decf depth) (princ (char line ptr)))
		        (setq done t))
		    (princ (char line ptr))) )
	    (incf ptr))
	  (if html
	      (format t "</a><P>~%")
	      (format t "~60T& ~D \\\\~%" n))  ) ) ) ))

(defvar *prefix* "")
(defvar *outdir* "")
(defvar *feof* nil)
(defvar *done* nil)
(defvar *pagenumber* 0)
(defvar *firstpage* 1)
(defvar *lastpage* 999)
(defvar *center* nil)
(defvar *modestack* nil)
(defvar *verbatim* nil)
(defvar *ignore* t)
(defvar *specials* nil)
; &notin &there4 &nsub &copy &deg
(setq *specials* '(("pm" "&plusmn") ("cdot" "&middot") ("cap" "&cap")
 ("cup" "&cup") ("vee" "&or") ("wedge" "&and") ("leq" "&le") ("geq" "&ge")
 ("subset" "&sub") ("subseteq" "&sube") ("supset" "&sup")
 ("supseteq" "&supe") ("in" "&isin") ("perp" "&perp") ("cong" "&cong")
 ("sim" "&tilde") ("neq" "&ne") ("mid" "|") ("leftarrow" "&larr")
 ("rightarrow" "&rarr") ("leftrightarrow" "&harr") ("Leftarrow" "&lArr")
 ("Rightarrow" "&rArr") ("Leftrightarrow" "&hArr") ("uparrow" "&uarr")
 ("downarrow" "&darr") ("surd" "&radic ") ("emptyset" "&empty")
 ("forall" "&forall") ("exists" "&exist") ("neg" "&not") ("Box" "&#9633")
 ("models" "&#8872") ("vdash" "&#8866")
 ("filledBox" "&#9632") ("sum" "&sum") ("prod" "&prod") ("int" "&int")
 ("infty" "&infin") ("times" "X") ("sqrt" "&radic ") ("ll" "&lt &lt ")
 ("alpha" "&alpha") ("beta" "&beta") ("gamma" "&gamma") ("delta" "&delta")
 ("epsilon" "&epsilon") ("zeta" "&zeta") ("eta" "&eta") ("theta" "&theta")
 ("iota" "&iota") ("kappa" "&kappa") ("lambda" "&lambda") ("mu" "&mu")
 ("nu" "&nu") ("xi" "&xi") ("pi" "&pi") ("rho" "&rho") ("sigma" "&sigma")
 ("tau" "&tau") ("upsilon" "&upsilon") ("phi" "&phi") ("chi" "&chi")
 ("psi" "&psi") ("omega" "&omega")
 ("Alpha" "&Alpha") ("Beta" "&Beta") ("Gamma" "&Gamma") ("Delta" "&Delta")
 ("Epsilon" "&Epsilon") ("Zeta" "&Zeta") ("Eta" "&Eta") ("Theta" "&Theta")
 ("Iota" "&Iota") ("Kappa" "&Kappa") ("Lambda" "&Lambda") ("Mu" "&Mu")
 ("Nu" "&Nu") ("Xi" "&Xi") ("Pi" "&Pi") ("Rho" "&Rho") ("Sigma" "&Sigma")
 ("Tau" "&Tau") ("Upsilon" "&Upsilon") ("Phi" "&Phi") ("Chi" "&Chi")
 ("Psi" "&Psi") ("Omega" "&Omega") ("vert" "|")
) )

; 28 Apr 00; 07 Aug 00
; Translate a file of LaTex slides to HTML
; prefix is a prefix string for output files
; pagenumber is first page number.
(defun tohtml (filenm prefix &optional (pagenumber 1) (outdir prefix))
  (let (c)
    (setq *pagenumber* pagenumber)
    (setq *prefix* (stringify prefix))
    (setq *outdir* (stringify outdir))
    (setq *feof* nil)
    (setq *ignore* t)
    (setq *center* nil)
    (setq *modestack* nil)
    (setq *verbatim* nil)
    (with-open-file (infile filenm :direction :input :if-does-not-exist nil)
    ; skip initial stuff
      (while (and *ignore*
		  (not (or (null infile)
			   (eq (setq *line* (read-line infile nil 'zzeofvalue))
			       'zzeofvalue) )))
	(setq *lng* (length *line*))
	(setq *ptr* 0)
	(while (< *ptr* *lng*)
	  (setq c (char *line* *ptr*))
	  (incf *ptr*)
	  (if (and (char= c #\%) (not *verbatim*))
	      (flushline)
	    (if (char= c #\\)
		(if (alpha-char-p (safe-char))
		    (docommand nil) ) ) ) ) )
      (while (not *feof*) (dohtml infile)) ) ))

; 08 Aug 00; 18 Aug 00; 21 Aug 00; 07 Sep 00; 24 Jul 02; 25 Jul 02; 13 Jan 06
; Process input to produce one .html file
(defvar c)
(defun dohtml (infile)
  (let (c)
    (setq *done* nil)
    (with-open-file (outfile (concatenate 'string *outdir* *prefix*
					  (stringify *pagenumber*) ".html")
			     :direction :output :if-exists :supersede)
      (princ "<HTML> <HEAD> <TITLE>" outfile)
      (princ *prefix* outfile)
      (princ "  p. " outfile)
      (princ (stringify *pagenumber*) outfile)
      (princ "</TITLE> </HEAD>" outfile)
      (terpri outfile)
      (princ "<BODY>" outfile) (terpri outfile)
      (terpri outfile)
      (while (not (or *done* *feof*
		      (setq *feof*
			(eq (setq *line* (read-line infile nil 'zzeofvalue))
			    'zzeofvalue))))
	(doline outfile)
	(terpri outfile) )
      ; *pagenumber* is too large by 1 at this point...
      (if *feof* (incf *pagenumber*))
      (format outfile
	      "<a href=\"~Acontents.html\">Contents</a>&nbsp&nbsp&nbsp~%"
	      *prefix*)
      (if (>= *pagenumber* (+ *firstpage* 11))
	  (format outfile "<a href=\"~A~D.html\">Page-10</a>&nbsp&nbsp&nbsp~%"
		  *prefix* (- *pagenumber* 11)))
      (if (>= *pagenumber* (+ *firstpage* 2))
	  (format outfile "<a href=\"~A~D.html\">Prev</a>&nbsp&nbsp&nbsp~%"
		  *prefix* (- *pagenumber* 2)))
      (if (<= *pagenumber* *lastpage*)
	  (format outfile "<a href=\"~A~D.html\">Next</a>&nbsp&nbsp&nbsp~%"
		  *prefix* *pagenumber*))
      (if (<= *pagenumber* (- *lastpage* 9))
	  (format outfile "<a href=\"~A~D.html\">Page+10</a>&nbsp&nbsp&nbsp~%"
		  *prefix* (+ *pagenumber* 9)))
      (format outfile
	      "<a href=\"~Aindex.html\">Index</a>&nbsp&nbsp&nbsp~%" *prefix*)
      (princ "</BODY></HTML>" outfile) (terpri outfile)
      )
    ))

; 13 Jan 06
; process *line*
(defun doline (outfile)
  (let ()
	(setq *lng* (length *line*))
	(setq *ptr* 0)
	(if (and (= *lng* 0) (not *verbatim*))
	    (princ "<P>" outfile))
	(while (< *ptr* *lng*)
	  (setq c (char *line* *ptr*))
	  (incf *ptr*)
	  (if (and (char= c #\%) (not *verbatim*))
	      (flushline)
	    (if (char= c #\\)
		(if (alpha-char-p (setq c (safe-char)))
		    (docommand outfile)
		  (if (char= c #\\)
		      (progn (termline outfile) (incf *ptr*))
		    (if (char= c #\/)
			(progn (princ "&nbsp" outfile) (incf *ptr*))
		        (if (char= c #\[)
			    (progn (pushfont '$ outfile) (incf *ptr*))
			  (if (char= c #\])
			      (progn (popenv outfile) (incf *ptr*))
			    (progn (if *verbatim* (princ #\\ outfile))
				   (princ c outfile) (incf *ptr*)))))))
	      (if (char= c #\&)
		  (princ "</TD><TD>" outfile)
		(if (char= c #\{)
		    (if *verbatim*
			(princ #\{ outfile)
		        (pushenv nil))
		  (if (char= c #\})
		      (if *verbatim*
			  (princ #\} outfile)
			  (popenv outfile))
		    (if (and (char= c #\$) (not *verbatim*))
			(if (eq (car *modestack*) '$)
			    (popenv outfile)
			  (pushfont '$ outfile))
		      (if (and (or (char= c #\^) (char= c #\_))
			       (eq (car *modestack*) '$))
			  (progn
			    (pushfont (if (char= c #\^) 'sup 'sub) outfile)
			    (searchfor #\{))
		      (princ (if (char= c #\>) "&gt "
			       (if (char= c #\<) "&lt "
				 c))
			     outfile))))))))) ))

; 24 Jul 02; 25 Jul 02; 29 Jul 02; 12 Feb 03; 28 Aug 03
(defun docommand (outfile)
  (let (wordstring word subword termch done tmp c pair (saveptr (1- *ptr*)))
    (setq wordstring (car (parse-word nil)))
    (setq word (intern (string-upcase wordstring)))
    (case word
      ((documentstyle pagestyle setlength hyphenpenalty sloppy
	       large)
        (flushline))
      (setcounter (searchfor #\{)
		  (setq subword (intern (car (parse-word t))))
		  (when (eq subword 'page)
		    (searchfor #\{)
		    (setq *pagenumber* (1- (parse-int))) ; assumes pagebreak
		    (flushline)) )
      (addtocounter (searchfor #\{)
		  (setq subword (intern (car (parse-word t))))
		  (when (eq subword 'page)
		    (searchfor #\{)
		    (setq *pagenumber* (+ *pagenumber* (parse-int)))
		    (flushline)) )
      (includegraphics (searchfor #\{) (searchforalpha)
		       (setq done nil)
		       (while (not done)
			 (setq tmp (parse-word nil))
			 (if (char= (cadr tmp) #\})
			     (setq done t)
			   (if (char= (cadr tmp) #\.)
			       (progn (setq done t)
				      (princ "<IMG src=" outfile)
				      (princ #\" outfile)
				      (princ (car tmp) outfile)
				      (princ ".gif" outfile)
				      (princ #\" outfile)
				      (princ ">" outfile)
				      (terpri outfile)
				      (flushline) )
			     (incf *ptr*)))))
      (begin (searchfor #\{)
	     (setq subword (intern (car (parse-word t))))
	     (searchfor #\})
	 ;    (format t "subword = ~s~%" subword)
	(case subword
	  (document (setq *ignore* nil))
	  (center (pushenv 'center))
	  (itemize (princ "<UL>" outfile) (terpri outfile))
	  (enumerate (princ "<OL>" outfile) (terpri outfile))
	  (verbatim (princ "<PRE>" outfile) (terpri outfile)
		    (setq *verbatim* t))
	  (tabular (dotabular outfile))
	  ((quotation abstract quote)
	    (princ "<BLOCKQUOTE>" outfile) (terpri outfile))
	  ))
      (end (searchfor #\{)
	(setq subword (intern (car (parse-word t))))
	(searchfor #\})
	(case subword
	  (document (setq *feof* t))
	  (center (popenv outfile))
	  (itemize (princ "</UL>" outfile) (terpri outfile))
	  (enumerate (princ "</OL>" outfile) (terpri outfile))
	  (verbatim (princ "</PRE>" outfile) (terpri outfile)
		    (setq *verbatim* nil))
	  (tabular (princ "</TABLE>" outfile) (terpri outfile)
		   (popenv outfile))
	  ((quotation abstract quote)
	    (princ "</BLOCKQUOTE>" outfile) (terpri outfile))
	  ))
      (item (princ "<LI>" outfile))
      (pagebreak (setq *done* t) (incf *pagenumber*))
      ((bf tt em it) (pushfont word outfile))
      ((title section subsection subsubsection paragraph)
         (searchfor #\{)
         (pushfont (cadr (assoc word '((title h1) (section h2)
				       (subsection h3) (subsubsection h4)
				       (paragraph b))))
		   outfile))
      ((vspace vspace*) (searchfor #\})
        (princ "<P>" outfile) (terpri outfile))
      ((hspace hspace*) (searchfor #\})
        (dotimes (i 8) (princ "&nbsp" outfile)))
      ((index) (searchfor #\}))    ; ignore and consume
      (verb (setq termch (char *line* *ptr*))
	    (incf *ptr*)
	    (pushfont 'tt outfile)
	    (xferchars outfile termch)
	    (popenv outfile) )
      ((cite bibitem) (searchfor #\{)
	    (princ "[" outfile)
	    (xferchars outfile #\})
	    (princ "]" outfile) )
      (footnote (searchfor #\{)
		(princ "[" outfile)
		(pushenv 'footnote))
      (t (if *verbatim*
	     (while (< saveptr *ptr*)
	       (princ (char *line* saveptr) outfile)
	       (incf saveptr))
	     (if (setq pair (assoc wordstring *specials* :test #'string=))
		 (princ (cadr pair) outfile)) ) ) ) ))

; push a new item on the mode stack
(defun pushenv (item)
  (if (and *modestack* (eq (car *modestack*) nil))
      (setf (car *modestack*) item)
    (push item *modestack*)))

; 24 Jul 02; 25 Jul 02
(defun popenv (outfile)
  (let ((item (pop *modestack*)) new)
    (setq new (cadr (assoc item '((em i) (bf b) (it i) ($ i)))))
    (case item
      ((bf tt it em $ h1 h2 h3 h4 sub sup)
        (princ "</" outfile)
        (princ (or new item) outfile)
        (princ ">" outfile))
      (footnote (princ "]" outfile))
      )
    item))

(defun pushfont (word outfile)
  (let ((new (cadr (assoc word '((em i) (bf b) (it i) ($ i))))))
    (pushenv word)
    (princ "<" outfile) (princ (or new word) outfile)
    (princ ">" outfile) ))

; transfer chars to output until termch
(defun xferchars (outfile termch)
  (let (done)
    (while (and (< *ptr* *lng*) (not done))
      (setq c (char *line* *ptr*))
      (incf *ptr*)
      (if (char= c termch)
	  (setq done t)
	  (princ c outfile)) ) ))

(defun dotabular (outfile)
  (let ((ncols 0) done)
    (searchfor #\{)
    (while (and (< *ptr* *lng*) (not done))
      (setq c (char *line* *ptr*))
      (incf *ptr*)
      (if (char= c #\})
	  (setq done t)
	(if (or (char= c #\l) (char= c #\r) (char= c #\c))
	    (incf ncols))) )	  
    (princ "<TABLE>" outfile)
    (terpri outfile)
    (princ "<TR><TD>" outfile)
    (pushenv 'table)
    ))

(defun termline (outfile)
  (if (eq (car *modestack*) 'table)
      (progn (princ "</TD></TR>" outfile)
	     (terpri outfile)
	     (princ "<TR><TD>" outfile))
      (progn (princ "<BR>" outfile) (terpri outfile) )))

(defun safe-char ()
  (if (< *ptr* *lng*)
      (char *line* *ptr*)
      #\Space))

; Parse a word of alpha/num characters
; Returns ("word" ch) where ch is the terminating character
(defun parse-word (upper)
  (let (c res)
    (while (and (< *ptr* *lng*)
		(or (alpha-char-p (setq c (char *line* *ptr*)))
		    (and res (digit-char-p c))
		    (char= c #\*)))
      (push (if upper (char-upcase c) c) res)
      (incf *ptr*))
    (if res (list (coerce (nreverse res) 'string)
		  (and (not (alpha-char-p c)) c))) ))

(defun searchfor (ch)
  (let (c)
    (while (and (< *ptr* *lng*)
		(setq c (char *line* *ptr*))
		(not (char= ch c)))
      (incf *ptr*))
    (if (and c (char= ch c)) (incf *ptr*))
    c))

(defun searchforalpha ()
  (while (and (< *ptr* *lng*)
	      (not (alpha-char-p (char *line* *ptr*))))
    (incf *ptr*)))

(defun flushline () (setq *lng* 0))

(defun stringify (x)
  (cond ((stringp x) x)
        ((symbolp x) (symbol-name x))
	(t (princ-to-string x))))

; Parse an integer
(defun parse-int ()
  (let (c (n 0) digit found)
    (while (and (< *ptr* *lng*)
		(setq digit (digit-char-p
			     (setq c (char *line* *ptr*)))))
      (setq found (or found digit))
      (setq n (+ (* n 10) digit))
      (incf *ptr*))
    (if found n) ))
