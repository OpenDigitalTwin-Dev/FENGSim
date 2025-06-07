; index.lsp       Gordon S. Novak Jr.       08 Dec 00; 18 May 06

; This program processes LaTeX index entries, printing an index in
; either LaTeX or HTML form.

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


; To use: Gather the LaTeX index data: use \index{foo} within the
; text and include a \makeindex command at the top of the file,
; producing a file <file>.idx when the file is run through LaTeX.
; Use an editor to change the index data from LaTeX form to Lisp:
; \indexentry{combination}{37}     LaTeX
; ((combination) 37)               Lisp

; We assume that indexdata is a list of such entries, as illustrated
; at the end of this file.

; Warning: quote characters or apostrophes within the indexed
; entries will not read into Lisp as expected.
; Get rid of ' or change it to \'

; Start /p/bin/gcl
; (load "index.lsp")
; (printindex indexdata)          ; for LaTeX output
; (printindex indexdata "prefix") ; for HTML output
;    where "prefix" is the file name prefix for HTML files.

; Print index for LaTeX given a list of items ((words ...) page-number)
(in-package "XLIB")
(defun printindex (origlst &optional html)
  (let (lst top)
    (setq lst
	  (sort origlst
		#'(lambda (x y) (or (wordlist< (car x) (car y))
				    (and (equal (car x) (car y))
					 (< (cadr x) (cadr y)))))))
    (terpri)
    (while lst
      (setq top (pop lst))
      (if (not html)
	  (princ "\\item "))
      (dolist (word (car top))
	(princ (string-downcase (symbol-name word))) (princ " "))
      (printindexn (cadr top) html nil)
      (while (equal (caar lst) (car top))
	(setq top (pop lst))
	(printindexn (cadr top) html t) )
      (if html
	  (format t "<P>~%")
	  (terpri)) ) ))

(defun wordlist< (x y)
  (and (consp x) (consp y)
       (or (string< (symbol-name (car x))
		    (symbol-name (car y)))
	   (and (eq (car x) (car y))
		(or (and (null (cdr x)) (cdr y))
		    (and (cdr x) (cdr y)
			 (wordlist< (cdr x) (cdr y))))))))

(defun printindexn (n html comma)
  (if comma (princ ", "))
  (if html
      (format t "<a href=\"~A~D.html\">~D</a>" html n n)
      (princ n)) )

(setq indexdata '(

; Insert index entry data here.  Data should look like:
; ((isomorphism) 20)
; ((artificial intelligence) 30)

))
