; lispserver.lsp         Gordon S. Novak Jr.             ; 26 Jan 06

; Copyright (c) 2006 Gordon S. Novak Jr. and The University of Texas at Austin.

; 06 Jun 02

; See the file gnu.license .

; This program is free software; you can redistribute it and/or modify
; it under the terms of the GNU General Public License as published by
; the Free Software Foundation; either version 1, or (at your option)
; any later version.

; This program is distributed in the hope that it will be useful,
; but WITHOUT ANY WARRANTY; without even the implied warranty of
; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
; GNU General Public License for more details.

; You should have received a copy of the GNU General Public License
; along with this program; if not, write to the Free Software
; Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

; Written by: Gordon S. Novak Jr., Department of Computer Sciences,
; University of Texas at Austin  78712.    novak@cs.utexas.edu

;------------------------------------------------------------------------

; This is an example of a simple interactive graphical interface
; to a Lisp program.  It reads Lisp expressions from the user,
; evaluates them, and prints the result.

; Stand-alone usage using XGCL (edit file paths as appropriate):
; (load "/u/novak/X/xgcl-2/dwsyms.lsp")
; (load "/u/novak/X/xgcl-2/dwimports.lsp")
; (load "/u/novak/X/solaris/dwtrans.o")
; (load "/u/novak/glisp/menu-settrans.lsp")
; (load "/u/novak/glisp/lispservertrans.lsp")
; (lisp-server)

; Usage with the WeirdX Java emulation of an X server begins with
; the web page example.html and uses the files lispserver.cgi ,
; nph-lisp-action.cgi , and lispdemo.lsp .

;------------------------------------------------------------------------

(defvar *wio-window*           nil)
(defvar *wio-window-width*     500)
(defvar *wio-window-height*    300)
(defvar *wio-menu-set*         nil)
(defvar *wio-font* '8x13)

(glispglobals (*wio-window*           window)
	      (*wio-window-width*     integer)
	      (*wio-window-height*    integer)
	      (*wio-menu-set*         menu-set) )

(defmacro while (test &rest forms)
  `(loop (unless ,test (return)) ,@forms) )

; 18 Apr 95; 20 Apr 95; 08 May 95; 31 May 02
; Make a window to use.
(setf (glfnresulttype 'wio-window) 'window)
(defun wio-window (&optional title width height (posx 0) (posy 0) font)
  (if width (setq *wio-window-width* width))
  (if height (setq *wio-window-height* height))
  (or *wio-window*
      (setq *wio-window*
	    (window-create *wio-window-width* *wio-window-height* title
			   nil posx posy font))) )

; 19 Apr 95
(defun wio-init-menus (w commands)
  (let ()
    (window-clear w)
    (setq *wio-menu-set* (menu-set-create w nil))
    (menu-set-add-menu *wio-menu-set* 'command nil "Commands"
		       commands (list 0 0))
    (menu-set-adjust *wio-menu-set* 'command 'top nil 2)
    (menu-set-adjust *wio-menu-set* 'command 'right nil 2)
    ))

; 19 Apr 95; 20 Apr 95; 25 Apr 95; 02 May 95; 29 May 02
; Lisp server example
(gldefun lisp-server ()
  (let (w inputm done sel (redraw t) str result)
    (w = (wio-window "Lisp Server"))
    (open w)
    (clear w)
    (set-font w *wio-font*)
    (wio-init-menus w '(("Quit" . quit)))
    (window-print-lines w
      '("Click mouse in the input box, then enter"
	"a Lisp expression followed by Return."
	""
	"Input:   e.g.  (+ 3 4)  or  (sqrt 2)")
      10 (- *wio-window-height* 20))
    (window-printat-xy w "Result:" 10 (- *wio-window-height* 150))
    (inputm = (textmenu-create (- *wio-window-width* 100) 30 nil w
				 20 (- *wio-window-height* 110) t t '9x15 t))
    (add-item *wio-menu-set* 'input nil inputm)
    (while ~ done do
      (sel = (menu-set-select *wio-menu-set* redraw))
      (redraw = nil)
      (case (menu-name sel)
	(command
	  (case (port sel)
	    (quit    (done = t))
	    ))
	(input (str = (port sel))
	       (result = (catch 'error
			     (eval (safe-read-from-string str))))
	       (erase-area-xy w 20 2 (- *wio-window-width* 20)
			      (- *wio-window-height* 160))
	       (window-print-line w (write-to-string result :pretty t)
				  20 (- *wio-window-height* 170)))
	) )
    (close w)
    ))

; 25 Apr 95; 14 Mar 01
(defun safe-read-from-string (str)
  (if (and (stringp str) (> (length str) 0))
      (read-from-string str nil 'read-error)))

(defun compile-lispserver ()
  (glcompfiles *directory*
	       '("glisp/vector.lsp")   ; auxiliary files
               '("glisp/lispserver.lsp")      ; translated files
	       "glisp/lispservertrans.lsp")       ; output file
  )
