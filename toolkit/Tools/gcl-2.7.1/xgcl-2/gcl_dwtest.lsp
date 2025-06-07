; dwtest.lsp             Gordon S. Novak Jr.                 10 Jan 96

; Some examples for testing the window interface in dwindow.lsp / dwtrans.lsp

; Copyright (c) 1996 Gordon S. Novak Jr. and The University of Texas at Austin.
; Copyright (c) 2024 Camm Maguire

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

(use-package :xlib)
(defun user::xgcl-demo nil
 (wtesta)
 (wtestb)
 (format t "Try (wtestc) ... (wtestk) for more examples."))

(defmacro while (test &rest forms)
  `(loop (unless ,test (return)) ,@forms) )

(defvar *myw*)  ; my window
(defvar myw)

; Make a window to play in.
(defun wtesta ()
  (setq myw (setq *myw* (window-create 300 300 "test window"))) )

; 15 Aug 91; 12 Sep 91; 05 Oct 94; 06 Oct 94
; Draw some basic things in the window
(defun wtestb ()
  (window-clear *myw*)
  (window-draw-box-xy *myw* 50 50 50 20 1)
  (window-printat *myw* "howdy" '(58 55))
  (window-draw-line *myw* '(100 70) '(200 170))
  (window-draw-arrow-xy *myw* 200 170 165 205)
  (window-draw-circle-xy *myw* 200 170 50 2)
  (window-draw-ellipse-xy *myw* 100 170 40 20 1)
  (window-printat-xy *myw* "ellipse" 70 165)
  (window-draw-arc-xy *myw* 100 250 20 20 0 90 1)
  (window-draw-arc-xy *myw* 100 250 20 20 0 -90 1)
  (window-printat-xy *myw* "arcs" 80 244)
  (window-printat-xy *myw* "invert" 54 200)
  (window-invert-area-xy *myw* 50 160 60 60)
  (window-copy-area-xy *myw* 40 150 200 50 60 40)
  (window-printat-xy *myw* "copy" 210 100)
  (window-set-color-rgb *myw* 65535 0 0)       ; red foreground
  (window-printat-xy *myw* "Red" 20 20)
  (window-draw-rcbox-xy *myw* 15 15 32 20 5)
  (window-set-color-rgb *myw* 0 0 65535 t)     ; blue background
  (window-set-color-rgb *myw* 0 65535 0)       ; green foreground
  (window-printat-xy *myw* "Green" 120 20)
  (window-set-color-rgb *myw* 0 65535 0 t)     ; green background
  (window-set-color-rgb *myw* 0 0 65535)       ; blue foreground
  (window-printat-xy *myw* "Blue" 220 20)
  (window-reset-color *myw*)
  (window-force-output *myw*) )

; 15 Aug 91; 19 Aug 91; 03 Sep 91; 21 Apr 95
; Illustrate mouse interaction:
; click in window *myw* (2 times for line, 3 times for region).
(defun wtestc ()
  (let (mymenu result start done)
    (setq mymenu (menu-create '(quit point line box region) "Choose One:"))
    (while (not done)
      (setq result
	    (case (menu-select mymenu)
	      (quit   (setq done t))
	      (point  (window-get-point *myw*))
	      (line   (setq start (window-get-point *myw*))
		      (list start
			    (window-get-line-position *myw* (car start)
						            (cadr start))))
	      (box    (window-get-box-position *myw* 40 20))
	      (region (window-get-region *myw*)) ))
      (format t "Result: ~A~%" result) )
    (menu-destroy mymenu) ))

; 09 Sep 91
; Illustrate icons in menus
(defun wtestd ()
  (menu '(("Triangle" . triangle)
	  (dwtest-square . square)
	  (dwtest-circle . circle)
	  hexagon)
	"Icons in Menu") )

(defun dwtest-square (w x y)  (window-draw-box-xy w x y 20 20 1))
(setf (get 'dwtest-square 'display-size) '(20 20))

(defun dwtest-circle (w x y)  (window-draw-circle-xy w (+ x 10) (+ y 10) 10 1))
(setf (get 'dwtest-circle 'display-size) '(20 20))

(defvar mypms nil)
; 09 Sep 91; 11 Sep 91; 12 Sep 91; 14 Sep 91
; Illustrate a diagrammatic menu-like object: square with sensitive spots
(defun wteste ()
  (let (pm val)
    (or mypms (mypms-init))
    (setq pm (picmenu-create-from-spec mypms "Points on Square"))
    (setq val (picmenu-select pm))
    (picmenu-destroy pm)
    val ))

; 14 Sep 91
(defun mypms-init ()
  (setq mypms (picmenu-create-spec
	       '((bottom-left   ( 20  20))
		 (center-left   ( 20  70))
		 (top-left      ( 20 120))
		 (bottom-center ( 70  20))
		 (center        ( 70  70) (20 20))  ; larger
		 (top-center    ( 70 120))
		 (bottom-right  (120  20))
		 (center-right  (120  70))
		 (top-right     (120 120)))
	       140 140 'wteste-draw-square t)) )

(defvar mypm nil)
; 10 Sep 91; 11 Sep 91; 12 Sep 91; 14 Sep 91; 17 Sep 91
; A picmenu that is "flat" within another window, in this case *myw*.
; Must do (wtesta) first.
(defun wtestf ()
  (or mypms (mypms-init))
  (or mypm (setq mypm (picmenu-create-from-spec mypms "Points on Square"
						*myw* 50 50 nil t t)))
  (picmenu-select mypm))

(defun wteste-draw-square (w x y)
  (window-draw-box-xy w (+ x 20) (+ y 20) 100 100 1))

(defvar mym nil)
; 10 Sep 91; 17 Sep 91
; A menu that is "flat" within another window, in this case *myw*.
; Must do (wtesta) first.
(defun wtestg ()
  (or mym (setq mym (menu-create '(red white blue) "Flag" *myw* 50 50 nil t)))
  (menu-select mym))

; 09 Oct 91
; Demonstrate arrows.  Optional arg is line width.
(defun wtesth ( &optional (lw 1))
  (window-clear *myw*)
  (dotimes (i 5) (window-draw-arrow-xy *myw* 100 100 (+ 40 (* i 30)) 160 lw))
  (dotimes (i 5) (window-draw-arrow-xy *myw* 100 100 (+ 40 (* i 30)) 40 lw))
  (dotimes (i 5) (window-draw-arrow-xy *myw* 100 100 40 (+ 40 (* i 30)) lw))
  (dotimes (i 5) (window-draw-arrow-xy *myw* 100 100 160 (+ 40 (* i 30)) lw))
  (dotimes (i 5) (window-draw-arrow-xy *myw* 200 (+ 40 (* i 30))
				           240 (+ 40 (* i 30))
					   (1+ i) ))
  (window-force-output *myw*) )

; 04 Jan 94
; Redo some of the arrows from wtesth in color
(defun wtesti ()
  (window-set-color-rgb *myw* 65535 0 0)
  (window-draw-arrow-xy *myw* 200 70 240 70 2)
  (window-set-color-rgb *myw* 0 65535 0)
  (window-draw-arrow-xy *myw* 200 100 240 100 3)
  (window-set-color-rgb *myw* 0 0 65535)
  (window-draw-arrow-xy *myw* 200 130 240 130 4)
  (window-reset-color *myw*)
  (window-force-output *myw*) )

; 04 Jan 94
; Get text from a window.  Move mouse pointer into test window.
; Add characters and/or backspace, Return.
; Note: it might be necessary to change the keyboard mapping, using
; (window-init-keyboard-mapping *myw*) and (window-print-keyboard-mapping)
(defun wtestj () (window-input-string *myw* "Foo" 50 200 200))

; 04 Jan 94
; Change foreground and background colors and input a string
(defun wtestk ()
  (window-set-color-rgb *myw* 0 65535 0)    ; green foreground
  (window-set-color-rgb *myw* 0 0 65535 t)  ; blue background
  (prog1 (window-input-string *myw* "Foo" 50 200 200)
    (window-reset-color *myw*)
    (window-force-output *myw*) ) )
