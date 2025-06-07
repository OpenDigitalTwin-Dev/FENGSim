; pcalc.lsp             Gordon S. Novak Jr.                 20 Oct 94

; Pocket calculator implemented using a picmenu.  Entry is (pcalc) .

; Copyright (c) 1994 Gordon S. Novak Jr. and The University of Texas at Austin.

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


(defvar *pcalcw* nil) ; window
(defvar *pcalcm* nil) ; picmenu

(defun pcalc-draw (w x y)
  (let (items item over up)
    (window-open w)
    (window-clear w)
    (window-draw-rcbox-xy *pcalcw* 0 0 170 215 10 2)
    (window-draw-rcbox-xy *pcalcw* 10 180 150 25 6)
    (setq items '(0 \. = + 1 2 3 - 4 5 6 * 7 8 9 / off ac ce +-))
    (dotimes (i 5)
      (setq up (+ 10 (* i 35)))
      (dotimes (j 4)
	(setq over (+ 10 (* j 40)))
	(setq item (pop items))
	(window-printat-xy *pcalcw* item
			   (+ over 15 (* (if (numberp item) 1
					     (length (stringify item)))
					 -5)) (+ up 3))
	(window-draw-rcbox-xy *pcalcw* over up 28 20 6) ))
    (window-force-output) ))

(defun pcalc-init ()
  (prog ((n 15))
    (setq *pcalcw* (window-create 170 215 "pcalc" nil nil nil '9x15))
 lp (when (and (> n 0) (null (window-wait-exposure *pcalcw*)))
      (sleep 1.0) (decf n) (go lp))
  (setq *pcalcm*
	(picmenu-create
	  '((0    (24  20) (24 16))
	    (\.   (64  20) (24 16))
	    (=   (104  20) (24 16))
	    (+   (144  20) (24 16))
	    (1    (24  55) (24 16))
	    (2    (64  55) (24 16))
	    (3   (104  55) (24 16))
	    (-   (144  55) (24 16))
	    (4    (24  90) (24 16))
	    (5    (64  90) (24 16))
	    (6   (104  90) (24 16))
	    (*   (144  90) (24 16))
	    (7    (24 125) (24 16))
	    (8    (64 125) (24 16))
	    (9   (104 125) (24 16))
	    (/   (144 125) (24 16))
	    (off  (24 160) (24 16))
	    (ac   (64 160) (24 16))
	    (ce  (104 160) (24 16))
	    (+-  (144 160) (24 16)))
	  170 215 'pcalc-draw nil nil *pcalcw* 0 0 t t)) ))

(defun pcalc-display (val)
  (let (str)
    (window-erase-area-xy *pcalcw* 15 182 140 20)
    (setq str (if (integerp val)
		  (princ-to-string val)
		  (format nil "~8,4F" val)))
    (window-printat-xy *pcalcw* str (- 131 (* 9 (length str))) 185)
    (window-force-output) ))


(defun pcalc ()
  (prog (key (ent 0) (ac 0) decpt lastop lastkey)
    (or *pcalcw* (pcalc-init))
    (pcalc-draw *pcalcw* 0 0)
    (pcalc-display ent)
 lp (setq key (picmenu-select *pcalcm*))
    (if (numberp key)
	(progn (when (eq lastkey '=)
		 (setq ent 0) (setq decpt nil) (setq ac 0) (setq lastop nil))
	       (if decpt
		   (progn (setq ent (+ ent (* key decpt)))
			  (setq decpt (/ decpt 10.0)) )
		   (setq ent (+ key (* ent 10))) )
	       (pcalc-display ent))
	(case key
	  ((+ - * /)
	    (if lastop
	        (progn (setq ac (if (eq lastop '/)
				    (/ (float ac) ent)
				    (funcall lastop ac ent)))
		       (pcalc-display ac))
		(setq ac ent))
	    (setq lastop key)
	    (setq ent 0)
	    (setq decpt nil))
	  (=  (if lastop
		  (progn (setq ent (if (eq lastop '/)
				    (/ (float ac) ent)
				    (funcall lastop ac ent)))
			 (pcalc-display ent)))
	      (setq lastop nil))
	  (\. (when (eq lastkey '=)
		 (setq ent 0) (setq ac 0) (setq lastop nil))
	      (setq decpt 0.1)
	      (setq ent (float ent))
	      (pcalc-display ent))
	  (+- (setq ent (- ent))
	      (pcalc-display ent))
	  (ce (setq ent 0) (setq decpt nil) (pcalc-display ent))
	  (ac (setq ent 0) (setq decpt nil) (setq ac 0) (setq lastop nil)
	      (pcalc-display ent))
	  (off (window-close *pcalcw*)
	       (return nil)) ) )
    (setq lastkey key)
    (go lp) ))

