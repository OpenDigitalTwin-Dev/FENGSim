; draw.lsp                  Gordon S. Novak Jr.       ; 06 Dec 07

; Functions to make drawings interactively

; Copyright (c) 2007 Gordon S. Novak Jr. and The University of Texas at Austin.
; Copyright (c) 2024 Camm Maguire

; 11 Nov 94; 05 Jan 95; 15 Jan 98; 09 Feb 99; 04 Dec 00; 28 Feb 02; 05 Jan 04
; 27 Jan 06

; See the file gnu.license

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


; Use (draw 'foo) to make a drawing named foo.
; When finished with the drawing, give commands "Origin - to zero", "Program".
; This will produce a program (DRAW-FOO w x y) to make the drawing.
; The LaTex command will print Latex input to make the drawing
; (but LaTex cannot draw things as well as the draw program).
; (draw-output <file> &optional names) will save things in a file for later.

; The small square in the drawing menu is a "button" for picture menus.
; If buttons are used, a picmenu-spec will be produced with the program.

(defvar *draw-window*        nil)
(defvar *draw-window-width*  600)
(defvar *draw-window-height* 600)
(defvar *draw-leave-window*  nil)  ; t to leave window displayed at end
(defvar *draw-menu-set*      nil)
(defvar *draw-zero-vector*   '(0 0) )
(defvar *draw-latex-factor*  1)    ; multiplier from pixels to LaTex
(defvar *draw-snap-flag*     t)
(defvar *draw-objects*       nil)
(defvar *draw-latex-mode*    nil)

(glispglobals (*draw-window* window) )

(defmacro draw-descr (name) `(get ,name 'draw-descr))

(glispobjects

(draw-desc (listobject (name    symbol)
		       (objects (listof draw-object))
		       (offset  vector)
		       (size    vector))
  prop   ((fnname      draw-desc-fnname)
	  (refpt       draw-desc-refpt))
  msg    ((draw        draw-desc-draw)
	  (snap        draw-desc-snap)
	  (find        draw-desc-find)
	  (delete      draw-desc-delete))  )

(draw-object (listobject (offset    vector)
			 (size      vector)
			 (contents  anything)
			 (linewidth integer))
  default ((linewidth 1))
  prop    ((region      ((virtual region with start = offset size = size)))
	   (vregion     ((virtual region with start = vstart size = vsize)))
	   (vstart      ((virtual vector with
			   x = (min (x offset) ((x offset) + (x size))) - 2
			   y = (min (y offset) ((y offset) + (y size))) - 2)))
	   (vsize       ((virtual vector with x = (abs (x size)) + 4
				              y = (abs (y size)) + 4))) )
  msg     ((erase       draw-object-erase)
	   (draw        draw-object-draw)
	   (snap        draw-object-snap)
	   (selectedp   draw-object-selectedp)
	   (move        draw-object-move))  )

(draw-line   (listobject (offset  vector)
			 (size    vector)
			 (contents anything)
			 (linewidth integer))
  prop   ((line      ((virtual line-segment with p1 = offset
				p2 = (offset + size)))))
  msg    ((draw       draw-line-draw)
	  (snap       draw-line-snap)
	  (selectedp  draw-line-selectedp) )
  supers (draw-object)    )

(draw-arrow  (listobject (offset  vector)
			 (size    vector)
			 (contents anything)
			 (linewidth integer))
  prop   ((line      ((virtual line-segment with p1 = offset
				p2 = (offset + size)))))
  msg    ((draw       draw-arrow-draw)
	  (snap       draw-line-snap)
	  (selectedp  draw-line-selectedp) )
  supers (draw-object)    )

(draw-box   (listobject (offset  vector)
			 (size    vector)
			 (contents anything)
			 (linewidth integer))
  msg    ((draw       draw-box-draw)
	  (snap       draw-box-snap)
	  (selectedp  draw-box-selectedp) )
  supers (draw-object)    )

(draw-rcbox  (listobject (offset  vector)
			 (size    vector)
			 (contents anything)
			 (linewidth integer))
  msg    ((draw       draw-rcbox-draw)
	  (snap       draw-rcbox-snap)
	  (selectedp  draw-rcbox-selectedp) )
  supers (draw-object)    )

(draw-erase  (listobject (offset  vector)
			 (size    vector)
			 (contents anything)
			 (linewidth integer))
  msg    ((draw       draw-erase-draw)
	  (snap       draw-no-snap)
	  (selectedp  draw-erase-selectedp) )
  supers (draw-object)    )

(draw-circle (listobject (offset  vector)
			 (size    vector)
			 (contents anything)
			 (linewidth integer))
  prop   ((radius     ((x size) / 2))
	  (center     (offset + size / 2)))
  msg    ((draw       draw-circle-draw)
	  (snap       draw-circle-snap)
	  (selectedp  draw-circle-selectedp) )
  supers (draw-object)   )

(draw-ellipse (listobject (offset  vector)
			  (size    vector)
			  (contents anything)
			  (linewidth integer))
  prop   ((radiusx    ((x size) / 2))
	  (radiusy    ((y size) / 2))
	  (radius     ((max radiusx radiusy)))
	  (center     (offset + size / 2))
	  (delta      ((sqrt (abs (radiusx ^ 2 - radiusy ^ 2)))))
	  (p1         ((if (radiusx > radiusy)                ; 05 Jan 04
			   (a vector x = (x center) - delta
				     y = (y center))
			   (a vector x = (x center)
				     y = (y center) - delta))))
	  (p2         ((if (radiusx > radiusy)
			   (a vector x = (x center) + delta
			             y = (y center))
			   (a vector x = (x center)
			             y = (y center) + delta)))) )
  msg    ((draw       draw-ellipse-draw)
	  (snap       draw-ellipse-snap)
	  (selectedp  draw-ellipse-selectedp) )
  supers (draw-object)   )

(draw-dot    (listobject (offset  vector)
			 (size    vector)
			 (contents anything)
			 (linewidth integer))
  msg    ((draw       draw-dot-draw)
	  (snap       draw-dot-snap)
	  (selectedp  draw-button-selectedp) )
  supers (draw-object)    )

(draw-button (listobject (offset  vector)
			 (size    vector)
			 (contents anything)
			 (linewidth integer))
  msg    ((draw       draw-button-draw)
	  (snap       draw-dot-snap)
	  (selectedp  draw-button-selectedp) )
  supers (draw-object)    )

(draw-text   (listobject (offset  vector)
			 (size    vector)
			 (contents anything)
			 (linewidth integer))
  msg    ((draw       draw-text-draw)
	  (snap       draw-no-snap)
	  (selectedp  draw-text-selectedp) )
  supers (draw-object)    )

; null object: no image, cannot be selected.
(draw-null   (listobject (offset  vector)
			 (size    vector)
			 (contents anything)
			 (linewidth integer))
  msg    ((draw       draw-null-draw)
	  (snap       draw-no-snap)
	  (selectedp  draw-null-selectedp) )
  supers (draw-object)    )

(draw-refpt (listobject (offset  vector)
			(size    vector)
			(contents anything)
			(linewidth integer))
  msg    ((draw       draw-refpt-draw)
	  (snap       draw-refpt-snap)
	  (selectedp  draw-refpt-selectedp) )
  supers (draw-object)   )

; multi-item drawing group
(draw-multi  (listobject (offset  vector)
			 (size    vector)
			 (contents (listof draw-object))
			 (linewidth integer))
  msg    ((draw       draw-multi-draw)
	  (snap       draw-no-snap)
	  (selectedp  draw-multi-selectedp) )
  supers (draw-object)    )


) ; glispobjects

; 05 Jan 04
; Get drawing description associated with name
(gldefun draw-desc ((name symbol))
  (result draw-desc)
  (let ((dd draw-desc))
    (dd = (draw-descr name))
    (if ~ dd (progn (dd = (a draw-desc with name = name))
		    (setf (draw-descr name) dd)))
    dd))

; Make a window to draw in.
(setf (glfnresulttype 'draw-window) 'window)
(defun draw-window ()
  (or *draw-window*
      (setq *draw-window*
	    (window-create *draw-window-width* *draw-window-height*
			   "Draw window"))) )

; 09 Sep 92; 11 Sep 92; 14 Sep 92; 16 Sep 92; 21 Oct 92; 21 May 93; 17 Dec 93
; 05 Jan 04
(gldefun draw ((name symbol))
  (let (w dd done sel (redraw t) (new draw-object))
    (w = (draw-window))
    (open w)
    (or *draw-menu-set* (draw-init-menus))
    (dd = (draw-desc name))
    (unless (member name *draw-objects*)
      (setq *draw-objects* (nconc *draw-objects* (list name))))
    (draw dd w)
    (while ~ done do
      (sel = (menu-set-select *draw-menu-set* redraw))
      (redraw = nil)
      (case (menu-name sel)
	(command
	  (case (port sel)
	    (done    (done = t))
	    (move    (draw-desc-move dd w))
	    (delete  (draw-desc-delete dd w))
	    (copy    (draw-desc-copy dd w))
	    (redraw  (clear w)
		     (setq redraw t)
		     (draw dd w))
	    (origin  (draw-desc-origin dd w)
		     (clear w)
		     (setq redraw t)
		     (draw dd w))
	    (program (draw-desc-program dd))
	    (latex   (draw-desc-latex dd))
	    (latexmode (setq *draw-latex-mode* (not *draw-latex-mode*))
		       (format t "Latex Mode is now ~A~%" *draw-latex-mode*))
	    ))
	(draw
	  (new = nil)
	  (case (port sel)
	    (rectangle (new = (draw-box-get dd w)))
	    (rcbox     (new = (draw-rcbox-get dd w)))
	    (circle    (new = (draw-circle-get dd w)))
	    (ellipse   (new = (draw-ellipse-get dd w)))
	    (line      (new = (draw-line-get dd w)))
	    (arrow     (new = (draw-arrow-get dd w)))
	    (dot       (new = (draw-dot-get dd w)))
	    (erase     (new = (draw-erase-get dd w)))
	    (button    (new = (draw-button-get dd w)))
	    (text      (new = (draw-text-get dd w)))
	    (refpt     (new = (draw-refpt-get dd w))))
	  (if new
	      (progn ((offset new) _- (offset dd))
		     ((objects dd) _+ new)
		     (draw new w (offset dd)))))
	(background nil)) )
    (setf (draw-descr name) dd)
    (unless *draw-leave-window* (close w))
    name ))

; 06 Dec 07
; Copy a draw description to another name
(defun copy-draw-desc (from to)
  (let (old)
    (setq old (copy-tree (get from 'draw-descr)))
    (setf (get to 'draw-descr) 
          (cons (car old) (cons to (cddr old))) ) ))

; 09 Sep 92
(gldefun draw-desc-draw ((dd draw-desc) (w window))
  (let ( (off (offset dd)) )
    (clear w)
    (for obj in (objects dd) (draw obj w off))
    (force-output w)  ))

; 11 Sep 92; 12 Sep 92; 06 Oct 92; 05 Jan 04
; Find a draw-object such that point p selects it
(gldefun draw-desc-selected ((dd draw-desc) (p vector))
  (result draw-object)
  (let (objs objsb obj)
    (objs = (for obj in objects when (selectedp obj p (offset dd))
		   collect obj))
    (if objs
        (if (null (rest objs))
	    (obj = (first objs))
	    (progn (objsb = (for z in objs
				      when (member (first z)
						   '(draw-button draw-dot))
				      collect z))
		   (if (and objsb (null (rest objsb)))
		       (obj = (first objsb)))) ) )
    obj))

; 11 Sep 92; 12 Sep 92; 13 Sep 92; 05 Jan 04
; Find a draw-object such that point p selects it
(gldefun draw-desc-find ((dd draw-desc) (w window) &optional (crossflg boolean))
  (result draw-object)
  (let (p obj)
    (while ~ obj do
      (p = (if crossflg (draw-get-cross dd w)
	                (draw-get-crosshairs dd w)))
      (obj = (draw-desc-selected dd p)) )
    obj))

; 15 Sep 92
(gldefun draw-get-cross ((dd draw-desc) (w window))
  (result vector)
  (draw-desc-snap dd (window-get-cross w)))

; 15 Sep 92
(gldefun draw-get-crosshairs ((dd draw-desc) (w window))
  (result vector)
  (draw-desc-snap dd (window-get-crosshairs w)))

; 12 Sep 92; 14 Sep 92; 06 Oct 92
; Delete selected object
(gldefun draw-desc-delete ((dd draw-desc) (w window))
  (let (obj)
    (obj = (draw-desc-find dd w t))
    (erase obj w (offset dd))
    ((objects dd) _- obj) ))

; 12 Sep 92; 07 Oct 92
; Copy selected object
(gldefun draw-desc-copy ((dd draw-desc) (w window))
  (let (obj (objb draw-object))
    (obj = (draw-desc-find dd w))
    (objb = (copy-tree obj))
    (draw-get-object-pos objb w)
    ((offset objb) _- (offset dd))
    (draw objb w (offset dd))
    (force-output w)
    ((objects dd) _+ objb) ))

; 12 Sep 92; 13 Sep 92; 07 Oct 92; 05 Jan 04
; Move selected object
(gldefun draw-desc-move ((dd draw-desc) (w window))
  (let (obj)
    (if (obj = (draw-desc-find dd w))
        (move obj w (offset dd)))  ))

; 14 Sep 92; 28 Feb 02; 05 Jan 04; 27 Jan 06
; Reset origin of object group
(gldefun draw-desc-origin ((dd draw-desc) (w window))
  (let (sel)
    (draw-desc-bounds dd)
    (sel = (menu '(("To zero" . tozero) ("Select" . select))))
    (if (sel == 'select)
	((offset dd) = (get-box-position w (x (size dd)) (y (size dd))))
        (if (sel == 'tozero) ((offset dd) = (a vector x 0 y 0)) ) )))

; 14 Sep 92
; Compute boundaries of objects in a drawing; set offset and size of
; the draw-desc and reset offsets of items relative to it.
(gldefun draw-desc-bounds ((dd draw-desc))
  (let ((xmin 9999) (ymin 9999) (xmax 0) (ymax 0) basev)
    (for obj in objects do
      (xmin = (min xmin (x (offset obj))
		     ((x (offset obj)) + (x (size obj)))))
      (ymin = (min ymin (y (offset obj))
		     ((y (offset obj)) + (y (size obj)))))
      (xmax = (max xmax (x (offset obj))
		     ((x (offset obj)) + (x (size obj)))))
      (ymax = (max ymax (y (offset obj))
		     ((y (offset obj)) + (y (size obj))))) )
    ((x (size dd)) = (xmax - xmin))
    ((y (size dd)) = (ymax - ymin))
    (basev = (a vector with x = xmin y = ymin))
    ((offset dd) = basev)
    (for obj in objects do ((offset obj) _- basev)) ))

; 14 Sep 92; 16 Sep 92; 19 Dec 93; 15 Jan 98; 06 Dec 07
; Produce LaTex output for object group.
; LaTex can only *approximately* reproduce the picture.
(gldefun draw-desc-latex ((dd draw-desc))
  (let (base bx by sx sy)
    (format t "   \\begin{picture}(~5,0F,~5,0F)(0,0)~%"
	      (* (x (size dd)) *draw-latex-factor*)
	      (* (y (size dd)) *draw-latex-factor*) )
    (for obj in (objects dd) do
      (base = (offset dd) + (offset obj))
      (bx = (x base) * *draw-latex-factor*)
      (by = (y base) * *draw-latex-factor*)
      (sx = (x (size obj)) * *draw-latex-factor*)
      (sy = (y (size obj)) * *draw-latex-factor*)
      (case (first obj)
	(draw-line (latex-line (x base) (y base)
			       ((x base) + sx) ((y base) + sy)))
	(draw-arrow (latex-line (x base) (y base)
			       ((x base) + sx) ((y base) + sy) t) )
	(draw-box
	  (format t "   \\put(~5,0F,~5,0F) {\\framebox(~5,0F,~5,0F)}~%"
		  bx by sx sy))
	(draw-rcbox
	  (format t "   \\put(~5,0F,~5,0F) {\\oval(~5,0F,~5,0F)}~%"
		  (bx + sx / 2) (by + sy / 2) sx sy))
	(draw-circle
	  (format t "   \\put(~5,0F,~5,0F) {\\circle{~5,0F}}~%"
		  (bx + sx / 2) (by + sy / 2) sx))
	(draw-ellipse
	  (format t "   \\put(~5,0F,~5,0F) {\\oval(~5,0F,~5,0F)}~%"
		  (bx + sx / 2) (by + sy / 2) sx sy))
	(draw-button
	  (format t "   \\put(~5,0F,~5,0F) {\\framebox(~5,0F,~5,0F)}~%"
		  bx by sx sy))
	(draw-erase )
	(draw-dot
	  (format t "   \\put(~5,0F,~5,0F) {\\circle*{~5,0F}}~%"
		  (bx + sx / 2) (by + sy / 2) sx))
	(draw-text
	  (format t "   \\put(~5,0F,~5,0F) {~A}~%"
		  bx (by + 4 * *draw-latex-factor*) (contents obj)) ) ) )
    (format t "   \\end{picture}~%")  ))

; 14 Sep 92; 15 Sep 92; 16 Sep 92; 05 Oct 92; 17 Dec 93; 21 Dec 93; 28 Feb 02
; 05 Jan 04
; Produce program to draw object group
(gldefun draw-desc-program ((dd draw-desc))
  (let (base bx by sx sy tox toy r rx ry s code fncode fnname cd)
    (code = (for obj in (objects dd) when
		(cd = (progn
		  (base = (offset dd) + (offset obj) - (refpt dd))
		  (bx = (x base))
		  (by = (y base))
		  (sx = (x (size obj)))
		  (sy = (y (size obj)))
		  (tox = bx + sx)
		  (toy = by + sy)
		  (if ((car obj) == 'draw-circle)
		      (r = (x (size obj)) / 2))
		  (if ((car obj) == 'draw-ellipse)
		      (progn (rx = (x (size obj)) / 2)
			     (ry = (y (size obj)) / 2)))
		  (draw-optimize
		    (case (first obj)
		      (draw-line `(window-draw-line-xy w (+ x ,bx)  (+ y ,by)
						       (+ x ,tox) (+ y ,toy)))
		      (draw-arrow `(window-draw-arrow-xy w (+ x ,bx)  (+ y ,by)
							(+ x ,tox) (+ y ,toy)))
		      (draw-box `(window-draw-box-xy w (+ x ,bx) (+ y ,by)
						     ,sx ,sy))
		      (draw-rcbox `(window-draw-rcbox-xy w (+ x ,bx) (+ y ,by)
						         ,sx ,sy 8))
		      (draw-circle `(window-draw-circle-xy w (+ x ,(+ r bx))
							   (+ y ,(+ r by)) ,r))
		      (draw-ellipse `(window-draw-ellipse-xy w (+ x ,(+ rx bx))
							     (+ y ,(+ ry by))
							     ,rx ,ry))
		      ((draw-button draw-refpt)
		         nil)  ; let picmenu draw the buttons
		      (draw-erase `(window-erase-area-xy w (+ x ,bx) (+ y ,by)
						         ,sx ,sy))
		      (draw-dot `(window-draw-dot-xy w (+ x ,(+ 2 bx))
						     (+ y ,(+ 2 by))))
		      (draw-text (s = (stringify (contents obj)))
				 `(window-printat-xy w ,s (+ x ,bx) (+ y ,by)))
		      )) ))
		collect cd))
    (fncode = (cons 'lambda (cons (list 'w 'x 'y)
				    (nconc code
					   (list (list 'window-force-output
						       'w))))))
    (fnname = (fnname dd))
    (setf (symbol-function fnname) fncode)
    (format t "Constructed program (~A w x y)~%" fnname)
    (draw-desc-picmenu dd)
  ))

; 21 Dec 93
; Optimize code if GLISP is present
(defun draw-optimize (x)  (if (fboundp 'glunwrap) (glunwrap x nil) x))

; 14 Sep 92
(gldefun draw-desc-fnname ((dd draw-desc))
  (intern (concatenate 'string "DRAW-" (symbol-name (name dd)))) )

; 14 Sep 92; 06 Oct 92; 08 Apr 93; 28 Feb 02; 05 Jan 04
; Produce a picmenu-spec from the buttons of a drawing description
(gldefun draw-desc-picmenu ((dd draw-desc))
  (let (buttons)
    (buttons = (for obj in (objects dd) when ((first obj) == 'draw-button)
		      collect (list (contents obj)
				    ((a vector x 2 y 2) + (offset obj)
				      + (offset dd) )) ) )
    (if buttons
        (setf (get (name dd) 'picmenu-spec)
	      (list 'picmenu-spec (x (size dd)) (y (size dd)) buttons
		    t (fnname dd) '9x15))) ))

; 15 Sep 92; 05 Jan 04
(gldefun draw-desc-snap ((dd draw-desc) (p vector))
  (result vector)
  (let (psnap obj (objs (objects dd)) )
    (if *draw-snap-flag*
        (while objs and ~ psnap do
          (obj = (pop objs))
	  (psnap = (draw-object-snap obj p (offset dd))) ) )
    (or psnap p) ))

; 10 Sep 92; 12 Sep 92
; Move specified object
(gldefun draw-object-move ((d draw-object) (w window) (off vector))
  (let ()
    (erase d w off)
    (draw-get-object-pos d w)
    ((offset d) _- off)
    (draw d w off)
    (force-output w) ))

; 12 Sep 92; 13 Sep 92; 15 Sep 92
; Draw an object at specified (x y) by calling its drawing function
(defun draw-object-draw-at (w x y d)
  (setf (second d) (list x y))
  (draw-object-draw d w *draw-zero-vector*) )

; 15 Sep 92
; Simulate glsend of draw message to an object
(defun draw-object-draw (d w off)
  (funcall (glmethod (car d) 'draw) d w off) )

; 15 Sep 92
; Simulate glsend of snap message to an object
(defun draw-object-snap (d p off)
  (funcall (glmethod (car d) 'snap) d p off) )

; 15 Sep 92
; Simulate glsend of selectedp message to an object
(defun draw-object-selectedp (d w off)
  (funcall (glmethod (car d) 'selectedp) d w off) )

; 12 Sep 92; 07 Oct 92; 28 Feb 02; 05 Jan 04; 06 Dec 07
(gldefun draw-get-object-pos ((d draw-object) (w window))
  (window-get-icon-position w 
    (if ((first d) == 'draw-text) #'draw-text-draw-outline
                                  #'draw-object-draw-at)
    (list d)) )

; 10 Sep 92; 15 Sep 92; 05 Jan 04
(gldefun draw-object-erase ((d draw-object) (w window) (off vector))
  (let ()
    (if ((first d) <> 'draw-erase)
	(progn (set-xor w)
	       (draw d w off)
	       (unset w)) )))

; 09 Sep 92; 17 Dec 93; 19 Dec 93; 04 Dec 00
(gldefun draw-line-draw ((d draw-line) (w window) (off vector))
  (let ((from (off + (offset d))) (to ((off  + (offset d)) + (size d))) )
    (draw-line-xy w (x from) (y from) (x to) (y to)) ))

; 11 Sep 92; 17 Dec 93; 19 Dec 93; 04 Dec 00
(gldefun draw-arrow-draw ((d draw-arrow) (w window) (off vector))
  (let ((from (off + (offset d))) (to ((off  + (offset d)) + (size d))) )
    (draw-arrow-xy w (x from) (y from) (x to) (y to)) ))

; 09 Sep 92; 10 Sep 92; 12 Sep 92
(gldefun draw-line-selectedp ((d draw-line) (pt vector) (off vector))
  (let ((ptp (pt - off)))
    (and (contains? (vregion d) ptp)
	 ((distance (line d) ptp) < 5) ) ))

; 09 Sep 92; 10 Sep 92; 15 Sep 92; 17 Dec 93; 05 Jan 04
(gldefun draw-line-get ((dd draw-desc) (w window))
  (let (from to)
    (from = (draw-get-crosshairs dd w))
    (to   = (if *draw-latex-mode*
	        (window-get-latex-position w (x from) (y from) nil)
		(draw-desc-snap dd 
				(window-get-line-position w (x from) (y from)))))
    (a draw-line with offset = from  size = (to - from)) ))

; 11 Sep 92; 15 Sep 92; 17 Dec 93; 05 Jan 04
(gldefun draw-arrow-get ((dd draw-desc) (w window))
  (let (from to)
    (from = (draw-get-crosshairs dd w))
    (to   = (if *draw-latex-mode*
	        (window-get-latex-position w (x from) (y from) nil)
	        (draw-desc-snap dd 
			  (window-get-line-position w (x from) (y from)))))
    (a draw-arrow with offset = from  size = (to - from)) ))

; 09 Sep 92
(gldefun draw-box-draw ((d draw-box) (w window) (off vector))
  (draw-box w (off + (offset d)) (size d)) )

; 09 Sep 92; 11 Sep 92
(gldefun draw-box-selectedp ((d draw-box) (p vector) (off vector))
  (let ((pt (p - off)))
    (or (and ((y pt) < (top (vregion d)) + 5)
	     ((y pt) > (bottom (vregion d)) - 5)
	     (or ((abs (x pt) - (left (vregion d))) < 5)
		 ((abs (x pt) - (right (vregion d))) < 5)))
	(and ((x pt) < (right (vregion d)) + 5)
	     ((x pt) > (left (vregion d)) - 5)
	     (or ((abs (y pt) - (top (vregion d))) < 5)
		 ((abs (y pt) - (bottom (vregion d))) < 5))) ) ))

; 11 Sep 92
(gldefun draw-box-get ((dd draw-desc) (w window))
  (let (box)
    (box = (window-get-region w))
    (a draw-box with offset = (start box)  size = (size box)) ))

; (dotimes (i 10) (print (draw-box-selectedp db (window-get-point dw))))

; 16 Sep 92
(gldefun draw-rcbox-draw ((d draw-box) (w window) (off vector))
  (draw-rcbox-xy w ((x off) + (x (offset d))) ((y off) + (y (offset d)))
		   (x (size d)) (y (size d)) 8) )

; 16 Sep 92
(gldefun draw-rcbox-selectedp ((d draw-box) (p vector) (off vector))
  (let ((pt (p - off)))
    (or (and ((y pt) < (top (vregion d)) - 3)
	     ((y pt) > (bottom (vregion d)) + 3)
	     (or ((abs (x pt) - (left (vregion d))) < 5)
		 ((abs (x pt) - (right (vregion d))) < 5)))
	(and ((x pt) < (right (vregion d)) - 3)
	     ((x pt) > (left (vregion d)) + 3)
	     (or ((abs (y pt) - (top (vregion d))) < 5)
		 ((abs (y pt) - (bottom (vregion d))) < 5))) ) ))

; 16 Sep 92
(gldefun draw-rcbox-get ((dd draw-desc) (w window))
  (let (box)
    (box = (window-get-region w))
    (a draw-rcbox with offset = (start box)  size = (size box)) ))

; 09 Sep 92
(gldefun draw-circle-draw ((d draw-circle) (w window) (off vector))
  (draw-circle w (off + (center d)) (radius d)) )

; 09 Sep 92; 11 Sep 92; 17 Sep 92
(gldefun draw-circle-selectedp ((d draw-circle) (p vector) (off vector))
  ((abs (radius d) - (magnitude ((center d) + off) - p)) < 5) )

; 11 Sep 92; 15 Sep 92
(gldefun draw-circle-get ((dd draw-desc) (w window))
  (let (cir cent)
    (cent = (draw-get-crosshairs dd w))
    (cir = (window-get-circle w cent))
    (a draw-circle with
       offset = (a vector with x = ( (x (center cir)) - (radius cir) )
		               y = ( (y (center cir)) - (radius cir) ))
       size   = (a vector with x = 2 * (radius cir) y = 2 * (radius cir))) ))

; 11 Sep 92
(gldefun draw-ellipse-draw ((d draw-ellipse) (w window) (off vector))
  (let ((c (off + (center d))))
    (draw-ellipse-xy w (x c) (y c) (radiusx d) (radiusy d)) ))

; 11 Sep 92; 15 Sep 92; 17 Sep 92
; Uses the fact that sum of distances from foci is constant.
(gldefun draw-ellipse-selectedp ((d draw-ellipse) (p vector) (off vector))
  (let ((pt (p - off)))
    ( (abs ( (magnitude ((p1 d) - pt)) +  (magnitude ((p2 d) - pt)) )
      - 2 * (radius d)) < 2) ))

; print out what the "boundary" of an ellipse looks like via selectedp
(defun draw-test-ellipse-selectedp (e)
  (let ( (size (third e)) (offset (second e)) )
    (dotimes (y (+ (cadr size) 10))
      (dotimes (x (+ (car size) 10))
	(princ (if (draw-ellipse-selectedp e
		     (list (+ x (car offset) -5) (+ y (cadr offset) -5))
		     (list 0 0))
		   "T" " ")))
      (terpri)) ))

; 11 Sep 92
(gldefun draw-ellipse-get ((dd draw-desc) (w window))
  (let (ell cent)
    (cent = (draw-get-crosshairs dd w))
    (ell = (window-get-ellipse w cent))
    (a draw-ellipse with
       offset = (a vector with x = ( (x (center ell)) - (x (halfsize ell)) )
		               y = ( (y (center ell)) - (y (halfsize ell)) ))
       size   = (a vector with x = 2 * (x (halfsize ell))
		               y = 2 * (y (halfsize ell)))) ))
      
; 10 Sep 92
(gldefun draw-null-draw ((d draw-null) (w window) (off vector)) nil)

; 10 Sep 92; 11 Sep 92
(gldefun draw-null-selectedp ((d draw-null) (pt vector) (off vector)) nil)

; 11 Sep 92
(gldefun draw-button-draw ((d draw-button) (w window) (off vector))
  (draw-box w (off + (offset d)) (a vector x = 4 y = 4)) )

; 11 Sep 92
(gldefun draw-button-selectedp ((d draw-button) (p vector) (off vector))
  (let ( (ptx (((x p) - (x off)) - (x (offset d))))
	 (pty (((y p) - (y off)) - (y (offset d)))) )
    (and (ptx > -2) (ptx < 6) (pty > -2) (pty < 6) ) ))
 ))

; 11 Sep 92
(gldefun draw-button-get ((dd draw-desc) (w window))
  (let (cent var)
    (princ "Enter button name: ")
    (var = (read))
    (cent = (draw-get-crosshairs dd w))
    (a draw-button with
       offset = (a vector with x = ((x cent) - 2) y = ((y cent) - 2))
       size   = (a vector with x = 4 y = 4)
       contents = var) ))

; 14 Sep 92
(gldefun draw-erase-draw ((d draw-box) (w window) (off vector))
  (erase-area w (off + (offset d)) (size d)) )

; 14 Sep 92
(gldefun draw-erase-selectedp ((d draw-box) (p vector) (off vector))
  (let ((pt (p - off)))
    (contains? (region d) pt) ))

; 14 Sep 92
(gldefun draw-erase-get ((dd draw-desc) (w window))
  (let (box)
    (box = (window-get-region w))
    (a draw-erase with offset = (start box)  size = (size box)) ))

; 11 Sep 92; 14 Sep 92
(gldefun draw-dot-draw ((d draw-dot) (w window) (off vector))
  (window-draw-dot-xy w ((x off) + (x (offset d)) + 2)
		        ((y off) + (y (offset d)) + 2) ) )

; 11 Sep 92; 15 Sep 92
(gldefun draw-dot-get ((dd draw-desc) (w window))
  (let (cent)
    (cent = (draw-get-crosshairs dd w))
    (a draw-dot with
       offset = (a vector with x = ((x cent) - 2) y = ((y cent) - 2))
       size   = (a vector with x = 4 y = 4)) ))

; 17 Dec 93
(gldefun draw-refpt-draw ((d draw-refpt) (w window) (off vector))
  (window-draw-crosshairs-xy w ((x off) + (x (offset d)))
		               ((y off) + (y (offset d))) ) )

; 17 Dec 93
(gldefun draw-refpt-selectedp ((d draw-button) (p vector) (off vector))
  (let ( (ptx (((x p) - (x off)) - (x (offset d))))
	 (pty (((y p) - (y off)) - (y (offset d)))) )
    (and (ptx > -3) (ptx < 3) (pty > -3) (pty < 3) ) ))

; 17 Dec 93; 05 Jan 04
(gldefun draw-refpt-get ((dd draw-desc) (w window))
  (let (cent refpt)
    (if (refpt = (assoc 'draw-refpt (objects dd)))
	(progn (set-erase *draw-window*)
	       (draw refpt *draw-window* (a vector with x = 0 y = 0))
	       (unset *draw-window*)
	       ((objects dd) _- refpt) ) )
    (cent = (draw-get-crosshairs dd w))
    (a draw-refpt with offset = cent
		       size   = (a vector with x = 0 y = 0)) ))

; 17 Dec 93; 05 Jan 04
(gldefun draw-desc-refpt ((dd draw-desc)) (result vector)
  (let (refpt)
    (refpt = (assoc 'draw-refpt (objects dd)))
    (if refpt (offset refpt)
              (a vector x = 0 y = 0)) ))

; 11 Sep 92; 06 Oct 92; 19 Dec 93; 11 Nov 94
(gldefun draw-text-draw ((d draw-text) (w window) (off vector))
  (printat-xy w (contents d) ((x off) + (x (offset d)))
	                     ((y off) + (y (offset d)))) )

; 07 Oct 92
(gldefun draw-text-draw-outline ((w window) (x integer) (y integer) (d draw-text))
  (setf (second d) (list x y))
  (draw-box-xy w x (y + 2) (x (size d)) (y (size d))) )

; define compiled version directly to avoid repeated recompilation
(defun draw-text-draw-outline (W X Y D)
  (SETF (SECOND D) (LIST X Y))
  (WINDOW-DRAW-BOX-XY W X (+ 2 Y) (CAADDR D) (CADR (CADDR D))))

; 11 Sep 92
(gldefun draw-text-selectedp ((d draw-text) (pt vector) (off vector))
  (let ((ptp (pt - off)))
    (contains? (vregion d) ptp)))

; 11 Sep 92; 17 Sep 92; 06 Oct 92; 11 Nov 94
(gldefun draw-text-get ((dd draw-desc) (w window))
  (let (txt lng off)
    (princ "Enter text string: ")
    (txt = (stringify (read)))
    (lng = (string-width w txt))
    (off = (get-box-position w lng 14))
    (a draw-text with  offset   = (off + (a vector x 0 y 4))
                       size     = (a vector with x = lng y = 14)
                       contents = txt) ))

; 15 Sep 92; 05 Jan 04
; Test if a point p1 is close to a point p2.  If so, result is p2, else nil.
(gldefun draw-snapp ((p1 vector) (off vector) (p2x integer) (p2y integer))
  (if (and ((abs ((x p1) - (x off) - p2x)) < 4)
	   ((abs ((y p1) - (y off) - p2y)) < 4) )
      (a vector with x = ((x off) + p2x) y = ((y off) + p2y)) ))

; 15 Sep 92
(gldefun draw-dot-snap ((d draw-dot) (p vector) (off vector))
  (draw-snapp p off ((x (offset d)) + 2)
		    ((y (offset d)) + 2) ) )

; 17 Dec 93
(gldefun draw-refpt-snap ((d draw-refpt) (p vector) (off vector))
  (draw-snapp p off (x (offset d)) (y (offset d)) ) )

; 15 Sep 92
(gldefun draw-line-snap ((d draw-line) (p vector) (off vector))
  (or (draw-snapp p off (x (offset d)) (y (offset d)))
      (draw-snapp p off ( (x (offset d)) + (x (size d)) )
		        ( (y (offset d)) + (y (size d)) ) ) ))

; 15 Sep 92; 19 Dec 93
; Snap for square: corners, middle of sides.
(gldefun draw-box-snap ((d draw-box) (p vector) (off vector))
  (let ((xoff (x (offset d))) (yoff (y (offset d)))
	(xsize (x (size d)) ) (ysize (y (size d)) ) )
    (or (draw-snapp p off xoff yoff)
	(draw-snapp p off (xoff + xsize) (yoff + ysize))
	(draw-snapp p off (xoff + xsize) yoff)
	(draw-snapp p off xoff (yoff + ysize))
	(draw-snapp p off (xoff + xsize / 2) yoff)
	(draw-snapp p off xoff (yoff + ysize / 2))
	(draw-snapp p off (xoff + xsize / 2) (yoff + ysize))
	(draw-snapp p off (xoff + xsize) (yoff + ysize / 2)) ) ))

; 15 Sep 92
(gldefun draw-circle-snap ((d draw-circle) (p vector) (off vector))
  (or (draw-snapp p off ( (x (offset d)) + (radius d) )
		        ( (y (offset d)) + (radius d) ) )
      (draw-snapp p off ( (x (offset d)) + (radius d) )
		        (y (offset d)) )
      (draw-snapp p off (x (offset d))
		        ( (y (offset d)) + (radius d) ) )
      (draw-snapp p off ( (x (offset d)) + (radius d) )
		        ( (y (offset d)) + (y (size d)) ) )
      (draw-snapp p off ( (x (offset d)) + (x (size d)) )
		        ( (y (offset d)) + (radius d) ) ) ))

; 15 Sep 92
(gldefun draw-ellipse-snap ((d draw-ellipse) (p vector) (off vector))
  (or (draw-snapp p off ( (x (offset d)) + (radiusx d) )
		        ( (y (offset d)) + (radiusy d) ) )
      (draw-snapp p off ( (x (offset d)) + (radiusx d) )
		        (y (offset d)) )
      (draw-snapp p off (x (offset d))
		        ( (y (offset d)) + (radiusy d) ) )
      (draw-snapp p off ( (x (offset d)) + (radiusx d) )
		        ( (y (offset d)) + (y (size d)) ) )
      (draw-snapp p off ( (x (offset d)) + (x (size d)) )
		        ( (y (offset d)) + (radiusy d) ) ) ))

; 16 Sep 92
(gldefun draw-rcbox-snap ((d draw-rcbox) (p vector) (off vector))
  (let ( (rx ((x (size d)) / 2)) (ry ((y (size d)) / 2)) )
    (or (draw-snapp p off ( (x (offset d)) + rx ) (y (offset d)) )
	(draw-snapp p off (x (offset d)) ( (y (offset d)) + ry ) )
	(draw-snapp p off ( (x (offset d)) + rx )
		          ( (y (offset d)) + (y (size d)) ) )
	(draw-snapp p off ( (x (offset d)) + (x (size d)) )
		          ( (y (offset d)) + ry ) )  ) ))

; 15 Sep 92
(gldefun draw-no-snap ((d draw-ellipse) (p vector) (off vector)) nil)

; 11 Sep 92
(gldefun draw-multi-draw ((d draw-multi) (w window) (off vector))
  (let ( (totaloff ((offset d) + off)) )
    (for subd in (contents d) do
      (draw subd w totaloff)) ))

; 11 Sep 92; 13 Sep 92; 15 Sep 92; 16 Sep 92; 29 Sep 92; 17 Dec 93; 07 Jan 94
; Initialize drawing and command menus
(defun draw-init-menus ()
  (let ((w (draw-window)))
    (window-clear w)
    (dolist (fn '(draw-menu-rectangle draw-menu-circle draw-menu-ellipse
	          draw-menu-line      draw-menu-arrow  draw-menu-dot
		  draw-menu-button    draw-menu-text))
      (setf (get fn 'display-size) '(30 20)) )
    (setq *draw-menu-set* (menu-set-create w nil))
    (menu-set-add-menu *draw-menu-set* 'draw nil "Draw"
		       '((draw-menu-rectangle . rectangle)
			 (draw-menu-rcbox     . rcbox)
			 (draw-menu-circle    . circle)
			 (draw-menu-ellipse   . ellipse)
			 (draw-menu-line      . line)
			 (draw-menu-arrow     . arrow)
			 (draw-menu-dot       . dot)
			 (" "                 . erase)
			 (draw-menu-button    . button)
			 (draw-menu-text      . text)
			 (draw-menu-refpt     . refpt))
		       (list 0 0))
    (menu-set-adjust *draw-menu-set* 'draw 'top nil 1)
    (menu-set-adjust *draw-menu-set* 'draw 'right nil 2)
    (menu-set-add-menu *draw-menu-set* 'command nil "Commands"
		       '(("Done" . done)       ("Move" . move)
			 ("Delete" . delete)   ("Copy" . copy)
			 ("Redraw" . redraw)   ("Origin" . origin)
			 ("LaTex Mode" . latexmode)
			 ("Make Program" . program) ("Make LaTex" . latex))
		        (list 0 0))
    (menu-set-adjust *draw-menu-set* 'command 'top 'draw 5)
    (menu-set-adjust *draw-menu-set* 'command 'right nil 2) ))


; 10 Sep 92
(defun draw-menu-rectangle (w x y)
  (window-draw-box-xy w (+ x 3) (+ y 3) 24 14 1))
(defun draw-menu-rcbox (w x y)
  (window-draw-rcbox-xy w (+ x 3) (+ y 3) 24 14 3 1))
(defun draw-menu-circle (w x y)
  (window-draw-circle-xy w (+ x 15) (+ y 10) 8 1))
(defun draw-menu-ellipse (w x y)
  (window-draw-ellipse-xy w (+ x 15) (+ y 10) 12 8 1))
(defun draw-menu-line (w x y)
  (window-draw-line-xy w (+ x 4) (+ y 4) (+ x 26) (+ y 16) 1))
(defun draw-menu-arrow (w x y)
  (window-draw-arrow-xy w (+ x 4) (+ y 4) (+ x 26) (+ y 16) 1))
(defun draw-menu-dot (w x y) (window-draw-dot-xy w (+ x 15) (+ y 10)) )
(defun draw-menu-button (w x y)
  (window-draw-box-xy w (+ x 14) (+ y 5) 4 4 1))
(defun draw-menu-text (w x y)
  (window-printat-xy w "A" (+ x 12) (+ y 5)))
(defun draw-menu-refpt (w x y)
  (window-draw-crosshairs-xy w (+ x 15) (+ y 9))
  (window-draw-circle-xy w (+ x 15) (+ y 9) 2))

; 14 Sep 92; 15 Jan 98
; Draw a line or arrow in LaTex form
(defun latex-line (fromx fromy x y &optional arrowflg)
  (let (dx dy sx sy siz err errb)
    (setq dx (- x fromx))
    (setq dy (- y fromy))
    (if (= dx 0)
	(progn (setq sx 0)
	       (setq sy (if (>= dy 0) 1 -1))
	       (setq siz (* (abs dy) *draw-latex-factor*)))
	(if (= dy 0)
	    (progn (setq sx (if (>= dx 0) 1 -1))
		   (setq sy 0)
		   (setq siz (* (abs dx) *draw-latex-factor*)))
	    (progn
	      (setq err 9999)
	      (setq siz (* (abs dx) *draw-latex-factor*))
	      (dotimes (i (if arrowflg 4 6))
		(dotimes (j (if arrowflg 4 6))
		  (setq errb (abs (- (/ (float (1+ i))
					(float (1+ j)))
				     (abs (/ (float dx)
					     (float dy))))))
		  (if (and (= (gcd (1+ i) (1+ j)) 1)
			   (< errb err))
		      (progn (setq err errb)
			     (setq sx (1+ i))
			     (setq sy (1+ j))))))
	      (setq sx (* sx (latex-sign dx)))
	      (setq sy (* sy (latex-sign dy))) )))
    (format t "   \\put(~5,0F,~5,0F) {\\~A(~D,~D){~5,0F}}~%"
	    (* fromx *draw-latex-factor*) (* fromy *draw-latex-factor*)
	    (if arrowflg "vector" "line") sx sy siz)  ))

(defun latex-sign (x) (if (>= x 0) 1 -1))


; 16 Sep 92; 30 Sep 92; 02 Oct 92; 07 Oct 92
(defun draw-output (outfilename &optional names)
  (prog (prettysave lengthsave d fnname code)
    (or names (setq names *draw-objects*))
    (if (symbolp names) (setq names (list names)))
    (with-open-file (outfile outfilename
			     :direction :output
			     :if-exists :supersede)
         (setq prettysave *print-pretty*)
	 (setq lengthsave *print-length*)
	 (setq *print-pretty* t)
	 (setq *print-length* 80)
	 (format outfile "; ~A   ~A~%" 
		          outfilename (draw-get-time-string))
	 (dolist (name names)
	   (if (setq d (get name 'draw-descr))
	       (progn (terpri outfile)
		      (print `(setf (get ',name 'draw-descr) ',d) outfile)
		      (if (and (setq fnname (draw-desc-fnname d))
			       (setq code (symbol-function fnname)))
			  (progn (terpri outfile)
				 (print (cons 'defun
					      (if (eq (car code) 'lambda-block)
						  (cdr code)
						  (cons fnname (cdr code))))
					outfile)) )))
	   (if (setq d (get name 'picmenu-spec))
	       (progn (terpri outfile)
		      (print `(setf (get ',name 'picmenu-spec) ',d) outfile))))
	 (terpri outfile)
	 (setq *print-pretty* prettysave)
	 (setq *print-length* lengthsave)  )
    (return outfilename) ))

; 09 Sep 92
(defun draw-get-time-string ()
  (let (second minute hour date month year)
    (multiple-value-setq (second minute hour date month year)
			 (get-decoded-time))
    (format nil "~2D ~A ~4D ~2D:~2D:~2D"
	    date (nth (1- month) '("Jan" "Feb" "Mar" "Apr" "May" "Jun" "Jul"
				   "Aug" "Sep" "Oct" "Nov" "Dec"))
	    year hour minute second) ))

; 14 Sep 92; 16 Sep 92; 13 July 93
; Compile the draw.lsp and menu-set files into a plain Lisp file
(defun compile-draw ()
  (glcompfiles *directory*
	       '("glisp/vector.lsp"          ; auxiliary files
                 "X/dwindow.lsp")
	       '("glisp/menu-set.lsp"        ; translated files
		 "glisp/draw.lsp")
	       "glisp/drawtrans.lsp"         ; output file
	       "glisp/draw-header.lsp")      ; header file
  (cf drawtrans) )

(defun compile-drawb ()
  (glcompfiles *directory*
	       '("glisp/vector.lsp"          ; auxiliary files
                 "X/dwindow.lsp" "X/dwnoopen.lsp")
	       '("glisp/menu-set.lsp"        ; translated files
		 "glisp/draw.lsp")
	       "glisp/drawtrans.lsp"         ; output file
	       "glisp/draw-header.lsp")      ; header file
  )

; 16 Nov 92; 08 Apr 93; 08 Oct 93; 20 Apr 94; 29 Oct 94; 09 Feb 99
; Output drawing descriptions and functions to the specified file
(defun draw-out (&optional names file)
  (or names (setq names *draw-objects*))
  (if (not (consp names)) (setq names (list names)))
  (draw-output (or file "glisp/draw.del") names)
  (setq *draw-objects* (set-difference *draw-objects* names))
  names )
