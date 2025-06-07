; editors.lsp               Gordon S. Novak Jr.         ; 08 Dec 08

; Copyright (c) 2008 Gordon S. Novak Jr. and The University of Texas at Austin.

; 13 Apr 95; 02 Jan 97; 28 Feb 02; 08 Jan 04; 03 Mar 04; 26 Jan 06; 27 Jan 06

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

; Graphical editor functions

; (edit-thermom 75 myw 20 20 150 250)
; (window-draw-thermometer myw 0 20 5 50 50 50 232)
; (window-adjust-thermometer myw 0 20 5 50 50 50 232)

; 20 Nov 91; 03 Dec 91; 27 Dec 91; 26 Dec 93; 28 Feb 02; 08 Jan 04
; Edit an integer with a thermometer-like display
(gldefun edit-thermom ((num number) (w window)
		       &optional (offsetx integer) (offsety integer)
		                 (sizex integer) (sizey integer))
  (prog (nmin ndel ndiv range pten drange pair neww (res num) off)
    (if ~ sizex (progn (sizex = 150) (sizey = 250)))
    (if ~ offsetx
	(progn (off = (centeroffset w (a vector with x = sizex y = sizey)))
	     (offsetx = (x off))
	     (offsety = (y off))))
    (neww = (window-create sizex sizey nil (parent w) offsetx offsety))
    (window-draw-button neww "Typein" 80 20 50 25)
    (window-draw-button neww "Adjust" 80 70 50 25)
    (window-draw-button neww "Done"   80 120 50 25)
 rn (range = (abs res) * 2)
    (if (range == 0) (range = 50))
    (if ((range < 8) and (integerp num)) (range = 10))
    (pten = (expt 10 (truncate (log range 10))))
    (drange = (range * 10) / pten)
    (setq pair (car (some #'(lambda (x) (> (car x) drange))
			  '((14 2) (20 4) (40 5) (70 10) (101 20)))))
    (setq ndel ((cadr pair) * pten / 10))
    (setq ndiv (ceiling (range / ndel)))
    (setq nmin (if (>= res 0)
		   0
		   (- ndel * ndiv)))
    (window-draw-thermometer neww nmin ndel ndiv res 10 10 (sizey - 20))
 lp (case (button-select neww '((done (84 124) (42 17))
				(adjust (84 74) (42 17))
				(typein (84 24) (42 17))))
      (done (destroy neww) (return res))
      (adjust (setq res (window-adjust-thermometer neww nmin ndel ndiv res
						   10 10 (sizey - 20)))
	      (go lp))
      (typein (princ "Enter new value: ")
	      (setq res (read))
	      (if ((res >= nmin) and (res <= (nmin + ndel * ndiv)))
		 (progn (window-set-thermometer neww nmin ndel ndiv res
				               10 10 (sizey - 20))
                       (go lp))
		 (go rn)) ) ) ))

; 20 Nov 91; 04 Dec 91
; Draw a button-like icon
(gldefun window-draw-button ((w window) (s string)
				       (offsetx integer) (offsety integer)
				       (sizex integer) (sizey integer))
  (let (sw)
    (erase-area-xy w offsetx offsety sizex sizey 8)
    (draw-rcbox-xy w offsetx offsety sizex sizey 8)
    (sw = (string-width w s))
    (printat-xy w s (offsetx + (sizex - sw) / 2) (offsety + 8))
    (force-output w)))

; 17 Dec 91
; Print in the center of a specified region
(gldefun window-center-print ((w window) (s string)
				        (offsetx integer) (offsety integer)
				        (sizex integer) (sizey integer))
  (let (sw)
    (erase-area-xy w offsetx offsety sizex sizey 8)
    (sw = (string-width w s))
    (printat-xy w s (offsetx + (sizex - sw) / 2)
		    (offsety + (sizey - 10) / 2) )
    (force-output w)))

; 20 Nov 91; 03 Dec 91; 26 Dec 93
; Draw a thermometer-like icon
(gldefun window-draw-thermometer ((w window) (nmin integer) (ndel integer)
					    (ndiv integer) (val number)
					    (offsetx integer) (offsety integer)
					    (sizey integer))
  (let (hdel marky)
    (erase-area-xy w offsetx offsety 66 sizey)
    (editors-print-in-box val w offsetx offsety 40 20)
    (draw-arc-xy w (offsetx + 12) (offsety + 36) 12 12 132 276)
    (draw-line-xy w (offsetx + 4) (offsety + 44)
		    (offsetx + 4) (offsety + sizey - 8) )
    (draw-line-xy w (offsetx + 20) (offsety + 44)
		    (offsetx + 20) (offsety + sizey - 8) )
    (draw-arc-xy w (offsetx + 12) (offsety + sizey - 8) 8 8 0 180)
    (draw-circle-xy w (offsetx + 12) (offsety + 36) 4 7)
    (hdel = (sizey - 56) / ndiv)
    (draw-line-xy w (offsetx + 12) (offsety + 35)
		    (offsetx + 12)
		    (offsety + 48 + hdel * ((val - nmin) / ndel)) 7)
    (dotimes (i (1+ ndiv))
      (marky = (offsety + 48 + i * hdel))
      (draw-line-xy w (offsetx + 24) marky (offsetx + 34) marky)
      (printat-xy w (nmin + i * ndel) (offsetx + 36) (marky - 6)) )
    (force-output w)))


; 20 Nov 91; 03 Dec 91; 13 Apr 95
; Draw value for a thermometer-like icon
(gldefun window-set-thermometer ((w window) (nmin integer) (ndel integer)
					   (ndiv integer) (val number)
					   (offsetx integer) (offsety integer)
					   (sizey integer))
  (let (hdel)
    (hdel = (sizey - 56) / ndiv)
    (erase-area-xy w (offsetx + 7) (offsety + 48)
		     10 (sizey - 56))
    (draw-line-xy w (offsetx + 12) (offsety + 35)
		    (offsetx + 12)
		    (offsety + 48 + hdel * ((val - nmin) / ndel)) 7)
    (editors-update-in-box val w offsetx offsety 40 20))))


; 20 Nov 91; 03 Dec 91; 15 Oct 93; 02 Dec 93; 08 Jan 04
; Adjust a thermometer-like icon with the mouse.  Returns new value.
(gldefun window-adjust-thermometer ((w window) (nmin integer) (ndel integer)
					      (ndiv integer) (val number)
					      (offsetx integer) (offsety integer)
					      (sizey integer))
  (let (hdel (lasty integer) xmin xmax ymin ymax inside (newval number))
    (hdel = (sizey - 56) / ndiv)
    (lasty = (truncate (offsety + 48 + hdel * ((val - nmin) / ndel))))
    (xmin = offsetx + 4)
    (xmax = offsetx + 20)
    (ymin = offsety + 48)
    (ymax = offsety + sizey - 8)
    (window-track-mouse w 
	    #'(lambda (x y code)
		(inside = (and (>= x xmin) (<= x xmax)
				 (>= y ymin) (<= y ymax)))
		(when (and inside (/= y lasty))
		  (if (> y lasty)
		      (draw-line-xy w (offsetx + 12) lasty (offsetx + 12) y 7)
		      (erase-area-xy w (offsetx + 7) (y + 1)
					    10 (- lasty y)))
		  (lasty = y)
		  (newval = ( ( (lasty - (offsety + 48))
				  / (float hdel)) * ndel) + nmin)
		  (if (integerp val) (newval = (truncate newval)))
		  (editors-update-in-box newval w offsetx offsety 40 20))
		(not (zerop code))))
    (if inside
	newval
        val)  ))

; 20 Nov 91; 15 Oct 93; 08 Jan 04; 26 Jan 06
; Get a mouse selection from a button area.  cf. picmenu-select
(gldefun button-select ((mw window) (buttons (listof picmenu-button)))
  (let ((current-button picmenu-button) item items (val picmenu-button)
	   xzero yzero inside)
    (xzero = 0) ; (menu-x m 0)
    (yzero = 0) ; (menu-y m 0)
    (track-mouse mw
      #'(lambda (x y code)
	  (x = (x - xzero))
	  (y = (y - yzero))
	  (if ((x >= 0) and (y >= 0))
	      (inside = t))
	  (if current-button
	      (if ~ (button-containsxy? current-button x y)
		 (progn (button-invert mw current-button)
		       (current-button = nil))))
	  (if ~ current-button
	      (progn (items = buttons)
	           (while ~ current-button and (item -_ items) do
			  (if (button-containsxy? item x y)
			      (progn (current-button = item)
				     (button-invert mw current-button) )))))
	  (if (> code 0)
	      (progn (if current-button
			 (button-invert mw current-button) )
		     (val = (or current-button *picmenu-no-selection*)) )))
      t)
    (if (val <> *picmenu-no-selection*) (buttonname val)) ))

; 03 Dec 91
(gldefun button-invert ((w window) (button picmenu-button))
  (window-invert-area w (offset button) (size button)) )

(gldefun window-undraw-box ((w window) offset size &optional lw)
  (set-erase w)
  (window-draw-box w offset size lw)
  (unset w) )

; 20 Nov 91; 08 Jan 04
(gldefun button-containsxy? ((b picmenu-button) (x integer) (y integer))
  (let ((xsize 6) (ysize 6))
    (if (size b)
	(progn (xsize = (x (size b)))
	       (ysize = (y (size b)))))
    ((x >= (x (offset b))) and (x <= ((x (offset b)) + xsize)) and
     (y >= (y (offset b))) and (y <= ((y (offset b)) + ysize)) ) ))


(glispobjects

(menu-item (z anything)
  prop ((value      ((if z is atomic
			 z
			 (cdr z)))) )
  msg  ((print-size menu-item-print-size)
	(draw       menu-item-draw)) )

) ; glispobjects

(gldefun menu-item-print-size ((item menu-item) (w window))
  (result vector)
  (let (siz)
    (if item is atomic
        (a vector with x = (string-width w item) y = 11)
        (if (car item) is a string
	    (a vector with x = (string-width w (car item)) y = 11)
	    (if ((symbolp (car item))
			   and (siz = (get (car item) 'display-size)))
		siz
	        (a vector with x = 50 y = 11)))) ))

; 17 Dec 91; 08 Jan 04
(gldefun menu-item-draw ((item menu-item) (w window)
					 (offsetx integer) (offsety integer)
					 (sizex integer) (sizey integer))
    (if item is atomic
        (window-center-print w item offsetx offsety sizex sizey)
        (if ((symbolp (car item)) and (fboundp (car item)))
	    (funcall (car item) w offsetx offsety)
	    (window-center-print w (car item) offsetx offsety
					   sizex sizey))) )

; 03 Dec 91; 26 Dec 93; 08 Jan 04
(gldefun pick-one-size ((items (listof menu-item)) (w window))
  (let (wid)
    (for item in items do
      (wid = (if wid
		 (max wid (x (print-size item w)))
		 (x (print-size item w))) ) )
    (a vector with x = wid y = 11) ))

; 03 Dec 91; 26 Dec 93; 29 Jul 94; 28 Feb 02
(gldefun draw-pick-one ((items (listof menu-item)) (val anything) (w window)
				 &optional (offsetx integer) (offsety integer)
				           (sizex integer) (sizey integer))
  (let (itm)
    (if (itm = (that item with (value (that item)) == val))
	(draw itm w offsetx offsety sizex sizey))))

; 04 Dec 91; 26 Dec 93; 29 Jul 94; 08 Jan 04
(gldefun edit-pick-one ((items (listof menu-item)) (val anything) (w window)
				 &optional (offsetx integer) (offsety integer)
				           (sizex integer) (sizey integer))
  (let (newval)
    (if ((length items) <= 3)
	(if (equal val (value (first items)))
	    (newval = (value (second items)))
	    (if (equal val (value (second items)))
		(newval = (if (third items)
			      (value (third items))
			      (value (first items))))
	        (newval = (value (first items)))))
        (newval = (menu items)) )
    (draw-pick-one newval w items offsetx offsety sizex sizey)
    newval  ))


; 13 Dec 91; 26 Dec 93; 28 Jul 94; 28 Feb 02; 08 Jan 04
(gldefun draw-black-white ((items (listof menu-item)) (val anything) (w window)
				 &optional (offsetx integer) (offsety integer)
				           (sizex integer) (sizey integer))
  (let (itm)
    (erase-area-xy w offsetx offsety sizex sizey)
    (if (itm = (that item with (value (that item)) == val))
        (if (eql (if (consp itm) 
		     (car itm)
		     itm)
		 1)
	    (invert-area-xy w offsetx offsety sizex sizey)) ) ))

; 13 Dec 91; 15 Dec 91; 26 Dec 93; 28 Jul 94; 08 Jan 04
(gldefun edit-black-white ((items (listof menu-item)) (val anything) (w window)
				 &optional (offsetx integer) (offsety integer)
				           (sizex integer) (sizey integer))
  (let (newval)
    (if (equal val (value (first items)))
	(newval = (value (second items)))
        (if (equal val (value (second items)))
	    (newval = (value (first items)))))
    (draw-black-white items newval w offsetx offsety sizex sizey)
    newval  ))

; 23 Dec 91; 26 Dec 93
(gldefun draw-integer ((val integer) (w window)
				 &optional (offsetx integer) (offsety integer)
				           (sizex integer) (sizey integer))
  (editors-anything-print val w offsetx offsety sizex sizey)  )

; 24 Dec 91; 26 Dec 93
(defun draw-real (val w &optional offsetx offsety sizex sizey)
  (let (str nc lng fmt)
    (if (null sizex) (setq sizex 50))
    (setq nc (max 1 (truncate sizex 7)))
    (setq str (princ-to-string val))
    (setq lng (length str))
    (if (> lng nc)
	(if (or (find #\. str :start nc)
		(find #\E str)
		(find #\L str))
	    (if (>= nc 8)
		(progn (setq fmt (cadr (or (assoc nc '((8 "~8,2E")
						 (9 "~9,2E")   (10 "~10,2E")
						 (11 "~11,2E") (12 "~12,2E")
						 (13 "~13,2E") (14 "~14,2E")))
					   '(15 "~15,2E"))))
		       (setq str (format nil fmt val)))
		(setq str "*******"))
	    (setq str (subseq str 0 nc)) ))
    (editors-anything-print w str offsetx offsety sizex sizey)  ))

; 09 Dec 91; 10 Dec 91; 23 Dec 91; 26 Dec 93; 22 Jul 94
; Display function for use when a more specific one is not found.
(gldefun editors-anything-print (obj (w window) offsetx offsety sizex sizey)
  (let ((s (stringify obj)) swidth smax dx dy)
    (erase-area-xy w offsetx offsety sizex sizey)
    (swidth = (string-width w s))
    (smax = (min swidth sizex))
    (dx = (sizex - smax) / 2)
    (dy = (max 0 ((sizey - 10) / 2)))
    (printat-xy w (editors-string-limit obj w smax)
		(offsetx + dx) (offsety + dy))
   ))

; 26 Dec 93
(gldefun editors-print-in-box (obj (w window) offsetx offsety sizex sizey)
  (printat-xy w (editors-string-limit obj w sizex)
	      (offsetx + 4) (offsety + (sizey - 10) / 2))
  (draw-box-xy w offsetx offsety sizex sizey)  )

; 26 Dec 93
(gldefun editors-update-in-box (obj (w window) offsetx offsety sizex sizey)
  (erase-area-xy w (offsetx + 3) (offsety + 3) (sizex - 6) (sizey - 6))
  (printat-xy w (editors-string-limit obj w sizex)
	      (offsetx + 4) (offsety + (sizey - 10) / 2)) )

; 28 Oct 91; 26 Dec 93; 08 Jan 04
; Limit string to a specified number of pixels
(gldefun editors-string-limit ((s string) (w window) (max integer))
  (result string)
  (let ((str (stringify s)) (lng integer) (nc integer))
    (lng = (string-width w str))
    (if (lng > max)
	(progn (nc = (((length str) * max) / lng))
	       (subseq str 0 nc))
        str) ))

(defvar *edit-color-menu-set* nil)
(defvar *edit-color-rmenu* nil)
(defvar *edit-color-old-color* nil)
(glispglobals (*edit-color-menu-set* menu-set)
	      (*edit-color-rmenu* barmenu))

; 03 Jan 94; 04 Jan 94; 05 Jan 94; 08 Dec 08
(gldefun edit-color-init ((w window))
  (let (rm gm bm rgb)
    (rgb = (a rgb))
    ;; (glcc 'edit-color-red)
    ;; (glcc 'edit-color-green)
    ;; (glcc 'edit-color-blue)
    (*edit-color-menu-set* = (menu-set-create w nil))
    (rm = (barmenu-create 256 200 10 "" nil #'edit-color-red (list rgb) w
			    120 40 nil t (a rgb with red = 65535)))
    (*edit-color-rmenu* = rm)
    (gm = (barmenu-create 256 50 10 "" nil #'edit-color-green (list rgb) w
			    170 40 nil t (a rgb with green = 65535)))
    (bm = (barmenu-create 256 250 10 "" nil #'edit-color-blue (list rgb) w
			    220 40 nil t (a rgb with blue = 65535)))
    (add-barmenu *edit-color-menu-set* 'red   nil rm "Red"   '(120 40))
    (add-barmenu *edit-color-menu-set* 'green nil gm "Green" '(170 40))
    (add-barmenu *edit-color-menu-set* 'blue  nil bm "Blue"  '(220 40))
    (add-menu *edit-color-menu-set* 'done nil "" '(("Done" . done)) '(30 150))
    (edit-color-red   200 rgb)
    (edit-color-green  50 rgb)
    (edit-color-blue  250 rgb)
  ))

; 03 Jan 94; 04 Jan 94
(gldefun edit-color-red ((val integer) (color rgb))
  (let ((w (window *edit-color-menu-set*)))
    (printat-xy w (format nil "~3D" val) 113 20)
    ((red color) = (max 0 (val * 256 - 1)))
    (edit-display-color w color) ))

; 03 Jan 94; 04 Jan 94
(gldefun edit-color-green ((val integer) (color rgb))
  (let ((w (window *edit-color-menu-set*)))
    (printat-xy w (format nil "~3D" val) 163 20)
    ((green color) = (max 0 (val * 256 - 1)))
    (edit-display-color w color) ))

; 03 Jan 94; 04 Jan 94
(gldefun edit-color-blue ((val integer) (color rgb))
  (let ((w (window *edit-color-menu-set*)))
    (printat-xy w (format nil "~3D" val) 213 20)
    ((blue color) = (max 0 (val * 256 - 1)))
    (edit-display-color w color) ))

; 03 Jan 94
(gldefun edit-display-color ((w window) (color rgb))
  (window-set-color w color)
  (window-draw-line-xy w 50 40 50 100 60)
  (window-reset-color w)
  (if *edit-color-old-color* (window-free-color w *edit-color-old-color*))
  (*edit-color-old-color* = *window-xcolor*) )

; 03 Jan 94; 04 Jan 94; 05 Jan 94; 28 Feb 02
(gldefun edit-color ((w window))
  (let (done (color rgb) sel)
    (if (or (null *edit-color-menu-set*)
	    (not (eq w (menu-window (menu (first (menu-items
						  *edit-color-menu-set*)))))))
	(edit-color-init w))
    (color = (first (subtrackparms *edit-color-rmenu*)))
    (draw *edit-color-menu-set*)
    (edit-color-red   (truncate (1+ (red color)) 256) color)
    (edit-color-green (truncate (1+ (green color)) 256) color)
    (edit-color-blue  (truncate (1+ (blue color)) 256) color)
    (while ~ done
      (sel = (select *edit-color-menu-set*))
      (done = (and sel ((first sel) == 'done))) )
    color))

; 08 Dec 08
(gldefun color-dot ((w window) (x integer) (y integer) (color symbol))
  (let (rgb)
    (setq rgb (cdr (assoc color '((red 65535 0 0)
                                  (yellow 65535 57600 0)
                                  (green 0 50175 12287)
                                  (blue 0 0 65535)))))
    (or rgb (setq rgb '(30000 30000 30000)))
    (set-color w rgb)
    (draw-dot-xy w x y)
    (reset-color w) ))

; 15 Oct 93; 26 Jan 06
; Compile the editors.lsp file into a plain Lisp file
(defun compile-editors ()
  (glcompfiles *directory*
	       '("glisp/vector.lsp"          ; auxiliary files
                 "X/dwindow.lsp")
	       '("glisp/editors.lsp")        ; translated files
	       "glisp/editorstrans.lsp"         ; output file
	       "glisp/gpl.txt")      ; header file
  (cf editorstrans) )

; Compile the editors.lsp file into a plain Lisp file for XGCL
(defun compile-editorsb ()
  (glcompfiles *directory*
	       '("glisp/vector.lsp"          ; auxiliary files
                 "X/dwindow.lsp" "X/dwnoopen.lsp")
	       '("glisp/editors.lsp")        ; translated files
	       "glisp/editorstrans.lsp"         ; output file
	       "glisp/gpl.txt")      ; header file
  )
