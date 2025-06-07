(in-package :XLIB)
; Xakcl.example.lsp        Hiep Huu Nguyen                      27 Aug 92

; Copyright (c) 1994 Hiep Huu Nguyen and The University of Texas at Austin.

; See the files gnu.license and dec.copyright .

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

; Some of the files that interface to the Xlib are adapted from DEC/MIT files.
; See the file dec.copyright for details.

;;;;;;;;;;;;;;;;;;;;;;
;;this is an example of getting a geometry feature of a drawable there
;;is also XGetWindowAttributes for just windows.  See reference manual
;;on X lib.  it is probably more efficient to use XGetGeometry function
;;once when a lot of geometry information is needed since, XGetGeometry
;;returns many values.  also as can be noticed, XGetGeometry needs C
;;Pointers, so it is best to allocate these pointers as globals so that
;;they won't have to be created and destroyed all the time, taking time
;;and fragmenting memory

(defun drawable-height (a-drawable &key (display *default-display*))
  (XGetGeometry display a-drawable *root-return* *x-return* *y-return* *width-return* 
		  *height-return* *border-width-return* *depth-return*)
    (int-pos   *height-return* 0))



;;;;;;;;;;;;;;;;;;;;;;
;;this function is a simple application of line drawing. it uses the
;;drawable-height function and the default globals like
;;*default-display* and *default-GC*

(defun graph-x-y (info &key (test #'first) (scale 10) (displ 0) (invert t))

  (let* ((info (sort info #'< :key test))
	 (first-x-y (first info))
	 (prev-x (* (first first-x-y)  scale))
	 (mid-height ( / (drawable-height a-window) 2))
	 (prev-y (if invert 
		     (-  mid-height (* (+ (second first-x-y)  displ) scale))
		     (* (+ (second first-x-y) displ) scale))))
    (print info)
    (dolist (next-x-y (rest info))
      (let ((pres-x  (* (first next-x-y) scale))
	    (pres-y  (if invert 
			 (-  mid-height (* (+ (second next-x-y)  displ) scale))
			 (* (+ (second next-x-y) displ) scale))))

	;;	    (format t "~%prev-x : ~a prev-y: ~a pres-x: ~a pres-y: ~a" prev-x prev-y pres-x pres-y)
	(Xdrawline   *default-display* a-window  *default-GC*
		     prev-x prev-y pres-x pres-y)
	(Xflush  *default-display*)
	(setq prev-x pres-x)
	(setq prev-y pres-y)))))



;;;;;;;;;;;;;;;;;;;;;;
;; here's an example of getting values stored in a certain GC
;; the structure XGCValues contain values for a GC
(defun get-foreground-of-gc (display GC) 
  (XGetGCValues display GC (+ GCForeground) *GC-Values*)
  (XGCValues-foreground  *GC-Values*))


;;;;;;;;;;;;;;;;;;;;;;
;;this is an example of changing the graphics context and allocating a
;;color for drawing.  this is also an example of setting the line
;;attributes this function changes the graphics context so becareful.
;;also notice that c-types Xcolor is created and freed.  again it is
;;possible to make them global, because they could be used often.  this
;;function was fixed to have no side effects.  Side effects are a danger
;;with passing C structures.  the structures could be changed as a side
;;effect if you're not careful

(defun my-draw-line (&key (display *default-display*) (GC  *default-GC*) x1 y1 x2 y2 (width 0) (color "BLACK")
			  (line-style LineSolid) (cap-style CapRound) (join-style JoinRound) (colormap *default-colormap*)
			  window)

  (let ((pixel-xcolor (make-Xcolor))
	(exact-rgb  (make-Xcolor))
	(prev-fore-pixel (get-foreground-of-gc display GC)))
    (XSetLineAttributes display GC width line-style cap-style join-style)
    (XAllocNamedColor display colormap  (get-c-string color) pixel-xcolor exact-rgb)
    (Xsetforeground display GC (Xcolor-pixel  pixel-xcolor))
    (XDrawLine  display  window GC x1 y1 x2 y2)
    (Xflush display)
    (free pixel-xcolor)
    (free exact-rgb)
    (XSetForeground display GC prev-fore-pixel)))



(defun colors ()
  (let ((pixel-xcolor (make-Xcolor))
	(y 0)
	(r 0)
	(b 0)
	(g 0))
    (dotimes (g 65535)
;;	  (format t "~% ~a ~a ~a" r b g)
      (set-Xcolor-red pixel-xcolor r)
      (set-Xcolor-blue pixel-xcolor b)
      (set-Xcolor-green pixel-xcolor g)
      (if (not (eql 0 (XallocColor *default-display* *default-colormap* pixel-xcolor)))
	  (progn (Xsetforeground  *default-display* *default-GC* (Xcolor-pixel  pixel-xcolor))
		 (XDrawLine  *default-display* a-window *default-GC*  0 0 200 y)
		 (Xflush *default-display*)
		 (incf y 1))
	  ;;	      (format t "~%error in reading color")
	  ))))
     
    
(defun return-r-b-g (color &key (display *default-display*) (GC  *default-GC*) (colormap *default-colormap*)
			 )
   (let ((pixel-xcolor (make-Xcolor))
	(exact-rgb  (make-Xcolor)))
	(XAllocNamedColor display colormap  (get-c-string color) pixel-xcolor pixel-xcolor)
	(format t "~% red: ~a  blue: ~a  green: ~a" (Xcolor-red pixel-xcolor)
		 (Xcolor-blue pixel-xcolor) (Xcolor-green pixel-xcolor))))
   
;;;;;;;;;;;;;;;;;;;;;;
;;this function tracks the mouse.  when the mouse button is pressed a
;;line is drawn from the previous position to the current position.
;;this function also shows a way of handling exposure events.  The
;;positions are remembered in order to redraw the contents of the window
;;when it is exposed.  this function handles events in two windows, the
;;quit window and the draw window.  there is an example of setting the
;;input for a window.  the draw window can have button press events,
;;pointer motion events and exposure events, while the quit window
;;(button) only needs button press events, and exposure events.  notice
;;that the event queue is actually flushed at the beginng of the
;;functions.  There is also an example of drawing and inverting text.
;;and handling sub windows.  the sub windows are destroyed at the end of
;;the function.

(defun track-mouse (a-window)
    (Xsync *default-display* 1)                    ;; this clears the event queue so that previous 
                                                   ;; motion events won't show up 
    (XClearWindow  *default-display* a-window)
   
    ;; create two sub window

    (let ((quit-window (XCreateSimpleWindow
			*default-display*  a-window
			2 2 50 20 1  *black-pixel*  *white-pixel*))
	  (draw-window (XCreateSimpleWindow
			*default-display*  a-window
			2 32 220 350 1  *black-pixel*  *white-pixel*)))
      (Xselectinput *default-display* quit-window (+  ButtonpressMask  ExposureMask))
      (Xselectinput *default-display* draw-window 
		    (+ ButtonpressMask PointerMotionMask ExposureMask))
 
      (XMapWindow   *default-display*  quit-window) 
      (XMapWindow   *default-display*  draw-window)
      (Xflush   *default-display* ) 
      (XDrawString  *default-display*   quit-window  *default-GC*  10 15 (get-c-string "Quit") 4)
      (Xflush   *default-display* ) 
      (do ((exit nil)
	   (lines-list nil)  
	   (prev-x nil)
	   (prev-y nil))
	(exit)
	(XNextEvent  *default-display*  *default-event*)
	(let ((type (XAnyEvent-type  *default-event*))
	      (active-window (XAnyevent-window  *default-event*)))
	  (cond ((eql draw-window active-window)
		 (cond 	
;;; draw a line
		  ((eql type ButtonPress)
		   (let ((x (XButtonEvent-x  *default-event*))
			 (y (XButtonEvent-y  *default-event*)))	      
		     (if prev-x
			 (XDrawLine *default-display* draw-window  *default-GC*  prev-x prev-y x y))
		     (setq prev-x x)
		     (setq prev-y y)
		     (push (list x y) lines-list)))
;;; track the mouse
		  ((eql type MotionNotify)
		   (let ((x (XMotionEvent-x  *default-event*))
			 (y (XMotionEvent-y  *default-event*))
			 (time (XmotionEvent-time *default-event*)))
		     ;;trace the mouse
		     ;;(format t "~% pos-x: ~a  pos-y: ~a" x y)
		     ;;(format t "~%time: ~a" time)
		     ))

;;;; redraw window after expose event

		  ((eql type Expose)
		   (let* ((first-xy (first lines-list))
			  (prev-x (first first-xy))
			  (prev-y (second first-xy)))
		     (dolist (an-xy (rest lines-list))
		       (let ((x (first an-xy))
			     (y (second an-xy)))
			 (XDrawLine *default-display* draw-window  *default-GC*  prev-x prev-y x y)
			 (setq prev-x x)
			 (setq prev-y y)))))))

		;; exit if the quit button is pressed

		((eql quit-window active-window)
		 (cond ((eql type ButtonPress)
			(setq exit t)
			(XSetForeground *default-display* 
					*default-GC* *white-pixel*)
			(XSetBackground *default-display* 
					*default-GC* *black-pixel*)
			(XDrawImageString  *default-display*   quit-window  *default-GC*  10 15 (get-c-string "Quit") 4)
			(Xflush *default-display*)

;;the drawing goes so fast that you can't see the text invert, so the
;;function wiats for for about .2 seconds.  but it would be better to
;;keep the text inverted until the button is released this is done by
;;setting the quit window to have button release events as well and
;;handling it appropriately

			(dotimes (i 1500))


			(XSetForeground *default-display* 
					*default-GC*  *black-pixel*)
			(XSetBackground *default-display* 
					*default-GC* *white-pixel*) 
			(XDrawImageString  *default-display*   quit-window  *default-GC*  10 15 (get-c-string "Quit") 4)
			(Xflush *default-display*))

;; do quit window expose event
		       ((eql type Expose) 
			(XDrawString  *default-display*   quit-window  *default-GC*  10 15 (get-c-string "Quit") 4)))))))
      (XDestroySubWindows *default-display* a-window)
      (Xflush *default-display*)))
    

;;;;;;;;;;;;;;;;;;;;;;
;;this function demonstrtes using different fonts of text

(defun basic-text (a-window &key (display *default-display*) (GC *default-GC* ))
  (my-load-font "9x15"  :display display  :GC GC)
  (Xdrawstring  display  a-window  GC 50 100  (get-c-string "hello") 5)
  (my-load-font "*-*-courier-bold-r-*-*-12-*-*-*-*-*-iso8859-1" :display display  :GC GC)
  (Xdrawstring  display  a-window  GC 50 150  (get-c-string "hello") 5)
  (Xflush display))


;;;;;;;;;;;;;;;;;;;;;;
;;this function demonstartes getting different fonts and setting them in a GC

(defun my-load-font (a-string  &key (display *default-display*) (GC *default-GC* ))
  (let ((font-info  (XloadQueryFont  display 	(get-c-string a-string))))
    (if (not (eql 0 font-info))
	(XsetFont  display GC (Xfontstruct-fid font-info))
	(format t "~%can't open font ~a" a-string))))


;;;;;;;;;;;;;;;;;;;;;;
;;this function draws a ghst line by setting the X function to GXXor. and the
;;foreground color to th logxor of the back and foreground pixel
;;this function actually changes the graphics context. and does not change it back
;;to use the ghost method and switch back to regular drawing. set the function
;;back to GXcopy and the foregorund pixel appropriately

(defun do-ghost-line-1 (a-window)
  (Xsync *default-display* 1);; this clears the event queue so that previous 
  ;; motion events won't show up 
  (XClearWindow  *default-display* a-window)
   
  (XdrawRectangle *default-display* a-window *default-GC* 
		  0 0 100 100)
  (Xdrawarc  *default-display* a-window  *default-GC* 100 200 100 100 0 (* 360 64))

  (Xsetfunction  *default-display* *default-GC* GXxor)
  (Xsetforeground  *default-display* *default-GC* (logxor *black-pixel* *white-pixel*))
  (Xselectinput *default-display* a-window    PointerMotionMask )
  (do ((exit nil)
       (prev-x 0)
       (prev-y 0))
    (exit)
    (XNextEvent  *default-display*  *default-event*)
    (let ((type (XAnyEvent-type  *default-event*)))
      (cond

       ;;draw ghost line
       ((eql type MotionNotify)
	(let ((x (XMotionEvent-x  *default-event*))
	      (y (XMotionEvent-y  *default-event*))
	      (time (XmotionEvent-time *default-event*)))
	  (Xdrawline  *default-display* a-window *default-GC* 0 0 prev-x prev-y)
	  (Xdrawline  *default-display* a-window *default-GC* 0 0 x y)
	  (setq prev-x x)
	  (setq prev-y y)
	  ))))))





  ;;example of a circle 
  ;;position 100 100 diameter 100

  ;;(XdrawArc *default-display* a-window  *default-GC* 100 100 100 100 0 (* 360 64))

  ;;example of font

  ;;(XloadFont *default-display* (get-c-string "8x10"))



  ;; set a pixel 

  ;;(XallocNamedColor *default-display* *default-colormap* (get-c-string "aquamarine") a b)
